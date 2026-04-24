import os
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from prefect import flow, task
from prefect.logging import get_run_logger
from prefect.task_runners import ThreadPoolTaskRunner

import openfda_pipeline as op

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", getattr(op, "MONGO_URI", "mongodb://localhost:27017"))
DB_NAME = os.getenv("DB_NAME", getattr(op, "DB_NAME", "Api_openfda"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", getattr(op, "COLLECTION_NAME", "Drug_event_data"))
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", getattr(op, "OPENFDA_API_KEY", ""))
DEFAULT_SORT = getattr(op, "DEFAULT_SORT", "receivedate:desc")
DEFAULT_LIMIT = getattr(op, "DEFAULT_LIMIT", 100)


@task(retries=3, retry_delay_seconds=10, timeout_seconds=30)
def t_fetch_page(
    page_number: int,
    search_query: Optional[str],
    limit: int,
    sort: Optional[str],
    api_key: str,
) -> Dict[str, Any]:
    """
    Extrae una página de openFDA.
    """
    skip = (page_number - 1) * limit
    return op.fetch_adverse_events(
        search_query=search_query,
        limit=limit,
        skip=skip,
        sort=sort,
        api_key=api_key,
    )


@task
def t_stage(
    adverse_event_json: Dict[str, Any],
    page_number: int,
    search_query: Optional[str],
    limit: int,
) -> pd.DataFrame:
    """
    Usa op.to_staging_df().
    """
    return op.to_staging_df(
        adverse_event_json,
        search_query=search_query,
        page_number=page_number,
        limit=limit,
    )


@task
def t_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Usa op.transform().
    """
    return op.transform(df)


@task
def t_concat(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        return pd.DataFrame()
    non_empty = [df for df in dfs if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


@task(retries=3, retry_delay_seconds=10)
def t_load(df: pd.DataFrame, mongo_uri: str, db_name: str, collection_name: str) -> Dict[str, int]:
    """
    Usa op.load_to_mongo() (incluye upsert + índices).
    """
    if df is None or df.empty:
        return {"matched": 0, "modified": 0, "upserted": 0}

    return op.load_to_mongo(df, mongo_uri=mongo_uri, db_name=db_name, collection_name=collection_name)


@flow(
    name="openfda-drug-event-to-mongo",
    task_runner=ThreadPoolTaskRunner(max_workers=4),
    log_prints=True,
)
def openfda_flow(
    search_query: Optional[str] = None,
    pages: int = 1,
    limit: int = DEFAULT_LIMIT,
    sort: Optional[str] = DEFAULT_SORT,
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = COLLECTION_NAME,
    api_key: str = OPENFDA_API_KEY,
) -> Dict[str, Any]:
    """
    Orquestación:
    fetch (por página en paralelo) -> stage -> transform -> concat -> load
    """
    logger = get_run_logger()
    logger.info(f"DB={db_name} | COLLECTION={collection_name}")
    logger.info(f"search_query={search_query} | pages={pages} | limit={limit} | sort={sort}")

    futures = []
    for page_number in range(1, pages + 1):
        fetch_f = t_fetch_page.submit(page_number, search_query, limit, sort, api_key)
        stage_f = t_stage.submit(fetch_f, page_number, search_query, limit)
        clean_f = t_transform.submit(stage_f)
        futures.append(clean_f)

    dfs = [f.result() for f in futures]
    final_df = t_concat(dfs)
    mongo_stats = t_load(final_df, mongo_uri, db_name, collection_name)

    report = {
        "rows_total": int(final_df.shape[0]) if final_df is not None else 0,
        "mongo": mongo_stats,
        "pages_requested": pages,
        "search_query": search_query,
    }
    logger.info(f"REPORT: {report}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prefect Flow: openFDA Drug Adverse Event -> MongoDB")
    parser.add_argument("--search", default=None, help="Query openFDA. Ej: 'serious:1' o 'patient.reaction.reactionmeddrapt:\"HEADACHE\"'")
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sort", default=DEFAULT_SORT)

    args = parser.parse_args()

    openfda_flow(
        search_query=args.search,
        pages=args.pages,
        limit=args.limit,
        sort=args.sort,
    )
