import os
from datetime import datetime, timezone
from typing import Iterable, Dict, Any, List, Optional

import requests
import pandas as pd
from pymongo import MongoClient, ASCENDING, UpdateOne
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "Api_openfda")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "Drug_event_data")
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")

OPENFDA_EVENT_URL = "https://api.fda.gov/drug/event.json"
DEFAULT_SEARCH_QUERY = None
DEFAULT_SORT = "receivedate:desc"
DEFAULT_LIMIT = 100
MAX_LIMIT = 1000
MAX_SKIP = 25000


# -------------------------
# 1) EXTRACT - openFDA Drug Adverse Event
# -------------------------
def fetch_adverse_events(
    search_query: Optional[str] = DEFAULT_SEARCH_QUERY,
    limit: int = DEFAULT_LIMIT,
    skip: int = 0,
    sort: Optional[str] = DEFAULT_SORT,
    api_key: str = OPENFDA_API_KEY,
) -> Dict[str, Any]:
    """
    Extrae reportes desde openFDA Drug Adverse Event.

    Ejemplos de search_query:
    - None  -> trae los reportes más recientes (si usas sort)
    - 'patient.reaction.reactionmeddrapt:"HEADACHE"'
    - 'serious:1+AND+receivedate:[20250101+TO+20251231]'
    """
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"limit debe estar entre 1 y {MAX_LIMIT}. Valor recibido: {limit}")
    if skip < 0 or skip > MAX_SKIP:
        raise ValueError(f"skip debe estar entre 0 y {MAX_SKIP}. Valor recibido: {skip}")

    params: Dict[str, Any] = {
        "limit": limit,
        "skip": skip,
    }
    if search_query:
        params["search"] = search_query
    if sort:
        params["sort"] = sort
    if api_key:
        params["api_key"] = api_key

    r = requests.get(OPENFDA_EVENT_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


# -------------------------
# 2) HELPERS - flatten / limpieza básica
# -------------------------
def _safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []



def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()



def _join_unique(values: Iterable[Any], sep: str = " | ") -> str:
    seen = set()
    out: List[str] = []
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return sep.join(out)



def _flag_01(value: Any) -> int:
    return 1 if str(value).strip() == "1" else 0



def _to_datetime_yyyymmdd(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce")


# -------------------------
# 3) STAGE - dict/json -> DataFrame temporal
# -------------------------
def to_staging_df(
    adverse_event_json: Dict[str, Any],
    search_query: Optional[str] = None,
    page_number: int = 1,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """
    Convierte la respuesta JSON a un DataFrame "staging" (temporal).
    Aplana la estructura anidada a nivel de safety report.
    """
    results = adverse_event_json.get("results", []) or []
    meta = adverse_event_json.get("meta", {}) or {}
    meta_results = meta.get("results", {}) or {}

    rows: List[Dict[str, Any]] = []
    ingested_at_utc = datetime.now(timezone.utc).replace(microsecond=0)

    for report in results:
        patient = report.get("patient", {}) or {}
        primarysource = report.get("primarysource", {}) or {}

        reactions = _safe_list(patient.get("reaction"))
        drugs = _safe_list(patient.get("drug"))

        reaction_terms = [r.get("reactionmeddrapt") for r in reactions if isinstance(r, dict)]
        medicinal_products = [d.get("medicinalproduct") for d in drugs if isinstance(d, dict)]
        drug_indications = [d.get("drugindication") for d in drugs if isinstance(d, dict)]
        administration_routes = [d.get("drugadministrationroute") for d in drugs if isinstance(d, dict)]
        drug_roles = [str(d.get("drugcharacterization", "")).strip() for d in drugs if isinstance(d, dict)]

        row = {
            "safetyreportid": _clean_text(report.get("safetyreportid")),
            "companynumb": _clean_text(report.get("companynumb")),
            "occurcountry": _clean_text(report.get("occurcountry")),
            "receivedate": _clean_text(report.get("receivedate")),
            "receiptdate": _clean_text(report.get("receiptdate")),
            "transmissiondate": _clean_text(report.get("transmissiondate")),
            "primarysource_reportercountry": _clean_text(primarysource.get("reportercountry")),
            "primarysource_qualification": _clean_text(primarysource.get("qualification")),
            "patient_onset_age_raw": _clean_text(patient.get("patientonsetage")),
            "patient_onset_age_unit": _clean_text(patient.get("patientonsetageunit")),
            "patient_sex": _clean_text(patient.get("patientsex")),
            "patient_weight": _clean_text(patient.get("patientweight")),
            "serious": _flag_01(report.get("serious")),
            "seriousness_death": _flag_01(report.get("seriousnessdeath")),
            "seriousness_life_threatening": _flag_01(report.get("seriousnesslifethreatening")),
            "seriousness_hospitalization": _flag_01(report.get("seriousnesshospitalization")),
            "seriousness_disabling": _flag_01(report.get("seriousnessdisabling")),
            "seriousness_congenital_anomali": _flag_01(report.get("seriousnesscongenitalanomali")),
            "seriousness_other": _flag_01(report.get("seriousnessother")),
            "num_reactions": len(reaction_terms),
            "num_drugs": len(drugs),
            "num_suspect_drugs": sum(1 for x in drug_roles if x == "1"),
            "num_concomitant_drugs": sum(1 for x in drug_roles if x == "2"),
            "num_interacting_drugs": sum(1 for x in drug_roles if x == "3"),
            "reaction_terms": _join_unique(reaction_terms),
            "medicinal_products": _join_unique(medicinal_products),
            "drug_indications": _join_unique(drug_indications),
            "administration_routes": _join_unique(administration_routes),
            "api_page_number": int(page_number),
            "api_limit": int(limit),
            "api_total_records": meta_results.get("total"),
            "api_last_updated": meta.get("last_updated"),
            "search_query_used": search_query or "",
            "ingested_at_utc": ingested_at_utc,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "safetyreportid",
                "companynumb",
                "occurcountry",
                "receivedate",
                "receiptdate",
                "transmissiondate",
                "primarysource_reportercountry",
                "primarysource_qualification",
                "patient_onset_age_raw",
                "patient_onset_age_unit",
                "patient_sex",
                "patient_weight",
                "serious",
                "seriousness_death",
                "seriousness_life_threatening",
                "seriousness_hospitalization",
                "seriousness_disabling",
                "seriousness_congenital_anomali",
                "seriousness_other",
                "num_reactions",
                "num_drugs",
                "num_suspect_drugs",
                "num_concomitant_drugs",
                "num_interacting_drugs",
                "reaction_terms",
                "medicinal_products",
                "drug_indications",
                "administration_routes",
                "api_page_number",
                "api_limit",
                "api_total_records",
                "api_last_updated",
                "search_query_used",
                "ingested_at_utc",
            ]
        )

    return df


# -------------------------
# 4) TRANSFORM - limpieza + tipos + features
# -------------------------
def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transformación típica:
    - Tipos
    - Fechas
    - Limpieza básica
    - Features útiles para el modelo y análisis
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Fechas openFDA vienen como YYYYMMDD
    out["receivedate_dt"] = _to_datetime_yyyymmdd(out["receivedate"])
    out["receiptdate_dt"] = _to_datetime_yyyymmdd(out["receiptdate"])
    out["transmissiondate_dt"] = _to_datetime_yyyymmdd(out["transmissiondate"])

    # Numéricos
    numeric_cols = [
        "patient_onset_age_raw",
        "patient_weight",
        "num_reactions",
        "num_drugs",
        "num_suspect_drugs",
        "num_concomitant_drugs",
        "num_interacting_drugs",
        "api_page_number",
        "api_limit",
        "api_total_records",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # Limpieza mínima
    out["safetyreportid"] = out["safetyreportid"].astype(str).str.strip()
    out = out[out["safetyreportid"].ne("")].copy()
    out = out.drop_duplicates(subset=["safetyreportid"]).reset_index(drop=True)

    # Features simples para análisis / modelo
    out["report_year"] = out["receivedate_dt"].dt.year
    out["report_month"] = out["receivedate_dt"].dt.month
    out["has_age"] = out["patient_onset_age_raw"].notna().astype(int)
    out["has_weight"] = out["patient_weight"].notna().astype(int)
    out["has_country"] = out["occurcountry"].fillna("").ne("").astype(int)
    out["num_reaction_terms_text"] = out["reaction_terms"].fillna("").str.count(r"\|") + out["reaction_terms"].fillna("").ne("").astype(int)
    out["num_medicinal_products_text"] = out["medicinal_products"].fillna("").str.count(r"\|") + out["medicinal_products"].fillna("").ne("").astype(int)

    # Target sugerido para Random Forest
    out["target_serious"] = out["serious"].astype(int)

    final_cols = [
        "safetyreportid",
        "companynumb",
        "occurcountry",
        "primarysource_reportercountry",
        "primarysource_qualification",
        "receivedate",
        "receiptdate",
        "transmissiondate",
        "receivedate_dt",
        "receiptdate_dt",
        "transmissiondate_dt",
        "patient_onset_age_raw",
        "patient_onset_age_unit",
        "patient_sex",
        "patient_weight",
        "serious",
        "seriousness_death",
        "seriousness_life_threatening",
        "seriousness_hospitalization",
        "seriousness_disabling",
        "seriousness_congenital_anomali",
        "seriousness_other",
        "num_reactions",
        "num_drugs",
        "num_suspect_drugs",
        "num_concomitant_drugs",
        "num_interacting_drugs",
        "reaction_terms",
        "medicinal_products",
        "drug_indications",
        "administration_routes",
        "report_year",
        "report_month",
        "has_age",
        "has_weight",
        "has_country",
        "num_reaction_terms_text",
        "num_medicinal_products_text",
        "target_serious",
        "api_page_number",
        "api_limit",
        "api_total_records",
        "api_last_updated",
        "search_query_used",
        "ingested_at_utc",
    ]
    return out[final_cols]


# -------------------------
# 5) LOAD - DataFrame -> MongoDB (upsert + index)
# -------------------------
def ensure_indexes(collection) -> None:
    """
    Índices principales para evitar duplicados y facilitar consultas.
    """
    collection.create_index(
        [("safetyreportid", ASCENDING)],
        unique=True,
        name="uniq_safetyreportid",
    )
    collection.create_index([("receivedate_dt", ASCENDING)], name="idx_receivedate_dt")
    collection.create_index([("target_serious", ASCENDING)], name="idx_target_serious")
    collection.create_index([("occurcountry", ASCENDING)], name="idx_occurcountry")



def _to_python_datetime(value: Any) -> Any:
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else value



def load_to_mongo(
    df: pd.DataFrame,
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = COLLECTION_NAME,
) -> Dict[str, int]:
    """
    Inserta/actualiza en Mongo usando upsert por safetyreportid.
    """
    if df is None or df.empty:
        return {"matched": 0, "modified": 0, "upserted": 0}

    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[collection_name]

    ensure_indexes(coll)

    df2 = df.drop_duplicates(subset=["safetyreportid"]).copy()

    ops: List[UpdateOne] = []
    for row in df2.to_dict(orient="records"):
        doc = dict(row)
        for date_col in ["receivedate_dt", "receiptdate_dt", "transmissiondate_dt", "ingested_at_utc"]:
            if date_col in doc:
                doc[date_col] = _to_python_datetime(doc[date_col])

        flt = {"safetyreportid": doc["safetyreportid"]}
        upd = {
            "$set": doc,
            "$setOnInsert": {"created_at_utc": datetime.now(timezone.utc).replace(microsecond=0)},
        }
        ops.append(UpdateOne(flt, upd, upsert=True))

    if not ops:
        return {"matched": 0, "modified": 0, "upserted": 0}

    res = coll.bulk_write(ops, ordered=False)
    return {
        "matched": int(res.matched_count),
        "modified": int(res.modified_count),
        "upserted": int(len(res.upserted_ids) if res.upserted_ids else 0),
    }


# -------------------------
# Orquestación simple (sin Prefect aún)
# -------------------------
def run_etl(
    search_query: Optional[str] = DEFAULT_SEARCH_QUERY,
    pages: int = 1,
    limit: int = DEFAULT_LIMIT,
    sort: Optional[str] = DEFAULT_SORT,
    mongo_uri: str = MONGO_URI,
    db_name: str = DB_NAME,
    collection_name: str = COLLECTION_NAME,
    api_key: str = OPENFDA_API_KEY,
) -> Dict[str, Any]:
    """
    Corre el ETL para una búsqueda en openFDA.

    pages=3 y limit=100 traerá hasta 300 reportes.
    """
    all_dfs = []
    fetched_pages = 0

    for page_number in range(1, pages + 1):
        skip = (page_number - 1) * limit
        api_json = fetch_adverse_events(
            search_query=search_query,
            limit=limit,
            skip=skip,
            sort=sort,
            api_key=api_key,
        )
        staging = to_staging_df(
            api_json,
            search_query=search_query,
            page_number=page_number,
            limit=limit,
        )
        clean = transform(staging)

        if clean.empty:
            break

        all_dfs.append(clean)
        fetched_pages += 1

        # Si llegó menos de limit, ya no hay más páginas útiles.
        if clean.shape[0] < limit:
            break

    if not all_dfs:
        return {
            "rows_total": 0,
            "mongo": {"matched": 0, "modified": 0, "upserted": 0},
            "pages_fetched": fetched_pages,
            "search_query": search_query,
        }

    final_df = pd.concat(all_dfs, ignore_index=True)
    mongo_stats = load_to_mongo(
        final_df,
        mongo_uri=mongo_uri,
        db_name=db_name,
        collection_name=collection_name,
    )

    return {
        "rows_total": int(final_df.shape[0]),
        "mongo": mongo_stats,
        "pages_fetched": fetched_pages,
        "search_query": search_query,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ETL openFDA Drug Adverse Event -> MongoDB")
    parser.add_argument("--search", default=None, help="Query openFDA. Ej: 'serious:1' o 'patient.reaction.reactionmeddrapt:\"HEADACHE\"'")
    parser.add_argument("--pages", type=int, default=1, help="Cantidad de páginas a traer (default: 1)")
    parser.add_argument("--limit", type=int, default=100, help=f"Registros por página (1..{MAX_LIMIT})")
    parser.add_argument("--sort", default=DEFAULT_SORT, help="Orden openFDA. Ej: receivedate:desc")

    args = parser.parse_args()

    report = run_etl(
        search_query=args.search,
        pages=args.pages,
        limit=args.limit,
        sort=args.sort,
    )

    print("\n=== ETL REPORT ===")
    print("Search query:", report["search_query"])
    print("Pages fetched:", report["pages_fetched"])
    print("Rows total:", report["rows_total"])
    print("Mongo:", report["mongo"])
