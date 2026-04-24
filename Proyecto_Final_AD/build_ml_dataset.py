import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "Proyecto_Final_AD")
RAW_COLLECTION = os.getenv("COLLECTION_NAME", "cluster-drug-adverse")

CURATED_CSV = "drug_event_curated.csv"
MODEL_READY_CSV = "drug_event_model_ready.csv"


def keep_top_n(series: pd.Series, n: int = 10, other_label: str = "OTHER") -> pd.Series:
    series = series.fillna("UNKNOWN").astype(str).str.strip()
    top_vals = series.value_counts(dropna=False).head(n).index
    return series.where(series.isin(top_vals), other_label)


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    coll = db[RAW_COLLECTION]

    docs = list(
        coll.find(
            {},
            {
                "_id": 0,
                "safetyreportid": 1,
                "target_serious": 1,
                "serious": 1,
                "administration_routes": 1,
                "occountry": 1,
                "patient_onset_age_raw": 1,
                "patient_onset_age_unit": 1,
                "patient_sex": 1,
                "has_age": 1,
                "has_country": 1,
                "has_weight": 1,
                "num_reactions": 1,
                "num_drugs": 1,
                "num_suspect_drugs": 1,
                "num_concomitant_drugs": 1,
                "num_interacting_drugs": 1,
                "num_medicinal_products_text": 1,
                "receivedate": 1,
                "api_last_updated": 1,
            },
        )
    )

    df = pd.DataFrame(docs)

    if df.empty:
        raise ValueError("No hay documentos en Mongo para construir el dataset.")

    # ---------- TARGET ----------
    if "target_serious" not in df.columns:
        if "serious" in df.columns:
            df["target_serious"] = pd.to_numeric(df["serious"], errors="coerce")
        else:
            raise ValueError("No existe target_serious ni serious en la colección.")

    df["target_serious"] = pd.to_numeric(df["target_serious"], errors="coerce")
    df = df[df["target_serious"].isin([0, 1])].copy()
    df["target_serious"] = df["target_serious"].astype(int)

    # ---------- DUPLICADOS ----------
    if "safetyreportid" in df.columns:
        df = df.drop_duplicates(subset=["safetyreportid"]).copy()

    # ---------- NUMÉRICAS ----------
    numeric_cols = [
        "patient_onset_age_raw",
        "has_age",
        "has_country",
        "has_weight",
        "num_reactions",
        "num_drugs",
        "num_suspect_drugs",
        "num_concomitant_drugs",
        "num_interacting_drugs",
        "num_medicinal_products_text",
    ]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Imputación simple
    for col in [
        "patient_onset_age_raw",
        "num_reactions",
        "num_drugs",
        "num_suspect_drugs",
        "num_concomitant_drugs",
        "num_interacting_drugs",
        "num_medicinal_products_text",
    ]:
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0
        df[col] = df[col].fillna(median_val)

    for col in ["has_age", "has_country", "has_weight"]:
        df[col] = df[col].fillna(0).astype(int)

    # ---------- FECHA ----------
    date_source = None
    if "receivedate" in df.columns:
        date_source = "receivedate"
    elif "api_last_updated" in df.columns:
        date_source = "api_last_updated"

    if date_source:
        df[date_source] = pd.to_datetime(df[date_source], errors="coerce")
        df["report_year"] = df[date_source].dt.year.fillna(0).astype(int)
        df["report_month"] = df[date_source].dt.month.fillna(0).astype(int)
    else:
        df["report_year"] = 0
        df["report_month"] = 0

    # ---------- CATEGÓRICAS ----------
    categorical_cols = ["administration_routes", "occountry", "patient_onset_age_unit", "patient_sex"]

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()

    # Reducir cardinalidad
    df["administration_routes"] = keep_top_n(df["administration_routes"], n=10)
    df["occountry"] = keep_top_n(df["occountry"], n=20)
    df["patient_onset_age_unit"] = keep_top_n(df["patient_onset_age_unit"], n=10)
    df["patient_sex"] = keep_top_n(df["patient_sex"], n=5)

    # ---------- DATASET CURADO ----------
    curated_cols = [
        "safetyreportid",
        "target_serious",
        "patient_onset_age_raw",
        "patient_onset_age_unit",
        "patient_sex",
        "administration_routes",
        "occountry",
        "has_age",
        "has_country",
        "has_weight",
        "num_reactions",
        "num_drugs",
        "num_suspect_drugs",
        "num_concomitant_drugs",
        "num_interacting_drugs",
        "num_medicinal_products_text",
        "report_year",
        "report_month",
    ]

    curated_df = df[curated_cols].copy()
    curated_df.to_csv(CURATED_CSV, index=False)

    # ---------- MODEL READY ----------
    model_df = curated_df.drop(columns=["safetyreportid"]).copy()

    model_df = pd.get_dummies(
        model_df,
        columns=["patient_onset_age_unit", "patient_sex", "administration_routes", "occountry"],
        drop_first=False,
        dtype=int,
    )

    model_df.to_csv(MODEL_READY_CSV, index=False)

    print("\n=== DATASET BUILD REPORT ===")
    print("Rows:", len(model_df))
    print("Curated CSV:", CURATED_CSV)
    print("Model-ready CSV:", MODEL_READY_CSV)
    print("\nTarget distribution:")
    print(curated_df["target_serious"].value_counts(dropna=False))


if __name__ == "__main__":
    main()