from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_PATH = Path("random_forest_openfda.pkl")
DATASET_PATH = Path("drug_event_model_ready.csv")
IMPORTANCE_PATH = Path("random_forest_feature_importance.csv")

NUMERIC_FEATURES = [
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
    "report_year",
    "report_month",
]

CATEGORICAL_PREFIXES = {
    "administration_routes": "administration_routes_",
    "occountry": "occountry_",
    "patient_onset_age_unit": "patient_onset_age_unit_",
    "patient_sex": "patient_sex_",
}

IMPORTANT_DEFAULTS = {
    "patient_onset_age_raw": 45,
    "num_medicinal_products_text": 1,
    "num_reactions": 2,
    "num_drugs": 2,
    "num_suspect_drugs": 1,
    "num_concomitant_drugs": 0,
    "num_interacting_drugs": 0,
    "report_year": 2025,
    "report_month": 4,
    "has_age": 1,
    "has_country": 1,
    "has_weight": 0,
}

CHART_BG = "#120d12"
PLOT_BG = "#181118"
TEXT = "#f6f0f0"
ACCENT = "#ff6b3d"
ACCENT_2 = "#ff944d"
ACCENT_SOFT = "#ffb38f"
GRID = "rgba(255,255,255,0.10)"


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_reference_dataset(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


@st.cache_data
def load_importance(csv_path: str) -> pd.DataFrame | None:
    p = Path(csv_path)
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data
def infer_feature_schema(df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    feature_columns = [c for c in df.columns if c != "target_serious"]
    category_options: Dict[str, List[str]] = {}

    for logical_name, prefix in CATEGORICAL_PREFIXES.items():
        raw_values = []
        for col in feature_columns:
            if col.startswith(prefix):
                raw_values.append(col[len(prefix) :])
        category_options[logical_name] = sorted(raw_values)

    return feature_columns, category_options


@st.cache_data
def summarize_reference(reference_df: pd.DataFrame, importance_df: pd.DataFrame | None) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    summary["rows"] = int(len(reference_df))
    summary["target_share"] = (
        float(reference_df["target_serious"].mean()) if "target_serious" in reference_df.columns else None
    )
    summary["serious_counts"] = (
        reference_df["target_serious"].value_counts().rename(index={0: "No serio", 1: "Serio"}).reset_index()
        if "target_serious" in reference_df.columns
        else pd.DataFrame(columns=["target_serious", "count"])
    )
    if not summary["serious_counts"].empty:
        summary["serious_counts"].columns = ["Clase", "Cantidad"]

    numeric_candidates = [
        "patient_onset_age_raw",
        "num_reactions",
        "num_drugs",
        "num_suspect_drugs",
        "num_medicinal_products_text",
    ]
    means = (
        reference_df[numeric_candidates]
        .mean(numeric_only=True)
        .rename(
            {
                "patient_onset_age_raw": "Edad",
                "num_reactions": "Reacciones",
                "num_drugs": "Medicamentos",
                "num_suspect_drugs": "Sospechosos",
                "num_medicinal_products_text": "Productos texto",
            }
        )
        .round(2)
        .reset_index()
    )
    means.columns = ["Variable", "Promedio"]
    summary["numeric_means"] = means
    summary["top_features"] = importance_df.head(8).copy() if importance_df is not None and not importance_df.empty else None
    return summary



def build_input_row(feature_columns: List[str], form_values: Dict[str, object]) -> pd.DataFrame:
    row = {col: 0 for col in feature_columns}

    for col in NUMERIC_FEATURES:
        if col in row:
            row[col] = form_values.get(col, 0)

    for logical_name, prefix in CATEGORICAL_PREFIXES.items():
        selected_value = str(form_values.get(logical_name, ""))
        dummy_col = f"{prefix}{selected_value}"
        if dummy_col in row:
            row[dummy_col] = 1

    return pd.DataFrame([row], columns=feature_columns)



def label_for_option(value: str) -> str:
    mapping = {
        "": "No especificado",
        "1": "Masculino (1)",
        "2": "Femenino (2)",
        "800": "Décadas / años agrupados (800)",
        "801": "Años (801)",
        "802": "Meses (802)",
        "803": "Semanas (803)",
        "804": "Días (804)",
        "805": "Horas (805)",
        "806": "Minutos (806)",
        "807": "Segundos (807)",
    }
    return mapping.get(value, value)



def yes_no_to_int(value: str) -> int:
    return 1 if value == "Sí" else 0



def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #09070b 0%, #130d14 100%);
            color: {TEXT};
        }}
        h1, h2, h3, h4, .stMarkdown, label, p, div {{
            color: {TEXT};
        }}
        .accent-text {{ color: {ACCENT}; }}
        .mini-card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,107,61,0.28);
            border-radius: 16px;
            padding: 14px 16px;
            margin-bottom: 10px;
        }}
        .result-serious {{
            background: rgba(255,107,61,0.12);
            border: 1px solid {ACCENT};
            border-radius: 16px;
            padding: 18px;
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffb29b;
        }}
        .result-safe {{
            background: rgba(255,255,255,0.04);
            border: 1px solid {ACCENT_2};
            border-radius: 16px;
            padding: 18px;
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffd0c2;
        }}
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,107,61,0.28);
            padding: 10px 14px;
            border-radius: 16px;
        }}
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {{
            color: {TEXT} !important;
        }}
        .stButton > button, .stFormSubmitButton > button {{
            background: linear-gradient(90deg, {ACCENT} 0%, {ACCENT_2} 100%);
            color: white;
            border: 0;
            border-radius: 12px;
            font-weight: 700;
        }}
        .stTabs [data-baseweb="tab"] {{ color: {TEXT}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )



def apply_plot_style(fig):
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=PLOT_BG,
        font_color=TEXT,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig



def main():
    st.set_page_config(page_title="openFDA Serious Event Predictor", page_icon="💊", layout="wide")
    inject_custom_css()

    st.markdown("<h1><span class='accent-text'>💊</span> Predicción de severidad de eventos adversos</h1>", unsafe_allow_html=True)
    st.caption("Interfaz simplificada para predecir si un reporte openFDA se clasifica como serio o no serio.")

    missing = [str(p) for p in [MODEL_PATH, DATASET_PATH] if not p.exists()]
    if missing:
        st.error("Faltan archivos necesarios en esta misma carpeta: " + ", ".join(missing))
        st.stop()

    model = load_model(str(MODEL_PATH))
    reference_df = load_reference_dataset(str(DATASET_PATH))
    feature_columns, category_options = infer_feature_schema(reference_df)
    importance_df = load_importance(str(IMPORTANCE_PATH))
    summary = summarize_reference(reference_df, importance_df)

    dashboard_tab, predictor_tab = st.tabs(["Mini dashboard", "Predicción"])

    with dashboard_tab:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Registros", f"{summary['rows']:,}")
        with m2:
            target_share = summary["target_share"]
            st.metric("Proporción de casos serios", f"{target_share:.1%}" if target_share is not None else "N/D")
        with m3:
            st.metric("Variables del modelo", len(feature_columns))

        c1, c2 = st.columns([0.95, 1.05])
        with c1:
            st.markdown("### Distribución del target")
            serious_counts = summary["serious_counts"]
            if not serious_counts.empty:
                fig_target = px.bar(
                    serious_counts,
                    x="Clase",
                    y="Cantidad",
                    color="Clase",
                    color_discrete_sequence=[ACCENT_2, ACCENT],
                    text="Cantidad",
                )
                fig_target.update_traces(textposition="outside")
                fig_target.update_layout(showlegend=False)
                st.plotly_chart(apply_plot_style(fig_target), use_container_width=True)

        with c2:
            st.markdown("### Promedios del dataset")
            numeric_means = summary["numeric_means"]
            fig_means = px.line_polar(
                numeric_means,
                r="Promedio",
                theta="Variable",
                line_close=True,
            )
            fig_means.update_traces(fill="toself", line_color=ACCENT, fillcolor="rgba(255,107,61,0.25)")
            fig_means.update_layout(
                polar=dict(
                    bgcolor=PLOT_BG,
                    radialaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                    angularaxis=dict(gridcolor=GRID, tickfont=dict(color=TEXT)),
                )
            )
            st.plotly_chart(apply_plot_style(fig_means), use_container_width=True)

        st.markdown("### Variables más influyentes")
        top_features = summary.get("top_features")
        if top_features is not None:
            fig_imp = px.bar(
                top_features.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[ACCENT_SOFT, ACCENT],
            )
            fig_imp.update_layout(coloraxis_showscale=False)
            st.plotly_chart(apply_plot_style(fig_imp), use_container_width=True)
        else:
            st.info("No se encontró el archivo de importancia de variables.")

    with predictor_tab:
        left, right = st.columns([1.05, 0.95])

        with left:
            st.markdown("## Entradas principales")

            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)

                with c1:
                    patient_onset_age_raw = st.number_input(
                        "Edad de inicio del evento",
                        min_value=0,
                        max_value=120,
                        value=int(IMPORTANT_DEFAULTS["patient_onset_age_raw"]),
                        step=1,
                    )
                    num_reactions = st.number_input(
                        "Cantidad de reacciones",
                        min_value=0,
                        max_value=50,
                        value=int(IMPORTANT_DEFAULTS["num_reactions"]),
                        step=1,
                    )
                    num_drugs = st.number_input(
                        "Cantidad total de medicamentos",
                        min_value=0,
                        max_value=50,
                        value=int(IMPORTANT_DEFAULTS["num_drugs"]),
                        step=1,
                    )

                with c2:
                    num_suspect_drugs = st.number_input(
                        "Medicamentos sospechosos",
                        min_value=0,
                        max_value=20,
                        value=int(IMPORTANT_DEFAULTS["num_suspect_drugs"]),
                        step=1,
                    )
                    num_concomitant_drugs = st.number_input(
                        "Medicamentos concomitantes",
                        min_value=0,
                        max_value=50,
                        value=int(IMPORTANT_DEFAULTS["num_concomitant_drugs"]),
                        step=1,
                    )
                    num_medicinal_products_text = st.number_input(
                        "Productos medicinales en texto",
                        min_value=0,
                        max_value=50,
                        value=int(IMPORTANT_DEFAULTS["num_medicinal_products_text"]),
                        step=1,
                    )

                with c3:
                    has_age_text = st.selectbox("¿Tiene edad?", ["Sí", "No"], index=0)
                    patient_sex = st.selectbox(
                        "Sexo del paciente",
                        options=category_options.get("patient_sex", [""]),
                        format_func=label_for_option,
                    )
                    patient_onset_age_unit = st.selectbox(
                        "Unidad de edad",
                        options=category_options.get("patient_onset_age_unit", [""]),
                        index=0 if "801" not in category_options.get("patient_onset_age_unit", []) else category_options.get("patient_onset_age_unit", []).index("801"),
                        format_func=label_for_option,
                    )

                st.markdown("### Contexto del reporte")
                c4, c5, c6 = st.columns(3)
                with c4:
                    administration_routes = st.selectbox(
                        "Ruta de administración",
                        options=category_options.get("administration_routes", [""]),
                        format_func=label_for_option,
                    )
                with c5:
                    report_year = st.number_input(
                        "Año del reporte",
                        min_value=2000,
                        max_value=2035,
                        value=int(IMPORTANT_DEFAULTS["report_year"]),
                        step=1,
                    )
                with c6:
                    report_month = st.number_input(
                        "Mes del reporte",
                        min_value=1,
                        max_value=12,
                        value=int(IMPORTANT_DEFAULTS["report_month"]),
                        step=1,
                    )

                with st.expander("Ajustes avanzados"):
                    a1, a2, a3 = st.columns(3)
                    with a1:
                        has_country_text = st.selectbox("¿Tiene país?", ["Sí", "No"], index=0)
                        occountry = st.selectbox(
                            "País del reporte",
                            options=category_options.get("occountry", [""]),
                            format_func=label_for_option,
                        )
                    with a2:
                        has_weight_text = st.selectbox("¿Tiene peso?", ["Sí", "No"], index=1)
                        num_interacting_drugs = st.number_input(
                            "Medicamentos con interacción",
                            min_value=0,
                            max_value=20,
                            value=int(IMPORTANT_DEFAULTS["num_interacting_drugs"]),
                            step=1,
                        )
                    with a3:
                        st.markdown(
                            "<div class='mini-card'><b>Tip:</b><br>Si no conoces un dato, déjalo con el valor por defecto.</div>",
                            unsafe_allow_html=True,
                        )

                submitted = st.form_submit_button("Predecir severidad", use_container_width=True)

        with right:
            st.markdown("## Resumen rápido")
            st.markdown(
                "<div class='mini-card'><b>Objetivo</b><br>Clasificar si un evento adverso reportado tiende a ser serio o no serio.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='mini-card'><b>Modelo</b><br>Random Forest entrenado con 5000 registros curados de openFDA.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div class='mini-card'><b>Uso</b><br>Se muestran solo las variables más relevantes para que la predicción sea rápida de probar.</div>",
                unsafe_allow_html=True,
            )

            if importance_df is not None and not importance_df.empty:
                st.markdown("### Top 5 variables")
                top5 = importance_df.head(5).copy()
                fig_top5 = px.pie(top5, names="feature", values="importance", hole=0.45, color_discrete_sequence=[ACCENT, ACCENT_2, "#ffb38f", "#ff7f50", "#ffd0c2"])
                fig_top5.update_traces(textinfo="percent+label")
                fig_top5.update_layout(paper_bgcolor=CHART_BG, font_color=TEXT, margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig_top5, use_container_width=True)

        if submitted:
            form_values = {
                "patient_onset_age_raw": int(patient_onset_age_raw),
                "has_age": yes_no_to_int(has_age_text),
                "has_country": yes_no_to_int(has_country_text),
                "has_weight": yes_no_to_int(has_weight_text),
                "num_reactions": int(num_reactions),
                "num_drugs": int(num_drugs),
                "num_suspect_drugs": int(num_suspect_drugs),
                "num_concomitant_drugs": int(num_concomitant_drugs),
                "num_interacting_drugs": int(num_interacting_drugs),
                "num_medicinal_products_text": int(num_medicinal_products_text),
                "report_year": int(report_year),
                "report_month": int(report_month),
                "administration_routes": administration_routes,
                "occountry": occountry,
                "patient_onset_age_unit": patient_onset_age_unit,
                "patient_sex": patient_sex,
            }

            input_df = build_input_row(feature_columns, form_values)
            pred = int(model.predict(input_df)[0])
            proba = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else None

            st.markdown("---")
            st.markdown("## Resultado")
            r1, r2 = st.columns([1.2, 0.8])
            with r1:
                if pred == 1:
                    st.markdown("<div class='result-serious'>El modelo predice: CASO SERIO</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='result-safe'>El modelo predice: CASO NO SERIO</div>", unsafe_allow_html=True)
            with r2:
                if proba is not None:
                    st.metric("Probabilidad de caso serio", f"{proba:.2%}")

            with st.expander("Ver registro que entra al modelo"):
                st.dataframe(input_df, use_container_width=True)

            with st.expander("Explicación simple"):
                st.write(
                    "La predicción se apoya sobre todo en edad, cantidad de medicamentos, cantidad de reacciones y ruta de administración."
                )


if __name__ == "__main__":
    main()
