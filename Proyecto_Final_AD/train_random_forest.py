import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def load_dataset(csv_path: str, target_col: str = "target_serious"):
    """
    Carga el dataset model-ready y separa X/y.
    """
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"No se encontró la columna target '{target_col}' en {csv_path}")

    if df.empty:
        raise ValueError("El dataset está vacío.")

    if df[target_col].isna().any():
        raise ValueError(f"La columna target '{target_col}' contiene valores nulos.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].astype(int).copy()

    # Aseguramos que todas las columnas predictoras sean numéricas
    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(
            "Todavía existen columnas no numéricas en el dataset model-ready: "
            + ", ".join(non_numeric)
        )

    return df, X, y



def train_model(X_train, y_train, random_state: int = 42):
    """
    Entrena un Random Forest base, suficientemente sólido para una primera versión del proyecto.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test):
    """
    Calcula métricas principales del modelo.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        metrics["roc_auc"] = None

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    return metrics, cm, report, y_pred, y_proba



def save_feature_importance(model, feature_names, output_csv: str, top_n: int = 25):
    """
    Guarda las importancias de variables en CSV.
    """
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(output_csv, index=False)
    return importance_df.head(top_n)



def save_predictions(X_test, y_test, y_pred, y_proba, output_csv: str):
    """
    Guarda predicciones del conjunto de prueba para revisión.
    """
    pred_df = X_test.copy()
    pred_df["target_real"] = y_test.values
    pred_df["target_predicho"] = y_pred
    if y_proba is not None:
        pred_df["probabilidad_clase_1"] = y_proba
    pred_df.to_csv(output_csv, index=False)



def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de Random Forest para drug_event_model_ready.csv")
    parser.add_argument("--input", default="drug_event_model_ready.csv", help="Ruta al CSV model-ready")
    parser.add_argument("--target", default="target_serious", help="Nombre de la columna target")
    parser.add_argument("--test-size", type=float, default=0.20, help="Proporción para test (default: 0.20)")
    parser.add_argument("--random-state", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument("--model-out", default="random_forest_openfda.pkl", help="Archivo de salida del modelo")
    parser.add_argument(
        "--importance-out",
        default="random_forest_feature_importance.csv",
        help="CSV de importancias de variables",
    )
    parser.add_argument(
        "--predictions-out",
        default="random_forest_test_predictions.csv",
        help="CSV con predicciones del set de prueba",
    )

    args = parser.parse_args()

    _, X, y = load_dataset(args.input, target_col=args.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = train_model(X_train, y_train, random_state=args.random_state)
    metrics, cm, report, y_pred, y_proba = evaluate_model(model, X_test, y_test)

    # Guardados
    joblib.dump(model, args.model_out)
    top_importance = save_feature_importance(model, X.columns.tolist(), args.importance_out)
    save_predictions(X_test, y_test, y_pred, y_proba, args.predictions_out)

    # Reporte en consola
    print("\n=== RANDOM FOREST TRAIN REPORT ===")
    print(f"Input dataset: {args.input}")
    print(f"Filas totales: {len(X)}")
    print(f"Columnas predictoras: {X.shape[1]}")
    print(f"Train size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")
    print(f"Model saved to: {Path(args.model_out).resolve()}")
    print(f"Feature importance CSV: {Path(args.importance_out).resolve()}")
    print(f"Predictions CSV: {Path(args.predictions_out).resolve()}")

    print("\nMétricas:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")

    print("\nMatriz de confusión:")
    print(cm)

    print("\nClassification report:")
    print(report)

    print("\nTop 15 variables más importantes:")
    print(top_importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
