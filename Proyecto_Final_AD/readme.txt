Predicción de eventos adversos de medicamentos (openFDA)

Proyecto que construye un pipeline completo desde una API médica hasta una app con IA:

API → ETL → MongoDB → Dataset → Modelo → App

Tecnologías
- Python
- MongoDB
- Prefect
- Scikit-learn (Random Forest)
- Streamlit

Ejecución paso a paso

1. Cargar datos desde la API:
python openfda_pipeline.py --pages 30 --limit 100

2. (Opcional) Ejecutar con Prefect:
prefect server start
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
python prefect_openfda_flow.py --pages 30 --limit 100

3. Construir dataset:
python build_ml_dataset.py

4. Entrenar modelo:
python train_random_forest.py

5. Ejecutar app:
streamlit run app.py

Resultado

El modelo predice si un evento adverso será serio o no serio, mostrando:
- probabilidad
- variables importantes
- visualizaciones del dataset

Flujo final

openFDA → ETL → Mongo → Dataset → Random Forest → Streamlit

Proyecto académico
Administración de Datos  
Fuente: openFDA Drug Adverse Event
