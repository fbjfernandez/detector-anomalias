# AGENTS.md

## Project state
- Backend completo: EDA, entrenamiento supervisado, generalización cruzada, detección no supervisada, API FastAPI y suite de pruebas. No hay frontend ni CI/lint configurados.
- `README.md` es la referencia funcional y de métricas; `AGENTS.md` prioriza cómo trabajar sin romper el sistema.

## Layout
- `src/preprocessing.py` — `CICIDSPreprocessor`: strip de headers, `±inf`→`NaN`, `dropna`, `drop_duplicates`, 78 features numéricas y mapeo `BENIGN=0` + `positive_labels→1`. Default `("DDoS",)`; PortScan usa `("PortScan",)`.
- `src/train.py` — `SupervisedTrainer`: split estratificado, RandomForest de 200 árboles y persistencia.
- `src/evaluate_generalization.py` — `GeneralizationEvaluator`: aplica el modelo DDoS a PortScan sin reentrenar.
- `src/train_unsupervised.py` — `UnsupervisedAnomalyTrainer`: `StandardScaler` + `IsolationForest` ajustados solo con filas BENIGN del train.
- `src/api.py` — servicio de inferencia dual con `lifespan`, esquemas Pydantic y endpoints `/health`, `/predict`, `/metrics`.
- `src/metrics.py` — `ModelMetrics` y `calculate_metrics`; reutilizar en lugar de recalcular métricas.
- `src/artifacts.py` — `load_model_artifact` y `validate_feature_order`; usar siempre al cargar modelos o puntuar un dataset nuevo.
- `tests/test_api.py` — pruebas de integración con `TestClient`.
- `data/` y `models/` están ignorados por git; los CSV se aportan localmente y `models/` se regenera.

## Commands
- Instalar dependencias: `venv\Scripts\python -m pip install -r requirements.txt`.
- Entrenar supervisado: `venv\Scripts\python -m src.train`.
- Evaluar generalización: `venv\Scripts\python -m src.evaluate_generalization`.
- Entrenar no supervisado: `venv\Scripts\python -m src.train_unsupervised`.
- Levantar API: `venv\Scripts\python -m uvicorn src.api:app --host 127.0.0.1 --port 8000`.
- Ejecutar pruebas: `venv\Scripts\python -m pytest -v` (deben pasar las 8).
- No hay CI, linter ni formatter. La verificación es ejecutar los comandos y comparar contra los baselines del README: recall DDoS ≈0.9999, PortScan cruzado ≈0.0006, Isolation Forest DDoS ≈0.165.

## API
- El `lifespan` carga `models/rf_ddos_model.joblib` y `models/iforest_anomaly_model.joblib`, valida 78 features y ejecuta `NetworkFlowRequest.configure(feature_names)`.
- Por eso `TestClient` debe usarse como context manager (`with TestClient(app):`); sin lifespan, un payload válido falla con "aún no ha cargado las características".
- `NetworkFlowRequest` exige exactamente las 78 features, sin `NaN`/`inf`. No eliminar el handler `validation_exception_handler`: sin él, FastAPI falla al serializar la respuesta 422 cuando el cuerpo contiene `inf`/`NaN`.
- `anomaly_score` es `-decision_function` (mayor = más anómalo). `risk_level`: ALTO si ambas señales alertan, MEDIO si una, BAJO si ninguna.
- `/metrics` devuelve las claves `supervised_ddos`, `generalization_portscan` y `unsupervised_isolation_forest`.
- La API resuelve rutas desde la raíz del proyecto; debe arrancarse desde ahí o con el paquete `src` importable.

## Environment
- Python 3.13; `requirements.txt` fija las versiones instaladas en `venv/` (incluye `pytest` y `httpx`).
- Datasets: `data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` y `...-PortScan.pcap_ISCX.csv`.
- El CSV crudo usa `' Label'` con espacio inicial; el preprocesador limpia headers y luego expone `'Label'`.
- Orden de limpieza: inf→NaN, `dropna()`, `drop_duplicates()`, split. Etiquetas no configuradas lanzan `ValueError`; añadir ataques vía `positive_labels`.

## Conventions
- Docs, mensajes de consola y errores en español.
- Sin comentarios ni docstrings en `.py`; nombres autoexplicativos.
- Manejo explícito de excepciones: `FileNotFoundError` para datasets/artefactos ausentes, `ValueError`/`TypeError` para datos inválidos, `RuntimeError` para fallos de lectura/escritura, `HTTPException` en la API.
- SOLID/SRP: una responsabilidad por clase, helpers privados `_metodo`.
- No versionar `data/`, `models/`, `venv/` ni `.pytest_cache/`.
