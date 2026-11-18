# Detector de Anomalías en Tráfico de Red

Sistema de detección de intrusiones orientado a equipos SOC/Blue Team. Combina un clasificador supervisado de DDoS con un detector no supervisado de anomalías, y expone ambos mediante una API REST en FastAPI.

## 1. Problema de negocio

Los SOC corporativos y bancarios enfrentan dos problemas simultáneos: la fatiga por alertas (demasiados falsos positivos) y los ataques desconocidos que ningún modelo entrenado ha visto antes.

Este proyecto aborda ambos con una arquitectura híbrida sobre el dataset **CIC-IDS-2017**:

- **Modelo supervisado:** clasifica tráfico benigno vs. DDoS con alta precisión.
- **Modelo no supervisado:** aprende únicamente el comportamiento benigno y marca desviaciones, incluyendo ataques sin etiqueta.
- **Evaluación de generalización:** mide cuánto se degrada el clasificador ante un ataque no visto (PortScan).
- **API de inferencia dual:** entrega una señal combinada con nivel de riesgo operativo.

## 2. Arquitectura modular

| Módulo | Responsabilidad |
| --- | --- |
| `src/preprocessing.py` | `CICIDSPreprocessor`: limpieza única y reutilizable (strip de headers, `±inf`→`NaN`, `dropna`, `drop_duplicates`), validación de 78 features numéricas y etiquetado configurable (`BENIGN=0`, `positive_labels→1`). |
| `src/train.py` | `SupervisedTrainer`: split estratificado 80/20, `RandomForestClassifier` de 200 árboles, métricas y persistencia del artefacto. |
| `src/evaluate_generalization.py` | `GeneralizationEvaluator`: evalúa el Random Forest entrenado en DDoS directamente sobre PortScan, sin reentrenar. |
| `src/train_unsupervised.py` | `UnsupervisedAnomalyTrainer`: ajusta `StandardScaler` e `IsolationForest` solo con filas BENIGN del train. |
| `src/api.py` | Servicio FastAPI con carga de modelos en `lifespan`, validación estricta de payload y endpoints `/health`, `/predict`, `/metrics`. |
| `src/metrics.py` | `ModelMetrics` y cálculo compartido de precisión, recall, F1 y matriz de confusión. |
| `src/artifacts.py` | Carga segura de artefactos y validación del orden exacto de `feature_names`. |
| `tests/test_api.py` | Suite de integración con `TestClient` sobre los endpoints y la validación de entrada. |

Decisiones de diseño:

- El preprocesamiento es una única fuente de verdad para entrenamiento e inferencia, parametrizable con `positive_labels`.
- La deduplicación ocurre **antes** del split para evitar *data leakage*.
- Los artefactos incluyen el orden exacto de las 78 features; la API rechaza cualquier modelo incompatible.
- La API carga ambos modelos una sola vez al arrancar y valida la compatibilidad entre ellos.

## 3. Datos

Se utilizan dos archivos de CIC-IDS-2017, no versionados en el repositorio:

- `data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`

Cada flujo contiene 79 columnas: 78 features numéricas y la etiqueta `Label`. Tras limpiar y deduplicar, el conjunto DDoS queda en 223,082 filas; el de PortScan en 213,777.

## 4. Resultados reales

Split común: `test_size=0.2`, `random_state=42`, estratificado.

| Experimento | Modelo | Evaluado en | Precisión | Recall | F1 | TN | FP | FN | TP |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervisado DDoS | Random Forest (200) | Test DDoS | 1.0000 | 0.9999 | 0.9999 | 19,013 | 1 | 3 | 25,600 |
| Generalización DDoS→PortScan | Random Forest (sin reentrenar) | PortScan | 0.8615 | 0.0006 | 0.0012 | 123,074 | 9 | 90,638 | 56 |
| No supervisado | Isolation Forest (contaminación 0.05) | Test DDoS | 0.8092 | 0.1654 | 0.2747 | 18,015 | 999 | 21,367 | 4,236 |
| No supervisado | Isolation Forest (mismo modelo) | PortScan | 0.0191 | 0.0013 | 0.0024 | 117,219 | 5,864 | 90,580 | 114 |

**Hallazgo principal:** el clasificador supervisado pasa de un recall de 0.9999 en DDoS a 0.0006 en PortScan, una caída de **99.93 puntos porcentuales**. El modelo aprende patrones específicos de DDoS y es prácticamente ciego ante un ataque que no vio durante el entrenamiento.

## 5. Hallazgos técnicos y lecciones de ingeniería

- **Data leakage:** CIC-IDS-2017 contiene flujos duplicados. Deduplicar antes del split evita que el test comparta filas con el entrenamiento y reduce métricas infladas.
- **Domain shift:** un ataque nuevo cambia la distribución de features. La generalización cero de DDoS→PortScan demuestra que un clasificador supervisado no sustituye la detección de anomalías.
- **Arquitectura híbrida:** el Random Forest aporta precisión sobre ataques conocidos; Isolation Forest aporta cobertura ante comportamiento desconocido. La API combina ambas señales en `BAJO`, `MEDIO` y `ALTO`.
- **Costo de la detección no supervisada:** con 5% de contaminación, Isolation Forest genera 999 falsos positivos en benignos para detectar el 16.5% de los DDoS. Es una señal complementaria, no un reemplazo.
- **Limitaciones:** el dataset es histórico y no refleja tráfico actual; faltan autenticación, rate limiting, monitoreo de drift y validación temporal (split por tiempo en lugar de aleatorio).

## 6. API REST

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/health` | Estado del servicio, modelos cargados y features esperadas (78). |
| `POST` | `/predict` | Inferencia dual en una llamada; devuelve predicción, probabilidad, anomalía, score, riesgo y latencia. |
| `GET` | `/metrics` | Métricas consolidadas de los tres experimentos. |

Ejemplo de respuesta de `/predict`:

```json
{
  "supervised_prediction": "BENIGN",
  "ddos_probability": 0.01,
  "unsupervised_anomaly": false,
  "anomaly_score": -0.12,
  "risk_level": "BAJO",
  "latency_ms": 3.42
}
```

Contrato de entrada:

- `features` debe contener **exactamente** las 78 features del modelo, con valores finitos.
- Faltantes, features desconocidas, `NaN` o `inf` devuelven HTTP 422 con detalle en español.
- `risk_level` es `ALTO` si ambas señales alertan, `MEDIO` si alerta una y `BAJO` si ninguna.

Documentación interactiva disponible en `http://127.0.0.1:8000/docs`.

## 7. Guía rápida

### Instalación

```powershell
python -m venv venv
.\venv\Scripts\python -m pip install -r requirements.txt
```

Requiere Python 3.13. Coloca los CSV en `data/raw/` antes de entrenar.

### Entrenamiento y evaluación

```powershell
.\venv\Scripts\python -m src.train
.\venv\Scripts\python -m src.evaluate_generalization
.\venv\Scripts\python -m src.train_unsupervised
```

Artefactos generados en `models/`:

- `rf_ddos_model.joblib`, `rf_metrics.json`
- `generalization_metrics.json`
- `iforest_anomaly_model.joblib`, `iforest_metrics.json`

### Ejecución de la API

```powershell
.\venv\Scripts\python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

### Pruebas

```powershell
.\venv\Scripts\python -m pytest -v
```

La suite cubre health, predicción válida, features faltantes, features no reconocidas, valores `NaN`/`inf` y consolidación de métricas.
