import json
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, ClassVar, Literal

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.artifacts import load_model_artifact, validate_feature_order
from src.preprocessing import CICIDSPreprocessor


class NetworkFlowRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        description="Mapa con las 78 características numéricas del flujo de red"
    )
    expected_features: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def configure(cls, feature_names: list[str]) -> None:
        cls.expected_features = tuple(feature_names)

    @field_validator("features")
    @classmethod
    def validate_feature_map(cls, features: dict[str, float]) -> dict[str, float]:
        expected = set(cls.expected_features)
        if not expected:
            raise ValueError(
                "El servicio aún no ha cargado las características del modelo"
            )

        missing = sorted(expected.difference(features))
        if missing:
            raise ValueError(
                f"Faltan {len(missing)} características requeridas: {missing}"
            )

        unexpected = sorted(set(features).difference(expected))
        if unexpected:
            raise ValueError(f"Características no reconocidas: {unexpected}")

        non_finite = sorted(
            name for name, value in features.items() if not math.isfinite(value)
        )
        if non_finite:
            raise ValueError(
                "Características con valores no finitos (NaN o inf): "
                f"{non_finite}"
            )
        return features


class PredictionResponse(BaseModel):
    supervised_prediction: Literal["BENIGN", "DDoS"] = Field(
        description="Clasificación del modelo supervisado"
    )
    ddos_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Probabilidad estimada de tráfico DDoS",
    )
    unsupervised_anomaly: bool = Field(
        description="Indica si el modelo no supervisado marcó el flujo como anómalo"
    )
    anomaly_score: float = Field(
        description="Puntaje de anomalía; valores mayores indican mayor rareza"
    )
    risk_level: Literal["BAJO", "MEDIO", "ALTO"] = Field(
        description="Nivel de riesgo combinado de ambas señales"
    )
    latency_ms: float = Field(
        ge=0.0,
        description="Latencia de inferencia en milisegundos",
    )


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    models_loaded: dict[str, bool]
    expected_features: int


def _risk_level(supervised_prediction: str, unsupervised_anomaly: bool) -> str:
    supervised_alert = supervised_prediction == "DDoS"
    if supervised_alert and unsupervised_anomaly:
        return "ALTO"
    if supervised_alert or unsupervised_anomaly:
        return "MEDIO"
    return "BAJO"


def _read_metrics_file(path: Path) -> dict[str, Any]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se encontró el archivo de métricas: {path}",
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo leer el archivo de métricas '{path}': {error}",
        ) from error
    if not isinstance(content, dict):
        raise HTTPException(
            status_code=500,
            detail=f"El archivo de métricas '{path}' no contiene un objeto JSON válido",
        )
    return content


def _sanitize_non_finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _sanitize_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_non_finite(item) for item in value]
    return value


async def validation_exception_handler(
    request: Request, error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": _sanitize_non_finite(jsonable_encoder(error.errors()))},
    )


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

    supervised_artifact = load_model_artifact(models_dir / "rf_ddos_model.joblib")
    unsupervised_artifact = load_model_artifact(
        models_dir / "iforest_anomaly_model.joblib",
        required_keys=("model", "scaler", "feature_names"),
    )

    feature_names = list(supervised_artifact["feature_names"])
    if len(feature_names) != CICIDSPreprocessor.EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            f"El modelo supervisado expone {len(feature_names)} características; "
            f"se esperaban {CICIDSPreprocessor.EXPECTED_FEATURE_COUNT}"
        )
    validate_feature_order(feature_names, unsupervised_artifact["feature_names"])

    NetworkFlowRequest.configure(feature_names)
    application.state.supervised_model = supervised_artifact["model"]
    application.state.unsupervised_model = unsupervised_artifact["model"]
    application.state.scaler = unsupervised_artifact["scaler"]
    application.state.feature_names = feature_names
    application.state.models_dir = models_dir
    yield


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    models_loaded = {
        "supervised": getattr(request.app.state, "supervised_model", None)
        is not None,
        "unsupervised": getattr(request.app.state, "unsupervised_model", None)
        is not None,
    }
    feature_names = getattr(request.app.state, "feature_names", [])
    return HealthResponse(
        status="ok" if all(models_loaded.values()) else "error",
        models_loaded=models_loaded,
        expected_features=len(feature_names),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: NetworkFlowRequest, request: Request) -> PredictionResponse:
    started_at = time.perf_counter()
    feature_names = request.app.state.feature_names

    try:
        frame = pd.DataFrame(
            [[payload.features[name] for name in feature_names]],
            columns=feature_names,
        )
        ddos_probability = float(
            request.app.state.supervised_model.predict_proba(frame)[0][1]
        )
        scaled_features = request.app.state.scaler.transform(frame)
        unsupervised_anomaly = bool(
            request.app.state.unsupervised_model.predict(scaled_features)[0] == -1
        )
        anomaly_score = float(
            -request.app.state.unsupervised_model.decision_function(scaled_features)[0]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno durante la inferencia: {error}",
        ) from error

    latency_ms = (time.perf_counter() - started_at) * 1000
    supervised_prediction = "DDoS" if ddos_probability >= 0.5 else "BENIGN"

    return PredictionResponse(
        supervised_prediction=supervised_prediction,
        ddos_probability=ddos_probability,
        unsupervised_anomaly=unsupervised_anomaly,
        anomaly_score=anomaly_score,
        risk_level=_risk_level(supervised_prediction, unsupervised_anomaly),
        latency_ms=latency_ms,
    )


@router.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    models_dir = request.app.state.models_dir
    return {
        "supervised_ddos": _read_metrics_file(models_dir / "rf_metrics.json"),
        "generalization_portscan": _read_metrics_file(
            models_dir / "generalization_metrics.json"
        ),
        "unsupervised_isolation_forest": _read_metrics_file(
            models_dir / "iforest_metrics.json"
        ),
    }


def create_app() -> FastAPI:
    application = FastAPI(
        title="Detector de Anomalías en Tráfico de Red",
        version="1.0.0",
        description=(
            "Servicio de inferencia dual para clasificación de DDoS y "
            "detección no supervisada de anomalías."
        ),
        lifespan=lifespan,
    )
    application.include_router(router)
    application.add_exception_handler(
        RequestValidationError, validation_exception_handler
    )
    return application


app = create_app()
