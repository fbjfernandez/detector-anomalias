import pickle
from pathlib import Path
from typing import Any, Sequence

import joblib


def load_model_artifact(
    path: str | Path,
    required_keys: tuple[str, ...] = ("model", "feature_names"),
) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el artefacto del modelo: {artifact_path}"
        )
    try:
        artifact = joblib.load(artifact_path)
    except (OSError, EOFError, pickle.UnpicklingError) as error:
        raise RuntimeError(
            f"No se pudo cargar el artefacto '{artifact_path}': {error}"
        ) from error
    if not isinstance(artifact, dict):
        raise ValueError(
            f"El artefacto '{artifact_path}' no contiene un diccionario válido"
        )
    missing_keys = [key for key in required_keys if key not in artifact]
    if missing_keys:
        raise ValueError(
            f"El artefacto '{artifact_path}' no contiene las claves requeridas: "
            f"{missing_keys}"
        )
    return artifact


def validate_feature_order(
    expected_names: Sequence[str], actual_names: Sequence[str]
) -> None:
    expected = list(expected_names)
    actual = list(actual_names)
    for position, (expected_name, actual_name) in enumerate(
        zip(expected, actual), start=1
    ):
        if expected_name != actual_name:
            raise ValueError(
                "El orden de las características no coincide con el entrenamiento: "
                f"en la posición {position} se esperaba '{expected_name}' "
                f"y se recibió '{actual_name}'"
            )
    if len(expected) != len(actual):
        raise ValueError(
            "El número de características no coincide con el entrenamiento: "
            f"se esperaban {len(expected)} y se recibieron {len(actual)}"
        )
