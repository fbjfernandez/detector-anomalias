import json

import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def valid_features(client):
    feature_names = client.app.state.feature_names
    features = {name: 0.0 for name in feature_names}
    features["Destination Port"] = 80.0
    features["Flow Duration"] = 1000.0
    return features


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == {"supervised": True, "unsupervised": True}
    assert body["expected_features"] == 78


def test_predict_with_valid_payload(client, valid_features):
    response = client.post("/predict", json={"features": valid_features})

    assert response.status_code == 200
    body = response.json()
    assert body["supervised_prediction"] in {"BENIGN", "DDoS"}
    assert 0.0 <= body["ddos_probability"] <= 1.0
    assert isinstance(body["unsupervised_anomaly"], bool)
    assert isinstance(body["anomaly_score"], float)
    assert body["risk_level"] in {"BAJO", "MEDIO", "ALTO"}
    assert body["latency_ms"] >= 0.0


def test_predict_with_missing_feature(client, valid_features):
    incomplete_features = dict(valid_features)
    incomplete_features.pop(next(iter(incomplete_features)))

    response = client.post("/predict", json={"features": incomplete_features})

    assert response.status_code == 422
    assert "Faltan" in response.text


def test_predict_with_unexpected_feature(client, valid_features):
    extra_features = dict(valid_features)
    extra_features["Unknown Feature"] = 1.0

    response = client.post("/predict", json={"features": extra_features})

    assert response.status_code == 422
    assert "no reconocidas" in response.text


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_predict_with_non_finite_features(client, valid_features, invalid_value):
    invalid_features = dict(valid_features)
    invalid_features[next(iter(invalid_features))] = invalid_value

    response = client.post(
        "/predict",
        content=json.dumps({"features": invalid_features}),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    assert "no finitos" in response.text


def test_metrics(client):
    response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "supervised_ddos",
        "generalization_portscan",
        "unsupervised_isolation_forest",
    }
    assert "recall" in body["supervised_ddos"]
    assert "recall_ddos_supervisado" in body["generalization_portscan"]
    assert {"ddos", "portscan"} <= set(body["unsupervised_isolation_forest"])
