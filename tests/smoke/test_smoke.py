"""tests/smoke/test_smoke.py — Smoke tests para verificar el entorno desplegado (Unidad 8).

Se ejecutan DESPUÉS de levantar la API (docker run / docker compose up) para
verificar que los servicios respondan correctamente.

Uso en Codespace (con el entorno levantado):
    pytest tests/smoke/ -v
    # o: make smoke

Variables de entorno configurables:
    API_URL    (default: http://localhost:8000)
    MLFLOW_URL (default: http://localhost:5000)
"""
import json
import os
import urllib.request

import pytest

API_URL    = os.getenv("API_URL",    "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000")

PAYLOAD_BAJO = {
    "Edad": 25, "Flag_hipertension": 0, "Flag_problem_cardiaco": 0,
    "Promedio_nivel_glucosa": 85.0, "IMC": 22.0,
    "Estados_civil": "No", "Tipo_trabajo": "Empresa_privada",
}
PAYLOAD_ALTO = {
    "Edad": 67, "Flag_hipertension": 1, "Flag_problem_cardiaco": 1,
    "Promedio_nivel_glucosa": 228.7, "IMC": 36.6,
    "Estados_civil": "Si", "Tipo_trabajo": "Empresa_privada",
}
DECISIONES = ["PROGRAMA PREVENCION", "SEGUIMIENTO", "RUTINA"]


def get_json(url: str) -> dict:
    """Hace GET a una URL y retorna el JSON parseado."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def post_json(url: str, payload: dict) -> dict:
    """Hace POST con JSON y retorna la respuesta."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


# ── Tests de la API ───────────────────────────────────────────────────────────

def test_api_root_responde():
    data = get_json(f"{API_URL}/")
    assert data.get("api") is not None
    assert data.get("version") == "1.0.0"


def test_api_health_ok():
    data = get_json(f"{API_URL}/health")
    assert data["status"] == "ok"
    assert "modelo" in data
    assert 0 < data["recall_test"] <= 1


def test_api_health_tiene_env():
    data = get_json(f"{API_URL}/health")
    assert "env" in data


def test_api_prediccion_payload_bajo():
    data = post_json(f"{API_URL}/predecir", PAYLOAD_BAJO)
    assert data["decision"] in DECISIONES
    assert 0.0 <= data["probabilidad_infarto"] <= 1.0


def test_api_prediccion_payload_alto():
    data = post_json(f"{API_URL}/predecir", PAYLOAD_ALTO)
    assert data["decision"] == "PROGRAMA PREVENCION"


def test_api_prediccion_es_determinista():
    r1 = post_json(f"{API_URL}/predecir", PAYLOAD_ALTO)
    r2 = post_json(f"{API_URL}/predecir", PAYLOAD_ALTO)
    assert r1["probabilidad_infarto"] == r2["probabilidad_infarto"]


def test_api_docs_disponible():
    with urllib.request.urlopen(f"{API_URL}/docs", timeout=10) as resp:
        assert resp.status == 200


# ── Tests de MLflow (solo en el stack pre-prod) ───────────────────────────────

def test_mlflow_ui_responde():
    try:
        with urllib.request.urlopen(f"{MLFLOW_URL}/health", timeout=10) as resp:
            assert resp.status == 200
    except Exception:
        pytest.skip("MLflow no disponible — verifica que el stack está levantado")


def test_mlflow_experimentos_accesibles():
    try:
        data = get_json(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/search?max_results=10")
        assert "experiments" in data
    except Exception:
        pytest.skip("MLflow API no disponible en este entorno")
