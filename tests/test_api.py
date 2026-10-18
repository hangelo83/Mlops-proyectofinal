"""
tests/test_api.py — Tests del endpoint /predecir con TestClient de FastAPI (Unidad 5).

No requiere Docker corriendo. Usa TestClient que levanta la app en memoria.
Requiere los artefactos del pipeline (python main.py); si no existen, se omiten.
Ejecutar: pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.predictor import MODEL_PATH, PREPRO_PATH

# ── Fixture ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def client():
    """Levanta la app completa con lifespan (carga el modelo en startup)."""
    if not (MODEL_PATH.exists() and PREPRO_PATH.exists()):
        pytest.skip("Artefactos no encontrados — ejecuta python main.py primero")
    with TestClient(app) as c:
        yield c

# ── Datos de prueba ───────────────────────────────────────────────────────────


PACIENTE_ALTO_RIESGO = {
    'Edad':                   67,
    'Flag_hipertension':      1,
    'Flag_problem_cardiaco':  1,
    'Promedio_nivel_glucosa': 228.7,
    'IMC':                    36.6,
    'Estados_civil':          'Si',
    'Tipo_trabajo':           'Empresa_privada',
}

PACIENTE_BAJO_RIESGO = {
    'Edad':                   25,
    'Flag_hipertension':      0,
    'Flag_problem_cardiaco':  0,
    'Promedio_nivel_glucosa': 85.0,
    'IMC':                    22.0,
    'Estados_civil':          'No',
    'Tipo_trabajo':           'Empresa_privada',
}

DECISIONES_VALIDAS     = {'PROGRAMA PREVENCION', 'SEGUIMIENTO', 'RUTINA'}
NIVELES_RIESGO_VALIDOS = {'BAJO', 'MODERADO', 'ALTO', 'MUY ALTO'}


def assert_respuesta_valida(body: dict):
    """Verifica estructura, tipos y coherencia score/decisión."""
    for campo in ['probabilidad_infarto', 'decision', 'nivel_riesgo', 'umbral_usado', 'modelo']:
        assert campo in body
    assert 0.0 <= body['probabilidad_infarto'] <= 1.0
    assert body['decision'] in DECISIONES_VALIDAS
    assert body['nivel_riesgo'] in NIVELES_RIESGO_VALIDOS
    score, umbral = body['probabilidad_infarto'], body['umbral_usado']
    if score >= umbral:
        assert body['decision'] == 'PROGRAMA PREVENCION'
    elif score >= 0.30:
        assert body['decision'] == 'SEGUIMIENTO'
    else:
        assert body['decision'] == 'RUTINA'

# ── Tests ─────────────────────────────────────────────────────────────────────


def test_health_ok(client):
    r = client.get('/health')
    assert r.status_code == 200
    data = r.json()
    assert data['status'] == 'ok'
    assert data['modelo'] == 'RandomForestClassifier'
    assert 0.0 < data['recall_test'] <= 1.0


def test_root_info(client):
    r = client.get('/')
    assert r.status_code == 200
    assert 'api' in r.json() and 'docs' in r.json()


def test_predecir_estructura(client):
    for payload in [PACIENTE_ALTO_RIESGO, PACIENTE_BAJO_RIESGO]:
        r = client.post('/predecir', json=payload)
        assert r.status_code == 200
        assert_respuesta_valida(r.json())


def test_score_es_determinista(client):
    r1 = client.post('/predecir', json=PACIENTE_ALTO_RIESGO).json()
    r2 = client.post('/predecir', json=PACIENTE_ALTO_RIESGO).json()
    assert r1['probabilidad_infarto'] == r2['probabilidad_infarto']


def test_imc_opcional(client):
    """Sin IMC la API imputa el promedio del entrenamiento y responde 200."""
    datos = {k: v for k, v in PACIENTE_ALTO_RIESGO.items() if k != 'IMC'}
    r = client.post('/predecir', json=datos)
    assert r.status_code == 200
    assert_respuesta_valida(r.json())


def test_validacion_campos_invalidos(client):
    r = client.post('/predecir', json={'Edad': -5})
    assert r.status_code == 422


def test_validacion_tipo_trabajo_desconocido(client):
    datos = dict(PACIENTE_BAJO_RIESGO, Tipo_trabajo='Astronauta')
    assert client.post('/predecir', json=datos).status_code == 422


def test_validacion_flag_no_binario(client):
    datos = dict(PACIENTE_BAJO_RIESGO, Flag_hipertension=3)
    assert client.post('/predecir', json=datos).status_code == 422


def test_openapi_schema(client):
    r = client.get('/openapi.json')
    assert r.status_code == 200
    paths = r.json()['paths']
    assert '/predecir' in paths and '/health' in paths


def test_alto_riesgo_score_mayor_que_bajo_riesgo(client):
    alto = client.post('/predecir', json=PACIENTE_ALTO_RIESGO).json()
    bajo = client.post('/predecir', json=PACIENTE_BAJO_RIESGO).json()
    assert alto['probabilidad_infarto'] > bajo['probabilidad_infarto'] + 0.20


def test_alto_riesgo_va_a_prevencion(client):
    """Detecta el desajuste train/serve de la Unidad 10: si la API no transformara
    las variables igual que el entrenamiento, este paciente no saldría en prevención."""
    r = client.post('/predecir', json=PACIENTE_ALTO_RIESGO)
    assert r.json()['decision'] == 'PROGRAMA PREVENCION'


def test_bajo_riesgo_no_va_a_prevencion(client):
    r = client.post('/predecir', json=PACIENTE_BAJO_RIESGO)
    assert r.json()['decision'] != 'PROGRAMA PREVENCION'
