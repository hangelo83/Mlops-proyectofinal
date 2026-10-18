"""
api/app.py — API REST del modelo de prevención de infartos.

Endpoints:
    POST /predecir  -> probabilidad de infarto y decisión de la aseguradora
    GET  /health    -> estado de la API y metadatos del modelo
    GET  /          -> info básica
    GET  /docs      -> Swagger UI (autogenerado por FastAPI)

Ejecutar en Codespace:
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Con Docker:
    docker build -t infartos-api . && docker run -p 8000:8000 infartos-api
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.predictor import predictor
from api.schemas import HealthResponse, PacienteInput, PrediccionOutput

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s | API | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar la API (startup event)."""
    log.info('Iniciando API en entorno: %s', os.getenv('ENV', 'dev'))
    predictor.cargar()
    log.info('API lista para recibir predicciones')
    yield
    log.info('API cerrando')


app = FastAPI(
    title='API Prevención de Infartos — Aseguradora',
    description=(
        'Probabilidad de infarto de un asegurado para incluirlo en el programa '
        'de prevención. Modelo entrenado con el pipeline MLOps del proyecto.'
    ),
    version='1.0.0',
    lifespan=lifespan,
)

# Permitir llamadas desde cualquier origen (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', tags=['Info'])
def root():
    """Información básica de la API."""
    return {
        'api':        'Prevención de Infartos',
        'env':        os.getenv('ENV', 'dev'),
        'version':    '1.0.0',
        'docs':       '/docs',
        'health':     '/health',
        'prediccion': 'POST /predecir',
    }


@app.get('/health', response_model=HealthResponse, tags=['Salud'])
def health():
    """Estado de la API y metadatos del modelo cargado."""
    if predictor.modelo is None:
        raise HTTPException(status_code=503, detail='Modelo no cargado')
    return HealthResponse(
        status='ok',
        modelo=type(predictor.modelo.named_steps['modelo']).__name__,
        origen_modelo=predictor.origen,
        version='1.0.0',
        recall_test=float(predictor.metricas.get('test_recall', 0.0)),
        auc_test=float(predictor.metricas.get('test_auc', 0.0)),
        env=os.getenv('ENV', 'dev'),
    )


@app.post('/predecir', response_model=PrediccionOutput, tags=['Prediccion'])
def predecir(paciente: PacienteInput):
    """
    Predice la probabilidad de infarto de un asegurado.

    - **probabilidad_infarto**: [0.0 - 1.0]
    - **decision**: PROGRAMA PREVENCION | SEGUIMIENTO | RUTINA
    - **nivel_riesgo**: BAJO | MODERADO | ALTO | MUY ALTO
    """
    try:
        resultado = predictor.predecir(paciente.model_dump())
        log.info('Predicción: prob=%.4f decision=%s',
                 resultado['probabilidad_infarto'], resultado['decision'])
        return PrediccionOutput(**resultado)
    except Exception as e:
        log.error('Error en predicción: %s', str(e))
        raise HTTPException(status_code=500, detail=str(e))
