"""
api/predictor.py — Carga el modelo y ejecuta predicciones (patrón Unidad 5).
El modelo se carga UNA sola vez al arrancar la API (startup event).

Origen del modelo (patrón Unidad 10):
  1. Si MLFLOW_TRACKING_URI está definido → MLflow Model Registry, stage Production.
  2. Si no, o si el registry no responde → artifacts/modelo_infartos.pkl.
"""
import json
import logging
import os
import pickle
from pathlib import Path

import pandas as pd

from src.preprocessing import transformar_variables

log = logging.getLogger(__name__)

# Rutas absolutas a partir de la carpeta del proyecto (funciona desde cualquier
# directorio: uvicorn, pytest o dentro del contenedor en /app)
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / 'artifacts' / 'modelo_infartos.pkl'
METRICS_PATH = BASE_DIR / 'artifacts' / 'eval_metrics.json'
PREPRO_PATH  = BASE_DIR / 'artifacts' / 'preprocesamiento.json'   # preprocess/run.py

# Umbrales de decisión. 0.50 es el corte con el que evaluate/run.py midió el recall.
UMBRAL_PREVENCION  = 0.50   # score >= 0.50 -> PROGRAMA PREVENCION
UMBRAL_SEGUIMIENTO = 0.30   # score >= 0.30 -> SEGUIMIENTO


class Predictor:
    """Singleton que encapsula la lógica de predicción."""

    def __init__(self):
        self.modelo   = None
        self.origen   = 'sin cargar'
        self.metricas = {}
        self.prepro   = {}

    def cargar(self) -> None:
        """Carga modelo y metadatos. Llamar UNA vez en startup."""
        # Parámetros de preprocesamiento: sin ellos no sabemos el orden de las
        # features ni con qué valor imputar un IMC ausente.
        if not PREPRO_PATH.exists():
            raise FileNotFoundError(
                f'No se encontró {PREPRO_PATH}. Ejecuta primero: python main.py')
        with open(PREPRO_PATH) as f:
            self.prepro = json.load(f)

        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', '')
        if mlflow_uri:
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_uri)
                nombre = os.getenv('MODEL_NAME', 'InfartosRiskModel')
                stage  = os.getenv('MODEL_STAGE', 'Production')
                self.modelo = mlflow.sklearn.load_model(f'models:/{nombre}/{stage}')
                self.origen = f'registry:{stage}'
            except Exception as e:
                log.warning('MLflow Registry no disponible (%s). Se usa el pickle local.', e)

        if self.modelo is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f'Modelo no encontrado: {MODEL_PATH}. Ejecuta primero: python main.py')
            with open(MODEL_PATH, 'rb') as f:
                self.modelo = pickle.load(f)
            self.origen = 'pickle'

        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                self.metricas = json.load(f)

        log.info('Modelo cargado desde %s: %s', self.origen, type(self.modelo).__name__)

    def predecir(self, datos: dict) -> dict:
        """Recibe dict con los campos crudos del paciente y retorna la predicción."""
        if self.modelo is None:
            raise RuntimeError('Modelo no cargado. Llama a cargar() primero.')

        datos = dict(datos)
        # IMC opcional: se imputa con el promedio del entrenamiento
        if datos.get('IMC') is None:
            datos['IMC'] = self.prepro['imc_avg']

        # Mismas transformaciones que preprocess/run.py → sin desajuste train/serve
        X = transformar_variables(pd.DataFrame([datos]))[self.prepro['features']]

        # Se redondea ANTES de decidir para que la decisión sea coherente con el score
        proba = round(float(self.modelo.predict_proba(X)[0][1]), 4)

        if proba >= UMBRAL_PREVENCION:
            decision     = 'PROGRAMA PREVENCION'
            nivel_riesgo = 'MUY ALTO' if proba >= 0.80 else 'ALTO'
        elif proba >= UMBRAL_SEGUIMIENTO:
            decision     = 'SEGUIMIENTO'
            nivel_riesgo = 'MODERADO'
        else:
            decision     = 'RUTINA'
            nivel_riesgo = 'BAJO'

        return {
            'probabilidad_infarto': proba,
            'decision':             decision,
            'nivel_riesgo':         nivel_riesgo,
            'umbral_usado':         UMBRAL_PREVENCION,
            'modelo':               type(self.modelo.named_steps['modelo']).__name__,
        }


# Instancia global — singleton
predictor = Predictor()
