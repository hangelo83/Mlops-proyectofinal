"""
data_loader.py — Carga y validación del dataset de prevención de infartos (patrón Unidad 2).
"""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

COLUMNAS_REQUERIDAS = {
    'ID', 'Genero', 'Edad', 'Flag_hipertension', 'Flag_problem_cardiaco',
    'Estados_civil', 'Tipo_trabajo', 'Zona_residencia',
    'Promedio_nivel_glucosa', 'IMC', 'Flag_fumador', 'Ataque_cardiaco',
}


def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el CSV de asegurados del caso Prevención de Infartos.

    Args:
        ruta: Ruta al archivo CSV (sep=';').
    Returns:
        DataFrame crudo con los datos originales.
    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas requeridas.
    """
    path = Path(ruta)
    if not path.exists():
        raise FileNotFoundError(f'Archivo no encontrado: {ruta}')
    logger.info('Cargando datos desde %s', ruta)
    df = pd.read_csv(path, sep=';')
    faltantes = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltantes:
        raise ValueError(f'Columnas faltantes en el dataset: {faltantes}')
    logger.info('Dataset cargado: %d filas x %d columnas', *df.shape)
    logger.info('Distribucion Ataque_cardiaco: %s',
                df['Ataque_cardiaco'].value_counts().to_dict())
    return df
