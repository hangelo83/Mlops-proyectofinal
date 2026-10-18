"""
preprocessing.py — Tratamiento de nulos y transformación de variables del caso Infartos.

Las reglas vienen del notebook del caso (secciones "Tratamiento de Datos (IMC)" y
"Transformar Variables"). La API importa estas mismas funciones: así el modelo
recibe en producción exactamente las variables con las que se entrenó
(evita el desajuste train/serve que documenta el PDF de la Unidad 10).
"""
import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

logger = logging.getLogger(__name__)

TARGET   = 'Ataque_cardiaco'
ID_COL   = 'ID'
# Variables finales del notebook (celda "Filtrar dataset para ejecución del modelo")
FEATURES = ['Flag_hipertension', 'Flag_problem_cardiaco', 'Edad_Encoded',
            'Gluocosa_Encoded', 'IMC_Encoded', 'Estado_Encoded', 'TipoTrabajo_Encoded']
ALL_COLS = [ID_COL] + FEATURES + [TARGET]


def imputar_imc(df: pd.DataFrame, imc_avg: float, imc_std: float,
                seed: int = 123) -> pd.DataFrame:
    """Reemplaza IMC vacío por valores uniformes en [promedio - desv, promedio + desv]."""
    df = df.copy()
    vacios = df['IMC'].isna()
    n = int(vacios.sum())
    if n > 0:
        rng = np.random.default_rng(seed)
        df.loc[vacios, 'IMC'] = rng.uniform(imc_avg - imc_std, imc_avg + imc_std, size=n)
        logger.info('  Imputados %d nulos en IMC (uniforme %.2f ± %.2f)', n, imc_avg, imc_std)
    return df


def transformar_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Crea las variables *_Encoded con los mismos cortes del notebook del caso."""
    df = df.copy()

    # Edad
    df['Edad_Encoded'] = 0
    df.loc[df['Edad'] < 5, 'Edad_Encoded'] = 1                              # Primera Infancia
    df.loc[(df['Edad'] >= 5) & (df['Edad'] <= 11), 'Edad_Encoded'] = 2      # Infancia
    df.loc[(df['Edad'] >= 12) & (df['Edad'] <= 18), 'Edad_Encoded'] = 3     # Adolescencia
    df.loc[(df['Edad'] >= 14) & (df['Edad'] <= 26), 'Edad_Encoded'] = 4     # Joven
    df.loc[(df['Edad'] >= 27) & (df['Edad'] <= 59), 'Edad_Encoded'] = 5     # Adultez
    df.loc[df['Edad'] >= 60, 'Edad_Encoded'] = 6                            # Vejez

    # Promedio nivel de glucosa (nombre de columna original del notebook)
    glucosa = df['Promedio_nivel_glucosa']
    df['Gluocosa_Encoded'] = 0
    df.loc[glucosa < 100, 'Gluocosa_Encoded'] = 1                           # Normal
    df.loc[(glucosa >= 100) & (glucosa <= 125), 'Gluocosa_Encoded'] = 2     # Prediabetes
    df.loc[glucosa >= 126, 'Gluocosa_Encoded'] = 3                          # Diabetes

    # IMC
    df['IMC_Encoded'] = 0
    df.loc[df['IMC'] < 18.5, 'IMC_Encoded'] = 1                             # Bajo Peso
    df.loc[(df['IMC'] >= 18.5) & (df['IMC'] <= 24.9), 'IMC_Encoded'] = 2    # Peso Saludable
    df.loc[(df['IMC'] >= 25) & (df['IMC'] <= 29.9), 'IMC_Encoded'] = 3      # Sobrepeso
    df.loc[df['IMC'] >= 30, 'IMC_Encoded'] = 4                              # Obesidad

    # Variables cualitativas
    df['Estado_Encoded'] = df['Estados_civil'].map({'Si': 1, 'No': 2, '': 2}).astype(int)
    df['TipoTrabajo_Encoded'] = df['Tipo_trabajo'].map({'Emprendedor': 1,
                                                        'Empresa_privada': 2,
                                                        'En_gobierno': 3,
                                                        'Nunca_trabajo': 4,
                                                        'cuidar_ninos': 4,
                                                        '': 4}).astype(int)
    return df


def validar_schema(df: pd.DataFrame) -> None:
    """Valida el dataset limpio. Lanza ValueError si falla — el pipeline no continúa."""
    errores = []
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        errores.append(f"NULOS RESIDUALES: {nulos}")
    missing = set(ALL_COLS) - set(df.columns)
    if missing:
        errores.append(f"COLUMNAS FALTANTES: {missing}")
    elif not set(df[TARGET].unique()).issubset({0, 1}):
        errores.append(f"TARGET no binario: {set(df[TARGET].unique())}")
    if errores:
        raise ValueError("VALIDACIÓN POST-PREPROCESS FALLIDA:\n" + "\n".join(errores))
    logger.info("  Schema OK — 0 nulos, todas las columnas presentes")


def obtener_balanceador(tecnica: str = 'undersampling', random_state: int = 123):
    """Devuelve la técnica de balanceo. tecnica: 'smote'|'undersampling'|'oversampling'."""
    sampler_map = {
        'smote':         SMOTE(random_state=random_state),
        'undersampling': RandomUnderSampler(random_state=random_state),
        'oversampling':  RandomOverSampler(random_state=random_state),
    }
    if tecnica not in sampler_map:
        raise ValueError(f"tecnica_balanceo desconocida: {tecnica}")
    return sampler_map[tecnica]
