"""tests/test_preprocessing.py — Tests unitarios de las funciones del pipeline (Unidad 2).

No requieren haber corrido el pipeline: usan DataFrames pequeños.
Ejecutar: pytest tests/test_preprocessing.py -v
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import cargar_datos
from src.preprocessing import (
    ALL_COLS, imputar_imc, obtener_balanceador, transformar_variables, validar_schema,
)

RAIZ = Path(__file__).resolve().parents[1]


def _cargar_drift():
    """drift/run.py no es un paquete: se importa por ruta (patrón tests Unidad 9)."""
    spec = importlib.util.spec_from_file_location("drift_run", RAIZ / "drift" / "run.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _paciente(**kw):
    base = {'Edad': 40, 'Promedio_nivel_glucosa': 90.0, 'IMC': 22.0,
            'Estados_civil': 'Si', 'Tipo_trabajo': 'Empresa_privada'}
    base.update(kw)
    return pd.DataFrame([base])


# ── cargar_datos ──────────────────────────────────────────────────────────────

def test_cargar_datos_archivo_inexistente():
    with pytest.raises(FileNotFoundError):
        cargar_datos('no_existe.csv')


def test_cargar_datos_columnas_faltantes(tmp_path):
    ruta = tmp_path / 'malo.csv'
    pd.DataFrame({'ID': [1], 'Edad': [30]}).to_csv(ruta, sep=';', index=False)
    with pytest.raises(ValueError, match='Columnas faltantes'):
        cargar_datos(str(ruta))


# ── imputar_imc ───────────────────────────────────────────────────────────────

def test_imputar_imc_sin_nulos_y_en_rango():
    df = pd.DataFrame({'IMC': [20.0, np.nan, 30.0, np.nan, np.nan]})
    r = imputar_imc(df, imc_avg=25.0, imc_std=5.0)
    assert r['IMC'].isnull().sum() == 0
    assert r['IMC'].between(20.0, 30.0).all()
    assert df['IMC'].isnull().sum() == 3, 'no debe modificar el DataFrame original'


def test_imputar_imc_reproducible():
    df = pd.DataFrame({'IMC': [np.nan] * 10})
    r1 = imputar_imc(df, 25.0, 5.0, seed=123)
    r2 = imputar_imc(df, 25.0, 5.0, seed=123)
    pd.testing.assert_frame_equal(r1, r2)


# ── transformar_variables (cortes del notebook del caso) ──────────────────────

@pytest.mark.parametrize('edad, esperado', [
    (2, 1), (8, 2), (12, 3), (16, 4), (26, 4), (27, 5), (59, 5), (60, 6), (82, 6)])
def test_edad_encoded(edad, esperado):
    assert transformar_variables(_paciente(Edad=edad))['Edad_Encoded'][0] == esperado


@pytest.mark.parametrize('glucosa, esperado', [(85, 1), (100, 2), (125, 2), (126, 3), (250, 3)])
def test_glucosa_encoded(glucosa, esperado):
    r = transformar_variables(_paciente(Promedio_nivel_glucosa=glucosa))
    assert r['Gluocosa_Encoded'][0] == esperado


@pytest.mark.parametrize('imc, esperado', [(17, 1), (22, 2), (27, 3), (35, 4)])
def test_imc_encoded(imc, esperado):
    assert transformar_variables(_paciente(IMC=imc))['IMC_Encoded'][0] == esperado


def test_tipo_trabajo_agrupa_nunca_trabajo_y_cuidar_ninos():
    r = transformar_variables(pd.concat([_paciente(Tipo_trabajo='Nunca_trabajo'),
                                         _paciente(Tipo_trabajo='cuidar_ninos')]))
    assert set(r['TipoTrabajo_Encoded']) == {4}


def test_estado_civil_encoded():
    r = transformar_variables(pd.concat([_paciente(Estados_civil='Si'),
                                         _paciente(Estados_civil='No')]))
    assert list(r['Estado_Encoded']) == [1, 2]


# ── validar_schema ────────────────────────────────────────────────────────────

def test_validar_schema_detecta_nulos():
    df = pd.DataFrame({c: [1, 0] for c in ALL_COLS})
    df.loc[0, 'IMC_Encoded'] = np.nan
    with pytest.raises(ValueError, match='NULOS'):
        validar_schema(df)


def test_validar_schema_detecta_columnas_faltantes():
    df = pd.DataFrame({c: [1, 0] for c in ALL_COLS}).drop(columns=['Edad_Encoded'])
    with pytest.raises(ValueError, match='COLUMNAS FALTANTES'):
        validar_schema(df)


# ── balanceo ──────────────────────────────────────────────────────────────────

def test_obtener_balanceador_desconocido():
    with pytest.raises(ValueError):
        obtener_balanceador('magia')


def test_undersampling_iguala_clases():
    X = pd.DataFrame({'a': range(100)})
    y = pd.Series([1] * 10 + [0] * 90)
    _, y_bal = obtener_balanceador('undersampling', 1).fit_resample(X, y)
    assert y_bal.value_counts().tolist() == [10, 10]


# ── PSI (drift/run.py, Unidad 9) ──────────────────────────────────────────────

def test_psi_misma_distribucion():
    drift = _cargar_drift()
    s = pd.Series(np.random.default_rng(0).integers(1, 7, 5000))
    assert drift.calcular_psi(s, s) < 0.01


def test_psi_detecta_cambio_de_distribucion():
    drift = _cargar_drift()
    ref = pd.Series([5] * 800 + [6] * 200)
    prod = pd.Series([5] * 400 + [6] * 600)
    assert drift.clasificar_psi(drift.calcular_psi(ref, prod)) == 'CRÍTICO'
