"""
check_data/test_data.py — Validación automática del dataset con pytest (Etapa 4).

Tests de schema, integridad, rangos y distribuciones.
Ejecutar: pytest check_data/test_data.py -v
"""
import pandas as pd

FEATURES = ['Flag_hipertension', 'Flag_problem_cardiaco', 'Edad_Encoded',
            'Gluocosa_Encoded', 'IMC_Encoded', 'Estado_Encoded', 'TipoTrabajo_Encoded']
TARGET   = "Ataque_cardiaco"
ALL_COLS = ["ID"] + FEATURES + [TARGET]

# Valores válidos de cada variable codificada. El 0 aparece en glucosa e IMC porque
# los cortes del notebook dejan huecos (125-126 y 24.9-25 / 29.9-30).
CATEGORIAS = {
    "Flag_hipertension":     {0, 1},
    "Flag_problem_cardiaco": {0, 1},
    "Edad_Encoded":          {1, 2, 3, 4, 5, 6},
    "Gluocosa_Encoded":      {0, 1, 2, 3},
    "IMC_Encoded":           {0, 1, 2, 3, 4},
    "Estado_Encoded":        {1, 2},
    "TipoTrabajo_Encoded":   {1, 2, 3, 4},
}


def test_columnas_requeridas(df_train):
    """Todas las columnas requeridas deben estar presentes."""
    for col in ALL_COLS:
        assert col in df_train.columns, f"Columna faltante: {col}"


def test_sin_nulos_train(df_train):
    """El conjunto de entrenamiento no debe tener nulos."""
    nulos = df_train.isnull().sum().sum()
    assert nulos == 0, f"Se encontraron {nulos} nulos en el set de entrenamiento"


def test_sin_nulos_test(df_test):
    """El conjunto de test no debe tener nulos."""
    nulos = df_test.isnull().sum().sum()
    assert nulos == 0, f"Se encontraron {nulos} nulos en el set de test"


def test_target_es_binario(df_train):
    """El target 'Ataque_cardiaco' debe ser binario (0 o 1)."""
    valores = set(df_train[TARGET].unique())
    assert valores.issubset({0, 1}), f"Target no binario: {valores}"


def test_tasa_infartos_razonable(df_train):
    """La tasa de infartos debe estar entre 0.5% y 5% (histórico del caso: 1.8%)."""
    tasa = df_train[TARGET].mean()
    assert 0.005 <= tasa <= 0.05, f"Tasa de infartos fuera de rango: {tasa:.2%}"


def test_variables_codificadas_en_rango(df_train):
    """Cada variable codificada solo toma los valores definidos en el notebook."""
    for col, validos in CATEGORIAS.items():
        valores = set(df_train[col].unique())
        assert valores.issubset(validos), f"Categorías inesperadas en {col}: {valores - validos}"


def test_id_unico(df_train):
    """Cada asegurado aparece una sola vez."""
    assert df_train["ID"].is_unique, "Hay IDs duplicados en train"


def test_train_test_sin_solapamiento(df_train, df_test):
    """Ningún asegurado debe estar en train y test a la vez."""
    merged = pd.merge(df_train[["ID"]], df_test[["ID"]], how="inner")
    assert len(merged) == 0, f"{len(merged)} asegurados solapados entre train y test"


def test_tamanio_minimo_train(df_train):
    """El set de entrenamiento debe tener al menos 1000 filas."""
    assert len(df_train) >= 1000, f"Set de entrenamiento muy pequeño: {len(df_train)}"


def test_minimo_casos_positivos(df_train):
    """Debe haber suficientes infartos para entrenar (al menos 100)."""
    positivos = int(df_train[TARGET].sum())
    assert positivos >= 100, f"Solo {positivos} casos positivos en train"


def test_distribucion_similar_train_test(df_train, df_test):
    """La tasa de infartos en train y test debe ser similar (±0.5 puntos)."""
    tasa_train = df_train[TARGET].mean()
    tasa_test  = df_test[TARGET].mean()
    assert abs(tasa_train - tasa_test) < 0.005, \
        f"Distribución muy diferente: train={tasa_train:.2%} test={tasa_test:.2%}"
