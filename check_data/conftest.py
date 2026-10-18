"""check_data/conftest.py — Fixtures para los tests de validación de datos."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONFIG   # noqa: E402


@pytest.fixture(scope="session")
def df_train():
    """Carga el conjunto de entrenamiento para los tests."""
    path = Path(CONFIG["data"]["train_path"])
    if not path.exists():
        pytest.skip(f"{path} no encontrado — ejecuta los pasos 1-3 primero")
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def df_test():
    """Carga el conjunto de test para los tests."""
    path = Path(CONFIG["data"]["test_path"])
    if not path.exists():
        pytest.skip(f"{path} no encontrado — ejecuta los pasos 1-3 primero")
    return pd.read_csv(path)
