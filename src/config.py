"""
src/config.py — Lee config.yaml para que todas las etapas usen los mismos parámetros.

Unidad 4 centralizaba la configuración en config.py; Unidad 10 en config.yaml.
Aquí config.yaml es la única fuente y este módulo solo la expone:

    from src.config import CONFIG
    CONFIG["data"]["train_path"]
"""
from pathlib import Path

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

with open(ROOT_DIR / "config.yaml", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)
