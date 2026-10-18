"""
preprocess/run.py — Etapa 2: Limpieza y transformación del dataset.

Operaciones (reglas del notebook del caso):
  - Imputar nulos de IMC (uniforme en promedio ± desviación)
  - Transformar Edad, glucosa, IMC, estado civil y tipo de trabajo a *_Encoded
  - Filtrar las variables del modelo + validación de schema post-limpieza
  - Guardar artifacts/preprocesamiento.json para que la API transforme igual (Unidad 5)

Flag_fumador (31% nulos) no se usa: el notebook la descarta del modelo.

Ejecutar: python preprocess/run.py
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONFIG              # noqa: E402
from src.preprocessing import (            # noqa: E402
    ALL_COLS, FEATURES, imputar_imc, transformar_variables, validar_schema,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | PREPROCESS | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=CONFIG["data"]["raw_path"])
    parser.add_argument("--output", default=CONFIG["data"]["clean_path"])
    parser.add_argument("--seed",   type=int, default=CONFIG["main"]["random_state"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"{args.input} no encontrado. Ejecuta: python data/run.py")

    log.info("Cargando: %s", args.input)
    df = pd.read_csv(args.input)
    log.info("Shape inicial: %d x %d | Nulos: %d", df.shape[0], df.shape[1], df.isnull().sum().sum())

    # Promedio y desviación del IMC (antes de imputar), igual que el notebook
    imc_avg = float(df['IMC'].mean())
    imc_std = float(df['IMC'].std())

    df = imputar_imc(df, imc_avg, imc_std, seed=args.seed)
    df = transformar_variables(df)
    df = df[ALL_COLS]
    validar_schema(df)

    df.to_csv(args.output, index=False)
    log.info("Dataset limpio guardado: %s (%d filas)", args.output, len(df))

    # Parámetros que la API necesita para transformar cada paciente igual que aquí
    ARTIFACTS.mkdir(exist_ok=True)
    prepro = {"imc_avg": round(imc_avg, 4), "imc_std": round(imc_std, 4), "features": FEATURES}
    with open(ARTIFACTS / "preprocesamiento.json", "w") as f:
        json.dump(prepro, f, indent=2)
    log.info("Parámetros guardados: artifacts/preprocesamiento.json")
