"""
segregate/run.py — Etapa 3: Train/Test split estratificado.

Split 80/20 con random_state fijo para reproducibilidad.
Usa stratify=y para mantener la proporción de infartos (1.8%) en ambos conjuntos.

Ejecutar: python segregate/run.py
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONFIG              # noqa: E402
from src.preprocessing import TARGET       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | SEGREGATE | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=CONFIG["data"]["clean_path"])
    parser.add_argument("--train_out",  default=CONFIG["data"]["train_path"])
    parser.add_argument("--test_out",   default=CONFIG["data"]["test_path"])
    parser.add_argument("--test_size",  type=float, default=CONFIG["data"]["test_size"])
    parser.add_argument("--seed",       type=int,   default=CONFIG["main"]["random_state"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"{args.input} no encontrado. Ejecuta: python preprocess/run.py")

    log.info("Cargando: %s", args.input)
    df = pd.read_csv(args.input)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test  = pd.concat([X_test,  y_test],  axis=1)

    df_train.to_csv(args.train_out, index=False)
    df_test.to_csv(args.test_out,   index=False)

    log.info("Train : %d filas | Tasa infartos: %.2f%%", len(df_train), df_train[TARGET].mean()*100)
    log.info("Test  : %d filas | Tasa infartos: %.2f%%", len(df_test),  df_test[TARGET].mean()*100)
    log.info("Guardado: %s y %s", args.train_out, args.test_out)
