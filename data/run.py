"""
data/run.py — Etapa 1: Ingesta del dataset de prevención de infartos.

En la Unidad 10 esta etapa generaba datos sintéticos. Aquí carga el CSV real del
caso (versionado en Git en data/raw/), valida sus columnas con cargar_datos()
(Unidad 2) y lo deja en formato estándar (sep=',') para las etapas siguientes.

Ejecutar: python data/run.py
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONFIG              # noqa: E402
from src.data_loader import cargar_datos   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DOWNLOAD | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Ingesta del dataset de infartos")
    parser.add_argument("--source", default=CONFIG["data"]["source_path"])
    parser.add_argument("--output", default=CONFIG["data"]["raw_path"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = cargar_datos(args.source)

    output = Path(args.output)
    output.parent.mkdir(exist_ok=True)
    df.to_csv(output, index=False, lineterminator="\n")

    log.info("Dataset guardado: %s", output)
    log.info("Shape: %d filas x %d columnas", *df.shape)
    log.info("Tasa de infartos: %.2f%%", df['Ataque_cardiaco'].mean() * 100)
    log.info("Nulos: %d", df.isnull().sum().sum())
