"""
main.py — Orquestador del pipeline MLOps completo (patrón Unidad 10).

Ejecuta las 7 etapas del pipeline en orden:
  1. download      → data/run.py
  2. preprocess    → preprocess/run.py
  3. segregate     → segregate/run.py
  4. check_data    → pytest check_data/
  5. random_forest → random_forest/run.py
  6. evaluate      → evaluate/run.py
  7. drift         → drift/run.py

Uso:
    python main.py                         # todas las etapas
    python main.py --steps download preprocess segregate
    python main.py --steps random_forest evaluate
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | PIPELINE | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline_run.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

ALL_STEPS = ["download", "preprocess", "segregate", "check_data", "random_forest", "evaluate", "drift"]

STEP_COMMANDS = {
    "download":      [sys.executable, "data/run.py"],
    "preprocess":    [sys.executable, "preprocess/run.py"],
    "segregate":     [sys.executable, "segregate/run.py"],
    "check_data":    [sys.executable, "-m", "pytest", "check_data/test_data.py", "-v", "--tb=short"],
    "random_forest": [sys.executable, "random_forest/run.py"],
    "evaluate":      [sys.executable, "evaluate/run.py"],
    "drift":         [sys.executable, "drift/run.py"],
}


def ejecutar_paso(nombre: str) -> tuple[bool, float]:
    """Ejecuta un paso del pipeline y retorna (éxito, duración)."""
    inicio = time.time()
    log.info(">>> Iniciando: %s", nombre)
    cmd = STEP_COMMANDS[nombre]

    # Backend local por defecto; respeta un MLFLOW_TRACKING_URI externo
    # (p. ej. http://mlflow:5000 en docker-compose.preprod.yml)
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    result = subprocess.run(cmd, capture_output=False, env=env)
    dur = round(time.time() - inicio, 2)
    ok  = result.returncode == 0
    if ok:
        log.info("<<< Completado: %s (%.2f s)", nombre, dur)
    else:
        log.error("XXX FALLO: %s (código: %d)", nombre, result.returncode)
    return ok, dur


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLOps End-to-End — Prevención de Infartos")
    parser.add_argument("--steps", nargs="+", default=ALL_STEPS,
                        choices=ALL_STEPS, help="Etapas a ejecutar")
    args = parser.parse_args()

    # Crear directorios necesarios
    for d in ["data", "artifacts", "reportes"]:
        Path(d).mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info(" PIPELINE MLOps — Prevención de Infartos")
    log.info(" Etapas: %s", " → ".join(args.steps))
    log.info("=" * 60)

    resumen = []
    for paso in args.steps:
        ok, dur = ejecutar_paso(paso)
        resumen.append({"paso": paso, "estado": "OK" if ok else "FALLO", "duracion_s": dur})
        if not ok:
            log.error("Pipeline detenido en: %s", paso)
            log.error("Corrige el error y reintenta con: python main.py --steps %s", paso)
            sys.exit(1)

    dur_total = sum(r["duracion_s"] for r in resumen)
    log.info("=" * 60)
    log.info(" PIPELINE COMPLETADO EN %.2f segundos", dur_total)
    log.info("=" * 60)
    for r in resumen:
        log.info("  [%s] %s (%.2f s)", r["estado"], r["paso"], r["duracion_s"])

    # Métricas finales (patrón run_pipeline.py de la Unidad 4)
    metrics_path = Path("artifacts/eval_metrics.json")
    if metrics_path.exists() and "evaluate" in args.steps:
        with open(metrics_path) as f:
            m = json.load(f)
        log.info("")
        log.info("  MÉTRICAS FINALES (test):")
        log.info("    AUC     : %.4f", m["test_auc"])
        log.info("    Recall  : %.4f", m["test_recall"])
        log.info("    FN      : %d infartos no detectados", m["FN"])
        log.info("    Costo   : $%s", f"{m['costo_estimado']:,}")

    log.info("")
    log.info("  Próximo paso: servir la API")
    log.info("  uvicorn api.app:app --host 0.0.0.0 --port 8000")
    log.info("  O con Docker: docker build -t infartos-api . && docker run -p 8000:8000 infartos-api")


if __name__ == "__main__":
    main()
