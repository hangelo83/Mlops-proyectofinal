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

Etapas opcionales (no corren por defecto ni en CI):
  drift_demo → drift/run.py --simular_drift (demo de la alerta de re-entrenamiento)
  docker     → build de la imagen + contenedor + smoke tests (Unidades 5 y 8)

Uso:
    python main.py                         # las 7 etapas (lo mismo que corre CI)
    python main.py --steps download preprocess segregate
    python main.py --steps random_forest evaluate
    python main.py --steps drift_demo docker
    python main.py --serve                 # 7 etapas + MLflow UI (5000) + API (8000)
    python main.py --demo                  # todo: 7 etapas + drift_demo + docker + --serve
"""
import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
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
    "drift_demo":    [sys.executable, "drift/run.py", "--simular_drift"],
}
EXTRA_STEPS = ["drift_demo", "docker"]

IMAGEN_API    = "infartos-api"
PUERTO_SMOKE  = 8001   # el contenedor de prueba no choca con la API de --serve
PUERTO_API    = int(os.getenv("API_PORT", "8000"))      # mismas variables que .env.preprod
PUERTO_MLFLOW = int(os.getenv("MLFLOW_PORT", "5000"))


def esperar_salud(url: str, proceso: subprocess.Popen = None, segundos: int = 90) -> bool:
    """Espera a que un servicio responda 200 en su endpoint de salud (corta si el proceso muere)."""
    limite = time.time() + segundos
    while time.time() < limite:
        if proceso is not None and proceso.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=3):
                return True
        except Exception:
            time.sleep(2)
    return False


def detener(proceso: subprocess.Popen) -> None:
    """Detiene el proceso y sus hijos (mlflow ui lanza su propio servidor como hijo)."""
    if proceso.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proceso.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(proceso.pid, signal.SIGTERM)


def smoke_tests(api_url: str) -> int:
    """Corre tests/smoke/ (Unidad 8) contra una API levantada."""
    env = os.environ.copy()
    env["API_URL"] = api_url
    cmd = [sys.executable, "-m", "pytest", "tests/smoke/", "-v", "--tb=short", "-p", "no:cacheprovider"]
    return subprocess.run(cmd, env=env).returncode


def paso_docker() -> int:
    """Build de la imagen de la API, contenedor de prueba y smoke tests (como el job build-docker de CI)."""
    if shutil.which("docker") is None:
        log.warning("Docker no está disponible en este entorno: etapa omitida")
        return 0
    nombre = f"{IMAGEN_API}-smoke"
    if subprocess.run(["docker", "build", "-t", IMAGEN_API, "."]).returncode != 0:
        return 1
    subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)
    run = subprocess.run(["docker", "run", "-d", "--name", nombre,
                          "-p", f"{PUERTO_SMOKE}:8000", IMAGEN_API])
    if run.returncode != 0:
        return 1
    try:
        api_url = f"http://localhost:{PUERTO_SMOKE}"
        if not esperar_salud(f"{api_url}/health"):
            log.error("El contenedor no respondió en %s/health", api_url)
            subprocess.run(["docker", "logs", nombre])
            return 1
        return smoke_tests(api_url)
    finally:
        subprocess.run(["docker", "rm", "-f", nombre], capture_output=True)


STEP_FUNCTIONS = {"docker": paso_docker}


def ejecutar_paso(nombre: str) -> tuple[bool, float]:
    """Ejecuta un paso del pipeline y retorna (éxito, duración)."""
    inicio = time.time()
    log.info(">>> Iniciando: %s", nombre)

    # Backend local por defecto; respeta un MLFLOW_TRACKING_URI externo
    # (p. ej. http://mlflow:5000 en docker-compose.preprod.yml)
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    if nombre in STEP_FUNCTIONS:
        returncode = STEP_FUNCTIONS[nombre]()
    else:
        returncode = subprocess.run(STEP_COMMANDS[nombre], capture_output=False, env=env).returncode
    dur = round(time.time() - inicio, 2)
    ok  = returncode == 0
    if ok:
        log.info("<<< Completado: %s (%.2f s)", nombre, dur)
    else:
        log.error("XXX FALLO: %s (código: %d)", nombre, returncode)
    return ok, dur


def servir() -> None:
    """Levanta MLflow UI y la API, corre smoke tests y queda escuchando hasta Ctrl+C."""
    hosts = "localhost:*,127.0.0.1:*,*.app.github.dev"   # *.app.github.dev = URL de Codespaces
    grupo = {} if os.name == "nt" else {"start_new_session": True}   # para detener los hijos
    api_url, mlflow_url = f"http://localhost:{PUERTO_API}", f"http://localhost:{PUERTO_MLFLOW}"
    procesos = {
        "MLflow UI": subprocess.Popen([sys.executable, "-m", "mlflow", "ui",
                                       "--backend-store-uri", "sqlite:///mlflow.db",
                                       "--host", "0.0.0.0", "--port", str(PUERTO_MLFLOW),
                                       "--allowed-hosts", hosts], **grupo),
        "API":       subprocess.Popen([sys.executable, "-m", "uvicorn", "api.app:app",
                                       "--host", "0.0.0.0", "--port", str(PUERTO_API)], **grupo),
    }
    try:
        for nombre, url in [("API", f"{api_url}/health"), ("MLflow UI", f"{mlflow_url}/health")]:
            if not esperar_salud(url, procesos[nombre]):
                log.error("%s no arrancó: ¿puerto ocupado? Libéralo o cambia API_PORT / MLFLOW_PORT", nombre)
                sys.exit(1)
        os.environ["MLFLOW_URL"] = mlflow_url
        if smoke_tests(api_url) != 0:
            log.error("Smoke tests fallidos contra la API")
            sys.exit(1)

        log.info("=" * 60)
        log.info(" SERVICIOS ARRIBA (en Codespaces: pestaña PUERTOS)")
        log.info("   API     : %s/docs", api_url)
        log.info("   MLflow  : %s", mlflow_url)
        log.info(" Ctrl+C para detenerlos")
        log.info("=" * 60)
        for proceso in procesos.values():
            proceso.wait()
    except KeyboardInterrupt:
        log.info("Deteniendo servicios...")
    finally:
        for proceso in procesos.values():
            detener(proceso)


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLOps End-to-End — Prevención de Infartos")
    parser.add_argument("--steps", nargs="+", default=ALL_STEPS,
                        choices=ALL_STEPS + EXTRA_STEPS, help="Etapas a ejecutar")
    parser.add_argument("--serve", action="store_true",
                        help="Al terminar, levantar MLflow UI y la API y correr smoke tests")
    parser.add_argument("--demo", action="store_true",
                        help="Todo: 7 etapas + drift_demo + docker + --serve")
    args = parser.parse_args()
    if args.demo:
        args.steps = ALL_STEPS + EXTRA_STEPS
        args.serve = True

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

    if args.serve:
        servir()
    else:
        log.info("")
        log.info("  Próximo paso: servir la API y MLflow UI")
        log.info("  python main.py --serve     (o --demo para correr además drift_demo y docker)")


if __name__ == "__main__":
    main()
