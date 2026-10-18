"""src/manage_versions.py — Gestión de versiones en MLflow Model Registry (Unidad 8).

Flujo de estados: None → Staging → Production → Archived
  - random_forest/run.py registra el mejor modelo y lo deja en Staging.
  - evaluate/run.py lo promueve a Production si pasa el quality gate.

Uso en Codespace:
    python src/manage_versions.py                       # backend local sqlite:///mlflow.db
    MLFLOW_TRACKING_URI=http://localhost:5000 python src/manage_versions.py   # stack pre-prod
"""
import os

import mlflow
from mlflow import MlflowClient

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_NAME = "InfartosRiskModel"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)


def listar_versiones(nombre: str = MODEL_NAME) -> None:
    """Lista todas las versiones registradas del modelo."""
    try:
        versiones = client.search_model_versions(f"name='{nombre}'")
        if not versiones:
            print(f"No hay versiones registradas para '{nombre}'")
            return
        print(f"\n=== Versiones de '{nombre}' ===")
        for v in sorted(versiones, key=lambda v: int(v.version)):
            print(f"  v{v.version} | {v.current_stage:12} | run_id: {v.run_id[:8]}...")
    except Exception as e:
        print(f"Error listando versiones: {e}")


def promover_a_staging(nombre: str, version: str) -> None:
    """Promueve una versión al stage Staging."""
    client.transition_model_version_stage(
        name=nombre, version=version, stage="Staging"
    )
    print(f"✓ {nombre} v{version} → Staging")


def promover_a_produccion(nombre: str, version: str) -> None:
    """Promueve una versión a Production y archiva la que estaba antes."""
    client.transition_model_version_stage(
        name=nombre, version=version, stage="Production",
        archive_existing_versions=True,
    )
    print(f"✓ {nombre} v{version} → Production")


def archivar(nombre: str, version: str) -> None:
    """Archiva una versión anterior."""
    client.transition_model_version_stage(
        name=nombre, version=version, stage="Archived"
    )
    print(f"✓ {nombre} v{version} → Archived")


def agregar_tags(nombre: str, version: str, tags: dict) -> None:
    """Agrega tags de trazabilidad a una versión."""
    for key, value in tags.items():
        client.set_model_version_tag(name=nombre, version=version, key=key, value=str(value))
    print(f"✓ Tags agregados a {nombre} v{version}: {tags}")


if __name__ == "__main__":
    print(f"MLflow URI: {MLFLOW_URI}")
    print(f"Modelo   : {MODEL_NAME}")
    listar_versiones(MODEL_NAME)

    # Rollback manual (descomentar según necesidad):
    # promover_a_produccion(MODEL_NAME, version="1")
    # archivar(MODEL_NAME, version="2")
