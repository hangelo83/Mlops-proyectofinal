"""
random_forest/run.py — Etapa 5: Tuning de Random Forest + MLflow tracking.

Unidad 10: GridSearchCV + MLflow + Model Registry.
Unidad 6 : un segundo run con RandomizedSearchCV, comparación de runs y
           selección programática del mejor; el mejor se registra y pasa a Staging.

El balanceo (undersampling, la mejor técnica en el notebook del caso) va DENTRO
del Pipeline de imblearn: así se aplica solo a los folds de entrenamiento de la
validación cruzada y nunca toca los datos de validación.

Baseline: antes del tuning se registra una Regresión Logística con los parámetros
de MODELOS_DEFAULT (Unidad 2) como línea base (Unidad 6, 01_base_pipeline.py).
Sirve de comparación y no compite por el Model Registry. La tabla de todos los
modelos queda en artifacts/comparativa_modelos.json.

Ejecutar: python random_forest/run.py
"""
import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.pipeline import Pipeline
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CONFIG                                  # noqa: E402
from src.manage_versions import agregar_tags, promover_a_staging   # noqa: E402
from src.preprocessing import FEATURES, TARGET, obtener_balanceador   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | TRAIN | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")
CFG = CONFIG["model"]

# Métricas de validación cruzada para la comparativa; el refit usa config.yaml model.scoring
METRICAS_CV = {"f1": "f1", "recall": "recall", "roc_auc": "roc_auc"}
CLAVE_CV    = {"f1": "cv_f1", "recall": "cv_recall", "roc_auc": "cv_auc"}

# Línea base: misma Regresión Logística de MODELOS_DEFAULT (Unidad 2, src/train.py)
MODELO_BASELINE = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",      default=CONFIG["data"]["train_path"])
    parser.add_argument("--experiment_name", default=CONFIG["main"]["experiment_name"])
    parser.add_argument("--model_name",      default=CFG["model_name"])
    parser.add_argument("--tecnica_balanceo", default=CFG["tecnica_balanceo"])
    parser.add_argument("--scoring",         default=CFG["scoring"])
    parser.add_argument("--seed",            type=int, default=CONFIG["main"]["random_state"])
    return parser.parse_args()


def construir_pipeline(tecnica: str, seed: int) -> Pipeline:
    """Balanceo + Random Forest en un solo estimador."""
    return Pipeline([
        ("balanceo", obtener_balanceador(tecnica, seed)),
        ("modelo",   RandomForestClassifier(random_state=seed, n_jobs=-1)),
    ])


def tipos_confiables(pipe: Pipeline) -> list:
    """Tipos que skops debe aceptar al serializar el modelo (MLflow 3.x, ver Unidad 6)."""
    sampler = type(pipe.named_steps["balanceo"])
    return ["imblearn.pipeline.Pipeline",
            f"{sampler.__module__}.{sampler.__qualname__}",
            "sklearn.tree._tree.Tree"]


def metricas_train(modelo, X, y) -> dict:
    """Métricas sobre el propio set de entrenamiento (para detectar sobreajuste)."""
    y_pred = modelo.predict(X)
    return {
        "train_f1":     round(f1_score(y, y_pred), 4),
        "train_recall": round(recall_score(y, y_pred), 4),
        "train_auc":    round(roc_auc_score(y, modelo.predict_proba(X)[:, 1]), 4),
    }


def metricas_cv(cv_results: dict, idx: int = None) -> dict:
    """Promedio de validación cruzada; idx = fila del mejor candidato de una búsqueda."""
    def media(nombre):
        valores = cv_results[f"mean_test_{nombre}"] if idx is not None else cv_results[f"test_{nombre}"].mean()
        return round(float(valores[idx] if idx is not None else valores), 4)
    return {"cv_f1": media("f1"), "cv_recall": media("recall"), "cv_auc": media("roc_auc")}


def entrenar_baseline(X, y, cv) -> dict:
    """Línea base sin tuning (Unidad 6): mismo balanceo, Regresión Logística de la Unidad 2."""
    pipe = Pipeline([("balanceo", obtener_balanceador(args.tecnica_balanceo, args.seed)),
                     ("modelo",   MODELO_BASELINE)])
    cv_m = metricas_cv(cross_validate(pipe, X, y, cv=cv, scoring=METRICAS_CV, n_jobs=-1))
    pipe.fit(X, y)
    train_m = metricas_train(pipe, X, y)
    with mlflow.start_run(run_name="Baseline-RegresionLogistica") as run:
        mlflow.set_tag("etapa", "random_forest")
        mlflow.log_param("metodo_busqueda",  "baseline")
        mlflow.log_param("algoritmo",        "LogisticRegression")
        mlflow.log_param("tecnica_balanceo", args.tecnica_balanceo)
        mlflow.log_param("cv_folds",         CFG["cv_folds"])
        mlflow.log_metrics({"cv_score": cv_m[CLAVE_CV[args.scoring]], **cv_m, **train_m})
    with open(ARTIFACTS / "modelo_baseline.pkl", "wb") as f:
        pickle.dump(pipe, f)
    log.info("[Baseline-RegresionLogistica] CV F1=%.4f Recall=%.4f AUC=%.4f | Run ID: %s",
             cv_m["cv_f1"], cv_m["cv_recall"], cv_m["cv_auc"], run.info.run_id)
    return {"modelo": "Regresión Logística (baseline)", "metodo": "baseline", **cv_m, **train_m}


def registrar_run(busqueda, nombre_run: str, metodo: str, X, y, extra_params: dict) -> dict:
    """Registra en MLflow los mejores parámetros, métricas CV/train y el modelo."""
    mejor = busqueda.best_estimator_
    cv_m = metricas_cv(busqueda.cv_results_, busqueda.best_index_)
    train_m = metricas_train(mejor, X, y)
    with mlflow.start_run(run_name=nombre_run) as run:
        mlflow.set_tag("etapa", "random_forest")
        mlflow.log_params({k.replace("modelo__", ""): v for k, v in busqueda.best_params_.items()})
        mlflow.log_param("metodo_busqueda",  metodo)
        mlflow.log_param("tecnica_balanceo", args.tecnica_balanceo)
        mlflow.log_param("cv_folds",         CFG["cv_folds"])
        mlflow.log_param("scoring",          args.scoring)
        mlflow.log_params(extra_params)

        mlflow.log_metrics({"cv_score": round(busqueda.best_score_, 4), **cv_m, **train_m})

        info = mlflow.sklearn.log_model(
            mejor,
            name="modelo_infartos",
            input_example=X.head(3),
            skops_trusted_types=tipos_confiables(mejor),
        )
        log.info("[%s] CV %s=%.4f | Run ID: %s", nombre_run, args.scoring,
                 busqueda.best_score_, run.info.run_id)
    return {"run_id": run.info.run_id, "model_uri": info.model_uri, "metodo": metodo,
            "cv_score": round(float(busqueda.best_score_), 4), "modelo": mejor,
            "metricas": {"modelo": f"Random Forest ({metodo})", "metodo": metodo, **cv_m, **train_m}}


if __name__ == "__main__":
    args = parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"{args.train_data} no encontrado. Ejecuta los pasos anteriores.")

    log.info("Cargando datos de entrenamiento: %s", args.train_data)
    df = pd.read_csv(args.train_data)
    X  = df[FEATURES]
    y  = df[TARGET]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment_name)

    cv = StratifiedKFold(CFG["cv_folds"], shuffle=True, random_state=args.seed)

    # ── Run 0: línea base sin tuning (referencia para los dos runs de tuning) ──
    baseline = entrenar_baseline(X, y, cv)

    # ── Run 1: GridSearchCV (búsqueda exhaustiva) ─────────────────────────────
    param_grid = {
        "modelo__n_estimators":      CFG["n_estimators"],
        "modelo__max_depth":         CFG["max_depth"],
        "modelo__min_samples_split": CFG["min_samples_split"],
    }
    n_combinaciones = (len(CFG["n_estimators"]) * len(CFG["max_depth"])
                       * len(CFG["min_samples_split"]))
    log.info("GridSearchCV: %d combinaciones x %d folds = %d entrenamientos",
             n_combinaciones, CFG["cv_folds"], n_combinaciones * CFG["cv_folds"])
    gs = GridSearchCV(construir_pipeline(args.tecnica_balanceo, args.seed), param_grid,
                      cv=cv, scoring=METRICAS_CV, refit=args.scoring, n_jobs=-1, verbose=1)
    gs.fit(X, y)
    run_gs = registrar_run(gs, "GridSearch-RandomForest", "GridSearchCV", X, y,
                           {"n_combinaciones": n_combinaciones})

    # ── Run 2: RandomizedSearchCV (muestreo aleatorio, Unidad 6) ──────────────
    param_dist = {
        "modelo__n_estimators":      randint(10, 300),
        "modelo__max_depth":         randint(3, 12),
        "modelo__min_samples_split": randint(2, 50),
        "modelo__min_samples_leaf":  randint(1, 20),
    }
    log.info("RandomizedSearchCV: %d combinaciones x %d folds", CFG["n_iter"], CFG["cv_folds"])
    rs = RandomizedSearchCV(construir_pipeline(args.tecnica_balanceo, args.seed), param_dist,
                            n_iter=CFG["n_iter"], cv=cv, scoring=METRICAS_CV, refit=args.scoring,
                            random_state=args.seed, n_jobs=-1, verbose=1)
    rs.fit(X, y)
    run_rs = registrar_run(rs, "RandomSearch-RandomForest", "RandomizedSearchCV", X, y,
                           {"n_iter": CFG["n_iter"]})

    # ── Selección del mejor run (por validación cruzada; el test queda para evaluate) ──
    mejor = max([run_gs, run_rs], key=lambda r: r["cv_score"])
    log.info("Mejor run: %s (CV %s=%.4f)", mejor["metodo"], args.scoring, mejor["cv_score"])
    log.info("Baseline : Regresión Logística (CV %s=%.4f)", args.scoring, baseline[CLAVE_CV[args.scoring]])

    # Tabla comparativa baseline vs tuning (README / Model Card); evaluate agrega el test
    comparativa = [baseline] + [dict(r["metricas"], seleccionado=r is mejor) for r in (run_gs, run_rs)]
    with open(ARTIFACTS / "comparativa_modelos.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(comparativa, f, indent=2, ensure_ascii=False)

    # ── Model Registry: registrar y promover a Staging ────────────────────────
    version = mlflow.register_model(mejor["model_uri"], args.model_name).version
    promover_a_staging(args.model_name, version)
    agregar_tags(args.model_name, version, {
        "metodo_busqueda":  mejor["metodo"],
        "tecnica_balanceo": args.tecnica_balanceo,
        "cv_score":         mejor["cv_score"],
        "git_commit":       os.getenv("GITHUB_SHA", "local"),
    })

    # Guardar modelo localmente también (lo usa evaluate y la API)
    with open(ARTIFACTS / "modelo_infartos.pkl", "wb") as f:
        pickle.dump(mejor["modelo"], f)

    # Guardar referencia al registro para la etapa evaluate
    with open(ARTIFACTS / "modelo_registrado.json", "w", newline="\n") as f:
        json.dump({"run_id": mejor["run_id"], "model_name": args.model_name,
                   "version": str(version), "metodo_busqueda": mejor["metodo"],
                   "cv_score": mejor["cv_score"]}, f, indent=2)

    log.info("Modelo guardado: artifacts/modelo_infartos.pkl")
    log.info("Registro guardado: artifacts/modelo_registrado.json (v%s en Staging)", version)
