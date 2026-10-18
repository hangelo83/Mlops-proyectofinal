"""
evaluate/run.py — Etapa 6: Evaluación en test set holdout + quality gate.

Carga el modelo entrenado, evalúa en el test set y registra métricas en MLflow.
Quality gate (Unidad 10 + validar_recall de la Unidad 4): el pipeline falla si
AUC, Recall o F1 quedan bajo su umbral. Si aprueba, la versión registrada pasa
de Staging a Production (Unidad 8).

También compara contra la línea base en el mismo test y audita fairness por
subgrupo (género y zona de residencia) → reportes/fairness.json.

Ejecutar: python evaluate/run.py
"""
import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import estilo_graficos as G                                     # noqa: E402
from src.config import CONFIG                                            # noqa: E402
from src.manage_versions import agregar_tags, promover_a_produccion      # noqa: E402
from src.metricas import metricas_por_grupo                              # noqa: E402
from src.preprocessing import FEATURES, ID_COL, TARGET                   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | EVALUATE | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")
REPORTS   = Path("reportes")
CFG       = CONFIG["model"]
COSTO_FN  = CONFIG["negocio"]["costo_fn"]
COSTO_FP  = CONFIG["negocio"]["costo_fp"]
# Atributos que el modelo NO usa, auditados para detectar trato desigual
GRUPOS_FAIRNESS = ["Genero", "Zona_residencia"]


def evaluar_baseline(X_test, y_test) -> dict:
    """Métricas de la línea base (random_forest/run.py) sobre el mismo test."""
    path = ARTIFACTS / "modelo_baseline.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        base = pickle.load(f)
    pred, proba = base.predict(X_test), base.predict_proba(X_test)[:, 1]
    return {
        "test_auc":       round(roc_auc_score(y_test, proba), 4),
        "test_recall":    round(recall_score(y_test, pred, zero_division=0), 4),
        "test_precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "test_f1":        round(f1_score(y_test, pred, zero_division=0), 4),
    }


def auditoria_fairness(df_test: pd.DataFrame, y_pred) -> dict:
    """Recall y tasa de marcados por subgrupo; los atributos vienen del CSV crudo vía ID."""
    crudo = pd.read_csv(CONFIG["data"]["raw_path"], usecols=[ID_COL] + GRUPOS_FAIRNESS)
    df = df_test[[ID_COL, TARGET]].merge(crudo, on=ID_COL, how="left")
    return {col: metricas_por_grupo(df[col], df[TARGET], y_pred) for col in GRUPOS_FAIRNESS}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",       default=CONFIG["data"]["test_path"])
    parser.add_argument("--model_path",      default="artifacts/modelo_infartos.pkl")
    parser.add_argument("--experiment_name", default=CONFIG["main"]["experiment_name"])
    parser.add_argument("--auc_threshold",    type=float, default=CFG["auc_threshold"])
    parser.add_argument("--recall_threshold", type=float, default=CFG["recall_threshold"])
    parser.add_argument("--f1_threshold",     type=float, default=CFG["f1_threshold"])
    return parser.parse_args()


def generar_reporte(metricas: dict, cm, umbrales: dict) -> Path:
    """Reporte visual PNG: métricas vs umbrales y matriz de confusión (Unidad 4)."""
    G.aplicar_estilo()
    fig, axes = G.plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.suptitle("Evaluación en test — Modelo de prevención de infartos",
                 fontsize=13, fontweight="bold", color=G.TEXTO)

    # Panel 1: métricas (una sola serie, un solo color)
    nombres = ["test_auc", "test_recall", "test_precision", "test_f1", "test_accuracy"]
    etiquetas = ["AUC", "Recall", "Precision", "F1", "Accuracy"]
    vals = [metricas[m] for m in nombres]
    ax = axes[0]
    bars = ax.bar(etiquetas, vals, color=G.SERIE_1, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=9, color=G.TEXTO)
    # Umbral del quality gate dibujado sobre cada barra que lo tiene
    for i, nombre in enumerate(nombres):
        if nombre in umbrales:
            u = umbrales[nombre]
            ax.hlines(u, i - 0.38, i + 0.38, colors=G.TEXTO_SEC, linestyles="--", linewidth=1.5)
            ax.text(i + 0.40, u, f"mín {u:.2f}", va="center", fontsize=8, color=G.TEXTO_SEC)
    ax.set_ylim(0, 1.12)
    ax.set_title("Métricas vs umbral del quality gate", loc="left")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)

    # Panel 2: matriz de confusión (rampa secuencial azul)
    ax = axes[1]
    ax.imshow(cm, cmap=G.AZULES)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                    fontweight="bold", color="white" if cm[i, j] > cm.max() / 2 else G.TEXTO)
    ax.set_xticks([0, 1], ["Sin infarto", "Infarto"])
    ax.set_yticks([0, 1], ["Sin infarto", "Infarto"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(0.5, -0.22,
            f"Costo estimado: FN={metricas['FN']}×${COSTO_FN:,} + FP={metricas['FP']}×${COSTO_FP:,}"
            f" = ${metricas['costo_estimado']:,}",
            transform=ax.transAxes, ha="center", fontsize=9, color=G.TEXTO_SEC)

    G.plt.tight_layout()
    REPORTS.mkdir(exist_ok=True)
    salida = REPORTS / "reporte_evaluacion.png"
    G.plt.savefig(salida, dpi=120, bbox_inches="tight")
    G.plt.close()
    return salida


if __name__ == "__main__":
    args = parse_args()

    for p in [args.test_data, args.model_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    log.info("Cargando modelo: %s", args.model_path)
    with open(args.model_path, "rb") as f:
        modelo = pickle.load(f)

    log.info("Cargando test set: %s", args.test_data)
    df_test = pd.read_csv(args.test_data)
    X_test  = df_test[FEATURES]
    y_test  = df_test[TARGET]

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    cm      = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()

    metricas = {
        "test_auc":       round(roc_auc_score(y_test, y_proba), 4),
        "test_recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "test_f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP),
        "costo_estimado": int(FN * COSTO_FN + FP * COSTO_FP),
        "modelo":         type(modelo.named_steps["modelo"]).__name__,
    }

    print("\n" + "="*50)
    print(" MÉTRICAS DE EVALUACIÓN (test set holdout)")
    print("="*50)
    for k in ["test_auc", "test_recall", "test_precision", "test_f1", "test_accuracy"]:
        print(f"  {k:<20}: {metricas[k]:.4f}")
    print(f"  {'FN / FP':<20}: {FN} / {FP}")
    print(f"  {'costo_estimado':<20}: ${metricas['costo_estimado']:,}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Sin infarto", "Infarto"]))

    # Línea base: todo modelo nuevo se compara con un baseline simple
    baseline = evaluar_baseline(X_test, y_test)
    if baseline:
        metricas["baseline"] = baseline
        print("="*50)
        print(" MODELO vs LÍNEA BASE (Regresión Logística)")
        print("="*50)
        for k in ["test_auc", "test_recall", "test_precision", "test_f1"]:
            print(f"  {k:<16}: {metricas[k]:.4f}  vs  {baseline[k]:.4f}")

    # Fairness por subgrupo
    fairness = auditoria_fairness(df_test, y_pred)
    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "fairness.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(fairness, f, indent=2, ensure_ascii=False)
    print("\n" + "="*50)
    print(" FAIRNESS POR SUBGRUPO (test)")
    print("="*50)
    for col, grupos in fairness.items():
        for grupo, m in grupos.items():
            nota = "  (muestra pequeña)" if m["muestra_pequena"] else ""
            print(f"  {col}={grupo:<10} n={m['n']:<5} infartos={m['infartos']:<4} "
                  f"recall={m['recall']:.3f}  marcados={m['tasa_marcados']:.1%}{nota}")
    log.info("Fairness guardado: reportes/fairness.json")

    umbrales = {"test_auc": args.auc_threshold, "test_recall": args.recall_threshold,
                "test_f1": args.f1_threshold}
    reporte = generar_reporte(metricas, cm, umbrales)
    log.info("Reporte guardado: %s", reporte)

    # Registrar en MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name="evaluate"):
        mlflow.set_tag("etapa", "evaluate")
        mlflow.log_metrics({k: v for k, v in metricas.items() if isinstance(v, (int, float))})
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline.items()})
        mlflow.log_metrics({f"recall_{col}_{grupo}": m["recall"]
                            for col, grupos in fairness.items() for grupo, m in grupos.items()})
        mlflow.log_params({k.replace("test_", "") + "_threshold": v for k, v in umbrales.items()})
        mlflow.log_artifact(str(reporte))
        mlflow.log_dict(fairness, "fairness.json")

    # Guardar métricas JSON (las lee la API en /health)
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "eval_metrics.json", "w", newline="\n") as f:
        json.dump(metricas, f, indent=2)
    log.info("Métricas guardadas: artifacts/eval_metrics.json")

    # Quality gate
    print("\n" + "="*50)
    print(" QUALITY GATE")
    print("="*50)
    fallos = []
    for nombre, umbral in umbrales.items():
        estado = "OK" if metricas[nombre] >= umbral else "FALLA"
        print(f"  {nombre:<12}: {metricas[nombre]:.4f} (umbral: >= {umbral})  [{estado}]")
        if metricas[nombre] < umbral:
            fallos.append(f"{nombre} {metricas[nombre]:.4f} < {umbral}")

    if fallos:
        log.error("QUALITY GATE FALLIDO: %s", " | ".join(fallos))
        log.error("El pipeline CI/CD se detiene. El modelo queda en Staging y no se despliega.")
        sys.exit(1)

    # Aprobado: Staging → Production en el Model Registry
    registro_path = ARTIFACTS / "modelo_registrado.json"
    if registro_path.exists():
        with open(registro_path) as f:
            registro = json.load(f)
        promover_a_produccion(registro["model_name"], registro["version"])
        agregar_tags(registro["model_name"], registro["version"], {
            "test_auc": metricas["test_auc"], "test_recall": metricas["test_recall"],
            "quality_gate": "aprobado",
        })

    print("\n  ✓ APROBADO — modelo en Production, listo para despliegue")
