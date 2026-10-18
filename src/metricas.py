"""
src/metricas.py — Métricas de clasificación global y por subgrupo.

calcular_metricas() viene de la Unidad 9 (02_entrenar_modelo.py).
metricas_por_grupo() la aplica a cada subgrupo (género, zona) para la
auditoría de fairness que la Unidad 2 deja como mitigación del riesgo de sesgo.
"""
import logging

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

log = logging.getLogger(__name__)

MIN_POSITIVOS = 30   # bajo este número de infartos el recall del grupo no es confiable


def calcular_metricas(y_true, y_pred, nombre: str) -> dict:
    """Calcula y retorna métricas de clasificación."""
    m = {
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
    }
    log.info("[%s] F1=%.4f | Recall=%.4f | Acc=%.4f", nombre, m["f1"], m["recall"], m["accuracy"])
    return m


def metricas_por_grupo(grupos: pd.Series, y_true: pd.Series, y_pred) -> dict:
    """Métricas por subgrupo + tasa de asegurados marcados como alto riesgo."""
    df = pd.DataFrame({"grupo": grupos.values, "y": y_true.values, "pred": pd.Series(y_pred).values})
    resultado = {}
    for grupo, d in df.groupby("grupo"):
        positivos = int(d["y"].sum())
        m = calcular_metricas(d["y"], d["pred"], str(grupo))
        resultado[str(grupo)] = {
            "n":              len(d),
            "infartos":       positivos,
            "recall":         m["recall"],
            "precision":      m["precision"],
            "tasa_marcados":  round(float(d["pred"].mean()), 4),
            "muestra_pequena": positivos < MIN_POSITIVOS,
        }
    return resultado
