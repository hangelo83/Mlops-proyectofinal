"""
drift/run.py — Etapa 7: Detección de drift con EvidentlyAI + PSI.

Compara la distribución del train set (referencia) vs el test set (producción).
Genera reporte HTML (Evidently), JSON y gráfico PSI por variable (Unidad 9).

La Unidad 10 usaba la API antigua de Evidently (evidently.report), que falla en
Evidently >= 0.5; aquí se usa la API nueva de la Unidad 9 y se conserva el
fallback KS por si Evidently no está instalado.

Ejecutar:
    python drift/run.py                  # train vs test
    python drift/run.py --simular_drift  # demo: población más envejecida y con más diabetes
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import estilo_graficos as G       # noqa: E402
from src.config import CONFIG              # noqa: E402
from src.preprocessing import FEATURES     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DRIFT | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

REPORTS = Path("reportes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",       default=CONFIG["drift"]["reference_path"])
    parser.add_argument("--current",         default=CONFIG["drift"]["current_path"])
    parser.add_argument("--drift_threshold", type=float, default=CONFIG["drift"]["drift_threshold"])
    parser.add_argument("--simular_drift",   action="store_true")
    return parser.parse_args()


def simular_drift(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Simula un lote de producción con más adultos mayores y más diabéticos (Unidad 9)."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    adultos = df.index[df["Edad_Encoded"] == 5]
    df.loc[rng.choice(adultos, size=len(adultos) // 2, replace=False), "Edad_Encoded"] = 6
    normales = df.index[df["Gluocosa_Encoded"] == 1]
    df.loc[rng.choice(normales, size=len(normales) * 3 // 10, replace=False), "Gluocosa_Encoded"] = 3
    return df


def calcular_psi(ref: pd.Series, prod: pd.Series, bins: int = 10) -> float:
    """Population Stability Index (PSI)."""
    breakpoints = np.linspace(
        min(ref.min(), prod.min()),
        max(ref.max(), prod.max()),
        bins + 1,
    )
    ref_pct  = np.histogram(ref,  bins=breakpoints)[0] / len(ref)
    prod_pct = np.histogram(prod, bins=breakpoints)[0] / len(prod)

    # Evitar log(0)
    ref_pct  = np.where(ref_pct  == 0, 1e-6, ref_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return round(float(psi), 4)


def clasificar_psi(psi: float) -> str:
    """Clasifica PSI según umbrales estándar."""
    if psi < 0.10:
        return "ESTABLE"
    if psi < 0.25:
        return "ALERTA"
    return "CRÍTICO"


def grafico_psi_barras(psis: dict) -> Path:
    """Barras horizontales de PSI por variable; el estado va escrito junto a cada barra."""
    G.aplicar_estilo()
    nombres = list(psis)[::-1]
    valores = [psis[n] for n in nombres]
    estados = [clasificar_psi(v) for v in valores]

    fig, ax = G.plt.subplots(figsize=(10, 4.5))
    ax.barh(nombres, valores, color=[G.ESTADO[e] for e in estados], height=0.55)
    tope = max(max(valores), 0.30) * 1.25
    for i, (v, e) in enumerate(zip(valores, estados)):
        ax.text(v + tope * 0.01, i, f"{v:.3f}  {e}", va="center", fontsize=9, color=G.TEXTO)
    for x, etiqueta in [(0.10, "0.10 alerta"), (0.25, "0.25 crítico")]:
        ax.axvline(x, color=G.TEXTO_SEC, linestyle="--", linewidth=1)
        ax.text(x + tope * 0.008, -0.9, etiqueta, ha="left", fontsize=8, color=G.TEXTO_SEC)
    ax.set_ylim(-1.1, len(nombres) - 0.5)
    ax.set_xlim(0, tope)
    ax.set_xlabel("PSI (referencia = train)")
    ax.set_title("Population Stability Index por variable", loc="left")
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    G.plt.tight_layout()
    salida = REPORTS / "psi_barras.png"
    G.plt.savefig(salida, dpi=120, bbox_inches="tight")
    G.plt.close()
    return salida


if __name__ == "__main__":
    args = parse_args()
    REPORTS.mkdir(exist_ok=True)

    for p in [args.reference, args.current]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    df_ref  = pd.read_csv(args.reference)[FEATURES]
    df_prod = pd.read_csv(args.current)[FEATURES]
    if args.simular_drift:
        df_prod = simular_drift(df_prod)
        log.info("Drift simulado en el lote actual (Edad_Encoded y Gluocosa_Encoded)")
    log.info("Referencia: %d | Actual: %d muestras", len(df_ref), len(df_prod))

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        snapshot = Report(metrics=[DataDriftPreset()]).run(
            reference_data=df_ref, current_data=df_prod)
        snapshot.save_html(str(REPORTS / "drift_report.html"))
        resumen_ev  = snapshot.dict()["metrics"][0]["value"]
        drift_n     = int(resumen_ev.get("count", 0))
        drift_share = float(resumen_ev.get("share", 0.0))
        metodo      = "evidently"
        log.info("Reporte HTML: reportes/drift_report.html")
    except ImportError:
        from scipy.stats import ks_2samp
        drift_n     = sum(1 for c in FEATURES if ks_2samp(df_ref[c], df_prod[c]).pvalue < 0.05)
        drift_share = drift_n / len(FEATURES)
        metodo      = "ks_2samp"

    psis = {c: calcular_psi(df_ref[c], df_prod[c]) for c in FEATURES}
    grafico = grafico_psi_barras(psis)
    log.info("Gráfico PSI: %s", grafico)

    # Alerta si muchas variables derivan (Evidently/KS) o si alguna tiene PSI crítico
    psi_critico = [c for c, v in psis.items() if clasificar_psi(v) == "CRÍTICO"]
    drift_det = drift_share > args.drift_threshold or bool(psi_critico)
    resumen = {"drift_detectado": drift_det, "features_con_drift": drift_n,
               "total_features": len(FEATURES), "share_drifted": round(drift_share, 4),
               "umbral": args.drift_threshold, "metodo": metodo, "psi_critico": psi_critico,
               "psi": {c: {"valor": v, "estado": clasificar_psi(v)} for c, v in psis.items()}}
    with open(REPORTS / "drift_summary.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\n  Drift detectado    : {drift_det}")
    print(f"  Features con drift : {drift_n}/{len(FEATURES)} ({drift_share*100:.0f}%)")
    for c, v in psis.items():
        print(f"    PSI {c:<24}: {v:.4f} [{clasificar_psi(v)}]")
    if drift_det:
        log.warning("ALERTA: drift significativo — considerar re-entrenamiento")
    else:
        print("  ✓ Drift dentro de límites aceptables")
