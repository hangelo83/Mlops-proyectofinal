"""
src/estilo_graficos.py — Paleta y estilo común de los reportes PNG del pipeline.

Paleta validada para daltonismo (azul/naranja ΔE CVD 24.7). Los colores de estado
(bien/alerta/crítico) siempre van acompañados de su etiqueta de texto.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SUPERFICIE  = "#fcfcfb"
TEXTO       = "#0b0b0b"
TEXTO_SEC   = "#52514e"
GRILLA      = "#e4e3df"
SERIE_1     = "#2a78d6"   # azul — serie principal / referencia
SERIE_2     = "#eb6834"   # naranja — segunda serie / datos actuales
ESTADO = {"ESTABLE": "#0ca30c", "ALERTA": "#fab219", "CRÍTICO": "#d03b3b"}

# Rampa secuencial azul (claro → oscuro) para la matriz de confusión
AZULES = LinearSegmentedColormap.from_list(
    "azules", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def aplicar_estilo() -> None:
    """Fondo claro, ejes y grilla discretos, texto en tinta neutra."""
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor":   SUPERFICIE,
        "axes.edgecolor":   GRILLA,
        "axes.labelcolor":  TEXTO_SEC,
        "axes.titlecolor":  TEXTO,
        "axes.titleweight": "bold",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "xtick.color":      TEXTO_SEC,
        "ytick.color":      TEXTO_SEC,
        "grid.color":       GRILLA,
        "text.color":       TEXTO,
        "font.size":        10,
    })
