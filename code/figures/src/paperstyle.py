"""paperstyle.py -- shared matplotlib style for the paper's PDF figures.

Discipline (per figure-spec): Times-family serif matching the body font
(Nimbus Roman + STIX mathtext), ~8pt base, Okabe-Ito colorblind-safe palette,
white background, thin spines, y-only light dashed grid, vector PDF output.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito
OI_BLUE = "#0072B2"
OI_VERMIL = "#D55E00"
OI_GREEN = "#009E73"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_PURPLE = "#CC79A7"


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": "0.85",
        "grid.linewidth": 0.5,
        "grid.linestyle": (0, (3, 3)),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })


def tidy(ax):
    """Thin spines, hidden top/right, y-only light dashed grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y")
    ax.set_axisbelow(True)

