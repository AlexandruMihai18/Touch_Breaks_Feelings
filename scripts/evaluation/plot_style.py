"""Shared publication-quality matplotlib style for evaluation figures.

Import and call apply() at the top of any evaluation script before
creating figures:

    import sys; sys.path.insert(0, ...)
    import plot_style
    plot_style.apply()
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Colour tokens ────────────────────────────────────────────────────────────

DARK  = "#111827"
GRAY  = "#6B7280"
LGRAY = "#D1D5DB"

C_RECALL = "#2563EB"
C_PREC   = "#EA580C"
C_F1     = "#16A34A"
C_ACC    = "#0891B2"

C_TP = "#15803D"
C_TN = "#1D4ED8"
C_FP = "#D97706"
C_FN = "#B91C1C"

METRIC_COLORS = {
    "recall":    C_RECALL,
    "precision": C_PREC,
    "f1":        C_F1,
    "accuracy":  C_ACC,
}

CONF_COLORS = {"tp": C_TP, "tn": C_TN, "fp": C_FP, "fn": C_FN}

# ── rcParams ─────────────────────────────────────────────────────────────────

_RC: dict = {
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size":        10,
    "text.color":       DARK,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.titlepad":    10,
    "axes.labelsize":   10,
    "axes.labelpad":    5,
    "axes.labelcolor":  DARK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "axes.edgecolor":    LGRAY,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.color":       GRAY,
    "ytick.color":       GRAY,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "axes.axisbelow":    True,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.color":        "#E5E7EB",
    "grid.linewidth":    0.6,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.08,
    "legend.frameon":        False,
    "legend.fontsize":       9,
    "legend.labelspacing":   0.35,
    "legend.handlelength":   1.4,
    "legend.handletextpad":  0.4,
    "figure.constrained_layout.use":    True,
    "figure.constrained_layout.h_pad": 0.08,
    "figure.constrained_layout.w_pad": 0.08,
}


def apply() -> None:
    """Apply publication-quality rcParams to all subsequent figures."""
    mpl.rcParams.update(_RC)


def recall_colors(values: list[float], vmin: float = 0.0, vmax: float = 1.0) -> list:
    """Map metric values to a muted RdYlGn colour scale."""
    cmap = plt.cm.RdYlGn
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return [cmap(norm(v)) for v in values]
