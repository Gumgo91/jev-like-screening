"""Shared figure style for the manuscript (ACS column widths, Arial, colour-blind-safe palette)."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SINGLE = 3.33   # inches, ACS single column
DOUBLE = 7.0    # inches, ACS double column

PAL = {
    "blue": "#1f5fa8", "orange": "#d9730d", "green": "#2a8a4a", "red": "#c0392b",
    "purple": "#6f4a9e", "grey": "#6b6b6b", "light": "#d9d9d9", "teal": "#1a8c8c",
    "black": "#1a1a1a",
}


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    })


def panel(ax, letter, dx=-0.16, dy=1.04):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
