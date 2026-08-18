"""Shared figure styling for the report.

Figures are vector PDFs sized for \\linewidth in a 10pt LaTeX document, so the type is
set at the size it will actually be read at rather than scaled down afterwards — scaling
a 300dpi PNG into a column is the usual reason report figures have illegible axes.

The categorical palette is the validated four-slot order (blue, orange, aqua, yellow),
which clears the colour-vision-deficiency separation floor on adjacent pairs. Two of its
slots fall below 3:1 contrast against the page, so every figure here also carries direct
labels — colour is never the only thing distinguishing a mark.
"""

from __future__ import annotations

import matplotlib as mpl

# Validated categorical order — assign by slot, never cycle.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

INK = "#1a1a19"
INK_MUTED = "#6b6b66"
GRID = "#e3e3df"
SURFACE = "#ffffff"


def use_report_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        # Serif, to sit with the report's body text rather than against it.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        # Recessive frame: keep the data the most prominent thing on the page.
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    })


def strip_frame(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
