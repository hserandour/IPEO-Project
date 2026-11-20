"""Shared plot style for the report figures: black and white, sans serif, drawn at printed size.

Half-line slot = 86 mm wide (two panels per row); text prints at 7-8 pt with no LaTeX rescaling.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4  # inches per mm
HALF = 86 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def new(w=HALF, h=62 * MM):
    """One panel at its printed size."""
    return plt.subplots(figsize=(w, h))


def save(fig, script):
    """Write <script>.pdf next to the script."""
    out = Path(script).with_suffix(".pdf")
    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out.name}")
