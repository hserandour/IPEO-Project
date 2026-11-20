"""Precision-recall curves, pixel-wise baseline vs U-Net (E0). ILLUSTRATIVE CURVES, to be replaced by results."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import new, save  # noqa: E402

PREVALENCE = 0.05  # dummy share of burned pixels in the test set
r = np.linspace(0, 1, 200)


def pr(k):
    """Smooth dummy PR curve that starts near 1 and falls to the prevalence."""
    return PREVALENCE + (1 - PREVALENCE) * (1 - r ** k) ** 1.5


fig, ax = new()
ax.plot(r, pr(1.6), "k-", lw=1.2, label="U-Net (AUPRC 0.58)")
ax.plot(r, pr(0.9), "k--", lw=1.2, label="Pixel-wise baseline (AUPRC 0.41)")
ax.axhline(PREVALENCE, color="k", ls=":", lw=0.8, label="No skill (prevalence)")
ax.set_xlabel("Recall [-]")
ax.set_ylabel("Precision [-]")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, color="0.55", lw=0.4)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fancybox=False, framealpha=1, edgecolor="none")
save(fig, __file__)
