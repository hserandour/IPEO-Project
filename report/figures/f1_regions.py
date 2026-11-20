"""F1 score per test region for experiments E0-E3. ILLUSTRATIVE NUMBERS, to be replaced by results."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from style import new, save  # noqa: E402

REGIONS = ["Africa", "S. Amer.", "N. Amer.", "Europe", "Asia", "Austr."]
F1 = {  # experiment -> F1 per region (dummy)
    "E0 all regions": [0.62, 0.55, 0.48, 0.41, 0.52, 0.57],
    "E1 w/o Africa": [0.38, 0.53, 0.47, 0.40, 0.49, 0.51],
    "E2 random removal": [0.55, 0.50, 0.44, 0.37, 0.47, 0.52],
    "E3 balanced": [0.58, 0.56, 0.50, 0.45, 0.53, 0.56],
}
SHADES = ["k", "0.4", "0.7", "white"]

fig, ax = new()
x = np.arange(len(REGIONS))
w = 0.2
for i, ((name, vals), c) in enumerate(zip(F1.items(), SHADES)):
    ax.bar(x + (i - 1.5) * w, vals, w, color=c, edgecolor="k", lw=0.5, label=name)
ax.set_xticks(x, REGIONS)
ax.set_ylabel("F1 score [-]")
ax.set_ylim(0, 0.8)
ax.yaxis.grid(True, color="0.55", lw=0.4)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.legend(ncol=2, fancybox=False, framealpha=1, edgecolor="none", loc="upper right", handlelength=1, columnspacing=1)
save(fig, __file__)
