"""SeasFire project roadmap (Gantt chart).

Self-contained: the plan is defined below and the only dependency is matplotlib.
Run `python roadmap/seasfire_roadmap.py` to write seasfire_roadmap.pdf next to this file.
"""

from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- plan data --
TITLE = "SeasFire spatial shift · A = Martin, B = Herve"
START, END, TODAY = date(2026, 11, 20), date(2027, 1, 9), date(2026, 11, 20)

PERIODS = [  # (start, end, label); a one-day period is shown as a label only
    (date(2026, 11, 27), date(2026, 11, 27), "Midterm"),
    (date(2026, 12, 24), date(2027, 1, 1), "Holidays"),
]

MILESTONES = [  # (date, label, is_deadline)
    (date(2026, 11, 26), "M1", False),
    (date(2026, 12, 4), "M2", False),
    (date(2026, 12, 11), "M3", False),
    (date(2026, 12, 18), "M4", False),
    (date(2026, 12, 23), "M5", False),
    (date(2027, 1, 6), "M6", False),
    (date(2027, 1, 8), "M7", True),
]

GROUPS = {  # group -> [(task, start, end)], dates as "MM-DD" within the project
    "Setup": [
        ("0.1 Confirm scope & team with TAs (A+B)", "11-20", "11-22"),
        ("0.2 Repo, env, compute (B)", "11-20", "11-24"),
    ],
    "Data pipeline": [
        ("1.1 Download cube & mask (A)", "11-20", "11-23"),
        ("1.2 Exploratory analysis (A)", "11-22", "11-27"),
        ("1.3 Preprocessing (A)", "11-24", "11-30"),
        ("1.4 Patches & dataloaders (A)", "11-28", "12-04"),
    ],
    "Models & training": [
        ("2.1 Literature review (B)", "11-20", "11-29"),
        ("2.2 Pixel-wise baseline (B)", "11-26", "12-02"),
        ("2.3 U-Net & imbalance loss (B)", "11-29", "12-08"),
        ("2.4 E0 all regions (B)", "12-04", "12-11"),
    ],
    "Evaluation": [
        ("3.1 Metrics per region (A)", "12-01", "12-07"),
        ("3.2 Maps, PR curves, timing (A)", "12-06", "12-13"),
    ],
    "Spatial experiments": [
        ("4.1 E1 without Africa (A)", "12-08", "12-15"),
        ("4.2 E2 random removal, 3 seeds (A)", "12-10", "12-18"),
        ("4.3 E3 balanced training (B)", "12-11", "12-21"),
        ("4.4 Consolidate & freeze (A+B)", "12-18", "12-22"),
    ],
    "Temporal bonus": [
        ("5.1 Year selection, stretch (A)", "12-18", "12-24"),
    ],
    "Report": [
        ("6.1 Template & outline (A)", "12-11", "12-14"),
        ("6.2 Intro & literature (B)", "12-14", "12-21"),
        ("6.3 Data & methods (A)", "12-14", "12-22"),
        ("6.4 Results (A)", "12-21", "12-28"),
        ("6.5 Discussion (B)", "12-22", "12-31"),
        ("6.6 Review, cut to 10 p (A+B)", "01-02", "01-07"),
    ],
    "Code & submission": [
        ("7.1 Clean code & README (B)", "12-28", "01-06"),
        ("7.2 Fresh-env reproduction (A)", "01-04", "01-07"),
        ("7.3 Submit on Moodle (A+B)", "01-07", "01-09"),
    ],
}


def d(mmdd):
    """'MM-DD' -> date, placing Jan dates in the year after the project start."""
    m, day = map(int, mmdd.split("-"))
    return date(START.year + (m < START.month), m, day)


# ------------------------------------------------------------------ drawing --
def build_rows():
    """Flatten the plan into rows, top to bottom: (kind, label, start, end)."""
    rows = [("milestones", "Milestones", None, None), ("gap", "", None, None)]
    for name, tasks in GROUPS.items():
        spans = [(d(s), d(e)) for _, s, e in tasks]
        rows.append(("group", name, min(s for s, _ in spans), max(e for _, e in spans)))
        rows += [("task", t, s, e) for (t, _, _), (s, e) in zip(tasks, spans)]
        rows.append(("gap", "", None, None))
    return rows[:-1]


def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 7})
    rows = build_rows()
    n = len(rows)
    x0, x1 = mdates.date2num(START), mdates.date2num(END)

    fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
    ax.set_xlim(x0, x1)
    ax.set_ylim(n - 0.5, -1.6)

    # background: holiday shading, group bands, task guide lines
    for s, e, label in PERIODS:
        if e > s:
            ax.axvspan(mdates.date2num(s), mdates.date2num(e), color="0.9", lw=0, zorder=0)
        ax.text(mdates.date2num(s + (e - s) / 2) + (0.5 if e == s else 0), -1.25, label,
                ha="center", va="center", style="italic", fontsize=6)
    for y, (kind, *_rest) in enumerate(rows):
        if kind in ("group", "milestones"):
            ax.axhspan(y - 0.5, y + 0.5, color="0.95", lw=0, zorder=0.5)
        elif kind == "task":
            ax.axhline(y, color="0.6", lw=0.5, ls=(0, (4, 3)), zorder=1)

    # bars and milestones
    for y, (kind, label, s, e) in enumerate(rows):
        if kind == "group":
            ax.barh(y, (e - s).days, left=mdates.date2num(s), height=0.3, color="k", zorder=3)
        elif kind == "task":
            ax.barh(y, (e - s).days, left=mdates.date2num(s), height=0.6,
                    color="0.85", edgecolor="k", lw=0.7, zorder=3)
        elif kind == "milestones":
            for md, ml, deadline in MILESTONES:
                xm = mdates.date2num(md)
                ax.plot(xm, y, "D", ms=6, mfc="k" if deadline else "0.75", mec="k", mew=0.7, zorder=4)
                ax.text(xm, y - 0.55, ml, ha="center", va="bottom", style="italic", fontsize=6)

    # today line
    ax.axvline(mdates.date2num(TODAY), color="k", lw=1.2, ls=(0, (2, 1.5)), zorder=5)
    ax.text(mdates.date2num(TODAY) + 0.3, -1.25, "Today", ha="left", va="center", style="italic",
            fontsize=5.5, bbox=dict(fc="white", ec="none", pad=0.5), zorder=6)

    # y labels: groups bold
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[1] for r in rows])
    for tick, (kind, *_rest) in zip(ax.get_yticklabels(), rows):
        if kind in ("group", "milestones"):
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.text(-0.01, -1.25, TITLE, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontweight="bold")

    # bottom axis: every second Monday; top axis: ISO week numbers centred on each week
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b"))
    ax.tick_params(axis="x", length=3)
    top = ax.secondary_xaxis("top")
    monday = START - timedelta(days=START.weekday())
    weeks = []
    while monday < END:
        weeks.append(monday)
        monday += timedelta(days=7)
    top.set_xticks([mdates.date2num(w) + 3.5 for w in weeks])
    top.set_xticklabels([f"W{w.isocalendar()[1]}" for w in weeks])
    top.tick_params(length=0)

    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    top.spines["top"].set_visible(False)
    ax.spines["left"].set_linestyle((0, (2, 1.5)))

    legend = [Line2D([], [], marker="D", ls="", mfc="0.75", mec="k", label="Milestone"),
              Line2D([], [], marker="D", ls="", mfc="k", mec="k", label="Deadline")]
    ax.legend(handles=legend, loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2,
              frameon=False, fontsize=6, handletextpad=0.3)

    fig.tight_layout()
    out = Path(__file__).with_suffix(".pdf")
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
