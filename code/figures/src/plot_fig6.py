#!/usr/bin/env python3
"""plot_fig6.py -- Finding 1 tau-trajectory figure (three seed panels).

Input:  ../data/fig6_trajectory.csv  (from make_fig6_data.py)
Output: ../fig6_tau_trajectory.pdf   (vector)

One message: in the deep-drift regime the merge's scale-invariant margin on
the endpoint-surviving cohorts separates ordinally below both endpoint
trajectories. x = tau in tokens (symlog; real token ticks); shaded region and
dashed reference line mark the deep-drift regime (tau >= 50M).
"""
import csv
import os

import matplotlib.pyplot as plt

import paperstyle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "fig6_trajectory.csv")
OUT = os.path.join(HERE, "..", "fig6_tau_trajectory.pdf")

SEEDS = [42, 1042, 2042]
TAU_LABELS = ["0", "3.1", "6.3", "12.5", "25", "50", "100", "200"]
DEEP = 50003968          # 50M tokens: deep-drift regime boundary
XMAX = 2.9e8

LINES = {
    "merge": dict(color=paperstyle.OI_VERMIL, ls="-", marker="o", ms=2.8,
                  lw=1.5, label=r"merge$(0.5)$", zorder=4),
    "endA": dict(color=paperstyle.OI_BLUE, ls=(0, (4, 2)), marker="s", ms=2.6,
                 lw=1.0, label="endpoint A", zorder=2, mfc="white",
                 mew=0.7),
    "endB": dict(color=paperstyle.OI_GREEN, ls=(0, (1, 1.2)), marker="^",
                 ms=2.6, lw=1.4, label="endpoint B", zorder=3),
}

rows = []
with open(DATA) as f:
    for r in csv.DictReader(f):
        rows.append({k: (float(v) if k not in ("line",) else v)
                     for k, v in r.items()})


def take(seed, line):
    sel = [r for r in rows if r["seed"] == seed and r["line"] == line]
    sel.sort(key=lambda r: r["tau_idx"])
    x = [r["tau_tokens"] for r in sel]
    y = [r["y"] for r in sel]
    lo = [r["ci_lo"] for r in sel]
    hi = [r["ci_hi"] for r in sel]
    return x, y, lo, hi


paperstyle.apply()
fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95), sharey=True)
fig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.30,
                    wspace=0.06)

for ax, seed in zip(axes, SEEDS):
    paperstyle.tidy(ax)
    ax.set_xscale("symlog", linthresh=1.2e6)
    ax.set_xlim(-1.0e6, XMAX)
    # deep-drift shading + reference line
    ax.axvspan(DEEP, XMAX, color="0.93", zorder=0)
    ax.axvline(DEEP, color="0.45", lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax.axhline(0.0, color="0.25", lw=0.6, ls=(0, (2, 2)), zorder=1)
    for line, st in LINES.items():
        x, y, lo, hi = take(seed, line)
        ax.fill_between(x, lo, hi, color=st["color"], alpha=0.16, lw=0,
                        zorder=1)
        ax.plot(x, y, color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=st["ms"], lw=st["lw"], label=st["label"],
                zorder=st["zorder"], markeredgewidth=st.get("mew", 0),
                markerfacecolor=st.get("mfc", st["color"]))
    ax.set_title("seed {}".format(seed), pad=2.5)
    ticks = [0, 3145728, 6258688, 12517376, 25001984, 50003968, 100007936,
             200015872]
    ax.set_xticks(ticks)
    ax.set_xticklabels(TAU_LABELS, rotation=90, fontsize=6.4)
    ax.set_xlabel(r"drift $\tau$ (M tokens)", labelpad=1.5)

axes[0].set_ylabel(r"median $m_{\mathrm{top1}}/|\gamma_{\mathrm{ghost}}|$"
                   " on $F_{\\mathrm{live}}$", fontsize=7.6)
# reference-line tag (panel 1 only): two-word label, not prose
axes[0].text(DEEP * 1.25, axes[0].get_ylim()[1], "deep-drift\nregime",
             fontsize=6.4, color="0.35", ha="left", va="top", linespacing=1.1)

handles = [plt.Line2D([], [], color=s["color"], ls=s["ls"], marker=s["marker"],
                      ms=s["ms"], lw=s["lw"], label=s["label"],
                      markeredgewidth=s.get("mew", 0),
                      markerfacecolor=s.get("mfc", s["color"]))
           for s in LINES.values()]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, -0.015), handlelength=2.2, columnspacing=1.6)

fig.savefig(OUT, format="pdf", bbox_inches="tight")
print("[write]", OUT)

