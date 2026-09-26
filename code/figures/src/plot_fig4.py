#!/usr/bin/env python3
"""plot_fig4.py -- barrier-survival dissociation, 24-cell alpha=0.5 version.

Input:  ../data/fig4_barrier.csv  (from make_fig4_data.py)
Output: ../fig4_barrier.pdf       (vector)

One message: near-zero or negative aggregate loss barriers coexist with large
fact-level loss. One point per (seed, tau) cell per loss caliber: filler-loss
barrier (circles) and fact-side answer-token NLL barrier (triangles); a thin
gray connector joins the two calibers of the same cell. x = relative barrier
at alpha=0.5, y = fraction of endpoint-surviving facts lost by the merge.
"""
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import paperstyle

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "fig4_barrier.csv")
OUT = os.path.join(HERE, "..", "fig4_barrier.pdf")

SEED_COLOR = {42: paperstyle.OI_BLUE, 1042: paperstyle.OI_VERMIL,
              2042: paperstyle.OI_GREEN}

rows = []
with open(DATA) as f:
    for r in csv.DictReader(f):
        rows.append({"seed": int(r["seed"]),
                     "tau_idx": int(r["tau_idx"]),
                     "barrier_filler": float(r["barrier_filler"]),
                     "barrier_fact": float(r["barrier_fact"]),
                     "lost": float(r["lost"])})

paperstyle.apply()
fig, ax = plt.subplots(figsize=(3.35, 2.75))
fig.subplots_adjust(left=0.14, right=0.975, top=0.97, bottom=0.15)
paperstyle.tidy(ax)

# same-cell connectors first (under the markers)
for r in rows:
    ax.plot([r["barrier_filler"], r["barrier_fact"]], [r["lost"], r["lost"]],
            color="0.72", lw=0.55, zorder=1)

for fam, key, marker in (("filler loss", "barrier_filler", "o"),
                         ("fact NLL", "barrier_fact", "^")):
    for seed, color in SEED_COLOR.items():
        xs = [r[key] for r in rows if r["seed"] == seed]
        ys = [r["lost"] for r in rows if r["seed"] == seed]
        ax.scatter(xs, ys, marker=marker, s=13, facecolor=color,
                   edgecolor="white", linewidth=0.35, alpha=0.92, zorder=3)

ax.axvline(0.0, color="0.25", lw=0.7, ls=(0, (4, 3)), zorder=2)
ax.set_xlabel(r"relative barrier at $\alpha{=}0.5$")
ax.set_ylabel("fraction of facts lost")
ax.set_ylim(-0.03, 0.9)
ax.set_xlim(-0.42, 0.075)

fam_handles = [Line2D([], [], marker="o", ls="", color="0.45", ms=4.5,
                      label="filler loss"),
               Line2D([], [], marker="^", ls="", color="0.45", ms=4.5,
                      label="fact NLL")]
seed_handles = [Line2D([], [], marker="o", ls="", color=c, ms=4.5,
                       label="seed {}".format(s))
                for s, c in SEED_COLOR.items()]
leg1 = ax.legend(handles=fam_handles, loc="upper left", frameon=False,
                 handletextpad=0.4, borderaxespad=0.1, title="barrier on",
                 title_fontsize=7.2)
leg1.get_title().set_fontsize(7.2)
ax.add_artist(leg1)
ax.legend(handles=seed_handles, loc="lower right", frameon=False,
          handletextpad=0.4, borderaxespad=0.1)

fig.savefig(OUT, format="pdf", bbox_inches="tight")
print("[write]", OUT)

