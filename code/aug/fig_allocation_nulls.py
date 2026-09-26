#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headline allocation figure renderer."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
FIG = "/var/tmp/<account>/work-20260825/overleaf/main/figures"

C_ACT = "#1f77b4"
C_MN = "#8c8c8c"
C_LN = "#d95f02"


def main():
    mn = json.load(open(os.path.join(HERE, "alloclaw_marginnull.json")))["summary"]
    ln = json.load(open(os.path.join(HERE, "alloclaw_logitnull.json")))["summary"]
    arms = ["conc", "s75", "s50"]
    labels = ["concentrated", "3/4 split", "balanced"]
    actual = [mn["%s:all" % a]["actual"] for a in arms]
    null_m = [mn["%s:all" % a]["null_norm"] for a in arms]
    null_l = [ln["%s:all" % a]["null_logit"] for a in arms]

    fu = json.load(open(os.path.join(HERE, "e6alloc_fu_agg.json")))["cells"]
    import ast
    by = {}
    for k, v in fu.items():
        si, ori = ast.literal_eval(k)
        by.setdefault(si, {})[ori] = v
    conc1b, bal1b = [], []
    for s in range(5):
        conc1b.append((by[0][0][s]["uncond"] + by[0][1][s]["uncond"]) / 2)
        bal1b.append((by[2][0][s]["uncond"] + by[2][1][s]["uncond"]) / 2)
    ev3 = json.load(open(os.path.join(HERE, "e6alloc_eval3_agg.json")))
    og_conc = [ev3[s]["merge"]["conc"] for s in ("s1", "s2", "s3", "s4", "s5")]
    og_bal = [ev3[s]["merge"]["s50"] for s in ("s1", "s2", "s3", "s4", "s5")]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})

    ax = axes[0]
    x = np.arange(3)
    w = 0.27
    ax.bar(x - w, actual, w, color=C_ACT, label="merge (measured)")
    ax.bar(x, null_m, w, color=C_MN, label="margin-linear null")
    ax.bar(x + w, null_l, w, color=C_LN, label="logit-ensemble null")
    for xi, v in zip(x - w, actual):
        ax.text(xi, v + 0.02, "%.2f" % v, ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("merge survival (assigned facts)", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) testbed allocation arms vs nulls", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    xg = [0, 1]
    ax.boxplot([conc1b, bal1b], positions=[-0.13, 0.87], widths=0.2,
               showfliers=False, medianprops={"color": C_ACT},
               boxprops={"color": C_ACT}, whiskerprops={"color": C_ACT},
               capprops={"color": C_ACT})
    ax.scatter([-0.13] * 5, conc1b, color=C_ACT, s=9, zorder=3)
    ax.scatter([0.87] * 5, bal1b, color=C_ACT, s=9, zorder=3,
               label="candidate strip (5 seeds)")
    ax.scatter([0.13] * 5, og_conc, color=C_LN, marker="s", s=9, zorder=3)
    ax.scatter([1.13] * 5, og_bal, color=C_LN, marker="s", s=9, zorder=3,
               label="open generation (5 seeds)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["concentrated", "balanced"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) 1B bridge: two readouts", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIG, "fig_allocation_nulls.pdf")
    fig.savefig(out)
    print("->", out)


if __name__ == "__main__":
    main()
