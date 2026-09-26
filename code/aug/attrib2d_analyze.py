#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import gzip
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)

SEED_DIRS = {"results_v2_reid": [42, 1042, 2042],
             "results_v2_c1x3": [6784, 403, 1189],
             "results_v2_aug3": [3184, 5383, 7192]}
TAU_IDXS = [0, 5]
TAU_LAB = {0: "0", 5: "50M"}
ALPHAS = [0.25, 0.5, 0.75]
BETAS = [0.0, 0.25, 0.5, 0.75]
TD = os.path.join(EXP, "results/2dattrib", "shards")

_cache = {}


def load(path):
    if path in _cache:
        return _cache[path]
    meta, rows = None, {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                rows[r["fact_id"]] = r
    _cache[path] = (meta, rows)
    return _cache[path]


def cell_path(res_dir, seed, ti, alpha, beta, br):
    """Analysis script for the merge-audit study."""
    root = os.path.join(EXP, res_dir, "eval")
    if beta == 0.0:
        return os.path.join(root, "margins.s{}-t{}-s{:.2f}-dis{}.jsonl.gz".format(
            seed, ti, alpha, br))
    if abs(beta - (1 - alpha)) < 1e-9:
        return os.path.join(root, "margins.s{}-t{}-a{:.2f}-merge.jsonl.gz".format(
            seed, ti, alpha))
    return os.path.join(TD, "s{}-t{}-a{}b{}.jsonl.gz".format(seed, ti, alpha, beta))


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cells = {}
    for res_dir, seeds in SEED_DIRS.items():
        for seed in seeds:
            for ti in TAU_IDXS:
                for br in "AB":
                    emeta, erows = load(os.path.join(
                        EXP, res_dir, "eval",
                        "margins.s{}-t{}-end{}.jsonl.gz".format(seed, ti, br)))
                    thE = emeta["theta_alive_top1_q990"]
                    L = [f for f, r in erows.items()
                         if r["set"] == "excl_" + br and r["m_top1"] >= thE]
                    if not L:
                        continue
                    grid = {}
                    for a in ALPHAS:
                        for b in BETAS:
                            p = cell_path(res_dir, seed, ti, a, b, br)
                            if not os.path.exists(p):
                                continue
                            mmeta, mrows = load(p)
                            thM = mmeta["theta_alive_top1_q990"]
                            surv = sum(1 for f in L
                                       if f in mrows
                                       and mrows[f]["m_top1"] >= thM) / len(L)
                            grid["{:.2f},{:.2f}".format(a, b)] = round(surv, 4)
                    cells["s{}-t{}-{}".format(seed, ti, br)] = {
                        "campaign": res_dir, "n_live": len(L), "grid": grid}

    agg = {}
    for k, c in cells.items():
        ti = int(k.split("-")[1][1:])
        for cell, v in c["grid"].items():
            a, b = cell.split(",")
            agg.setdefault((ti, a, b), []).append(v)
    print("[note]")
    print("tau \\ alpha,beta:", ["a{}b{}".format(a, b) for a in ("0.25","0.50","0.75") for b in ("0","0.25","0.50","0.75")])
    for ti in TAU_IDXS:
        row = []
        for a in ("0.25", "0.50", "0.75"):
            for b in ("0", "0.25", "0.50", "0.75"):
                v = agg.get((ti, a, b))
                row.append("{:.3f}({})".format(med(v), len(v)) if v else "-")
        print("tau={}:".format(TAU_LAB[ti]), row)
    out = {"definition": "cohort=own-endpoint-alive at this tau; survival under "
                         "each grid model's own ghost-Q99",
           "cells": cells}
    with open(os.path.join(HERE, "attrib2d_report.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("cells:", len(cells))


if __name__ == "__main__":
    main()
