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
sys.path.insert(0, EXP)
import run_merge_audit as R                                   # noqa: E402

C1X3 = os.path.join(EXP, "results_v2_c1x3")
SEEDS = [6784, 403, 1189]
TAUS = ["0", "3.1M", "6.3M", "12.5M", "25M", "50M", "100M", "200M"]
TGRID = [0.50 + 0.02 * k for k in range(101)]
RANK1 = R.RANK1_BAR_THRESH

_cache = {}


def load_train(seed, suffix):
    key = (seed, suffix)
    if key in _cache:
        return _cache[key]
    path = os.path.join(C1X3, "eval",
                        "margins.s{}-{}.jsonl.gz".format(seed, suffix))
    meta, rows = None, {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                rows[r["fact_id"]] = r
    _cache[key] = (meta, rows)
    return _cache[key]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cells = {}
    for seed in SEEDS:
        for ti, tau in enumerate(TAUS):
            for br in "AB":
                emeta, erows = load_train(seed, "t{}-end{}".format(ti, br))
                L = [fid for fid, r in erows.items()
                     if r["set"] == "excl_" + br and R._alive(r, emeta)]
                mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
                dmeta, drows = load_train(seed, "t{}-s0.50-dis{}".format(ti, br))
                gM = abs(mmeta["ghost_median_top1_train"])
                gD = abs(dmeta["ghost_median_top1_train"])
                n = len(L)
                dmul = med([mrows[f]["m_top1"] / gM - drows[f]["m_top1"] / gD
                            for f in L])
                raw = med([mrows[f]["m_top1"] - drows[f]["m_top1"] for f in L])
                fr_m = sum(1 for f in L if mrows[f]["rank_bar"] <= RANK1) / n
                fr_d = sum(1 for f in L if drows[f]["rank_bar"] <= RANK1) / n
                mr_m = med([mrows[f]["rank_bar"] for f in L])
                mr_d = med([drows[f]["rank_bar"] for f in L])
                lp = med([mrows[f]["logp"] - drows[f]["logp"] for f in L])
                mn = med([mrows[f]["m_nll"] - drows[f]["m_nll"] for f in L])
                cell = {"n_live": n, "D_mul": round(dmul, 4),
                        "dirs": {"raw_mtop1": raw < 0, "frac_rank1": fr_m < fr_d,
                                 "median_rank": mr_m > mr_d, "logp": lp < 0,
                                 "m_nll": mn < 0, "D_mul": dmul < 0},
                        "S_meas": round(sum(1 for f in L if mrows[f]["alive"])
                                        / n, 4),
                        "S_dis": round(sum(1 for f in L if drows[f]["alive"])
                                       / n, 4),
                        "scale_ratio": round(gM / gD, 4)}
                if ti == 0:

                    base = [mrows[f]["m_top1"] for f in L]
                    ref = [drows[f]["m_top1"] for f in L]
                    s1 = med([b - r for b, r in zip(base, ref)])
                    ts, prev_t, prev_v = None, None, None
                    for t in TGRID:
                        v = med([b - t * r for b, r in zip(base, ref)])
                        if (v < 0) != (s1 < 0):
                            ts = t
                            break
                        prev_t, prev_v = t, v
                    cell["t_star"] = round(ts, 4) if ts else None
                    cell["flips"] = ts is not None
                cells["s{}-t{}-{}".format(seed, ti, br)] = cell

    out = {"cells": cells, "definition": "same code path as "
           "aug/paper_backfill.py (frozen helpers)"}

    cons = {}
    for ti, tau in enumerate(TAUS):
        c = {}
        for ro in ("raw_mtop1", "frac_rank1", "median_rank", "logp", "m_nll",
                   "D_mul"):
            n = sum(1 for s in SEEDS if all(
                cells["s{}-t{}-{}".format(s, ti, br)]["dirs"][ro]
                for br in "AB"))
            c[ro] = n
        cons[tau] = c
    out["consensus_c1x3"] = cons
    ts_list = [c["t_star"] for c in cells.values() if c.get("t_star")]
    out["tstar"] = {"n_flip": len(ts_list), "n_cells": 6,
                    "list": sorted(ts_list),
                    "median": med(ts_list) if ts_list else None}
    out["ratio_gt_tstar"] = sum(
        1 for k, c in cells.items()
        if k.endswith("-t0-A") or k.endswith("-t0-B")
        for _ in [0] if False)  # placeholder, recomputed below
    hits = 0
    tot = 0
    for seed in SEEDS:
        for br in "AB":
            c = cells["s{}-t0-{}".format(seed, br)]
            if c.get("t_star") is None:
                continue
            tot += 1
            if c["scale_ratio"] > c["t_star"]:
                hits += 1
    out["ratio_gt_tstar"] = {"hits": hits, "of": tot}
    with open(os.path.join(HERE, "c1x3_backfill.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("consensus c1x3 (both-branch count of 3):")
    for tau in TAUS:
        print(" ", tau, cons[tau])
    print("t*:", out["tstar"])
    print("ratio>t*:", out["ratio_gt_tstar"])


if __name__ == "__main__":
    main()
