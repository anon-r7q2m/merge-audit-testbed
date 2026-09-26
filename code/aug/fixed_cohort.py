#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import gzip
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
import sys
sys.path.insert(0, EXP)
import run_merge_audit as R                                   # noqa: E402

REID = os.path.join(EXP, "results_v2_reid")
AUG3 = os.path.join(EXP, "results_v2_aug3")
SEED_DIR = {s: REID for s in (42, 1042, 2042)}
SEED_DIR.update({s: AUG3 for s in (3184, 5383, 7192)})
C1 = (42, 1042, 2042)
C2 = (3184, 5383, 7192)
TAUS = ["0", "3.1M", "6.3M", "12.5M", "25M", "50M", "100M", "200M"]

_cache = {}


def load_train(seed, suffix):
    key = (seed, suffix)
    if key in _cache:
        return _cache[key]
    path = os.path.join(SEED_DIR[seed], "eval",
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


def main():
    out = {"definition": "cohort = tau=0 own-endpoint alive (frozen _alive, "
                         "ghost Q99); D_mul with ghost-floor normalizer "
                         "(Eq. multform)", "cells": {}}
    for seed in C1 + C2:
        for br in "AB":
            emeta0, erows0 = load_train(seed, "t0-end" + br)
            C0 = [fid for fid, r in erows0.items()
                  if r["set"] == "excl_" + br and R._alive(r, emeta0)]
            for ti in range(8):
                mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
                dmeta, drows = load_train(seed, "t{}-s0.50-dis{}".format(ti, br))
                s_m = abs(mmeta["ghost_median_top1_train"])
                s_d = abs(dmeta["ghost_median_top1_train"])
                d = statistics.median(
                    [mrows[f]["m_top1"] / s_m - drows[f]["m_top1"] / s_d
                     for f in C0])
                out["cells"]["s{}-t{}-{}".format(seed, ti, br)] = {
                    "n_c0": len(C0), "D_mul_fixed": round(d, 6)}

    counts = {}
    for ti in range(8):
        for name, seeds in (("c1", C1), ("c2", C2)):
            n = sum(1 for s in seeds
                    if all(out["cells"]["s{}-t{}-{}".format(s, ti, br)]["D_mul_fixed"] < 0
                           for br in "AB"))
            counts["t{}".format(ti)] = dict(counts.get("t{}".format(ti), {}),
                                            **{name: n})
    out["direction_counts_per_campaign"] = counts
    with open(os.path.join(HERE, "fixed_cohort.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("direction (both-branch D_mul<0) per campaign:")
    for ti, tau in enumerate(TAUS):
        c = counts["t{}".format(ti)]
        print("  tau={:5s}  c1 {}/3  c2 {}/3".format(tau, c["c1"], c["c2"]))


if __name__ == "__main__":
    main()
