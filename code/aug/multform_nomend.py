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

REID = os.path.join(EXP, "results_v2_reid")
AUG3 = os.path.join(EXP, "results_v2_aug3")
SEED_DIR = {s: REID for s in (42, 1042, 2042)}
SEED_DIR.update({s: AUG3 for s in (3184, 5383, 7192)})
SEEDS = [42, 1042, 2042, 3184, 5383, 7192]
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
    table = {}
    for ti, tau in enumerate(TAUS):
        row = {}
        n_seed_neg = 0
        for seed in SEEDS:
            vals = {}
            for br in "AB":
                emeta, erows = load_train(seed, "t{}-end{}".format(ti, br))
                L = [fid for fid, r in erows.items()
                     if r["set"] == "excl_" + br and R._alive(r, emeta)]
                mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
                dmeta, drows = load_train(seed, "t{}-s0.50-dis{}".format(ti, br))
                s_m = abs(mmeta["ghost_median_top1_train"])
                s_d = abs(dmeta["ghost_median_top1_train"])
                d = statistics.median(
                    [mrows[f]["m_top1"] / s_m - drows[f]["m_top1"] / s_d
                     for f in L])
                vals[br] = round(d, 4)
            row[str(seed)] = vals
            if vals["A"] < 0 and vals["B"] < 0:
                n_seed_neg += 1
        row["seeds_both_neg"] = n_seed_neg
        table[tau] = row
    out = {"definition": "D_mul = median over F_live(tau) of "
                         "m_merge/|ghost_floor_merge| - m_dis/|ghost_floor_dis|"
                         " (Eq. multform; NO M_end presentation scale)",
           "table": table}
    with open(os.path.join(HERE, "multform_nomend.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    for tau in TAUS:
        row = table[tau]
        cells = []
        for seed in SEEDS:
            for br in "AB":
                v = row[str(seed)][br]
                cells.append(("$+$" if v >= 0 else "$-$") +
                             ("%.4f" % abs(v)))
        print("{} & {} & {} \\\\".format(tau, " & ".join(cells),
                                        row["seeds_both_neg"]))


if __name__ == "__main__":
    main()
