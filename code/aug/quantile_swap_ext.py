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
QN = [("median", None), ("q98", 0.98), ("q99", 0.99), ("q995", 0.995)]

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


def ghost_q(rows, q):
    return R.quantile([r["m_top1"] for r in rows.values()
                       if r["set"].startswith("ghost")], q)


def main():
    pos = {qk: 0 for qk, _ in QN}
    sign_t0 = {qk: {} for qk, _ in QN}
    mul_sign = {qk: {} for qk, _ in QN}
    gaps = {qk: [] for qk, _ in QN}
    deep_dir = {qk: 0 for qk, _ in QN}
    for seed in SEEDS:
        for ti in range(8):
            for br in "AB":
                emeta, erows = load_train(seed, "t{}-end{}".format(ti, br))
                L = [fid for fid, r in erows.items()
                     if r["set"] == "excl_" + br and R._alive(r, emeta)]
                mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
                dmeta, drows = load_train(seed, "t{}-s0.50-dis{}".format(ti, br))
                for qk, qv in QN:
                    tM = (med([r["m_top1"] for r in mrows.values()
                               if r["set"].startswith("ghost")])
                          if qv is None else ghost_q(mrows, qv))
                    tD = (med([r["m_top1"] for r in drows.values()
                               if r["set"].startswith("ghost")])
                          if qv is None else ghost_q(drows, qv))
                    gaps[qk].append(abs(tM - tD))
                    dv = med([(mrows[f]["m_top1"] - tM) - (drows[f]["m_top1"] - tD)
                              for f in L])
                    if dv > 0:
                        pos[qk] += 1
                    if ti == 0:
                        sign_t0[qk]["{}-{}".format(seed, br)] = (
                            1 if dv > 0 else (-1 if dv < 0 else 0))

                    dm = med([mrows[f]["m_top1"] / abs(tM)
                              - drows[f]["m_top1"] / abs(tD) for f in L]) \
                        if tM != 0 and tD != 0 else float("nan")
                    mul_sign[qk]["{}-{}-{}".format(seed, ti, br)] = (
                        1 if dm > 0 else (-1 if dm < 0 else 0))
    out = {"pos_cells_ext48": pos,
           "tau0_signs_ext": sign_t0,
           "ghost_gap_max_ext": {qk: round(max(gaps[qk]), 4) for qk, _ in QN},
           "note": "[note]"}

    same3 = same4 = 0
    for seed in SEEDS:
        for ti in range(8):
            for br in "AB":
                k = "{}-{}-{}".format(seed, ti, br)
                s3 = {mul_sign[q][k] for q in ("median", "q98", "q99")}
                s4 = s3 | {mul_sign["q995"][k]}
                if len(s3) == 1:
                    same3 += 1
                if len(s4) == 1:
                    same4 += 1
    out["mul_sign_unchanged_ext48"] = {"median_q98_q99": same3,
                                       "all_four": same4}
    with open(os.path.join(HERE, "quantile_swap_ext.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("pos_cells_ext48", "ghost_gap_max_ext",
                       "mul_sign_unchanged_ext48")}, indent=1))


if __name__ == "__main__":
    main()
