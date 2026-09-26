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
SEEDS = [42, 1042, 2042]
TAUS = ["0", "3.1M", "6.3M", "12.5M", "25M", "50M", "100M", "200M"]
_cache = {}


def load(seed, suffix):
    key = (seed, suffix)
    if key in _cache:
        return _cache[key]
    p = os.path.join(REID, "eval", "margins.s{}-{}.jsonl.gz".format(seed, suffix))
    meta, rows = None, {}
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                rows[r["fact_id"]] = r
    _cache[key] = (meta, rows)
    return _cache[key]


def med(x):
    return statistics.median(x) if x else float("nan")


def main():
    out = {}
    for seed in SEEDS:
        rows_out = []
        for ti, tau in enumerate(TAUS):
            ends, diss = {}, {}
            for br in "AB":
                ends[br] = load(seed, "t{}-end{}".format(ti, br))
                diss[br] = load(seed, "t{}-s0.50-dis{}".format(ti, br))
            _, fM = load(seed, "t{}-a0.50-merge".format(ti))
            live = {}
            for br in "AB":
                meta, facts = ends[br]
                live[br] = [fid for fid, r in facts.items()
                            if r["set"] == "excl_" + br and R._alive(r, meta)]
            d_lin, d_dis, d_obs = [], [], []
            for br, w in (("A", 0.5), ("B", 0.5)):
                other = "B" if br == "A" else "A"
                for fid in live[br]:
                    m_meas = fM[fid]["m_top1"]
                    m_lin = w * ends[br][1][fid]["m_top1"] \
                        + (1 - w) * ends[other][1][fid]["m_top1"]
                    m_dis = diss[br][1][fid]["m_top1"]
                    d_lin.append(m_meas - m_lin)
                    d_dis.append(m_meas - m_dis)
                    d_obs.append(m_dis - m_lin)
            lo, hi = R.bootstrap_median_ci(d_dis)
            rows_out.append({"tau": tau,
                             "NMAD_lin": round(med([abs(x) for x in d_lin]), 4),
                             "NMAD_dis": round(med([abs(x) for x in d_dis]), 4),
                             "D_lin": round(med(d_lin), 4),
                             "D_obs": round(med(d_obs), 4),
                             "D_dis": round(med(d_dis), 4),
                             "CI90": [round(lo, 4), round(hi, 4)]})
        out[str(seed)] = rows_out
    with open(os.path.join(HERE, "decomp_plain.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for seed in SEEDS:
        print("seed", seed)
        for r in out[str(seed)]:
            print("  {:5s}  Dlin {:+.4f}  Dobs {:+.4f}  Ddis {:+.4f}  CI [{:+.4f},{:+.4f}]".format(
                r["tau"], r["D_lin"], r["D_obs"], r["D_dis"], *r["CI90"]))


if __name__ == "__main__":
    main()
