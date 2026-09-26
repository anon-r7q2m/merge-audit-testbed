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

REID = os.path.join(EXP, "results_v2_reid")
AUG3 = os.path.join(EXP, "results_v2_aug3")
C1X3 = os.path.join(EXP, "results_v2_c1x3")
SEED_DIR = {s: REID for s in (42, 1042, 2042)}
SEED_DIR.update({s: AUG3 for s in (3184, 5383, 7192)})
SEED_DIR.update({s: C1X3 for s in (6784, 403, 1189)})
CAMPAIGN = {s: "c1" for s in (42, 1042, 2042)}
CAMPAIGN.update({s: "c2" for s in (3184, 5383, 7192)})
CAMPAIGN.update({s: "c1x" for s in (6784, 403, 1189)})
SEEDS = (42, 1042, 2042, 3184, 5383, 7192, 6784, 403, 1189)
TAU_IDXS = [0, 5, 7]
TAU_LAB = {0: "0", 5: "50M", 7: "200M"}
TD = os.path.join(EXP, "results/tiesdare")

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


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cells = {}
    for seed in SEEDS:
        for ti in TAU_IDXS:
            for br in "AB":
                emeta, erows = load(os.path.join(
                    SEED_DIR[seed], "eval",
                    "margins.s{}-t{}-end{}.jsonl.gz".format(seed, ti, br)))
                thE = emeta["theta_alive_top1_q990"]
                L = [f for f, r in erows.items()
                     if r["set"] == "excl_" + br and r["m_top1"] >= thE]
                if not L:
                    continue
                rec = {}
                # averaging
                mmeta, mrows = load(os.path.join(
                    SEED_DIR[seed], "eval",
                    "margins.s{}-t{}-a0.50-merge.jsonl.gz".format(seed, ti)))
                rec["avg"] = (mmeta["theta_alive_top1_q990"], mrows)
                for meth in ("ties", "dare"):
                    mp = os.path.join(TD, "shards",
                                      "s{}-t{}-{}.jsonl.gz".format(seed, ti, meth))
                    if os.path.exists(mp):
                        mm, rr = load(mp)
                        rec[meth] = (mm["theta_alive_top1_q990"], rr)
                cell = {"campaign": CAMPAIGN[seed], "n_live": len(L)}
                for op, (th, rows) in rec.items():
                    surv = [f for f in L if f in rows and rows[f]["m_top1"] >= th]
                    dn = [erows[f]["logp"] - rows[f]["logp"]
                          for f in L if f in rows]
                    cell[op] = {"survival": round(len(surv) / len(L), 4),
                                "dnll_med": round(med(dn), 4),
                                "dnll_alarm2": round(
                                    sum(1 for x in dn if x > 2.0) / len(dn), 4)}
                cells["s{}-t{}-{}".format(seed, ti, br)] = cell

    agg = {}
    for k, c in cells.items():
        key = (c["campaign"], k.split("-")[1])
        agg.setdefault(key, []).append(c)
    summ = {}
    for (camp, tlab), cs in sorted(agg.items()):
        e = {}
        for op in ("avg", "ties", "dare"):
            vals = [c[op] for c in cs if op in c]
            if vals:
                e[op] = {"survival_med": round(med([v["survival"] for v in vals]), 4),
                         "survival_min": round(min(v["survival"] for v in vals), 4),
                         "survival_max": round(max(v["survival"] for v in vals), 4),
                         "dnll_med": round(med([v["dnll_med"] for v in vals]), 4)}
        summ["{}-{}".format(camp, tlab)] = e
    out = {"definition": "cohort = own-endpoint-alive (train battery, frozen "
                         "rule); survival under each operator's own ghost-Q99; "
                         "dnll = logp_endOwn - logp_mergeOp",
           "cells": cells, "summary": summ}
    with open(os.path.join(HERE, "operator_survival.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k in sorted(summ):
        print(k, {op: v["survival_med"] for op, v in summ[k].items()},
              {op + "_dnll": v["dnll_med"] for op, v in summ[k].items()})


if __name__ == "__main__":
    main()
