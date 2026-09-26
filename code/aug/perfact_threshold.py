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
C1X3 = os.path.join(EXP, "results_v2_c1x3")
SEED_DIR = {s: REID for s in (42, 1042, 2042)}
SEED_DIR.update({s: AUG3 for s in (3184, 5383, 7192)})
SEED_DIR.update({s: C1X3 for s in (6784, 403, 1189)})
CAMPAIGN = {s: "c1" for s in (42, 1042, 2042)}
CAMPAIGN.update({s: "c2" for s in (3184, 5383, 7192)})
CAMPAIGN.update({s: "c1x" for s in (6784, 403, 1189)})
THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

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


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def sweep(pairs):
    """Analysis script for the merge-audit study."""
    n_dead = sum(1 for _, d in pairs if d)
    n_alive = len(pairs) - n_dead
    out = {}
    for t in THRESHOLDS:
        tp = sum(1 for x, d in pairs if d and x > t)
        fp = sum(1 for x, d in pairs if (not d) and x > t)
        out["{:.1f}".format(t)] = {
            "recall": round(tp / n_dead, 4) if n_dead else None,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "fpr": round(fp / n_alive, 4) if n_alive else None}
    return out


def main():
    cells = {}
    pool = {"c1": [], "c2": [], "c1x": []}
    for seed in (42, 1042, 2042, 3184, 5383, 7192, 6784, 403, 1189):
        camp = CAMPAIGN[seed]
        for ti in range(8):
            for br in "AB":
                emeta, erows = load_train(seed, "t{}-end{}".format(ti, br))
                mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
                L = [fid for fid, r in erows.items()
                     if r["set"] == "excl_" + br and R._alive(r, emeta)]
                if not L:
                    continue
                pairs = []
                for f in L:
                    dn = erows[f]["logp"] - mrows[f]["logp"]
                    dead = not R._alive(mrows[f], mmeta)
                    pairs.append((dn, dead))
                pool[camp].extend(pairs)
                dn_dead = [x for x, d in pairs if d]
                dn_aliv = [x for x, d in pairs if not d]
                cells["s{}-t{}-{}".format(seed, ti, br)] = {
                    "campaign": camp, "n_live": len(L),
                    "n_dead": len(dn_dead),
                    "barrier_own_logp": round(med([erows[f]["logp"] for f in L])
                                              - med([mrows[f]["logp"]
                                                     for f in L]), 4),
                    "dnll_dead_median": round(med(dn_dead), 4),
                    "dnll_aliv_median": round(med(dn_aliv), 4),
                    "sweep": sweep(pairs)}
    camps = {}
    for camp, pairs in pool.items():
        dn_dead = [x for x, d in pairs if d]
        dn_aliv = [x for x, d in pairs if not d]
        bvals = [c["barrier_own_logp"] for c in cells.values()
                 if c["campaign"] == camp]
        camps[camp] = {"n_facts": len(pairs), "n_dead": len(dn_dead),
                       "dnll_dead_median": round(med(dn_dead), 4),
                       "dnll_aliv_median": round(med(dn_aliv), 4),
                       "barrier_min": round(min(bvals), 4),
                       "barrier_med": round(med(bvals), 4),
                       "barrier_max": round(max(bvals), 4),
                       "barrier_pos": sum(1 for b in bvals if b > 0),
                       "n_cells": len(bvals),
                       "sweep": sweep(pairs)}

    allp = pool["c1"] + pool["c2"] + pool["c1x"]
    dn_dead = [x for x, d in allp if d]
    dn_aliv = [x for x, d in allp if not d]
    bvals = [c["barrier_own_logp"] for c in cells.values()]
    camps["all"] = {"n_facts": len(allp), "n_dead": len(dn_dead),
                    "dnll_dead_median": round(med(dn_dead), 4),
                    "dnll_aliv_median": round(med(dn_aliv), 4),
                    "barrier_min": round(min(bvals), 4),
                    "barrier_med": round(med(bvals), 4),
                    "barrier_max": round(max(bvals), 4),
                    "barrier_pos": sum(1 for b in bvals if b > 0),
                    "n_cells": len(bvals),
                    "sweep": sweep(allp)}


    w3 = {}
    for seed in (42, 1042, 2042, 3184, 5383, 7192, 6784, 403, 1189):
        for br in "AB":
            emeta, erows = load_train(seed, "t0-end{}".format(br))
            _, mrows = load_train(seed, "t0-a0.50-merge")
            _, drows = load_train(seed, "t0-s0.50-dis{}".format(br))
            L = [fid for fid, r in erows.items()
                 if r["set"] == "excl_" + br and R._alive(r, emeta)]
            w3["s{}-{}".format(seed, br)] = {
                "n_live": len(L),
                "m_merge_med": round(med([mrows[f]["m_top1"] for f in L]), 4),
                "m_dis_med": round(med([drows[f]["m_top1"] for f in L]), 4),
                "m_endown_med": round(med([erows[f]["m_top1"] for f in L]), 4)}
    out = {"definition": {
               "cohort": "own-endpoint-alive exclusive facts (frozen _alive: "
                         "m_top1 >= ghost-train Q99)",
               "dnll": "logp_endOwn - logp_merge (positive = merge worse)",
               "dead": "cohort fact not alive at merge"},
           "cells": cells, "campaigns": camps, "w3_tau0": w3}
    with open(os.path.join(HERE, "perfact_threshold.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for camp in ("c1", "c2", "c1x", "all"):
        c = camps[camp]
        print(camp, "n_facts", c["n_facts"], "n_dead", c["n_dead"],
              "dead_med", c["dnll_dead_median"], "aliv_med",
              c["dnll_aliv_median"], "barrier", c["barrier_min"],
              c["barrier_med"], c["barrier_max"], "pos",
              c["barrier_pos"], "/", c["n_cells"])
        for t, v in c["sweep"].items():
            print("   t={}: recall={} precision={} fpr={}".format(
                t, v["recall"], v["precision"], v["fpr"]))
    print("W3 (tau=0):")
    for k, v in w3.items():
        print("  {}: merge={} dis={} endOwn={}".format(
            k, v["m_merge_med"], v["m_dis_med"], v["m_endown_med"]))


if __name__ == "__main__":
    main()
