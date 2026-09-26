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


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    cells = {}
    for seed in C1 + C2:
        for ti in range(8):
            for br in "AB":
                own = "end" + br
                emeta, erows = load_train(seed, "t{}-{}".format(ti, own))
                mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
                _, orows = load_train(seed, "t{}-end{}".format(
                    ti, "B" if br == "A" else "A"))
                L = [fid for fid, r in erows.items()
                     if r["set"] == "excl_" + br and R._alive(r, emeta)]
                if not L:
                    continue
                lp_own = med([erows[f]["logp"] for f in L])
                lp_mrg = med([mrows[f]["logp"] for f in L])
                lp_oth = med([orows[f]["logp"] for f in L])

                nll_m, nll_o, nll_x = -lp_mrg, -lp_own, -lp_oth
                bar_endmean = (nll_m - 0.5 * (nll_o + nll_x)) / abs(
                    0.5 * (nll_o + nll_x))
                bar_own = lp_own - lp_mrg
                dead = [f for f in L if not R._alive(mrows[f], mmeta)]
                dn = [erows[f]["logp"] - mrows[f]["logp"] for f in dead]
                rec = {"n_live": len(L), "n_dead": len(dead),
                       "barrier_endmean_rel": round(bar_endmean, 4),
                       "barrier_own_logp": round(bar_own, 4),
                       "dnll_dead_recall": (round(sum(1 for x in dn if x > 0)
                                                  / len(dn), 4) if dn else None),
                       "dnll_dead_median": (round(med(dn), 4) if dn else None)}
                cells["s{}-t{}-{}".format(seed, ti, br)] = rec
    def pool(keys, field):
        vals = [cells[k][field] for k in keys if cells[k].get(field) is not None]
        return {"median": round(med(vals), 4), "min": round(min(vals), 4),
                "max": round(max(vals), 4), "n": len(vals)}
    out = {"definition": {"own": "median(logp_endOwn) - median(logp_merge) "
                                 "on own-endpoint-alive exclusive facts",
                          "endmean": "current paper's endpoint-mean relative "
                                     "NLL barrier", "dead": "alive at own "
                           "endpoint, not alive at merge (frozen rule)"},
           "cells": cells}
    for name, seeds in (("c1", C1), ("c2", C2)):
        keys = [k for k in cells if int(k.split("-")[0][1:]) in seeds]
        out[name] = {"own": pool(keys, "barrier_own_logp"),
                     "endmean": pool(keys, "barrier_endmean_rel"),
                     "n_cells": len(keys),
                     "own_positive": sum(1 for k in keys
                                         if cells[k]["barrier_own_logp"] > 0),
                     "endmean_nonpos": sum(1 for k in keys
                                           if cells[k]["barrier_endmean_rel"] <= 0),
                     "dead_recall": pool(keys, "dnll_dead_recall"),
                     "dead_dnll_median_neg": sum(
                         1 for k in keys
                         if (cells[k]["dnll_dead_median"] or 0) < 0)}
    with open(os.path.join(HERE, "own_endpoint_barrier.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("c1", "c2")}, ensure_ascii=False,
                     indent=1))


if __name__ == "__main__":
    main()
