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
POS = os.path.join(EXP, "results/posctrl", "eval")
SEEDS = (42, 1042, 2042)
MECHS = ("ascenddmg", "misinfodmg")
THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
TARGETS = None


def load(path):
    meta, rows = None, {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                rows[r["fact_id"]] = r
    return meta, rows


def main():
    import json as J
    tf = J.load(open(os.path.join(EXP, "results/posctrl", "damage",
                                  "target_facts.json")))
    print("target_facts.json type:", type(tf).__name__,
          "keys/len:", list(tf.keys())[:6] if isinstance(tf, dict) else len(tf))
    cells = {}
    pool = []
    for seed in SEEDS:
        _, pre = load(os.path.join(REID, "eval",
                                   "margins.s{}-t0-endA.jsonl.gz".format(seed)))
        for mech in MECHS:
            _, post = load(os.path.join(
                POS, "margins.posctrl-s{}-{}.jsonl.gz".format(seed, mech)))
            pairs = []
            for fid, rpre in pre.items():
                if not rpre["set"].startswith("excl_A"):
                    continue
                if fid not in post:
                    continue
                pre_k = rpre["rank_bar"] <= R.RANK1_BAR_THRESH
                post_k = post[fid]["rank_bar"] <= R.RANK1_BAR_THRESH
                if not pre_k:
                    continue
                dead = not post_k
                dn = rpre["logp"] - post[fid]["logp"]
                pairs.append((dn, dead))
            rec = {"n": len(pairs), "n_dead": sum(1 for _, d in pairs if d)}
            for t in THRESHOLDS:
                tp = sum(1 for x, d in pairs if d and x > t)
                fp = sum(1 for x, d in pairs if (not d) and x > t)
                nd = rec["n_dead"]
                nr = len(pairs) - nd
                rec["t{:.1f}".format(t)] = {
                    "tpr": round(tp / nd, 4) if nd else None,
                    "fpr": round(fp / nr, 4) if nr else None}
            cells["{}:s{}".format(mech.replace("dmg", ""), seed)] = rec
            pool.extend(pairs)
    outp = {"n": len(pool), "n_dead": sum(1 for _, d in pool if d)}
    for t in THRESHOLDS:
        tp = sum(1 for x, d in pool if d and x > t)
        fp = sum(1 for x, d in pool if (not d) and x > t)
        nd = outp["n_dead"]
        nr = len(pool) - nd
        outp["t{:.1f}".format(t)] = {"tpr": round(tp / nd, 4) if nd else None,
                                     "fpr": round(fp / nr, 4) if nr else None}
    out = {"definition": "labels = certified forgotten/retained (rank_bar "
                         "flip, frozen posctrl rule); score = per-fact "
                         "logp_pre - logp_post at the damaged endpoint",
           "cells": cells, "pooled": outp}
    with open(os.path.join(HERE, "dnll_posctrl.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(outp, ensure_ascii=False, indent=1))
    for k, v in cells.items():
        print(k, "n={} dead={}".format(v["n"], v["n_dead"]),
              "t2.0 tpr={} fpr={}".format(v["t2.0"]["tpr"], v["t2.0"]["fpr"]))


if __name__ == "__main__":
    main()
