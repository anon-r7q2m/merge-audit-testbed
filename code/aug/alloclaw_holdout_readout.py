#!/usr/bin/env python3
"""alloclaw_holdout_readout.py -- allocation law under the held-out-template battery.

Read-only re-aggregation of the released margin fields (battery="hold"),
mirroring alloclaw_marginnull.py's cell structure (mirrors pooled per seed,
unconditional over assigned facts). Alive = m_top1 >= Q99 of the model's own
ghost-pool margins *within the same battery* (convention verified against
meta theta_alive_top1_q990 for the train battery).
Output: aug/analysis-out/alloclaw_holdout.json
"""
import gzip, json, os
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SEEDS = [42, 1042, 2042]
KS = [128, 256]
SPLIT_NAMES = {(1.0, 0.0): "conc", (0.75, 0.25): "s75", (0.5, 0.5): "s50"}

def load_hold(path):
    rows, meta = {}, None
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]; continue
            if r.get("battery") != "hold":
                continue
            rows[r["fact_id"]] = r
    return meta, rows

def main():
    out = {"units": [], "by_split": defaultdict(list)}
    for K in KS:
        rdir = os.path.join(EXP, f"results_aug_alloclaw_k{K}")
        assign = json.load(open(os.path.join(rdir, "alloc_assignment.json")))
        groups = assign["groups"]
        splits = [tuple(s) for s in assign["splits"]]
        for seed in SEEDS:
            tag = f"k{K}-s{seed}-t0"
            mp = os.path.join(rdir, "eval", f"margins.{tag}-merge05.jsonl.gz")
            if not os.path.exists(mp):
                continue
            metaM, M = load_hold(mp)
            # merge model's ghost-Q99 threshold within the hold battery
            gh = [r["m_top1"] for r in M.values() if r["set"].startswith("ghost")]
            thr = float(np.quantile(np.array(gh), 0.99))
            for gkey, fidxs in groups.items():
                si, mi = gkey.split("|")
                sA, sB = splits[int(si)]
                split_name = SPLIT_NAMES.get((sA, sB))
                if split_name is None:
                    continue
                tot, alive = 0, 0
                for fi in fidxs:
                    fid = f"S:{fi}"
                    if fid not in M:
                        continue
                    tot += 1
                    if M[fid]["m_top1"] >= thr:
                        alive += 1
                rec = {"K": K, "seed": seed, "split": split_name,
                       "n": tot, "survival": alive / tot if tot else None,
                       "ghost_q99_hold": thr}
                out["units"].append(rec)
                out["by_split"][(K, split_name)].append(rec["survival"])
    # summary: per-(K, split) range across seeds (mirrors pooled per seed)
    summ = {}
    for (K, sp), vals in sorted(out["by_split"].items()):
        vals = [v for v in vals if v is not None]
        summ[f"K{K}-{sp}"] = {"n_cells": len(vals),
                              "min": min(vals), "max": max(vals),
                              "mean": sum(vals)/len(vals)}
    out["summary"] = summ
    out["by_split"] = {f"{k[0]}|{k[1]}": v for k, v in out["by_split"].items()}
    op = os.path.join(HERE, "analysis-out", "alloclaw_holdout.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()
