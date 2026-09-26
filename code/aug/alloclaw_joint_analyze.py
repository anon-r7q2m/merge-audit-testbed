#!/usr/bin/env python3
"""alloclaw_joint_analyze.py -- joint-baseline vs merge comparison.

Per (K, seed): joint model's unconditional survival on the allocatable pool
(overall and per assignment group), side by side with the existing merge05
cells. Also the TIES cross (ties05) vs merge05 where available.
Output: aug/analysis-out/alloclaw_joint_compare.json + stdout summary.
"""
import gzip
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SEEDS = [42, 1042, 2042]
KS = [128, 256]
SPLIT_NAMES = {(1.0, 0.0): "conc", (0.75, 0.25): "s75", (0.5, 0.5): "s50"}


def load_train(path):
    rows, meta = {}, None
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
                continue
            if r.get("battery") != "train":
                continue
            rows[r["fact_id"]] = r
    return meta, rows


def main():
    out = {"units": []}
    for K in KS:
        adir = os.path.join(EXP, "results_aug_alloclaw_k{}".format(K))
        jdir = os.path.join(EXP, "results_aug_alloclaw_joint_k{}".format(K))
        assign = json.load(open(os.path.join(adir, "alloc_assignment.json")))
        groups = assign["groups"]
        splits = [tuple(s) for s in assign["splits"]]
        for seed in SEEDS:
            tag = "k{}-s{}-t0".format(K, seed)
            jp = os.path.join(jdir, "eval",
                              "margins.{}-joint.jsonl.gz".format(tag))
            mp = os.path.join(adir, "eval",
                              "margins.{}-merge05.jsonl.gz".format(tag))
            tp = os.path.join(adir, "eval",
                              "margins.{}-ties05.jsonl.gz".format(tag))
            if not (os.path.exists(jp) and os.path.exists(mp)):
                continue
            metaJ, J = load_train(jp)
            metaM, M = load_train(mp)
            T = load_train(tp)[1] if os.path.exists(tp) else None
            thrJ = metaJ["theta_alive_top1_q990"]
            thrM = metaM["theta_alive_top1_q990"]
            thrT = load_train(tp)[0]["theta_alive_top1_q990"] if T else None
            alloc_ids = ["S:{}".format(i) for g in groups.values()
                         for i in g]
            def surv(rows, thr, ids):
                vs = [rows[i] for i in ids if i in rows]
                return sum(1 for r in vs if r["m_top1"] >= thr) / len(vs)
            rec = {"K": K, "seed": seed,
                   "joint_all": surv(J, thrJ, alloc_ids),
                   "merge_all": surv(M, thrM, alloc_ids),
                   "ties_all": surv(T, thrT, alloc_ids) if T else None}
            # per assignment group
            for gkey, fidxs in groups.items():
                si, ori = gkey.split("|")
                sp = SPLIT_NAMES.get(splits[int(si)])
                ids = ["S:{}".format(i) for i in fidxs]
                rec["joint_" + sp] = surv(J, thrJ, ids)
                rec["merge_" + sp] = surv(M, thrM, ids)
                if T:
                    rec["ties_" + sp] = surv(T, thrT, ids)
            out["units"].append(rec)
    op = os.path.join(HERE, "analysis-out", "alloclaw_joint_compare.json")
    with open(op, "w") as f:
        json.dump(out, f, indent=1)
    # summary
    for r in out["units"]:
        print("K={K} s{seed}: joint {joint_all:.3f} | merge conc {mc:.3f} "
              "s75 {m7:.3f} s50 {m5:.3f} | ties-all {ta}".format(
                  mc=r.get("merge_conc", float("nan")),
                  m7=r.get("merge_s75", float("nan")),
                  m5=r.get("merge_s50", float("nan")),
                  ta=("{:.3f}".format(r["ties_all"])
                      if r.get("ties_all") is not None else "n/a"),
                  **r))
    print("written:", op)


if __name__ == "__main__":
    main()
