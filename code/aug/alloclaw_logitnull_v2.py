#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ensemble-null scale-convention sensitivity and per-fact confusion."""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, EXP)
SEEDS = [42, 1042, 2042]
KS = [128, 256]
SPLIT_NAMES = {(1.0, 0.0): "conc", (0.75, 0.25): "s75", (0.5, 0.5): "s50"}


def load_dump(path):
    d = np.load(path, allow_pickle=False)
    return d["z"].astype(np.float32), d["fid"], d["ti"], d["val"]


def null_margins(zA, zB, vi, mode):
    """Ensemble-null scale-convention sensitivity and per-fact confusion."""
    if mode == "raw":
        zn = 0.5 * (zA + zB)
    elif mode == "stdnorm":
        zn = 0.5 * (zA / zA.std() + zB / zB.std())
    elif mode == "prob":
        def sm(z):
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)
        zn = np.log(np.maximum(0.5 * (sm(zA) + sm(zB)), 1e-12))
    else:
        raise ValueError(mode)
    n = zn.shape[0]
    z_vi = zn[np.arange(n), vi]
    mask = np.zeros_like(zn, dtype=bool)
    mask[np.arange(n), vi] = True
    z_rest = np.where(mask, -np.inf, zn).max(1)
    return z_vi - z_rest


def main():
    import run_merge_audit as R
    summary = {}
    confusion_rows = []
    for K in KS:
        rdir = os.path.join(EXP, "results/alloclaw_k{}".format(K))
        assign = json.load(open(os.path.join(rdir, "alloc_assignment.json")))
        groups = assign["groups"]
        splits = [tuple(s) for s in assign["splits"]]
        for seed in SEEDS:
            tag = "k{}-s{}-t0".format(K, seed)
            pA = os.path.join(rdir, "logitdump", "endA-s{}.npz".format(seed))
            pB = os.path.join(rdir, "logitdump", "endB-s{}.npz".format(seed))
            if not (os.path.exists(pA) and os.path.exists(pB)):
                continue
            zA, fidA, tiA, valA = load_dump(pA)
            zB, fidB, tiB, valB = load_dump(pB)

            keyA = {(fidA[j], int(tiA[j])): j for j in range(len(fidA))}
            keepB = [j for j in range(len(fidB)) if (fidB[j], int(tiB[j])) in keyA]
            jA = np.array([keyA[(fidB[j], int(tiB[j]))] for j in keepB])
            zAa = zA[jA]
            zBb = zB[keepB]
            fidv = fidB[keepB]
            viv = (valB[keepB] - R.VAL0).astype(int)

            def load_alive(tagm):
                rows = {}
                with gzip.open(os.path.join(rdir, "eval",
                                            "margins.{}-{}.jsonl.gz".format(tag, tagm)),
                               "rt") as f:
                    f.readline()
                    for line in f:
                        r = json.loads(line)
                        if r.get("battery") == "train":
                            rows[r["fact_id"]] = r["alive"]
                return rows
            aliveA, aliveB, aliveM = load_alive("endA"), load_alive("endB"), load_alive("merge05")


            variants = {}
            for mode in ("raw", "stdnorm", "prob"):
                m = null_margins(zAa, zBb, viv, mode)
                per_fact = defaultdict(list)
                for f, mm in zip(fidv, m):
                    per_fact[f].append(float(mm))
                per_fact = {f: float(np.mean(v)) for f, v in per_fact.items()}
                ghosts = np.array([v for f, v in per_fact.items() if f.startswith("G")])
                thr = float(np.quantile(ghosts, 0.99))
                variants[mode] = ({f: v >= thr for f, v in per_fact.items()}, thr)

            for gkey, fidxs in groups.items():
                si, mi = gkey.split("|")
                sname = SPLIT_NAMES[tuple(splits[int(si)])]
                fids = ["S:{}".format(fi) for fi in fidxs]
                fids = [f for f in fids if f in aliveM and f in variants["raw"][0]]
                for cname, cond in (
                    ("all", lambda a, b: True),
                    ("none_alive", lambda a, b: (not a) and (not b)),
                    ("exactly_one", lambda a, b: a != b),
                ):
                    sel = [f for f in fids if cond(aliveA.get(f, False), aliveB.get(f, False))]
                    if not sel:
                        continue
                    key = "{}:{}".format(sname, cname)
                    rec = summary.setdefault(key, {"n": 0, "actual": 0.0,
                                                   "raw": 0.0, "stdnorm": 0.0, "prob": 0.0})
                    rec["n"] += len(sel)
                    rec["actual"] += sum(aliveM[f] for f in sel)
                    for mode in ("raw", "stdnorm", "prob"):
                        rec[mode] += sum(variants[mode][0][f] for f in sel)

            allf = [f for f in variants["raw"][0] if f in aliveM and f.startswith("S:")]
            tp = sum(1 for f in allf if variants["raw"][0][f] and aliveM[f])
            fp = sum(1 for f in allf if variants["raw"][0][f] and not aliveM[f])
            fn = sum(1 for f in allf if not variants["raw"][0][f] and aliveM[f])
            tn = sum(1 for f in allf if not variants["raw"][0][f] and not aliveM[f])
            confusion_rows.append({"cell": tag, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    print("{:<16} {:>6} {:>7} {:>7} {:>8} {:>7}".format("cohort", "n", "actual", "raw", "stdnorm", "prob"))
    for k in sorted(summary):
        r = summary[k]
        n = r["n"]
        print("{:<16} {:>6} {:>7.3f} {:>7.3f} {:>8.3f} {:>7.3f}".format(
            k, n, r["actual"] / n, r["raw"] / n, r["stdnorm"] / n, r["prob"] / n))
    out = {"summary": {k: {"n": v["n"], "actual": round(v["actual"] / v["n"], 4),
                           "raw": round(v["raw"] / v["n"], 4),
                           "stdnorm": round(v["stdnorm"] / v["n"], 4),
                           "prob": round(v["prob"] / v["n"], 4)}
                       for k, v in summary.items()},
           "confusion_raw": confusion_rows}
    with open(os.path.join(HERE, "alloclaw_logitnull_v2.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("->", os.path.join(HERE, "alloclaw_logitnull_v2.json"))


if __name__ == "__main__":
    main()
