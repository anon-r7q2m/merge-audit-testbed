#!/usr/bin/env python3
"""alloclaw_ensemble_decomp.py -- allocation-gain vs ensemble-reference decomposition.

Per (K, seed) unit, on the assigned pool (all-cohort, unconditional):
  S_M(g)  = merge survival in allocation cell g (measured, margin shards)
  S_E(g)  = survival predicted by output-ensemble reference E on the same facts
            (per-fact logits, own ghost-Q99 threshold per variant)
  E in {raw logit-average, std-normalized logit-average, probability mixture}

Decomposition reported per unit and pooled:
  Delta_M  = S_M(balanced) - S_M(concentrated)
           = [S_E(bal) - S_E(conc)]          (the reference's own allocation gain)
           + [d_E(conc) - d_E(bal)]          (change in merge-relative deficit),
  d_E(g) = S_E(g) - S_M(g).

Read-only over the released margin shards + logit dumps. Output:
aug/analysis-out/alloclaw_ensemble_decomp.json
"""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
SEEDS = [42, 1042, 2042]
KS = [128, 256]
SPLIT_NAMES = {(1.0, 0.0): "conc", (0.75, 0.25): "s75", (0.5, 0.5): "s50"}


def load_dump(path):
    d = np.load(path, allow_pickle=False)
    return d["z"].astype(np.float32), d["fid"], d["ti"], d["val"]


def null_margins(zA, zB, vi, mode):
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


def load_alive(path):
    rows = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                continue
            if r.get("battery") == "train":
                rows[r["fact_id"]] = r["alive"]
    return rows


def main():
    import run_r2rank08 as R
    units = []
    for K in KS:
        rdir = os.path.join(EXP, "results_aug_alloclaw_k{}".format(K))
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
            zAa, zBb = zA[jA], zB[keepB]
            fidv = fidB[keepB]
            viv = (valB[keepB] - R.VAL0).astype(int)
            aliveM = load_alive(os.path.join(
                rdir, "eval", "margins.{}-merge05.jsonl.gz".format(tag)))

            variants = {}
            for mode in ("raw", "stdnorm", "prob"):
                m = null_margins(zAa, zBb, viv, mode)
                per_fact = defaultdict(list)
                for f, mm in zip(fidv, m):
                    per_fact[f].append(float(mm))
                per_fact = {f: float(np.mean(v)) for f, v in per_fact.items()}
                ghosts = np.array([v for f, v in per_fact.items()
                                   if f.startswith("G")])
                thr = float(np.quantile(ghosts, 0.99))
                variants[mode] = {f: v >= thr for f, v in per_fact.items()}

            # 逐 split 的"all cohort"存活
            per_split = {}
            for gkey, fidxs in groups.items():
                si, mi = gkey.split("|")
                sname = SPLIT_NAMES[tuple(splits[int(si)])]
                fids = ["S:{}".format(fi) for fi in fidxs]
                fids = [f for f in fids
                        if f in aliveM and f in variants["raw"]]
                if not fids:
                    continue
                rec = per_split.setdefault(sname, {"n": 0, "M": 0.0,
                                                   "raw": 0.0,
                                                   "stdnorm": 0.0,
                                                   "prob": 0.0})
                rec["n"] += len(fids)
                rec["M"] += sum(aliveM[f] for f in fids)
                for mode in ("raw", "stdnorm", "prob"):
                    rec[mode] += sum(variants[mode][f] for f in fids)
            for sname, rec in per_split.items():
                for k in ("M", "raw", "stdnorm", "prob"):
                    rec[k] = rec[k] / rec["n"]
            u = {"K": K, "seed": seed, "cells": per_split}
            # 分解（conc vs s50）
            if "conc" in per_split and "s50" in per_split:
                c, b = per_split["conc"], per_split["s50"]
                u["delta_M"] = b["M"] - c["M"]
                for mode in ("raw", "stdnorm", "prob"):
                    ref_gain = b[mode] - c[mode]
                    d_conc = c[mode] - c["M"]
                    d_bal = b[mode] - b["M"]
                    u[mode] = {"S_E_conc": c[mode], "S_E_bal": b[mode],
                               "ref_gain": ref_gain,
                               "deficit_conc": d_conc, "deficit_bal": d_bal,
                               "deficit_change": d_conc - d_bal}
            units.append(u)

    # 汇总打印
    print(f"{'K':>4} {'seed':>5} {'ΔM':>7} | {'ref_gain':>22} | "
          f"{'deficit_change':>22}")
    print(f"{'':>4} {'':>5} {'':>7} | {'raw / stdnorm / prob':>22} | ")
    for u in units:
        if "delta_M" not in u:
            continue
        rg = "{:+.3f} / {:+.3f} / {:+.3f}".format(u["raw"]["ref_gain"],
                                                  u["stdnorm"]["ref_gain"],
                                                  u["prob"]["ref_gain"])
        dc = "{:+.3f} / {:+.3f} / {:+.3f}".format(
            u["raw"]["deficit_change"], u["stdnorm"]["deficit_change"],
            u["prob"]["deficit_change"])
        print(f"K{u['K']:<3} {u['seed']:>5} {u['delta_M']:+.3f} | {rg} | {dc}")

    op = os.path.join(HERE, "analysis-out", "alloclaw_ensemble_decomp.json")
    with open(op, "w") as f:
        json.dump({"units": units}, f, indent=1)
    print("written:", op)


if __name__ == "__main__":
    main()
