#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Logit-ensemble null for the allocation arm."""
import gzip
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, EXP)
VAL0 = None

SEEDS = [42, 1042, 2042]
KS = [128, 256]
SPLIT_NAMES = {(1.0, 0.0): "conc", (0.75, 0.25): "s75", (0.5, 0.5): "s50"}


def load_dump(path):
    d = np.load(path, allow_pickle=False)
    return d["z"].astype(np.float32), d["fid"], d["ti"], d["val"]


def main():
    import run_merge_audit as R
    out = {}
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

            def load_alive(tagm):
                rows = {}
                with gzip.open(os.path.join(rdir, "eval",
                                            "margins.{}-{}.jsonl.gz".format(tag, tagm)),
                               "rt") as f:
                    meta = json.loads(f.readline())["__meta__"]
                    for line in f:
                        r = json.loads(line)
                        if r.get("battery") == "train":
                            rows[r["fact_id"]] = r["alive"]
                return rows, meta["theta_alive_top1_q990"]
            aliveA, _ = load_alive("endA")
            aliveB, _ = load_alive("endB")
            aliveM, _ = load_alive("merge05")

            keyA = {(fidA[j], int(tiA[j])): j for j in range(len(fidA))}
            nullm = defaultdict(list)   # fid -> [margin per template]
            for jB in range(len(fidB)):
                f = fidB[jB]
                k = (f, int(tiB[jB]))
                jA = keyA.get(k)
                if jA is None:
                    continue
                vi = int(valB[jB]) - R.VAL0
                zn = 0.5 * (zA[jA] + zB[jB])
                m = zn[vi] - np.concatenate([zn[:vi], zn[vi + 1:]]).max()
                nullm[f].append(float(m))
            nullm = {f: float(np.mean(v)) for f, v in nullm.items()}

            ghosts = [v for f, v in nullm.items() if f.startswith("G")]
            thr = float(np.quantile(ghosts, 0.99))
            for gkey, fidxs in groups.items():
                si, mi = gkey.split("|")
                sname = SPLIT_NAMES[tuple(splits[int(si)])]
                recs = []
                for fi in fidxs:
                    fid = "S:{}".format(fi)
                    if fid not in nullm or fid not in aliveM:
                        continue
                    recs.append({
                        "aliveA": aliveA.get(fid, False),
                        "aliveB": aliveB.get(fid, False),
                        "aliveM": aliveM[fid],
                        "null_logit": nullm[fid] >= thr,
                    })
                if not recs:
                    continue
                cell = {}
                for cname, cond in (
                    ("all", lambda r: True),
                    ("both_alive", lambda r: r["aliveA"] and r["aliveB"]),
                    ("exactly_one", lambda r: r["aliveA"] != r["aliveB"]),
                    ("none_alive", lambda r: (not r["aliveA"]) and (not r["aliveB"])),
                ):
                    rs = [r for r in recs if cond(r)]
                    if not rs:
                        continue
                    n = len(rs)
                    cell[cname] = {"n": n,
                                   "actual": round(sum(r["aliveM"] for r in rs) / n, 4),
                                   "null_logit": round(sum(r["null_logit"] for r in rs) / n, 4)}
                out["{}-{}|{}".format(tag, si, mi)] = {"split": list(splits[int(si)]),
                                                       "cells": cell}

    agg = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
    for key, cell in out.items():
        sname = SPLIT_NAMES[tuple(cell["split"])]
        for cohort, s in cell["cells"].items():
            a = agg[sname][cohort]
            a[0] += s["n"]; a[1] += s["actual"] * s["n"]; a[2] += s["null_logit"] * s["n"]
    print("{:<6} {:<12} {:>6} {:>8} {:>10}".format("split", "cohort", "n", "actual", "null_logit"))
    summary = {}
    for sname in ("conc", "s75", "s50"):
        for cohort in ("all", "both_alive", "exactly_one", "none_alive"):
            a = agg[sname].get(cohort)
            if not a or not a[0]:
                continue
            summary["{}:{}".format(sname, cohort)] = {
                "n": a[0], "actual": round(a[1] / a[0], 4),
                "null_logit": round(a[2] / a[0], 4)}
            print("{:<6} {:<12} {:>6} {:>8.3f} {:>10.3f}".format(
                sname, cohort, a[0], a[1] / a[0], a[2] / a[0]))
    with open(os.path.join(HERE, "alloclaw_logitnull.json"), "w") as f:
        json.dump({"cells": out, "summary": summary}, f, ensure_ascii=False, indent=1)
    print("->", os.path.join(HERE, "alloclaw_logitnull.json"))


if __name__ == "__main__":
    main()
