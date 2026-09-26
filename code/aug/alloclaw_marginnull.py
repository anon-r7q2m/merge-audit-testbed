#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Margin-linear null for the allocation arm."""
import gzip
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
SEEDS = [42, 1042, 2042]
KS = [128, 256]
SPLIT_NAMES = {(1.0, 0.0): "conc", (0.75, 0.25): "s75", (0.5, 0.5): "s50"}


def load_margins(path):
    rows = {}
    meta = None
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
    out = {}
    for K in KS:
        rdir = os.path.join(EXP, "results/alloclaw_k{}".format(K))
        assign = json.load(open(os.path.join(rdir, "alloc_assignment.json")))
        groups = assign["groups"]          # "splitIdx|mirrorIdx" -> [fact_idx]
        splits = [tuple(s) for s in assign["splits"]]
        for seed in SEEDS:
            tag = "k{}-s{}-t0".format(K, seed)
            try:
                metaA, A = load_margins(os.path.join(rdir, "eval", "margins.{}-endA.jsonl.gz".format(tag)))
                metaB, B = load_margins(os.path.join(rdir, "eval", "margins.{}-endB.jsonl.gz".format(tag)))
                metaM, M = load_margins(os.path.join(rdir, "eval", "margins.{}-merge05.jsonl.gz".format(tag)))
            except FileNotFoundError:
                continue
            tA = metaA["theta_alive_top1_q990"]
            tB = metaB["theta_alive_top1_q990"]
            tM = metaM["theta_alive_top1_q990"]
            for gkey, fidxs in groups.items():
                si, mi = gkey.split("|")
                sA, sB = splits[int(si)]
                recs = []
                for fi in fidxs:
                    fid = "S:{}".format(fi)
                    if fid not in M or fid not in A or fid not in B:
                        continue
                    a, b, m = A[fid], B[fid], M[fid]
                    mA, mB, mM = a["m_top1"], b["m_top1"], m["m_top1"]
                    recs.append({
                        "aliveA": a["alive"], "aliveB": b["alive"],
                        "aliveM": m["alive"],
                        "null_raw": 0.5 * mA + 0.5 * mB,
                        "null_norm": 0.5 * (mA / abs(tA)) + 0.5 * (mB / abs(tB)),
                    })
                if not recs:
                    continue
                thr_raw, thr_norm = tM, tM / abs(tM)

                def pred(r, kind):
                    if kind == "raw":
                        return r["null_raw"] >= tM
                    return r["null_norm"] >= (tM / abs(tM))
                def summarize(rs):
                    n = len(rs)
                    if not n:
                        return None
                    return {
                        "n": n,
                        "actual": round(sum(r["aliveM"] for r in rs) / n, 4),
                        "null_raw": round(sum(pred(r, "raw") for r in rs) / n, 4),
                        "null_norm": round(sum(pred(r, "norm") for r in rs) / n, 4),
                    }
                cell = {"split": [sA, sB], "all": summarize(recs)}
                for cname, cond in (
                    ("both_alive", lambda r: r["aliveA"] and r["aliveB"]),
                    ("exactly_one", lambda r: r["aliveA"] != r["aliveB"]),
                    ("none_alive", lambda r: (not r["aliveA"]) and (not r["aliveB"])),
                ):
                    cell[cname] = summarize([r for r in recs if cond(r)])
                out["{}-{}|{}".format(tag, si, mi)] = cell

    agg = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0, 0.0]))
    for key, cell in out.items():
        sname = SPLIT_NAMES[tuple(cell["split"])]
        for cohort in ("all", "both_alive", "exactly_one", "none_alive"):
            s = cell.get(cohort)
            if not s:
                continue
            a = agg[sname][cohort]
            a[0] += s["n"]; a[1] += s["actual"] * s["n"]
            a[2] += s["null_raw"] * s["n"]; a[3] += s["null_norm"] * s["n"]
    print("{:<6} {:<12} {:>6} {:>8} {:>9} {:>9}".format(
        "split", "cohort", "n", "actual", "null_raw", "null_norm"))
    summary = {}
    for sname in ("conc", "s75", "s50"):
        for cohort in ("all", "both_alive", "exactly_one", "none_alive"):
            a = agg[sname].get(cohort)
            if not a or not a[0]:
                continue
            n, ac, nr, nn = a
            summary["{}:{}".format(sname, cohort)] = {
                "n": n, "actual": round(ac / n, 4),
                "null_raw": round(nr / n, 4), "null_norm": round(nn / n, 4)}
            print("{:<6} {:<12} {:>6} {:>8.3f} {:>9.3f} {:>9.3f}".format(
                sname, cohort, n, ac / n, nr / n, nn / n))
    with open(os.path.join(HERE, "alloclaw_marginnull.json"), "w") as f:
        json.dump({"cells": out, "summary": summary}, f, ensure_ascii=False, indent=1)
    print("->", os.path.join(HERE, "alloclaw_marginnull.json"))


if __name__ == "__main__":
    main()
