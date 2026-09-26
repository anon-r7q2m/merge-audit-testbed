#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Battery-stratified probe-level bootstrap CIs for the real arms."""
import gzip
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
TD = os.path.join(EXP, "results/truemerge", "shards")
FAMS = ["tm1-qwen25-1.5b", "tm2-llama3-8b", "tm3-qwen3-8b",
        "tm4-mistral-7b", "tm5-qwen25-7b"]
RANK1 = 1.5
B = 10_000
SEED = 20260915


def load(tag):
    rows = {}
    with gzip.open(os.path.join(TD, tag + ".jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" not in r:
                rows[r["qid"]] = r
    return rows


def alive(r):
    return r is not None and r.get("rank") is not None and r["rank"] <= RANK1


def battery(qid):
    return qid.split(":", 1)[0]


def boot_ci(items, rng, B=B):
    """Battery-stratified probe-level bootstrap CIs for the real arms."""
    by_bat = defaultdict(list)
    for bat, a in items:
        by_bat[bat].append(a)
    bats = sorted(by_bat)
    pools = [np.asarray(by_bat[b], dtype=bool) for b in bats]
    ns = [len(p) for p in pools]
    n_tot = sum(ns)
    if n_tot == 0:
        return None
    idx = [rng.randint(0, n, size=(B, n)) for n in ns]

    est = np.zeros(B)
    for pool, ix, n in zip(pools, idx, ns):
        est += pool[ix].mean(axis=1) * n
    est /= n_tot
    lo, hi = np.percentile(est, [2.5, 97.5])
    return [round(float(lo), 3), round(float(hi), 3)]


def main():
    rng = np.random.RandomState(SEED)
    data = {fam: {tag: load(fam + "." + tag)
                  for tag in ("base", "ftA", "ftB", "merge05",
                              "disA05", "disB05")}
            for fam in FAMS}
    out = {}
    for fam in FAMS:
        D = data[fam]
        base, M = D["base"], D["merge05"]
        pair_pool = {"G+R": [], "S": []}
        for parent, other in (("ftA", "ftB"), ("ftB", "ftA")):
            P, Q = D[parent], D[other]
            Down = D["dis" + parent[-1] + "05"]
            qs = [q for q in P if q in M and q in Q and q in base]
            excl, ess = [], []
            for q in qs:
                a, o, b = alive(P[q]), alive(Q[q]), alive(base.get(q))
                if a and not o:
                    excl.append(q)
                elif a and o:
                    ess.append(q)
            cell = {}
            for name, cohort in (("G+R", excl), ("S", ess)):
                if not cohort:
                    continue
                it_m = [(battery(q), alive(M[q])) for q in cohort]
                it_d = [(battery(q), alive(Down[q])) for q in cohort]
                km = sum(a for _, a in it_m)
                rec = {"n": len(cohort),
                       "n_by_battery": {b: sum(1 for x, _ in it_m if x == b)
                                        for b in sorted({x for x, _ in it_m})},
                       "surv_merge": round(km / len(cohort), 4),
                       "surv_merge_ci95": boot_ci(it_m, rng)}
                if name == "G+R":
                    kd = sum(a for _, a in it_d)
                    rec["surv_dis05"] = round(kd / len(cohort), 4)
                    rec["surv_dis05_ci95"] = boot_ci(it_d, rng)
                cell[name] = rec
                pair_pool[name].extend((fam, parent, q) for q in cohort)
            out["{}:{}".format(fam, parent)] = cell

        pooled = {}
        for name, pool in pair_pool.items():
            if not pool:
                continue
            it_m = [(battery(q), alive(M[q])) for _, _, q in pool]
            km = sum(a for _, a in it_m)
            rec = {"n": len(pool),
                   "surv_merge": round(km / len(pool), 4),
                   "surv_merge_ci95": boot_ci(it_m, rng)}
            if name == "G+R":
                it_d = []
                for _, parent, q in pool:
                    Down = D["dis" + parent[-1] + "05"]
                    it_d.append((battery(q), alive(Down[q])))
                kd = sum(a for _, a in it_d)
                rec["surv_dis05"] = round(kd / len(pool), 4)
                rec["surv_dis05_ci95"] = boot_ci(it_d, rng)
            pooled[name] = rec
        out[fam + ":_pair"] = pooled
    path = os.path.join(HERE, "truemerge_bootstrap.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k in sorted(out):
        if k.endswith(":_pair"):
            gr = out[k].get("G+R", {})
            s = out[k].get("S", {})
            print("{}: excl n={} merge {} CI{} dis05 {} CI{} | S n={} merge {} CI{}".format(
                k, gr.get("n"), gr.get("surv_merge"), gr.get("surv_merge_ci95"),
                gr.get("surv_dis05"), gr.get("surv_dis05_ci95"),
                s.get("n"), s.get("surv_merge"), s.get("surv_merge_ci95")))
        else:
            gr = out[k].get("G+R", {})
            print("  {}: n={} merge {} CI{} dis05 {} CI{} bat={}".format(
                k, gr.get("n"), gr.get("surv_merge"), gr.get("surv_merge_ci95"),
                gr.get("surv_dis05"), gr.get("surv_dis05_ci95"),
                gr.get("n_by_battery")))
    print("->", path)


if __name__ == "__main__":
    main()
