#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Free-generation survival aggregation."""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from truemerge_survival_v2 import load, alive, wilson  # noqa: E402

EXP = os.path.dirname(HERE)
TDIR = os.path.join(EXP, "results/truemerge", "shards")
GDIR = os.path.join(EXP, "results/truegen", "shards")
FAMS = ["tm1-qwen25-1.5b", "tm2-llama3-8b", "tm3-qwen3-8b"]


def load_gen(fam, tag):
    p = os.path.join(GDIR, "{}.{}.jsonl.gz".format(fam, tag))
    if not os.path.exists(p):
        return None
    rows = {}
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" not in r:
                rows[r["qid"]] = bool(r["ok"])
    return rows


def main():
    out = {}
    for fam in FAMS:
        D = {tag: load("{}.{}".format(fam, tag))
             for tag in ("base", "ftA", "ftB", "merge05")}
        G = {tag: load_gen(fam, tag) for tag in ("base", "ftA", "ftB", "merge05")}
        if any(v is None for v in G.values()):
            print("[note]".format(fam))
            continue
        base, M = D["base"], D["merge05"]
        for parent, other in (("ftA", "ftB"), ("ftB", "ftA")):
            P, Q = D[parent], D[other]
            genP, genM = G[parent], G["merge05"]
            qs = [q for q in P if q in M and q in Q and q in base
                  and q in genP and q in genM]
            grp = {"G": [], "R": [], "S": []}
            for q in qs:
                a, o, b = alive(P[q]), alive(Q[q]), alive(base.get(q))
                if a and not o and not b:
                    grp["G"].append(q)
                elif a and not o and b:
                    grp["R"].append(q)
                elif a and o:
                    grp["S"].append(q)
            excl = grp["G"] + grp["R"]
            cell = {}
            for name, cohort in (("G+R", excl), ("S", grp["S"])):
                if not cohort:
                    continue
                km_strip = sum(1 for q in cohort if alive(M[q]))
                km_gen = sum(1 for q in cohort if genM[q])
                k_endp_gen = sum(1 for q in cohort if genP[q])
                n = len(cohort)
                cell[name] = {"n": n,
                              "merge_strip": round(km_strip / n, 4),
                              "merge_freegen": round(km_gen / n, 4),
                              "merge_freegen_ci90": wilson(km_gen, n),
                              "endpoint_freegen": round(k_endp_gen / n, 4)}

            cell["_agg"] = {
                "freegen_all": round(sum(genM.values()) / len(genM), 4),
                "freegen_ftA": round(sum(G["ftA"].values()) / len(G["ftA"]), 4),
                "freegen_ftB": round(sum(G["ftB"].values()) / len(G["ftB"]), 4),
            }
            out["{}:{}".format(fam, parent)] = cell
    path = os.path.join(HERE, "truegen_survival.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k, c in sorted(out.items()):
        gr = c.get("G+R", {})
        s = c.get("S", {})
        print("{}: excl n={} strip={} gen={} {} | S strip={} gen={} | agg merge/ftA/ftB {}".format(
            k, gr.get("n"), gr.get("merge_strip"), gr.get("merge_freegen"),
            gr.get("merge_freegen_ci90"), s.get("merge_strip"), s.get("merge_freegen"),
            c["_agg"]))
    print("->", path)


if __name__ == "__main__":
    main()
