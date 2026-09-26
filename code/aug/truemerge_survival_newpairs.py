#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Survival analysis for the two additional real pairs."""
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from truemerge_survival_v2 import load, alive, wilson, FAMS as _  # noqa: F401

EXP = os.path.dirname(HERE)
NEWFAMS = ["tm4-mistral-7b", "tm5-qwen25-7b"]


def main():
    out = {}
    for fam in NEWFAMS:
        D = {tag: load(fam + "." + tag)
             for tag in ("base", "ftA", "ftB", "merge05", "disA05", "disB05")}
        base, M = D["base"], D["merge05"]
        for parent, other in (("ftA", "ftB"), ("ftB", "ftA")):
            P, Q = D[parent], D[other]
            Down = D["dis" + parent[-1] + "05"]
            qs = [q for q in P if q in M and q in Q and q in base]
            grp = {"G": [], "R": [], "S": [], "N": []}
            for q in qs:
                a, o, b = alive(P[q]), alive(Q[q]), alive(base.get(q))
                if a and not o and not b:
                    grp["G"].append(q)
                elif a and not o and b:
                    grp["R"].append(q)
                elif a and o:
                    grp["S"].append(q)
                elif not a and not o:
                    grp["N"].append(q)
            excl = grp["G"] + grp["R"]
            cell = {"n_total": len(qs),
                    **{f"n_{k}": len(v) for k, v in grp.items()}}
            if grp["S"]:
                k = sum(1 for q in grp["S"] if alive(M[q]))
                cell["S"] = {"n": len(grp["S"]),
                             "surv_merge": round(k / len(grp["S"]), 4),
                             "ci90": wilson(k, len(grp["S"]))}
            for gname, qs2 in (("G", grp["G"]), ("R", grp["R"]), ("G+R", excl)):
                if not qs2:
                    continue
                km = sum(1 for q in qs2 if alive(M[q]))
                kd = sum(1 for q in qs2 if alive(Down[q]))
                cell[gname] = {"n": len(qs2),
                               "surv_merge": round(km / len(qs2), 4),
                               "surv_merge_ci90": wilson(km, len(qs2)),
                               "surv_dis05": round(kd / len(qs2), 4),
                               "surv_dis05_ci90": wilson(kd, len(qs2))}
            out["{}:{}".format(fam, parent)] = cell
    path = os.path.join(HERE, "truemerge_survival_newpairs.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for k, c in sorted(out.items()):
        gr = c.get("G+R", {})
        print("{}: G={} R={} S={}(surv {}) N={} | excl merge={} {} dis05={} {}".format(
            k, c["n_G"], c["n_R"], c["n_S"], c.get("S", {}).get("surv_merge"),
            c["n_N"], gr.get("surv_merge"), gr.get("surv_merge_ci90"),
            gr.get("surv_dis05"), gr.get("surv_dis05_ci90")))
    print("->", path)


if __name__ == "__main__":
    main()
