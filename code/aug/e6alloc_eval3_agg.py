#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open-generation aggregation for the 1B allocation arm."""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
CELL = 400
SPLITS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5)]
SNAME = {0: "conc", 1: "s75", 2: "s50"}


def cell_of(i):
    b = i // CELL
    return SNAME[b // 2], b % 2


def load(seed, model):
    p = os.path.join(EXP, "results/e6allocfu_s{}".format(seed),
                     "eval3", "{}.json".format(model))
    return json.load(open(p))


def acc_by_split(recs):
    """recs: {'shared_i|attr': 0/1} -> {split: (hits, n)}"""
    agg = defaultdict(lambda: [0, 0])
    for key, v in recs.items():
        i = int(key.split("|")[0].split("_")[1])
        sname, _ = cell_of(i)
        agg[sname][0] += v
        agg[sname][1] += 1
    return {k: (h / n, n) for k, (h, n) in sorted(agg.items())}


def main():
    out = {}
    for seed in (1, 2, 3, 4, 5):
        row = {}
        ghosts = []
        for model in ("A", "B", "merge"):
            d = load(seed, model)
            row[model] = {k: round(a, 4) for k, (a, n) in acc_by_split(d["alloc"]).items()}
            g = d.get("ghost", {})
            if g:
                ghosts.append(sum(g.values()) / len(g))
        row["ghost_mean"] = round(sum(ghosts) / len(ghosts), 4) if ghosts else None
        out["s{}".format(seed)] = row
        m = row["merge"]
        print("s{}: merge open conc={} s75={} s50={} | ghost {} | A {} B {}".format(
            seed, m.get("conc"), m.get("s75"), m.get("s50"), row["ghost_mean"],
            row["A"].get("conc"), row["B"].get("conc")))

    deltas = []
    for seed, row in out.items():
        m = row["merge"]
        if all(k in m for k in ("conc", "s75", "s50")):
            deltas.append({"seed": seed,
                           "d_s75": round(m["s75"] - m["conc"], 4),
                           "d_s50": round(m["s50"] - m["conc"], 4)})
    out["_deltas"] = deltas
    print("deltas:", deltas)
    path = os.path.join(HERE, "e6alloc_eval3_agg.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
