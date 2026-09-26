#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import gzip
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
import run_merge_audit as R                                   # noqa: E402

TD = os.path.join(EXP, "results/tiesdare")
SEED_DIRS = {"results_v2_reid": [42, 1042, 2042],
             "results_v2_c1x3": [6784, 403, 1189],
             "results_v2_aug3": [3184, 5383, 7192]}
TAU_IDXS = [0, 5, 7]
TAU_LAB = {0: "0", 5: "50M", 7: "200M"}
TDENSE = [0.50 + 0.02 * k for k in range(101)]

_cache = {}


def load(path):
    if path in _cache:
        return _cache[path]
    meta, rows = None, {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                rows[r["fact_id"]] = r
    _cache[path] = (meta, rows)
    return _cache[path]


def med(x):
    return statistics.median(x) if x else float("nan")


def main():
    out = {"cells": {}}
    for res_dir, seeds in SEED_DIRS.items():
        root = os.path.join(EXP, res_dir)
        for seed in seeds:
            for ti in TAU_IDXS:
                for method in ("ties", "dare"):
                    mp = os.path.join(TD, "shards",
                                      "s{}-t{}-{}.jsonl.gz".format(seed, ti, method))
                    if not os.path.exists(mp):
                        continue
                    _, mrows = load(mp)
                    for br in "AB":
                        emeta, erows = load(os.path.join(
                            root, "eval", "margins.s{}-t{}-end{}.jsonl.gz".format(seed, ti, br)))
                        L = [fid for fid, r in erows.items()
                             if r["set"] == "excl_" + br and R._alive(r, emeta)]
                        dmeta, drows = load(os.path.join(
                            root, "eval",
                            "margins.s{}-t{}-s0.50-dis{}.jsonl.gz".format(seed, ti, br)))
                        gM = abs(mrows and med([r["m_top1"] for r in mrows.values()
                                                if r["set"].startswith("ghost")]))
                        gD = abs(dmeta["ghost_median_top1_train"])
                        ds = [med([mrows[f]["m_top1"] - t * drows[f]["m_top1"]
                                   for f in L]) for t in TDENSE]
                        s1 = ds[TDENSE.index(1.0)]
                        ts = None
                        for i in range(1, len(TDENSE)):
                            if (ds[i] < 0) != (s1 < 0):
                                ts = TDENSE[i]
                                break
                        dmul = med([mrows[f]["m_top1"] / gM
                                    - drows[f]["m_top1"] / gD for f in L])
                        out["cells"]["s{}-t{}-{}-{}".format(seed, ti, method, br)] = {
                            "n_live": len(L), "D_add_t1": round(s1, 4),
                            "t_star": ts, "D_mul": round(dmul, 4)}
    flips = [c["t_star"] for c in out["cells"].values() if c["t_star"]]
    n = len(out["cells"])
    out["summary"] = {"n_cells": n, "n_flip": len(flips),
                      "flip_frac": round(len(flips) / max(1, n), 3),
                      "tstar_median": med(flips) if flips else None,
                      "tstar_range": [min(flips), max(flips)] if flips else None}
    with open(os.path.join(HERE, "tiesdare_report.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out["summary"], ensure_ascii=False))
    bym = {}
    for k, c in out["cells"].items():
        m = k.rsplit("-", 1)[0].split("-")[2]
        bym.setdefault(m, []).append(c["t_star"] is not None)
    for m, v in bym.items():
        print(m, "flips:", sum(v), "/", len(v))


if __name__ == "__main__":
    main()
