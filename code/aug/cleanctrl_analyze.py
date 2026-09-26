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

EV = os.path.join(EXP, "results/cleanctrl", "eval")
SEEDS = (881, 913, 733)
TAUS = ("t0", "t5")
RANK1 = 1.5
T_MAIN = 2.0
THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load(name):
    meta, rows = None, {}
    with gzip.open(os.path.join(EV, "margins.{}.jsonl.gz".format(name)),
                   "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                rows[r["fact_id"]] = r
    return meta, rows


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def sweep(pairs, t):
    nd = sum(1 for _, d in pairs if d)
    na = len(pairs) - nd
    tp = sum(1 for x, d in pairs if d and x > t)
    fp = sum(1 for x, d in pairs if (not d) and x > t)
    return {"recall": round(tp / nd, 4) if nd else None,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "fpr": round(fp / na, 4) if na else None,
            "n_dead": nd, "n_alive": na}


def main():
    out = {"storage": {}, "survival": {}, "threshold": {}, "shared_ctrl": {}}
    for seed in SEEDS:
        for tau in TAUS:
            eA, rA = load("u2-s{}-{}-endA".format(seed, tau))
            eB, rB = load("u2-s{}-{}-endB".format(seed, tau))
            eM, rM = load("u2-s{}-{}-merge05".format(seed, tau))
            key = "s{}-{}".format(seed, tau)

            def fr1(rows, sset):
                sel = [r for r in rows.values() if r["set"] == sset]
                return (round(sum(1 for r in sel
                                  if r["rank_bar"] <= RANK1) / len(sel), 4)
                        if sel else None)
            out["storage"][key] = {
                "endA_exclA": fr1(rA, "excl_A"), "endA_shared": fr1(rA, "shared"),
                "endB_exclB": fr1(rB, "excl_B"), "endB_shared": fr1(rB, "shared"),
                "endA_ghostA": fr1(rA, "ghost_A"),
                "endB_ghostB": fr1(rB, "ghost_B")}

            for br, er, mr in (("A", (eA, rA), (eM, rM)), ("B", (eB, rB), (eM, rM))):
                em, erows = er
                mm, mrows = mr
                thE = em["theta_alive_top1_q990"]
                thM = mm["theta_alive_top1_q990"]
                L = [f for f, r in erows.items()
                     if r["set"] == "excl_" + br and r["m_top1"] >= thE]
                if not L:
                    continue
                surv = sum(1 for f in L if mrows[f]["m_top1"] >= thM) / len(L)
                pairs = [(erows[f]["logp"] - mrows[f]["logp"],
                          mrows[f]["m_top1"] < thM) for f in L]
                ck = key + "-" + br
                out["survival"][ck] = {"n_live": len(L),
                                       "survival": round(surv, 4)}
                out["threshold"][ck] = {
                    "t{:.1f}".format(t): sweep(pairs, t) for t in THRESHOLDS}
                out["threshold"][ck]["dnll_dead_median"] = round(
                    med([x for x, d in pairs if d]), 4)
                out["threshold"][ck]["dnll_aliv_median"] = round(
                    med([x for x, d in pairs if not d]), 4)

            thEA = eA["theta_alive_top1_q990"]
            thEB = eB["theta_alive_top1_q990"]
            thM = eM["theta_alive_top1_q990"]
            sh = [f for f, r in rA.items() if r["set"] == "shared"]
            both_alive = [f for f in sh
                          if rA[f]["m_top1"] >= thEA and rB[f]["m_top1"] >= thEB]
            merge_alive = sum(1 for f in both_alive
                              if rM[f]["m_top1"] >= thM)
            out["shared_ctrl"][key] = {
                "n_shared": len(sh), "both_alive": len(both_alive),
                "both_alive_frac": round(len(both_alive) / max(1, len(sh)), 4),
                "merge_keeps": round(merge_alive / max(1, len(both_alive)), 4)}
    with open(os.path.join(HERE, "cleanctrl_report.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
