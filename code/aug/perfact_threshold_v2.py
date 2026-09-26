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

REID = os.path.join(EXP, "results_v2_reid")
AUG3 = os.path.join(EXP, "results_v2_aug3")
C1X3 = os.path.join(EXP, "results_v2_c1x3")
SEED_DIR = {s: REID for s in (42, 1042, 2042)}
SEED_DIR.update({s: AUG3 for s in (3184, 5383, 7192)})
SEED_DIR.update({s: C1X3 for s in (6784, 403, 1189)})
CAMPAIGN = {s: "c1" for s in (42, 1042, 2042)}
CAMPAIGN.update({s: "c2" for s in (3184, 5383, 7192)})
CAMPAIGN.update({s: "c1x" for s in (6784, 403, 1189)})
SEEDS = (42, 1042, 2042, 3184, 5383, 7192, 6784, 403, 1189)
THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
T_MAIN = 2.0

_cache = {}


def load_both(seed, suffix):
    """Analysis script for the merge-audit study."""
    key = (seed, suffix)
    if key in _cache:
        return _cache[key]
    path = os.path.join(SEED_DIR[seed], "eval",
                        "margins.s{}-{}.jsonl.gz".format(seed, suffix))
    meta, tr, ho = None, {}, {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                tr[r["fact_id"]] = r
            elif r["battery"] == "hold":
                ho[r["fact_id"]] = r

    gh = sorted(r["m_top1"] for r in ho.values() if r["set"].startswith("ghost"))
    th_hold = None
    if gh:
        k = int(0.99 * len(gh))
        th_hold = gh[min(k, len(gh) - 1)]
    th_train = meta["theta_alive_top1_q990"] if meta else None
    _cache[key] = (tr, ho, th_train, th_hold)
    return _cache[key]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def rates(pairs, t):
    """Analysis script for the merge-audit study."""
    n_dead = sum(1 for _, d in pairs if d)
    n_alive = len(pairs) - n_dead
    tp = sum(1 for x, d in pairs if d and x > t)
    fp = sum(1 for x, d in pairs if (not d) and x > t)
    return {"recall": round(tp / n_dead, 4) if n_dead else None,
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "fpr": round(fp / n_alive, 4) if n_alive else None,
            "n_dead": n_dead, "n_alive": n_alive, "tp": tp, "fp": fp}


def prec_at(recall, fpr, pi):
    if recall is None or fpr is None:
        return None
    num = recall * pi
    den = num + fpr * (1 - pi)
    return round(num / den, 4) if den else None


def main():
    cells = {}
    run_pool = {}       # seed -> {"tt": [], "th": []}  (score, dead)
    camp_pool = {"c1": {"tt": [], "th": []}, "c2": {"tt": [], "th": []},
                 "c1x": {"tt": [], "th": []}}
    base_pool = {"c1": [], "c2": [], "c1x": []}
    joint = {"nn": 0, "np": 0, "pn": 0, "pp": 0}  # sign(m_merge), sign(m_dis) @ τ=0
    for seed in SEEDS:
        camp = CAMPAIGN[seed]
        run_pool[seed] = {"tt": [], "th": []}
        for ti in range(8):
            for br in "AB":
                trE, hoE, thE_tr, thE_ho = load_both(seed, "t{}-end{}".format(ti, br))
                trM, hoM, thM_tr, thM_ho = load_both(seed, "t{}-a0.50-merge".format(ti))
                if not (trE and hoE and trM and hoM):
                    continue

                L_tr = [f for f, r in trE.items()
                        if r["set"] == "excl_" + br and r["m_top1"] >= thE_tr]

                L_ho = [f for f, r in hoE.items()
                        if r["set"] == "excl_" + br and r["m_top1"] >= thE_ho]
                pairs_tt, pairs_th = [], []
                for f in L_tr:
                    if f not in trM:
                        continue
                    dn = trE[f]["logp"] - trM[f]["logp"]
                    dead = trM[f]["m_top1"] < thM_tr
                    pairs_tt.append((dn, dead))
                    base_pool[camp].append({
                        "dnll": dn, "dead": dead,
                        "logp_m": trM[f]["logp"], "m_m": trM[f]["m_top1"],
                        "dm": trE[f]["m_top1"] - trM[f]["m_top1"]})
                for f in L_ho:
                    if f not in hoM or f not in trE or f not in trM:
                        continue
                    dn = trE[f]["logp"] - trM[f]["logp"]
                    dead = hoM[f]["m_top1"] < thM_ho
                    pairs_th.append((dn, dead))
                run_pool[seed]["tt"].extend(pairs_tt)
                run_pool[seed]["th"].extend(pairs_th)
                camp_pool[camp]["tt"].extend(pairs_tt)
                camp_pool[camp]["th"].extend(pairs_th)
                cells["s{}-t{}-{}".format(seed, ti, br)] = {
                    "n_tt": len(pairs_tt), "n_th": len(pairs_th),
                    "tt@2": rates(pairs_tt, T_MAIN),
                    "th@2": rates(pairs_th, T_MAIN)}
                if ti == 0:
                    trD, _, thD_tr, _ = load_both(seed, "t0-s0.50-dis{}".format(br))
                    for f in L_tr:
                        if f in trM and f in trD:
                            sm = 1 if trM[f]["m_top1"] > 0 else -1
                            sd = 1 if trD[f]["m_top1"] > 0 else -1
                            joint[{( -1, -1): "nn", (-1, 1): "np",
                                   (1, -1): "pn", (1, 1): "pp"}[(sm, sd)]] += 1
    out = {"cells": cells, "joint_sign_tau0": joint, "runs": {}, "campaigns": {},
           "baselines": {}, "prevalence": {}}
    for seed in SEEDS:
        out["runs"][str(seed)] = {
            "tt@2": rates(run_pool[seed]["tt"], T_MAIN),
            "th@2": rates(run_pool[seed]["th"], T_MAIN)}
    for camp in ("c1", "c2", "c1x"):
        tt, th = camp_pool[camp]["tt"], camp_pool[camp]["th"]
        out["campaigns"][camp] = {
            "n_tt": len(tt), "n_th": len(th),
            "sweep_tt": {"{:.1f}".format(t): rates(tt, t) for t in THRESHOLDS},
            "sweep_th": {"{:.1f}".format(t): rates(th, t) for t in THRESHOLDS}}

        r2 = rates(tt, T_MAIN)
        out["prevalence"][camp] = {
            "pi": round(r2["n_dead"] / max(1, r2["n_dead"] + r2["n_alive"]), 4),
            "prec@70%": prec_at(r2["recall"], r2["fpr"], 0.70),
            "prec@10%": prec_at(r2["recall"], r2["fpr"], 0.10),
            "prec@5%": prec_at(r2["recall"], r2["fpr"], 0.05),
            "prec@1%": prec_at(r2["recall"], r2["fpr"], 0.01)}

        rows = base_pool[camp]
        for name, key, sign in (("logp_merge", "logp_m", -1),
                                ("m_top1_merge", "m_m", -1),
                                ("d_mtop1", "dm", 1)):
            pairs = [(sign * r[key], r["dead"]) for r in rows]
            out["baselines"].setdefault(camp, {})[name] = {
                "{:.1f}".format(t): rates(pairs, t)
                for t in sorted({T_MAIN, med([p[0] for p in pairs])})}
    with open(os.path.join(HERE, "perfact_threshold_v2.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)

    print("joint_sign_tau0:", joint)
    for camp in ("c1", "c2", "c1x"):
        c = out["campaigns"][camp]
        print(camp, "n_tt", c["n_tt"], "n_th", c["n_th"])
        print("  tt@2.0:", c["sweep_tt"]["2.0"])
        print("  th@2.0:", c["sweep_th"]["2.0"])
        print("  prevalence:", out["prevalence"][camp])
        for bn, bv in out["baselines"][camp].items():
            print("  baseline", bn, bv)
    print("per-run:")
    for seed in SEEDS:
        r = out["runs"][str(seed)]
        print("  s{}: tt recall={} prec={} | th recall={} prec={}".format(
            seed, r["tt@2"]["recall"], r["tt@2"]["precision"],
            r["th@2"]["recall"], r["th@2"]["precision"]))


if __name__ == "__main__":
    main()
