#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""G/R/S/N survival analysis for the true-merge arm."""
import gzip
import json
import math
import os
import statistics

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
TD = os.path.join(EXP, "results/truemerge", "shards")
FAMS = ["tm1-qwen25-1.5b", "tm2-llama3-8b", "tm3-qwen3-8b"]
RANK1 = 1.5
T_MAIN = 2.0
TAUS = [0.0, 0.5, 1.0, 2.0]


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


def wilson(k, n, z=1.645):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - hw, 3), round(c + hw, 3)]


def auc(sl):
    pos = [s for s, l in sl if l]
    neg = [s for s, l in sl if not l]
    if not pos or not neg:
        return None
    w = sum(1 for a in pos for b in neg if a > b)
    t = sum(1 for a in pos for b in neg if a == b)
    return round((w + 0.5 * t) / (len(pos) * len(neg)), 4)


def ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xm = x - x.mean()
    return float((xm @ (y - y.mean())) / (xm @ xm)) if (xm @ xm) > 0 else None


def logistic_multi(Xcols, y, iters=300):
    X = np.column_stack([np.ones(len(y))] + Xcols)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = np.clip(X @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        W = np.maximum(p * (1 - p), 1e-6)
        H = (X.T * W) @ X + 1e-4 * np.eye(X.shape[1])
        g = X.T @ (y - p)
        try:
            w = w + np.linalg.solve(H, g)
        except Exception:
            break
    return w


def predict_auc(w, Xcols):
    X = np.column_stack([np.ones(len(Xcols[0]))] + Xcols)
    z = X @ w
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def main():
    out = {}

    data = {}
    for fam in FAMS:
        data[fam] = {tag: load(fam + "." + tag)
                     for tag in ("base", "ftA", "ftB", "merge05",
                                 "disA05", "disB05")}
    for fam in FAMS:
        D = data[fam]
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
            cell = {"n_total": len(qs), "n_G": len(grp["G"]),
                    "n_R": len(grp["R"]), "n_S": len(grp["S"]),
                    "n_N": len(grp["N"])}
            if grp["S"]:
                k = sum(1 for q in grp["S"] if alive(M[q]))
                cell["S"] = {"n": len(grp["S"]), "surv_merge": round(k / len(grp["S"]), 4),
                             "ci90": wilson(k, len(grp["S"]))}
            if grp["N"]:
                cell["bgflip_N"] = round(
                    sum(1 for q in grp["N"] if alive(M[q])) / len(grp["N"]), 4)
            for gname, qs2 in (("G", grp["G"]), ("R", grp["R"]), ("G+R", excl)):
                if not qs2:
                    continue
                km = sum(1 for q in qs2 if alive(M[q]))
                kd = sum(1 for q in qs2 if alive(Down[q]))
                pairs = [(P[q]["logp"] - M[q]["logp"], not alive(M[q]))
                         for q in qs2]
                nd = sum(1 for _, d in pairs if d)
                tp = sum(1 for x, d in pairs if d and x > T_MAIN)
                fp = sum(1 for x, d in pairs if (not d) and x > T_MAIN)
                cell[gname] = {"n": len(qs2), "n_dead": nd,
                               "surv_merge": round(km / len(qs2), 4),
                               "surv_merge_ci90": wilson(km, len(qs2)),
                               "surv_dis05": round(kd / len(qs2), 4),
                               "surv_dis05_ci90": wilson(kd, len(qs2)),
                               "auc": auc(pairs),
                               "t2": {"recall": round(tp / nd, 4) if nd else None,
                                      "precision": round(tp / (tp + fp), 4)
                                      if tp + fp else None}}

            if len(excl) > 30:
                mB = [Q[q]["m_top1"] for q in excl]
                m0 = [base[q]["m_top1"] for q in excl]
                mM = [M[q]["m_top1"] for q in excl]
                mD = [Down[q]["m_top1"] for q in excl]
                s_int = ols_slope([b - z for b, z in zip(mB, m0)],
                                  [m - d for m, d in zip(mM, mD)])
                s_scl = ols_slope([a - z for a, z in zip(
                    [P[q]["m_top1"] for q in excl], m0)],
                    [d - z for d, z in zip(mD, m0)])
                cell["additivity"] = {
                    "slope_merge_minus_dis05_vs_BminusBase": (
                        round(s_int, 3) if s_int is not None else None),
                    "slope_dis05_minus_base_vs_AminusBase": (
                        round(s_scl, 3) if s_scl is not None else None)}

                sweep = {}
                for t in TAUS:
                    co = [q for q in qs if P[q]["m_top1"] > t
                          and Q[q]["m_top1"] < -t]
                    if co:
                        k = sum(1 for q in co if alive(M[q]))
                        sweep[str(t)] = {"n": len(co),
                                         "surv_merge": round(k / len(co), 4)}
                cell["tau_sweep"] = sweep
            out["{}:{}".format(fam, parent)] = cell

    feats = {}
    for fam in FAMS:
        D = data[fam]
        for parent, other in (("ftA", "ftB"), ("ftB", "ftA")):
            P, Q, M, B = D[parent], D[other], D["merge05"], D["base"]
            rows = []
            for q in P:
                if q not in M or q not in Q or q not in B:
                    continue
                if alive(P[q]) and not alive(Q[q]):
                    rows.append(([P[q]["m_top1"], Q[q]["m_top1"],
                                 B[q]["m_top1"]],
                                0 if alive(M[q]) else 1))
            feats["{}:{}".format(fam, parent)] = rows
    xval = {}
    keys = sorted(feats)
    for tr in keys:
        for te in keys:
            if tr == te or not feats[tr] or not feats[te]:
                continue
            Xtr = [r[0] for r in feats[tr]]
            ytr = [r[1] for r in feats[tr]]
            if len(set(ytr)) < 2:
                continue
            w = logistic_multi([np.array([r[0] for r in Xtr]),
                                np.array([r[1] for r in Xtr]),
                                np.array([r[2] for r in Xtr])], ytr)
            Xte = [np.array([r[0][0] for r in feats[te]]),
                   np.array([r[0][1] for r in feats[te]]),
                   np.array([r[0][2] for r in feats[te]])]
            yte = [r[1] for r in feats[te]]
            p = predict_auc(w, Xte)
            xval["{}->{}".format(tr, te)] = {"auc": auc(list(zip(p, yte))),
                                             "n_te": len(yte),
                                             "n_tr": len(ytr)}
    out["_crossval_predictor"] = xval
    with open(os.path.join(HERE, "truemerge_survival_v2.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    for k, c in sorted(out.items()):
        if k.startswith("_"):
            continue
        gr = c.get("G+R", {})
        ad = c.get("additivity", {})
        print("{}: G={} R={} S={}(surv {}) N={}(flip {}) | excl surv_m={} dis05={} auc={} | slopes {}/{} | tau {}".format(
            k, c["n_G"], c["n_R"], c["n_S"], c.get("S", {}).get("surv_merge"),
            c["n_N"], c.get("bgflip_N"), gr.get("surv_merge"),
            gr.get("surv_dis05"), gr.get("auc"),
            ad.get("slope_merge_minus_dis05_vs_BminusBase"),
            ad.get("slope_dis05_minus_base_vs_AminusBase"),
            {t: v["surv_merge"] for t, v in c.get("tau_sweep", {}).items()}))
    print("crossval:")
    for k, v in sorted(xval.items()):
        print("  {}: auc={} (n_te={})".format(k, v["auc"], v["n_te"]))


if __name__ == "__main__":
    main()
