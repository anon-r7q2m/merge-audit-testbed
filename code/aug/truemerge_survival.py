#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import gzip
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
TD = os.path.join(EXP, "results/truemerge", "shards")
FAMS = ["tm1-qwen25-1.5b", "tm2-llama3-8b", "tm3-qwen3-8b"]
T_MAIN = 2.0
RANK1 = 1.5


def load(tag):
    rows = {}
    with gzip.open(os.path.join(TD, tag + ".jsonl.gz"), "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" not in r:
                rows[r["qid"]] = r
    return rows


def med(x):
    return statistics.median(x) if x else float("nan")


def alive(r):
    return r is not None and r.get("rank") is not None and r["rank"] <= RANK1


def auc(scores_labels):
    pos = [s for s, l in scores_labels if l]
    neg = [s for s, l in scores_labels if not l]
    if not pos or not neg:
        return None
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def mcc_at(pairs, t):
    tp = sum(1 for x, d in pairs if d and x > t)
    fp = sum(1 for x, d in pairs if (not d) and x > t)
    fn = sum(1 for x, d in pairs if d and x <= t)
    tn = sum(1 for x, d in pairs if (not d) and x <= t)
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return round(num / den, 4) if den else None


def logistic2(xs1, xs2, ys, iters=200):
    """P(y=1) ~ 1 + x1 + x2, IRLS。"""
    import numpy as np
    X = np.column_stack([np.ones(len(ys)), xs1, xs2])
    y = np.array(ys, dtype=float)
    w = np.zeros(3)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        W = np.maximum(p * (1 - p), 1e-6)
        H = (X.T * W) @ X
        g = X.T @ (y - p)
        try:
            w = w + np.linalg.solve(H + 1e-4 * np.eye(3), g)
        except Exception:
            break
    return w.tolist()


def main():
    out = {}
    for fam in FAMS:
        base = load(fam + ".base")
        for parent, other in (("ftA", "ftB"), ("ftB", "ftA")):
            P = load(fam + "." + parent)
            Q = load(fam + "." + other)
            M = load(fam + ".merge05")
            D = load(fam + ".dis" + parent[-1] + "05")   # dis05 of the own branch
            qs = [q for q in P if q in M and q in Q and q in base]
            bats = sorted({q.split(":")[0] for q in qs})
            for bat in bats + ["_all"]:
                sub = [q for q in qs if bat == "_all" or q.startswith(bat + ":")]

                grp = {"G": [], "R": [], "S": [], "N": []}
                for q in sub:
                    a, o, b = alive(P[q]), alive(Q[q]), alive(base.get(q))
                    if a and not o and not b:
                        grp["G"].append(q)
                    elif a and not o and b:
                        grp["R"].append(q)
                    elif a and o:
                        grp["S"].append(q)
                    elif not a and not o:
                        grp["N"].append(q)
                cell = {"n_total": len(sub),
                        "n_G": len(grp["G"]), "n_R": len(grp["R"]),
                        "n_S": len(grp["S"]), "n_N": len(grp["N"])}
                for gname, qs2 in (("G", grp["G"]), ("R", grp["R"]),
                                   ("S", grp["S"]), ("G+R", grp["G"] + grp["R"])):
                    if not qs2:
                        continue
                    surv_m = sum(1 for q in qs2 if alive(M[q])) / len(qs2)
                    surv_d = sum(1 for q in qs2 if alive(D[q])) / len(qs2)
                    pairs = [(P[q]["logp"] - M[q]["logp"], not alive(M[q]))
                             for q in qs2]
                    nd = sum(1 for _, d in pairs if d)
                    tp = sum(1 for x, d in pairs if d and x > T_MAIN)
                    fp = sum(1 for x, d in pairs if (not d) and x > T_MAIN)
                    na = len(pairs) - nd
                    cell[gname] = {
                        "n": len(qs2), "n_dead": nd,
                        "surv_merge": round(surv_m, 4),
                        "surv_dis05": round(surv_d, 4),
                        "dnll_dead_med": round(med([x for x, d in pairs if d]), 3),
                        "dnll_aliv_med": round(med([x for x, d in pairs if not d]), 3),
                        "auc": auc(pairs),
                        "t2": {"recall": round(tp / nd, 4) if nd else None,
                               "precision": round(tp / (tp + fp), 4)
                               if tp + fp else None,
                               "mcc": mcc_at(pairs, T_MAIN)},
                    }

                if grp["N"]:
                    cell["bgflip_N"] = round(
                        sum(1 for q in grp["N"] if alive(M[q])) / len(grp["N"]), 4)

                main_qs = grp["G"] + grp["R"]
                if len(main_qs) > 30:
                    mA = [P[q]["m_top1"] for q in main_qs]
                    mB = [Q[q]["m_top1"] for q in main_qs]
                    mM = [M[q]["m_top1"] for q in main_qs]
                    pred = [0.5 * (a + b) for a, b in zip(mA, mB)]
                    mu = statistics.mean(mM)
                    ss_tot = sum((v - mu) ** 2 for v in mM)
                    ss_res = sum((v - p) ** 2 for v, p in zip(mM, pred))
                    r2 = 1 - ss_res / ss_tot if ss_tot else None
                    pred_surv = sum(1 for a, b in zip(mA, mB)
                                    if a > abs(b)) / len(mA)
                    obs_surv = sum(1 for v in mM if v > 0) / len(mM)
                    deaths = [0 if v > 0 else 1 for v in mM]
                    w = logistic2(mA, mB, deaths) if len(set(deaths)) > 1 else None
                    cell["null_model"] = {
                        "r2": round(r2, 4) if r2 is not None else None,
                        "pred_survival": round(pred_surv, 4),
                        "obs_survival_margin": round(obs_surv, 4),
                        "logistic_coef": [round(x, 3) for x in w] if w else None,
                    }
                out["{}:{}:{}".format(fam, parent, bat)] = cell
    with open(os.path.join(HERE, "truemerge_survival.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    for k, c in sorted(out.items()):
        gr = c.get("G+R", {})
        nm = c.get("null_model", {})
        print("{}: G={} R={} S={} N={} | G+R surv_m={} dis05={} auc={} t2={} | nullR2={} predS={} obsS={} logit={}".format(
            k, c["n_G"], c["n_R"], c["n_S"], c["n_N"],
            gr.get("surv_merge"), gr.get("surv_dis05"), gr.get("auc"),
            gr.get("t2"), nm.get("r2"), nm.get("pred_survival"),
            nm.get("obs_survival_margin"), nm.get("logistic_coef")))


if __name__ == "__main__":
    main()
