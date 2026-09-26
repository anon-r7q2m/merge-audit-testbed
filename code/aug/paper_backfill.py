#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import gzip
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
os.environ.setdefault("TANH_R2RANK08_SEEDS", "42,1042,2042,3184,5383,7192")
import run_merge_audit as R

REID = os.path.join(EXP, "results_v2_reid")
AUG3 = os.path.join(EXP, "results_v2_aug3")
CAMPAIGNS = [("c1", REID, [42, 1042, 2042]), ("c2", AUG3, [3184, 5383, 7192])]
ALL_SEEDS = [42, 1042, 2042, 3184, 5383, 7192]
SEED_DIR = {s: REID for s in (42, 1042, 2042)}
SEED_DIR.update({s: AUG3 for s in (3184, 5383, 7192)})
SEED_CAMP = {s: ("c1" if s in (42, 1042, 2042) else "c2") for s in ALL_SEEDS}
TAU_TOKENS = list(R.TAU_GRID_TOKENS)
TAU_LABELS = ["0", "3.1M", "6.3M", "12.5M", "25M", "50M", "100M", "200M"]
TGRID8 = [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
RANK1 = R.RANK1_BAR_THRESH

_cache = {}


def load_train(seed, model_suffix):
    """load margins for model id 's{seed}-{suffix}' from the seed's campaign dir;
    returns (meta, {fid: row}) for the train battery only."""
    key = (seed, model_suffix)
    if key in _cache:
        return _cache[key]
    mid = "s{}-{}".format(seed, model_suffix)
    path = os.path.join(SEED_DIR[seed], "eval", "margins.{}.jsonl.gz".format(mid))
    meta, rows = None, {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            elif r["battery"] == "train":
                rows[r["fact_id"]] = r
    _cache[key] = (meta, rows)
    return _cache[key]


def cohort_L(seed, ti, br):
    """F_live: own-branch exclusive facts alive at the own endpoint (frozen rule:
    _alive with q=0.99 top1 on the endpoint's own meta)."""
    emeta, erows = load_train(seed, "t{}-end{}".format(ti, br))
    return [fid for fid, r in erows.items()
            if r["set"] == "excl_" + br and R._alive(r, emeta)], emeta, erows


def ghost_ids(seed, ti, br):
    _, erows = load_train(seed, "t{}-end{}".format(ti, br))
    return [fid for fid, r in erows.items() if r["set"].startswith("ghost")]


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


OUT = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "provenance": "read-only re-aggregation for PENDING.md backfill; "
                     "frozen helpers from run_merge_audit.py; campaigns: "
                     "c1=results_v2_reid (K=128), c2=results_v2_aug3 (K=64)"}

# ============================== A. dense t* scan ==============================
print("== A. dense t* scan ==", flush=True)
sham = json.load(open(os.path.join(EXP, "adjudication", "sham_scale_control.json")))


def raw_add_resid(mrows, drows, L, t):
    return R.median([mrows[f]["m_top1"] - t * drows[f]["m_top1"] for f in L])


A = {"cells": {}, "tgrid_dense": [0.50 + 0.02 * k for k in range(101)]}
for seed in ALL_SEEDS:
    for br in "AB":
        L, emeta, erows = cohort_L(seed, 0, br)
        _, mrows = load_train(seed, "t0-a0.50-merge")
        _, drows = load_train(seed, "t0-s0.50-dis" + br)
        mend_scale = R.median([erows[f]["m_top1"] for f in L])
        gmedM = R.median([mrows[g]["m_top1"] for g in ghost_ids(seed, 0, br)])
        gmedD = R.median([drows[g]["m_top1"] for g in ghost_ids(seed, 0, br)])
        gmedE = R.median([erows[g]["m_top1"] for g in ghost_ids(seed, 0, br)])
        Mend_cal = R.median([erows[f]["m_top1"] - gmedE for f in L])
        Mend_mult = R.median([erows[f]["m_top1"] / abs(gmedE) for f in L])

        def f_raw(t):
            return raw_add_resid(mrows, drows, L, t) / max(1e-9, mend_scale)

        s1 = sign(f_raw(1.0))
        # dense scan: first t with opposite sign to sign at t=1
        t_star, prev_t, prev_v = None, None, None
        for t in A["tgrid_dense"]:
            v = f_raw(t)
            if sign(v) != 0 and sign(v) != s1:
                lo, hi = (prev_t, t) if prev_t is not None else (A["tgrid_dense"][0], t)
                for _ in range(60):  # bisection refine on the piecewise-linear median
                    mid = 0.5 * (lo + hi)
                    if sign(f_raw(mid)) == s1 or f_raw(mid) == 0:
                        lo = mid
                    else:
                        hi = mid
                t_star = 0.5 * (lo + hi)
                break
            prev_t, prev_v = t, v
        # 8-point grid replication (three forms, sham_scale_control.py code path)
        grid8 = []
        for t in TGRID8:
            raw = R.signed_med([mrows[f]["m_top1"] - t * drows[f]["m_top1"] for f in L],
                               mend_scale)
            calA = R.signed_med(
                [(mrows[f]["m_top1"] - gmedM) - (t * drows[f]["m_top1"] - t * gmedD)
                 for f in L], Mend_cal)
            calM = R.signed_med(
                [mrows[f]["m_top1"] / abs(gmedM) - (t * drows[f]["m_top1"]) / abs(t * gmedD)
                 for f in L], Mend_mult)
            grid8.append({"t": t, "raw": round(raw, 4), "calA": round(calA, 4),
                          "calM": round(calM, 6)})
        cell = {"campaign": SEED_CAMP[seed], "n_live": len(L), "sign_at_t1": s1,
                "t_star": (round(t_star, 4) if t_star is not None else None),
                "flips_in_dense_scan": t_star is not None,
                "grid8": grid8,
                "grid8_raw_flip": len({g["raw"] > 0 for g in grid8}) > 1,
                "grid8_calA_flip": len({g["calA"] > 0 for g in grid8}) > 1,
                "grid8_calM_spread": max(g["calM"] for g in grid8) - min(g["calM"] for g in grid8)}
        A["cells"]["{}-{}".format(seed, br)] = cell
        # verify old cells reproduce the frozen artifact bitwise at 4dp
        if SEED_CAMP[seed] == "c1":
            frozen = sham["cells"]["{}-{}".format(seed, br)]["sham_temperature_sweep"]
            for g, fz in zip(grid8, frozen):
                assert abs(g["raw"] - fz["D_dis_raw"]) < 5e-5, (seed, br, g, fz)
                assert abs(g["calA"] - fz["D_dis_cal_additive"]) < 5e-5
                assert abs(g["calM"] - fz["D_dis_cal_mult"]) < 5e-6
        print("  [A] {}-{}: sign@1={:+d} t*={} grid8_raw_flip={} calA_flip={} calM_spread={:.2e}"
              .format(seed, br, s1, cell["t_star"], cell["grid8_raw_flip"],
                      cell["grid8_calA_flip"], cell["grid8_calM_spread"]), flush=True)

stars = [c["t_star"] for c in A["cells"].values() if c["t_star"] is not None]
A["summary"] = {
    "n_cells": len(A["cells"]),
    "n_flip_dense": len(stars),
    "tstar_median": round(R.median(stars), 4) if stars else None,
    "tstar_min": round(min(stars), 4) if stars else None,
    "tstar_max": round(max(stars), 4) if stars else None,
    "tstar_list": sorted(round(s, 4) for s in stars),
    "n_in_practical_band": sum(1 for s in stars if 0.7 <= s <= 1.3),
    "n_grid8_raw_flip": sum(1 for c in A["cells"].values() if c["grid8_raw_flip"]),
    "n_grid8_calA_flip": sum(1 for c in A["cells"].values() if c["grid8_calA_flip"]),
    "calM_max_spread_all": max(c["grid8_calM_spread"] for c in A["cells"].values()),
}
print("  [A] summary:", json.dumps(A["summary"]), flush=True)
OUT["A_dense_tstar"] = A

# ============================== B. reference-family scan ==============================
print("== B. reference-family scan ==", flush=True)
B = {"cells": {}}
for seed in ALL_SEEDS:
    for br in "AB":
        L, emeta, erows = cohort_L(seed, 0, br)
        mmeta, mrows = load_train(seed, "t0-a0.50-merge")
        gmedM = mmeta["ghost_median_top1_train"]
        refs = {}
        for a in ("0.25", "0.50", "0.75"):
            rmeta, rrows = load_train(seed, "t0-s{}-dis{}".format(a, br))
            gmedR = rmeta["ghost_median_top1_train"]
            add = R.median([mrows[f]["m_top1"] - rrows[f]["m_top1"] for f in L])
            mul = R.median([mrows[f]["m_top1"] / abs(gmedM)
                            - rrows[f]["m_top1"] / abs(gmedR) for f in L])
            refs[a] = {"ghost_floor": gmedR, "add": round(add, 5), "mul": round(mul, 6),
                       "add_sign": sign(add), "mul_sign": sign(mul)}
        floors = [refs[a]["ghost_floor"] for a in ("0.25", "0.50", "0.75")]
        mfloors = []
        for a in ("0.25", "0.50", "0.75"):
            mm, _ = load_train(seed, "t0-a{}-merge".format(a))
            mfloors.append(mm["ghost_median_top1_train"])
        B["cells"]["{}-{}".format(seed, br)] = {
            "campaign": SEED_CAMP[seed], "n_live": len(L),
            "merge_ghost_floor": gmedM, "refs": refs,
            "ref_floor_range": round(max(floors) - min(floors), 5),
            "merge_floor_range_over_alpha": round(max(mfloors) - min(mfloors), 5),
            "scale_ratio_m_over_d05": round(abs(gmedM) / abs(refs["0.50"]["ghost_floor"]), 4),
        }
        print("  [B] {}-{}: add signs {} mul signs {} ref_floor_range={:.4f} ratio={:.3f}"
              .format(seed, br,
                      [refs[a]["add_sign"] for a in ("0.25", "0.50", "0.75")],
                      [refs[a]["mul_sign"] for a in ("0.25", "0.50", "0.75")],
                      B["cells"]["{}-{}".format(seed, br)]["ref_floor_range"],
                      B["cells"]["{}-{}".format(seed, br)]["scale_ratio_m_over_d05"]),
              flush=True)
B["summary"] = {
    "n_cells": len(B["cells"]),
    "add_sign_varies_across_refs": sum(
        1 for c in B["cells"].values()
        if len({c["refs"][a]["add_sign"] for a in ("0.25", "0.50", "0.75")}) > 1),
    "mul_sign_varies_across_refs": sum(
        1 for c in B["cells"].values()
        if len({c["refs"][a]["mul_sign"] for a in ("0.25", "0.50", "0.75")}) > 1),
    "mul_signs_per_ref": {a: [B["cells"][k]["refs"][a]["mul_sign"] for k in sorted(B["cells"])]
                          for a in ("0.25", "0.50", "0.75")},
    "add_signs_per_ref": {a: [B["cells"][k]["refs"][a]["add_sign"] for k in sorted(B["cells"])]
                          for a in ("0.25", "0.50", "0.75")},
    "ref_floor_ranges": {k: c["ref_floor_range"] for k, c in sorted(B["cells"].items())},
    "merge_floor_ranges": {k: c["merge_floor_range_over_alpha"]
                           for k, c in sorted(B["cells"].items())},
    "scale_ratios": {k: c["scale_ratio_m_over_d05"] for k, c in sorted(B["cells"].items())},
}
print("  [B] summary:", json.dumps({k: v for k, v in B["summary"].items()
                                    if not k.endswith("per_ref")}), flush=True)
OUT["B_refscan"] = B

# ============================== C. quantile normalizer swap ==============================
print("== C. quantile normalizer swap (96 cells x 4 q) ==", flush=True)
QN = [("median", 0.5), ("q98", 0.98), ("q99", 0.99), ("q995", 0.995)]


def ghost_q(rows, q):
    """full-precision ghost-pool m_top1 quantile (train battery), same quantile
    convention as the frozen pipeline (R.quantile)."""
    return R.quantile([r["m_top1"] for r in rows.values()
                       if r["set"].startswith("ghost")], q)


C = {"cells": {}}
for seed in ALL_SEEDS:
    for ti in range(8):
        mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
        for br in "AB":
            L, emeta, erows = cohort_L(seed, ti, br)
            dmeta, drows = load_train(seed, "t{}-s0.50-dis{}".format(ti, br))
            per_q = {}
            for qk, qv in QN:
                gMr, gDr, gEr = ghost_q(mrows, qv), ghost_q(drows, qv), ghost_q(erows, qv)
                # sanity vs frozen meta (5dp rounding there)
                logged = {"median": mmeta["ghost_median_top1_train"], "q98": mmeta["theta_alive_top1_q980"],
                          "q99": mmeta["theta_alive_top1_q990"], "q995": mmeta["theta_alive_top1_q995"]}[qk]
                assert abs(gMr - logged) < 2e-4, (seed, ti, br, qk, gMr, logged)
                gM, gD, gE = abs(gMr), abs(gDr), abs(gEr)
                if min(gM, gD, gE) < 1e-6:
                    per_q[qk] = {"val": None, "sign": 0, "singular": True,
                                 "gM": round(gMr, 5), "gD": round(gDr, 5)}
                    continue
                Mend = R.median([erows[f]["m_top1"] / gE for f in L])
                dv = R.signed_med([mrows[f]["m_top1"] / gM - drows[f]["m_top1"] / gD
                                   for f in L], Mend)
                per_q[qk] = {"val": round(dv, 6), "sign": sign(dv), "singular": False,
                             "gM": round(gMr, 5), "gD": round(gDr, 5)}
            key = "{}-{}-t{}".format(seed, br, ti)
            C["cells"][key] = {"campaign": SEED_CAMP[seed], "per_q": per_q,
                               "sign_constant": len({per_q[qk]["sign"] for qk, _ in QN}) == 1}
C["summary"] = {
    "n_cells": len(C["cells"]),
    "n_sign_constant": sum(1 for c in C["cells"].values() if c["sign_constant"]),
    "n_singular_any_q": sum(1 for c in C["cells"].values()
                            if any(c["per_q"][qk].get("singular") for qk, _ in QN)),
    "signs_by_q": {qk: {"neg": sum(1 for c in C["cells"].values() if c["per_q"][qk]["sign"] < 0),
                        "pos": sum(1 for c in C["cells"].values() if c["per_q"][qk]["sign"] > 0),
                        "singular": sum(1 for c in C["cells"].values()
                                        if c["per_q"][qk].get("singular"))}
                   for qk, _ in QN},
    "n_sign_constant_excl_q995": sum(
        1 for c in C["cells"].values()
        if len({c["per_q"][qk]["sign"] for qk in ("median", "q98", "q99")}) == 1),
    "tau0_signs": {qk: [C["cells"]["{}-{}-t0".format(s, br)]["per_q"][qk]["sign"]
                        for s in ALL_SEEDS for br in "AB"] for qk, _ in QN},
    "cells_not_constant": [k for k, c in C["cells"].items() if not c["sign_constant"]],
}
print("  [C] summary:", json.dumps(C["summary"]), flush=True)

# ---- C2. additive ghost-calibrated form under the same quantile swap ----
# (frozen construction: adjudicate_ddis.py addendum; D = (mM - thetaM_q) - (mD - thetaD_q),
#  sign only; extends the paper's 48-cell campaign-1 counts to 96 cells)
C2 = {"pos_cells": {}, "tau0_negpos_seeds": {}}
for qk, qv in QN:
    npos = 0
    for seed in ALL_SEEDS:
        for ti in range(8):
            mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
            for br in "AB":
                L, emeta, erows = cohort_L(seed, ti, br)
                dmeta, drows = load_train(seed, "t{}-s0.50-dis{}".format(ti, br))
                if qk == "median":
                    tM = R.median([r["m_top1"] for r in mrows.values()
                                   if r["set"].startswith("ghost")])
                    tD = R.median([r["m_top1"] for r in drows.values()
                                   if r["set"].startswith("ghost")])
                else:
                    tM = ghost_q(mrows, qv)
                    tD = ghost_q(drows, qv)
                dv = R.median([(mrows[f]["m_top1"] - tM) - (drows[f]["m_top1"] - tD)
                               for f in L])
                npos += int(dv > 0)
                if ti == 0:
                    C2.setdefault("tau0_branch_signs", {}).setdefault(qk, {})[
                        "{}-{}".format(seed, br)] = sign(dv)
    C2["pos_cells"][qk] = npos
for qk, qv in QN:
    pats = C2["tau0_branch_signs"][qk]
    nneg = sum(1 for s in ALL_SEEDS
               if pats["{}-A".format(s)] < 0 and pats["{}-B".format(s)] < 0)
    npos = sum(1 for s in ALL_SEEDS
               if pats["{}-A".format(s)] > 0 and pats["{}-B".format(s)] > 0)
    C2["tau0_negpos_seeds"][qk] = [nneg, npos]
print("  [C2] additive calibrated positive cells of 96:", C2["pos_cells"],
      " tau0 neg:pos seeds:", C2["tau0_negpos_seeds"], flush=True)
C["additive_counterpart"] = C2
OUT["C_quantile_swap"] = C

# ============================== D. posctrl verification + strata ==============================
print("== D. posctrl stratified verification ==", flush=True)
PC = os.path.join(EXP, "results/posctrl")
rep = json.load(open(os.path.join(PC, "posctrl_report.json")))
targets = {t["id"] for t in json.load(open(os.path.join(PC, "damage", "target_facts.json")))["targets"]}
D = {"cells": {}}
for mech in ("ascend", "misinfo"):
    for seed in (42, 1042, 2042):
        pmeta, prows = load_train(seed, "t0-endA")          # pre-damage (frozen)
        path = os.path.join(PC, "eval", "margins.posctrl-s{}-{}dmg.jsonl.gz".format(seed, mech))
        qmeta, qrows = None, {}
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                if "__meta__" in r:
                    qmeta = r["__meta__"]
                elif r["battery"] == "train":
                    qrows[r["fact_id"]] = r
        th = qmeta["theta_alive_top1_q990"]
        own = [fid for fid, r in prows.items() if r["set"] == "excl_A"]
        pre_alive = [f for f in own if prows[f]["rank_bar"] <= RANK1]
        forgotten = [f for f in pre_alive if qrows[f]["rank_bar"] > RANK1]
        retained = [f for f in pre_alive if qrows[f]["rank_bar"] <= RANK1]
        alarm = lambda f: qrows[f]["m_top1"] < th  # noqa: E731
        tpr = sum(1 for f in forgotten if alarm(f)) / max(1, len(forgotten))
        fpr = sum(1 for f in retained if alarm(f)) / max(1, len(retained))
        fg_t = [f for f in forgotten if f in targets]
        fg_c = [f for f in forgotten if f not in targets]
        tpr_t = sum(1 for f in fg_t if alarm(f)) / max(1, len(fg_t))
        tpr_c = sum(1 for f in fg_c if alarm(f)) / max(1, len(fg_c))
        key = "{}:s{}".format(mech, seed)
        repc = rep["per_cell"][key]
        D["cells"][key] = {
            "n_pre_alive": len(pre_alive), "n_forgotten": len(forgotten),
            "n_retained": len(retained), "tpr": round(tpr, 4), "fpr": round(fpr, 4),
            "tpr_report": repc["tpr"], "fpr_report": repc["fpr"],
            "tpr_match": abs(tpr - repc["tpr"]) < 2e-3,
            "fpr_match": abs(fpr - repc["fpr"]) < 2e-3,
            "n_forgotten_target": len(fg_t), "n_forgotten_collateral": len(fg_c),
            "tpr_target": round(tpr_t, 4), "tpr_collateral": round(tpr_c, 4),
            "theta_q99": th,
        }
        print("  [D] {}: TPR={:.4f}(rep {:.4f}) FPR={:.4f} | target {}/{}={:.4f} collat {}/{}={:.4f}"
              .format(key, tpr, repc["tpr"], fpr,
                      sum(1 for f in fg_t if alarm(f)), len(fg_t), tpr_t,
                      sum(1 for f in fg_c if alarm(f)), len(fg_c), tpr_c), flush=True)
D["summary"] = {
    "all_tpr_match": all(c["tpr_match"] for c in D["cells"].values()),
    "all_fpr_match": all(c["fpr_match"] for c in D["cells"].values()),
    "tpr_target_range": [min(c["tpr_target"] for c in D["cells"].values()),
                         max(c["tpr_target"] for c in D["cells"].values())],
    "tpr_collateral_range": [min(c["tpr_collateral"] for c in D["cells"].values()),
                             max(c["tpr_collateral"] for c in D["cells"].values())],
    "fpr_max": max(c["fpr"] for c in D["cells"].values()),
    "pooled_tpr_report": rep["pooled"]["tpr"], "pooled_fpr_report": rep["pooled"]["fpr"],
    "pooled_n_forgotten": rep["pooled"]["n_forgotten"],
    "pooled_n_retained": rep["pooled"]["n_retained"],
    "gate": rep["thresholds"], "verdict": rep["verdict"],
    "a2_placebo_invariant_all": all(repc["placebo_alarm_set_invariant"]
                                    for repc in rep["per_cell"].values()),
    "a3_undamaged_alarm": rep["legs"]["A3_undamaged_alarm"],
    "per_cell_auc": {k: v["auc"] for k, v in rep["per_cell"].items()},
    "per_cell_tpr": {k: v["tpr"] for k, v in rep["per_cell"].items()},
}
print("  [D] summary:", json.dumps(D["summary"], ensure_ascii=False), flush=True)
OUT["D_posctrl"] = D

# ============================== E. task-arith arm ==============================
print("== E. task-arith arm ==", flush=True)
ta = json.load(open(os.path.join(EXP, "results/taskarith", "taskarith_report.json")))
lams = ["0.50", "0.75", "1.00", "1.25", "1.50"]
E = {"verdict_p2": ta["verdict_p2"], "p3_note": ta["p3_note"], "per_cell": {}}
swings = []
for k, c in sorted(ta["cells"].items()):
    g = [c["ghost_floor_by_lambda"][l] for l in lams]
    E["per_cell"][k] = {"g_by_lambda": g, "uplift": c["uplift_1.5_minus_0.5"],
                        "monotone": c["monotone_nondecreasing"],
                        "swing": round(max(g) - min(g), 5),
                        "dis_ghost_floor_frozen": c["dis_ghost_floor_frozen"],
                        "scale_ratio": c["scale_ratio_merge_over_dis05"]}
    swings.append(max(g) - min(g))
E["summary"] = {
    "n_cells": len(E["per_cell"]), "n_monotone": ta["verdict_p2"]["n_monotone"],
    "median_uplift": ta["verdict_p2"]["median_uplift"],
    "median_swing": round(R.median(swings), 4),
    "swing_range": [round(min(swings), 4), round(max(swings), 4)],
    "uplift_range": [round(min(c["uplift"] for c in E["per_cell"].values()), 4),
                     round(max(c["uplift"] for c in E["per_cell"].values()), 4)],
    "tau0_table": {k: E["per_cell"][k] for k in ("s42-t0", "s1042-t0", "s2042-t0")},
}
print("  [E] summary:", json.dumps({k: v for k, v in E["summary"].items() if k != "tau0_table"}),
      flush=True)
OUT["E_taskarith"] = E

# ============================== F. real bridge + realmerge ==============================
print("== F. real bridge ==", flush=True)
rb = json.load(open(os.path.join(EXP, "results/realbridge", "realbridge_report.json")))
rm = json.load(open(os.path.join(EXP, "results/realmerge", "realmerge_report.json")))
rm27 = json.load(open(os.path.join(EXP, "results/realmerge27", "realmerge_report.json")))
F = {"tier1": rb["tiers"]["tier1"], "tier2_skipped": rb["tiers"]["tier2"],
     "headline_limits": rb["headline_limits"],
     "realmerge_E1": rm["E1"], "realmerge_limits": rm["headline_limits"],
     "families": {}}
for fam in rm["families"]:
    name = fam.get("family")
    if fam.get("skipped"):
        F["families"][name] = {"skipped": True, "reason": fam.get("reason")}
        continue
    cells = {}
    for cell in (fam.get("headline_cells") or []):
        for pb in cell.get("per_battery", []):
            cells["{}@{}".format(cell["meas"], pb["battery"])] = {
                "n_live": pb.get("n_live"), "t_star": pb.get("t_star"),
                "low_power": pb.get("low_power", False),
                "note": pb.get("t_star_note", "")}
    F["families"][name] = {"skipped": False, "K": fam.get("K"), "cells": cells}
tstars = [c["t_star"] for fam in F["families"].values()
          if not fam.get("skipped") for c in fam["cells"].values() if c["t_star"] is not None]
F["realmerge_tstar_range"] = [min(tstars), max(tstars)]
F["realmerge27"] = {"E1": rm27.get("E1")}
print("  [F] tier1 t*={} ; realmerge t* range {} ; E1 {}".format(
    rb["tiers"]["tier1"]["t_star"], F["realmerge_tstar_range"],
    json.dumps(rm["E1"]["flip_frac_powered"])), flush=True)
OUT["F_realbridge"] = F

# ============================== G. Finding 1 + 2 on six seeds ==============================
print("== G. Finding 1/2 six-seed recompute ==", flush=True)
QS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def qqfit(xs, ys):
    px = [R.quantile(xs, q) for q in QS]
    py = [R.quantile(ys, q) for q in QS]
    n = len(QS)
    mx, my = sum(px) / n, sum(py) / n
    num = sum((a - mx) * (b - my) for a, b in zip(px, py))
    den = sum((a - mx) ** 2 for a in px)
    return (my - (num / den) * mx), (num / den if den else float("nan"))


G = {"per_seed_branch_tau": {}, "consensus": {}, "multform": {}, "n_live": {},
     "qq_slope_tau7": {}, "ordinal_divergence": {}, "survival": {}}
READOUTS = ["raw_mtop1", "frac_rank1", "median_rank", "logp", "m_nll", "D_mul"]
cons = {k: [0] * 8 for k in READOUTS + ["all_five"]}
per_seed_allfive = {s: [False] * 8 for s in ALL_SEEDS}
per_seed_survival_caliber = {s: [False] * 8 for s in ALL_SEEDS}
for seed in ALL_SEEDS:
    for ti in range(8):
        mmeta, mrows = load_train(seed, "t{}-a0.50-merge".format(ti))
        gM = mmeta["ghost_median_top1_train"]
        per_branch = {}
        for br in "AB":
            L, emeta, erows = cohort_L(seed, ti, br)
            dmeta, drows = load_train(seed, "t{}-s0.50-dis{}".format(ti, br))
            gD = dmeta["ghost_median_top1_train"]
            gE = emeta["ghost_median_top1_train"]
            Mend_mult = R.median([erows[f]["m_top1"] / abs(gE) for f in L])
            dmul = R.signed_med([mrows[f]["m_top1"] / abs(gM) - drows[f]["m_top1"] / abs(gD)
                                 for f in L], Mend_mult)
            n = len(L)
            fr_m = sum(1 for f in L if mrows[f]["rank_bar"] <= RANK1) / n
            fr_d = sum(1 for f in L if drows[f]["rank_bar"] <= RANK1) / n
            mr_m = R.median([mrows[f]["rank_bar"] for f in L])
            mr_d = R.median([drows[f]["rank_bar"] for f in L])
            lp = R.median([mrows[f]["logp"] - drows[f]["logp"] for f in L])
            mn = R.median([mrows[f]["m_nll"] - drows[f]["m_nll"] for f in L])
            raw = R.median([mrows[f]["m_top1"] - drows[f]["m_top1"] for f in L])
            dirs = {"raw_mtop1": raw < 0, "frac_rank1": fr_m < fr_d,
                    "median_rank": mr_m > mr_d, "logp": lp < 0, "m_nll": mn < 0,
                    "D_mul": dmul < 0}
            # survival under merge vs dis (own-model Q99 threshold, alive flag)
            s_m = sum(1 for f in L if mrows[f]["alive"]) / n
            s_d = sum(1 for f in L if drows[f]["alive"]) / n
            # normalized trajectory values (fig6 definition)
            y_m = R.median([mrows[f]["m_top1"] / abs(gM) for f in L])
            y_e = R.median([erows[f]["m_top1"] / abs(gE) for f in L])
            per_branch[br] = {"n_live": n, "dmul": round(dmul, 4), "dirs": dirs,
                              "S_merge": round(s_m, 4), "S_dis": round(s_d, 4),
                              "y_merge": round(y_m, 6), "y_end": round(y_e, 6),
                              "fr_m": round(fr_m, 4), "fr_d": round(fr_d, 4)}
            key = "{}-{}-t{}".format(seed, br, ti)
            G["per_seed_branch_tau"][key] = per_branch[br]
            G["multform"][key] = round(dmul, 4)
            G["n_live"][key] = n
            if ti == 7:
                gids = ghost_ids(seed, ti, br)
                _, s7 = qqfit([drows[g]["m_top1"] for g in gids],
                              [mrows[g]["m_top1"] for g in gids])
                G["qq_slope_tau7"]["{}-{}".format(seed, br)] = round(s7, 4)
        for k in READOUTS:
            cons[k][ti] += int(all(per_branch[b]["dirs"][k] for b in "AB"))
        five = ["frac_rank1", "median_rank", "logp", "m_nll", "D_mul"]
        af = all(per_branch[b]["dirs"][r] for b in "AB" for r in five)
        cons["all_five"][ti] += int(af)
        per_seed_allfive[seed][ti] = af
        per_seed_survival_caliber[seed][ti] = all(
            per_branch[b]["S_merge"] < per_branch[b]["S_dis"] for b in "AB")
        G["ordinal_divergence"]["{}-t{}".format(seed, ti)] = {
            "merge": round(R.median(
                [mrows[f]["m_top1"] / abs(gM)
                 for f in [x for b in "AB" for x in cohort_L(seed, ti, b)[0]]]), 6),
            "endA": per_branch["A"]["y_end"], "endB": per_branch["B"]["y_end"],
            "merge_below_both": None}
        G["ordinal_divergence"]["{}-t{}".format(seed, ti)]["merge_below_both"] = (
            G["ordinal_divergence"]["{}-t{}".format(seed, ti)]["merge"]
            < min(per_branch["A"]["y_end"], per_branch["B"]["y_end"]))
    print("  [G] seed {} done".format(seed), flush=True)

G["consensus"] = cons
# verify campaign-1 consensus reproduces the appendix table
EXP_CONS = {"raw_mtop1": [3, 3, 3, 3, 3, 3, 3, 3], "frac_rank1": [1, 1, 1, 1, 2, 3, 3, 3],
            "median_rank": [1, 1, 1, 2, 2, 2, 2, 3], "logp": [1, 1, 1, 1, 1, 3, 3, 3],
            "m_nll": [1, 1, 1, 1, 1, 3, 3, 3], "D_mul": [1, 1, 1, 2, 2, 3, 3, 3],
            "all_five": [1, 1, 1, 1, 1, 2, 2, 3]}
c1_ok = {}
for k, exp_row in EXP_CONS.items():
    got = []
    # recompute campaign-1-only counts from per-cell dirs
    for ti in range(8):
        cnt = 0
        for s in (42, 1042, 2042):
            if k == "all_five":
                five = ["frac_rank1", "median_rank", "logp", "m_nll", "D_mul"]
                ok = all(G["per_seed_branch_tau"]["{}-{}-t{}".format(s, b, ti)]["dirs"][r]
                         for b in "AB" for r in five)
            else:
                ok = all(G["per_seed_branch_tau"]["{}-{}-t{}".format(s, b, ti)]["dirs"][k]
                         for b in "AB")
            cnt += int(ok)
        got.append(cnt)
    c1_ok[k] = (got == exp_row)
    print("  [G] c1 consensus check {:<12s} got {} expected {} {}".format(
        k, got, exp_row, "OK" if got == exp_row else "MISMATCH"), flush=True)
G["consensus_c1_matches_appendix"] = c1_ok


def sustained_turning(flags):
    """first index i with flags[i] and all(flags[i:]); None if never sustained."""
    for i in range(len(flags)):
        if flags[i] and all(flags[i:]):
            return i
    return None


G["turning_allfive"] = {str(s): (lambda i: TAU_LABELS[i] if i is not None else None)(
    sustained_turning(per_seed_allfive[s])) for s in ALL_SEEDS}
G["turning_survival"] = {str(s): (lambda i: TAU_LABELS[i] if i is not None else None)(
    sustained_turning(per_seed_survival_caliber[s])) for s in ALL_SEEDS}
# deep-regime (ti>=5) counts per readout
G["deep_counts"] = {k: {"50M": cons[k][5], "100M": cons[k][6], "200M": cons[k][7]}
                    for k in READOUTS + ["all_five"]}
G["ordinal_div_all"] = all(v["merge_below_both"] for v in G["ordinal_divergence"].values())
G["ordinal_div_deep"] = all(G["ordinal_divergence"]["{}-t{}".format(s, ti)]["merge_below_both"]
                            for s in ALL_SEEDS for ti in (5, 6, 7))
print("  [G] consensus (6 seeds):", json.dumps(cons), flush=True)
print("  [G] turning allfive:", G["turning_allfive"], " survival:", G["turning_survival"],
      flush=True)
print("  [G] ordinal divergence all/deep:", G["ordinal_div_all"], G["ordinal_div_deep"],
      flush=True)
print("  [G] qq slopes tau7:", G["qq_slope_tau7"], flush=True)

# ---- Finding 2: survival surface on six seeds (own-model Q99 alive flag) ----
# w_own coordinate (fig3 convention): w=0.25 means the OWN branch weight is 0.25,
# i.e. A-facts under merge(alpha=0.25) and B-facts under merge(alpha=0.75).
for seed in ALL_SEEDS:
    for ti in range(8):
        _, mrows25 = load_train(seed, "t{}-a0.25-merge".format(ti))
        _, mrows50 = load_train(seed, "t{}-a0.50-merge".format(ti))
        _, mrows75 = load_train(seed, "t{}-a0.75-merge".format(ti))
        per_w = {}
        for br in "AB":
            L, _, _ = cohort_L(seed, ti, br)
            n = len(L)
            rows25 = mrows25 if br == "A" else mrows75
            rows75 = mrows75 if br == "A" else mrows25
            per_w["w25"] = per_w.get("w25", {})
            per_w["w25"][br] = {"S": round(sum(1 for f in L if rows25[f]["alive"]) / n, 4),
                                "n": n, "w_own": 0.25}
            per_w["w50"] = per_w.get("w50", {})
            per_w["w50"][br] = {"S": round(sum(1 for f in L if mrows50[f]["alive"]) / n, 4),
                                "n": n, "w_own": 0.5}
            per_w["w75"] = per_w.get("w75", {})
            per_w["w75"][br] = {"S": round(sum(1 for f in L if rows75[f]["alive"]) / n, 4),
                                "n": n, "w_own": 0.75}
        for wtag in ("w25", "w50", "w75"):
            G["survival"]["{}-t{}-{}".format(seed, ti, wtag)] = per_w[wtag]
print("  [G] survival done", flush=True)

# Finding 2 headline numbers
def pooled_S(seed, ti, wtag):
    c = G["survival"]["{}-t{}-{}".format(seed, ti, wtag)]
    return (c["A"]["S"] * c["A"]["n"] + c["B"]["S"] * c["B"]["n"]) / (c["A"]["n"] + c["B"]["n"])


G["finding2"] = {
    "tau0_a05_pooled": {str(s): round(pooled_S(s, 0, "w50"), 4) for s in ALL_SEEDS},
    "tau0_a05_pooled_c1": {str(s): round(pooled_S(s, 0, "w50"), 4) for s in (42, 1042, 2042)},
    "n_pooled_a05_cells_below_0.5": sum(
        1 for s in ALL_SEEDS for ti in range(8) if pooled_S(s, ti, "w50") < 0.5),
    "n_pooled_a05_cells": 6 * 8,
    "n_perbranch_w50_below_0.5": sum(
        1 for k, v in G["survival"].items() if k.endswith("w50")
        for b in "AB" if v[b]["S"] < 0.5),
    "n_perbranch_w50": 6 * 8 * 2,
    "w25_range_all": [round(min(v[b]["S"] for k, v in G["survival"].items()
                                if k.endswith("w25") for b in "AB"), 4),
                      round(max(v[b]["S"] for k, v in G["survival"].items()
                                if k.endswith("w25") for b in "AB"), 4)],
    "w75_range_all": [round(min(v[b]["S"] for k, v in G["survival"].items()
                                if k.endswith("w75") for b in "AB"), 4),
                      round(max(v[b]["S"] for k, v in G["survival"].items()
                                if k.endswith("w75") for b in "AB"), 4)],
    "w25_range_c1": [round(min(v[b]["S"] for k, v in G["survival"].items()
                               if k.endswith("w25") and int(k.split("-")[0]) in (42, 1042, 2042)
                               for b in "AB"), 4),
                     round(max(v[b]["S"] for k, v in G["survival"].items()
                               if k.endswith("w25") and int(k.split("-")[0]) in (42, 1042, 2042)
                               for b in "AB"), 4)],
    "w75_range_c1": [round(min(v[b]["S"] for k, v in G["survival"].items()
                               if k.endswith("w75") and int(k.split("-")[0]) in (42, 1042, 2042)
                               for b in "AB"), 4),
                     round(max(v[b]["S"] for k, v in G["survival"].items()
                               if k.endswith("w75") and int(k.split("-")[0]) in (42, 1042, 2042)
                               for b in "AB"), 4)],
    # range of pooled w50 survivals across all (seed, tau) cells (the "0.1841--0.4174" slot)
    "pooled_w50_range_all": [round(min(pooled_S(s, ti, "w50") for s in ALL_SEEDS
                                     for ti in range(8)), 4),
                             round(max(pooled_S(s, ti, "w50") for s in ALL_SEEDS
                                       for ti in range(8)), 4)],
    "pooled_w50_range_c1": [round(min(pooled_S(s, ti, "w50") for s in (42, 1042, 2042)
                                    for ti in range(8)), 4),
                            round(max(pooled_S(s, ti, "w50") for s in (42, 1042, 2042)
                                       for ti in range(8)), 4)],
}
print("  [G] finding2:", json.dumps(G["finding2"]), flush=True)

# ---- extras for the text ----
# (i) per-seed readout-split status at tau=0 (section 4.1 story): do the four
# non-mtop readouts all agree with the raw-mtop direction?
G["tau0_readout_split"] = {}
for s in ALL_SEEDS:
    agree = all(G["per_seed_branch_tau"]["{}-{}-t0".format(s, b)]["dirs"][r]
                for b in "AB" for r in ("frac_rank1", "median_rank", "logp", "m_nll"))
    G["tau0_readout_split"][str(s)] = {
        "all_four_agree_with_mtop": agree,
        "scale_ratio": OUT["B_refscan"]["cells"]["{}-A".format(s)]["scale_ratio_m_over_d05"]}
# (ii) deep-regime (ti 5..7) mult signs under the quantile normalizer family
C = OUT["C_quantile_swap"]
G["deep_quantile_signs"] = {
    qk: {"neg": sum(1 for k, c in C["cells"].items()
                    if int(k.split("-t")[1]) >= 5 and c["per_q"][qk]["sign"] < 0),
         "pos": sum(1 for k, c in C["cells"].items()
                    if int(k.split("-t")[1]) >= 5 and c["per_q"][qk]["sign"] > 0),
         "singular": sum(1 for k, c in C["cells"].items()
                         if int(k.split("-t")[1]) >= 5 and c["per_q"][qk].get("singular")),
         "of": 36}
    for qk, _ in [("median", 0), ("q98", 0), ("q99", 0), ("q995", 0)]}
# (iii) n_live trajectories (A branch; cohort appendix)
G["n_live_A"] = {str(s): [G["n_live"]["{}-A-t{}".format(s, ti)] for ti in range(8)]
                 for s in ALL_SEEDS}
G["n_live_B"] = {str(s): [G["n_live"]["{}-B-t{}".format(s, ti)] for ti in range(8)]
                 for s in ALL_SEEDS}
# (iv) mult signs at tau=0 (12 cells) for the "does not rubber-stamp" sentence
G["tau0_dmul_neg_cells"] = sum(1 for s in ALL_SEEDS for b in "AB"
                               if G["multform"]["{}-{}-t0".format(s, b)] < 0)
G["tau0_dmul_neg_seeds"] = sum(
    1 for s in ALL_SEEDS
    if all(G["multform"]["{}-{}-t0".format(s, b)] < 0 for b in "AB"))
print("  [G] extras:", json.dumps({"tau0_readout_split": G["tau0_readout_split"],
                                   "deep_quantile_signs": G["deep_quantile_signs"],
                                   "tau0_dmul_neg": [G["tau0_dmul_neg_cells"],
                                                     G["tau0_dmul_neg_seeds"]]}),
      flush=True)
OUT["G_findings"] = G

# ============================== write (atomic) ==============================
outp = os.path.join(HERE, "paper_backfill.json")
tmp = outp + ".tmp"
with open(tmp, "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
os.replace(tmp, outp)
print("[write]", outp, flush=True)
print("DONE", flush=True)
