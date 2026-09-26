#!/usr/bin/env python3
"""make_fig6_data.py -- Fig 6 (Finding 1 tau-trajectory) data layer.

Read-only re-aggregation of the frozen per-fact margin fields in
runs/merge_audit/results_v2_reid/eval/. For each (seed, tau) cell:

  y_line = median over the endpoint-surviving cohort F_live(tau) of the
  scale-invariant (multiplicative-form) margin  m_top1(f; M) / |gamma_M|,
  gamma_M = median ghost-pool m_top1 of model M  (Eq. multform normalizer),

for three lines: merge(alpha=0.5) [branches pooled by cohort size],
endpoint A [on F_live^A], endpoint B [on F_live^B]. CI = fact-level bootstrap
90% (B=2000, numpy rng seed 20260814; normalizer fixed at its point estimate),
matching the paper's "CIs cover fact-sampling noise only" semantics.

Verification block (same loaded data):
  * recomputes D_mul per (seed, branch, tau) with the same code path as
    adjudication/sham_scale_control.py (frozen helpers from run_merge_audit.py)
    and asserts equality with recompute/mult_form_deep_tau.json, the artifact
    behind appendix Table multform (3/3 seeds from tau >= 50M; 6/6 cells
    negative at tau = 200M);
  * recomputes the per-readout consensus counts behind appendix
    Table tab:consensus and asserts equality row by row;
  * checks the ordinal divergence plotted (merge below both endpoints at
    tau >= 50M, per seed);
  * checks survival_a05[0] against the DOSSIER values 0.4174/0.3930/0.3775.

Output: ../data/fig6_trajectory.csv
Columns: seed, tau_idx, tau_tokens, line, y, ci_lo, ci_hi, n_cohort
Lines: merge (pooled), merge_A, merge_B (per-branch detail), endA, endB.

Nothing outside figures/data/ is written. Frozen artifacts stay read-only.
"""
import json
import os
import sys

import numpy as np

EXP = "<REPO>"
RES = os.path.join(EXP, "results_v2_reid")
MULT_TAU_JSON = "/var/tmp/<account>/work-20260825/recompute/mult_form_deep_tau.json"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 1042, 2042]
sys.path.insert(0, EXP)
os.environ.setdefault("TANH_R2RANK08_SEEDS", ",".join(str(s) for s in SEEDS))
import run_merge_audit as R  # noqa: E402

BOOT_B, BOOT_SEED = 2000, 20260814


class ROCtx:
    def p(self, *parts):
        return os.path.join(RES, *parts)


CTX = ROCtx()

# tau axis: from metrics.json regime[].tau per the figure brief (checked
# against the frozen module constant).
MET = json.load(open(os.path.join(RES, "metrics.json")))
TAU_TOKENS = [r["tau"] for r in MET["per_seed"]["42"]["regime"]]
assert TAU_TOKENS == list(R.TAU_GRID_TOKENS), "tau grid mismatch vs frozen module"
for s in SEEDS:
    assert [r["tau"] for r in MET["per_seed"][str(s)]["regime"]] == TAU_TOKENS


def boot_ci(vals, b=BOOT_B, seed=BOOT_SEED):
    """Fact-level bootstrap 90% CI of the median (normalizer already folded
    into vals and held fixed); same B and quantile convention as the frozen
    pipeline's bootstrap_median_ci."""
    v = np.asarray(vals, dtype=np.float64)
    n = v.size
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    meds = np.median(v[idx], axis=1)
    return (float(R.quantile(list(meds), 0.05)),
            float(R.quantile(list(meds), 0.95)))


rows = []           # CSV rows
verify = {"dmul_max_abs_diff": 0.0, "dmul_cells": 0}
# Expected consensus rows: appendix Table tab:consensus (sections/appendix.tex),
# transcribed for a paper-vs-recomputation transcription check.
EXPECTED = {
    "raw_mtop1": [3, 3, 3, 3, 3, 3, 3, 3],
    "frac_rank1": [1, 1, 1, 1, 2, 3, 3, 3],
    "median_rank": [1, 1, 1, 2, 2, 2, 2, 3],
    "logp": [1, 1, 1, 1, 1, 3, 3, 3],
    "m_nll": [1, 1, 1, 1, 1, 3, 3, 3],
    "D_mul": [1, 1, 1, 2, 2, 3, 3, 3],
    "all_five": [1, 1, 1, 1, 1, 2, 2, 3],
}
consensus = {k: [0] * len(TAU_TOKENS) for k in EXPECTED}

ref = json.load(open(MULT_TAU_JSON))["cells"]

for seed in SEEDS:
    for ti, tau in enumerate(TAU_TOKENS):
        ids = R._model_ids(seed, ti)
        M = {}
        for k in ["endA", "endB", "merge0.50", "disA0.50", "disB0.50"]:
            meta, f = R.load_margins(CTX, ids[k])
            M[k] = (meta, f["train"])
        per_branch = {}
        for br in "AB":
            emeta, erow = M["end" + br]
            L = [fid for fid, r in erow.items()
                 if r["set"] == "excl_" + br and R._alive(r, emeta)]
            mrow = M["merge0.50"][1]
            drow = M["dis{}0.50".format(br)][1]
            gids = [fid for fid, r in erow.items() if r["set"].startswith("ghost")]
            gM = R.median([mrow[g]["m_top1"] for g in gids])
            gD = R.median([drow[g]["m_top1"] for g in gids])
            gE = R.median([erow[g]["m_top1"] for g in gids])
            # cross-check the recomputed ghost median against the logged meta
            for mk, gm in (("merge0.50", gM), ("dis{}0.50".format(br), gD),
                           ("end" + br, gE)):
                logged = M[mk][0]["ghost_median_top1_train"]
                assert abs(gm - logged) < 2e-4, (seed, ti, br, mk, gm, logged)
            # ---- verification: D_mul, same code path as the frozen sweep ----
            Mend_mult = R.median([erow[f]["m_top1"] / abs(gE) for f in L])
            dmul = R.signed_med(
                [mrow[f]["m_top1"] / abs(gM) - drow[f]["m_top1"] / abs(gD)
                 for f in L], Mend_mult)
            key = "{}-{}-t{}".format(seed, br, ti)
            dv = abs(dmul - ref[key]["D_dis_cal_mult"])
            verify["dmul_max_abs_diff"] = max(verify["dmul_max_abs_diff"], dv)
            verify["dmul_cells"] += 1
            # ---- verification: per-readout directions (appendix consensus) --
            n = len(L)
            fr_m = sum(1 for f in L
                       if mrow[f]["rank_bar"] <= R.RANK1_BAR_THRESH) / n
            fr_d = sum(1 for f in L
                       if drow[f]["rank_bar"] <= R.RANK1_BAR_THRESH) / n
            mr_m = R.median([mrow[f]["rank_bar"] for f in L])
            mr_d = R.median([drow[f]["rank_bar"] for f in L])
            lp = R.median([mrow[f]["logp"] - drow[f]["logp"] for f in L])
            mn = R.median([mrow[f]["m_nll"] - drow[f]["m_nll"] for f in L])
            raw = R.median([mrow[f]["m_top1"] - drow[f]["m_top1"] for f in L])
            per_branch[br] = {
                "L": L, "gM": gM, "gE": gE,
                "dir": {"raw_mtop1": raw < 0, "frac_rank1": fr_m < fr_d,
                        "median_rank": mr_m > mr_d, "logp": lp < 0,
                        "m_nll": mn < 0, "D_mul": dmul < 0},
            }
        for k in consensus:
            if k == "all_five":
                five = ["frac_rank1", "median_rank", "logp", "m_nll", "D_mul"]
                ok = all(per_branch[b]["dir"][r] for b in "AB" for r in five)
            else:
                ok = all(per_branch[b]["dir"][k] for b in "AB")
            consensus[k][ti] += int(ok)
        # ---- figure values: normalized median margins on the cohorts -------
        mrow = M["merge0.50"][1]
        LA, LB = per_branch["A"]["L"], per_branch["B"]["L"]
        gM = per_branch["A"]["gM"]  # merge ghost median (same file both br)
        vals = {
            "merge": [mrow[f]["m_top1"] / abs(gM) for f in LA + LB],
            "merge_A": [mrow[f]["m_top1"] / abs(gM) for f in LA],
            "merge_B": [mrow[f]["m_top1"] / abs(gM) for f in LB],
        }
        for br in "AB":
            erow = M["end" + br][1]
            gE = per_branch[br]["gE"]
            vals["end" + br] = [erow[f]["m_top1"] / abs(gE)
                                for f in per_branch[br]["L"]]
        for line, v in vals.items():
            y = R.median(v)
            lo, hi = boot_ci(v)
            rows.append({"seed": seed, "tau_idx": ti, "tau_tokens": tau,
                         "line": line, "y": y, "ci_lo": lo, "ci_hi": hi,
                         "n_cohort": len(v)})
        print("[s{} t{}] n={}/{}  merge={:+.4f}  endA={:+.4f}  endB={:+.4f}"
              .format(seed, ti, len(LA), len(LB),
                      R.median(vals["merge"]), R.median(vals["endA"]),
                      R.median(vals["endB"])), flush=True)

# ------------------------------------------------------------------ checks
ok = True
print("\n[verify] D_mul vs recompute/mult_form_deep_tau.json: cells={} "
      "max|diff|={:.3e}".format(verify["dmul_cells"], verify["dmul_max_abs_diff"]))
ok &= verify["dmul_cells"] == 48 and verify["dmul_max_abs_diff"] < 1e-6
print("[verify] consensus counts vs appendix Table tab:consensus:")
for k in EXPECTED:
    match = consensus[k] == EXPECTED[k]
    ok &= match
    print("  {:<12s} computed {}  expected {}  {}".format(
        k, consensus[k], EXPECTED[k], "OK" if match else "MISMATCH"))
# ordinal divergence actually plotted: merge below both endpoints, tau >= 50M
deep = [r for r in rows if r["tau_tokens"] >= 50003968]
div = {}
for r in deep:
    key = (r["seed"], r["tau_idx"])
    div.setdefault(key, {})[r["line"]] = r["y"]
div_ok = all(v["merge"] < v["endA"] and v["merge"] < v["endB"]
             for v in div.values())
print("[verify] ordinal divergence merge < min(endA, endB) at tau >= 50M: "
      "{} ({} cells)".format(div_ok, len(div)))
ok &= div_ok
s0 = [MET["per_seed"][str(s)]["survival_a05"][0] for s in SEEDS]
print("[verify] survival_a05[tau=0] = {} (DOSSIER: 0.4174/0.3930/0.3775)"
      .format(s0))
ok &= all(abs(a - b) < 5e-5 for a, b in
          zip(s0, [0.4174, 0.3930, 0.3775]))

# ------------------------------------------------------------------ write
p = os.path.join(OUT, "fig6_trajectory.csv")
with open(p, "w") as f:
    f.write("seed,tau_idx,tau_tokens,line,y,ci_lo,ci_hi,n_cohort\n")
    for r in rows:
        f.write("{seed},{tau_idx},{tau_tokens},{line},{y:.6f},{ci_lo:.6f},"
                "{ci_hi:.6f},{n_cohort}\n".format(**r))
print("[write]", p, "rows", len(rows))
print("ALL CHECKS PASS" if ok else "CHECK FAILURES -- see above")
sys.exit(0 if ok else 1)

