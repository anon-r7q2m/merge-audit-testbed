#!/usr/bin/env python3
"""Generate .dat files for the paper's pgfplots figures.

Read-only re-aggregation of frozen artifacts in runs/merge_audit/.
Outputs (written next to this script's ../data/ directory):
  fig1_temperature.dat   - temperature sweep, raw additive vs multiplicative form
  fig3_survival.dat      - survival surface S(tau, w_own), branches pooled by n_live
  fig4_barrier.dat       - (seed, alpha) scatter: relative filler barrier vs lost fraction

Nothing in runs/merge_audit/ is written. All numbers trace to:
  adjudication/sham_scale_control.json        (Fig 1)
  results_v2_reid/metrics.json                (Fig 3: symmetry cells + survival_a05)
  results_v2_reid/lmc_barrier.jsonl           (Fig 4: 11-point alpha loss curves)
"""
import json
import os

RUN = "<REPO>"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 1042, 2042]
TAU_TOKENS = [0, 3145728, 6258688, 12517376, 25001984, 50003968, 100007936, 200015872]
TAU_LABELS = ["0", "3.1M", "6.3M", "12.5M", "25M", "50M", "100M", "200M"]

# ---------------------------------------------------------------- Fig 1
# Temperature sweep applied to the reference (dis) logits, z -> t*z, tau=0,
# alpha=0.5. Raw additive form D_dis_raw crosses zero in 4/6 cells; the
# multiplicative form D_dis_cal_mult is flat in every cell.
sham = json.load(open(os.path.join(RUN, "adjudication/sham_scale_control.json")))
cells = sorted(sham["cells"].items())
tgrid = [row["t"] for row in cells[0][1]["sham_temperature_sweep"]]
with open(os.path.join(OUT, "fig1_temperature.dat"), "w") as f:
    hdr = "t " + " ".join("r" + c.replace("-", "") for c, _ in cells) + " " + \
          " ".join("m" + c.replace("-", "") for c, _ in cells)
    f.write("# " + hdr + "\n")
    f.write(hdr + "\n")
    for i, t in enumerate(tgrid):
        raw = [c[1]["sham_temperature_sweep"][i]["D_dis_raw"] for c in cells]
        mul = [c[1]["sham_temperature_sweep"][i]["D_dis_cal_mult"] for c in cells]
        f.write("{:.2f} ".format(t)
                + " ".join("{:.4f}".format(v) for v in raw) + " "
                + " ".join("{:.6f}".format(v) for v in mul) + "\n")
print("fig1_temperature.dat written;",
      "raw sign-change cells:", sham["summary"]["n_cells_raw_sign_changes_over_t"],
      "mult spread:", sham["summary"]["mult_max_spread_over_t"])

# ---------------------------------------------------------------- Fig 3
# Survival surface: pooled A/B survival per (seed, tau_idx, w_own).
# w=0.5 column: per_seed.<seed>.survival_a05 (verified grid order, DOSSIER App E.5)
# w=0.25/0.75: symmetry.cells[].P1_w25 / P3_w75, S_A / S_B, with n_liveA/B weights.
met = json.load(open(os.path.join(RUN, "results_v2_reid/metrics.json")))
surv = {s: {} for s in SEEDS}  # surv[seed][(ti, w)] = pooled survival
for s in SEEDS:
    ps = met["per_seed"][str(s)]
    for ti, v in enumerate(ps["survival_a05"]):
        surv[s][(ti, 0.5)] = v
    for cell in met["symmetry"]["cells"]:
        if cell["seed"] != s:
            continue
        ti = cell["tau_idx"]
        nA, nB = cell["n_liveA"], cell["n_liveB"]
        for key, w in (("P1_w25", 0.25), ("P3_w75", 0.75)):
            c = cell[key]
            pooled = (c["S_A"] * nA + c["S_B"] * nB) / (nA + nB)
            surv[s][(ti, w)] = pooled
with open(os.path.join(OUT, "fig3_survival.dat"), "w") as f:
    cols = ["s{}w{}".format(s, w) for s in SEEDS for w in (25, 50, 75)]
    f.write("# tidx " + " ".join(cols) + "   (tau labels: " + ",".join(TAU_LABELS) + ")\n")
    f.write("tidx " + " ".join(cols) + "\n")
    for ti in range(8):
        vals = [surv[s][(ti, w)] for s in SEEDS for w in (0.25, 0.5, 0.75)]
        f.write("{} ".format(ti) + " ".join("{:.4f}".format(v) for v in vals) + "\n")
print("fig3_survival.dat written; tau=0 w=0.5:",
      [round(surv[s][(0, 0.5)], 4) for s in SEEDS])

# ---------------------------------------------------------------- Fig 4
# One point per (seed, alpha) at tau=0: x = relative filler-loss barrier
#   barrier_rel(a) = (L(a) - mu) / mu,  mu = (L(0)+L(1))/2
# (alpha=0.5 is the frozen field barrier_rel_a05; alpha=0.25/0.75 linearly
# interpolated from the logged 11-point curve, disclosed in the caption);
# y = fraction of endpoint-surviving facts lost = 1 - S(tau=0, alpha).
lmc = [json.loads(l) for l in open(os.path.join(RUN, "results_v2_reid/lmc_barrier.jsonl"))]
with open(os.path.join(OUT, "fig4_barrier.dat"), "w") as f:
    f.write("# seed  alpha  barrier  lost   (tau = 0; barrier = relative filler-loss barrier)\n")
    f.write("seed alpha barrier lost\n")
    for s in SEEDS:
        row = next(r for r in lmc if r["seed"] == s and r["tau_idx"] == 0)
        alphas, losses = row["alphas"], row["filler_loss"]
        mu = (losses[0] + losses[-1]) / 2.0

        def barrier_at(a):
            if abs(a - 0.5) < 1e-9:
                return row["barrier_rel_a05"]
            # linear interpolation on the logged 11-point grid
            import bisect
            i = bisect.bisect_left(alphas, a)
            a0, a1 = alphas[i - 1], alphas[i]
            l0, l1 = losses[i - 1], losses[i]
            la = l0 + (l1 - l0) * (a - a0) / (a1 - a0)
            return (la - mu) / mu

        for a in (0.25, 0.5, 0.75):
            lost = 1.0 - surv[s][(0, a)]
            f.write("{} {:.2f} {:.4f} {:.4f}\n".format(s, a, barrier_at(a), lost))
print("fig4_barrier.dat written")
print("done.")

