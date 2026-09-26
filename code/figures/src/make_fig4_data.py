#!/usr/bin/env python3
"""make_fig4_data.py -- Fig 4 (barrier-survival dissociation), 24-cell
alpha=0.5 exact-field version.

Replaces the earlier 9-point tau=0 scatter (which linearly interpolated the
11-point filler curve at alpha=0.25/0.75): every (seed, tau) cell now uses the
exact logged alpha=0.5 fields, 24 points per loss caliber.

Per cell:
  x_filler = barrier_rel_a05 from results_v2_reid/lmc_barrier.jsonl
             (frozen field; rows indexed by .tau_idx, never row position);
  x_fact   = (NLL_bal(0.5) - mu) / mu, mu = (NLL_bal(0)+NLL_bal(1))/2, from
             reviewfix/fact_loss_alpha_path.json per_seed.NLL_bal (5-point
             alpha grid; facts' own balanced macro-averaged teacher-forced NLL
             over answer tokens on F_live);
  y        = 1 - survival_a05(seed, tau)  [metrics.json per_seed field; grid
             order verified per DOSSIER App E.5].

Cross-checks (read-only, against the reviewfix artifact's frozen 24-lists,
ordered seed-major / tau-minor): barrier_filler_frozen_24, barrier_fact_main_24,
survival_a05_24; and the Finding 3 headline numbers: filler median -0.0400
(n=24 even; interpolated median of the central pair -0.0410/-0.0390), 17/24
<= 0; fact-NLL median -0.2020, range [-0.3863, +0.0440], 22/24 <= 0.

Output: ../data/fig4_barrier.csv
Columns: seed, tau_idx, tau_tokens, barrier_filler, barrier_fact, lost
"""
import json
import os
import statistics
import sys

EXP = "<REPO>"
RES = os.path.join(EXP, "results_v2_reid")
REVIEWFIX = os.path.join(EXP, "reviewfix/fact_loss_alpha_path.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 1042, 2042]

met = json.load(open(os.path.join(RES, "metrics.json")))
TAU_TOKENS = [r["tau"] for r in met["per_seed"]["42"]["regime"]]

lmc = {}
for line in open(os.path.join(RES, "lmc_barrier.jsonl")):
    if line.strip():
        r = json.loads(line)
        lmc[(r["seed"], r["tau_idx"])] = r["barrier_rel_a05"]

fx = json.load(open(REVIEWFIX))

rows = []
for s in SEEDS:
    for ti, tau in enumerate(TAU_TOKENS):
        entry = fx["per_seed"][str(s)][ti]
        assert entry["tau_idx"] == ti, "tau order mismatch in reviewfix artifact"
        nll = entry["NLL_bal"]
        mu = (nll["0.0"] + nll["1.0"]) / 2.0
        bf = (nll["0.5"] - mu) / mu
        surv = met["per_seed"][str(s)]["survival_a05"][ti]
        rows.append({"seed": s, "tau_idx": ti, "tau_tokens": tau,
                     "barrier_filler": lmc[(s, ti)], "barrier_fact": bf,
                     "lost": 1.0 - surv})

ok = True
for name, mine, theirs in (
        ("barrier_filler", [r["barrier_filler"] for r in rows],
         fx["barrier_filler_frozen_24"]),
        ("barrier_fact", [r["barrier_fact"] for r in rows],
         fx["barrier_fact_main_24"])):
    d = max(abs(a - b) for a, b in zip(mine, theirs))
    print("[verify] {} vs reviewfix frozen 24-list: max|diff|={:.3e}"
          .format(name, d))
    ok &= d < 1e-4
surv_mine = [1.0 - r["lost"] for r in rows]
d = max(abs(a - b) for a, b in zip(surv_mine, fx["survival_a05_24"]))
print("[verify] survival_a05 vs reviewfix 24-list: max|diff|={:.3e}".format(d))
ok &= d < 5e-5

bf_filler = [r["barrier_filler"] for r in rows]
bf_fact = [r["barrier_fact"] for r in rows]
sf = sorted(bf_filler)
central = (sf[11], sf[12])
med_filler = statistics.median(bf_filler)
med_fact = statistics.median(bf_fact)
n_le_f = sum(1 for v in bf_filler if v <= 0)
n_le_x = sum(1 for v in bf_fact if v <= 0)
print("[verify] filler: central pair {:.4f}/{:.4f} median {:.4f} "
      "(paper: -0.0410/-0.0390, -0.0400); {}/24 <= 0 (paper: 17/24); "
      "range {:.4f}..{:.4f} (paper: -0.1357..+0.0217)"
      .format(central[0], central[1], med_filler, n_le_f,
              min(bf_filler), max(bf_filler)))
print("[verify] fact-NLL: median {:.4f} (paper: -0.2020); {}/24 <= 0 "
      "(paper: 22/24); range {:.4f}..{:.4f} (paper: -0.3863..+0.0440)"
      .format(med_fact, n_le_x, min(bf_fact), max(bf_fact)))
ok &= abs(med_filler - (-0.0400)) < 5e-5 and n_le_f == 17
ok &= abs(med_fact - (-0.2020)) < 5e-4 and n_le_x == 22
ok &= abs(min(bf_fact) - (-0.3863)) < 5e-5 and abs(max(bf_fact) - 0.0440) < 5e-5
ok &= abs(min(bf_filler) - (-0.1357)) < 5e-5 and abs(max(bf_filler) - 0.0217) < 5e-5

p = os.path.join(OUT, "fig4_barrier.csv")
with open(p, "w") as f:
    f.write("seed,tau_idx,tau_tokens,barrier_filler,barrier_fact,lost\n")
    for r in rows:
        f.write("{seed},{tau_idx},{tau_tokens},{barrier_filler:.6f},"
                "{barrier_fact:.6f},{lost:.6f}\n".format(**r))
print("[write]", p, "rows", len(rows))
print("ALL CHECKS PASS" if ok else "CHECK FAILURES -- see above")
sys.exit(0 if ok else 1)

