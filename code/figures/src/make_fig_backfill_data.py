#!/usr/bin/env python3
"""make_fig_backfill_data.py -- figure .dat files for the PENDING.md backfill.

Reads runs/merge_audit/aug/paper_backfill.json (produced by
runs/merge_audit/aug/paper_backfill.py; read-only re-aggregation of the
frozen margin fields, six seeds, two campaigns) and writes, atomically:

  ../data/fig1_temperature.dat  - 8-point temperature grid, 12 seed-branch
                                  cells (campaign 1 + campaign 2), raw additive
                                  and multiplicative columns. Campaign-1
                                  columns verified bitwise (4dp) against the
                                  previous fig1_temperature.dat content.
  ../data/fig2_refscan.dat      - reference-family scan: one row per
                                  (cell, reference alpha), additive and
                                  multiplicative residual signs/values.
  ../data/fig3_survival.dat     - survival surface, extended to six seeds
                                  (campaign-1 columns verified against the
                                  previous file at 4dp).
  ../data/fig5_power.dat        - power check: alarm rate per mechanism x seed
                                  cell on the three certified strata.

Nothing outside ../data/ is written.
"""
import json
import os

EXP = "<REPO>"
BK = json.load(open(os.path.join(EXP, "aug", "paper_backfill.json")))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 1042, 2042, 3184, 5383, 7192]
CELLS = ["{}-{}".format(s, b) for s in SEEDS for b in "AB"]
TAU_LABELS = ["0", "3.1M", "6.3M", "12.5M", "25M", "50M", "100M", "200M"]


def atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------- Fig 1
A = BK["A_dense_tstar"]["cells"]
tgrid = [g["t"] for g in A[CELLS[0]]["grid8"]]
rcols = ["r" + c.replace("-", "") for c in CELLS]
mcols = ["m" + c.replace("-", "") for c in CELLS]
lines = ["# t " + " ".join(rcols) + " " + " ".join(mcols),
         "t " + " ".join(rcols) + " " + " ".join(mcols)]
for i, t in enumerate(tgrid):
    raw = ["{:.4f}".format(A[c]["grid8"][i]["raw"]) for c in CELLS]
    mul = ["{:.6f}".format(A[c]["grid8"][i]["calM"]) for c in CELLS]
    lines.append("{:.2f} ".format(t) + " ".join(raw) + " " + " ".join(mul))
# verify campaign-1 columns reproduce the previous dat (same frozen source)
prev = open(os.path.join(OUT, "fig1_temperature.dat")).read().splitlines()
hdr = prev[1].split()
prev_cols = {h: [r.split()[j] for r in prev[2:]] for j, h in enumerate(hdr)}
new_map = {"r1042A": "r1042A", "r1042B": "r1042B", "r2042A": "r2042A",
           "r2042B": "r2042B", "r42A": "r42A", "r42B": "r42B",
           "m1042A": "m1042A", "m1042B": "m1042B", "m2042A": "m2042A",
           "m2042B": "m2042B", "m42A": "m42A", "m42B": "m42B"}
new_rows = [l.split() for l in lines[2:]]
new_hdr = lines[1].split()
for old, new in new_map.items():
    j = new_hdr.index(new)
    col = [r[j] for r in new_rows]
    assert col == prev_cols[old], (old, "campaign-1 column mismatch")
atomic_write(os.path.join(OUT, "fig1_temperature.dat"), "\n".join(lines) + "\n")
print("fig1_temperature.dat written; 12 cells; campaign-1 columns verified "
      "against previous dat")
print("  dense t*:", BK["A_dense_tstar"]["summary"]["tstar_list"])

# ---------------------------------------------------------------- Fig 2
B = BK["B_refscan"]["cells"]
lines = ["# cellidx cell ref add_sign mul_sign add_val mul_val add_cls mul_cls",
         "cellidx ref add_sign mul_sign add_val mul_val add_cls mul_cls"]
for i, c in enumerate(CELLS):
    for j, a in enumerate(("0.25", "0.50", "0.75")):
        r = B[c]["refs"][a]
        lines.append("{} {} {} {} {:.5f} {:.6f} {} {}".format(
            i, j, r["add_sign"], r["mul_sign"], r["add"], r["mul"],
            "p" if r["add_sign"] > 0 else "n", "p" if r["mul_sign"] > 0 else "n"))
atomic_write(os.path.join(OUT, "fig2_refscan.dat"), "\n".join(lines) + "\n")
print("fig2_refscan.dat written;",
      "add varies in {}/12 cells, mul varies in {}/12 cells".format(
          BK["B_refscan"]["summary"]["add_sign_varies_across_refs"],
          BK["B_refscan"]["summary"]["mul_sign_varies_across_refs"]))

# ---------------------------------------------------------------- Fig 3
G = BK["G_findings"]
wcols = ["s{}w{}".format(s, w) for s in SEEDS for w in (25, 50, 75)]
lines = ["# tidx " + " ".join(wcols) + "   (tau labels: " + ",".join(TAU_LABELS) + ")",
         "tidx " + " ".join(wcols)]
for ti in range(8):
    vals = []
    for s in SEEDS:
        for wtag in ("w25", "w50", "w75"):
            c = G["survival"]["{}-t{}-{}".format(s, ti, wtag)]
            pooled = (c["A"]["S"] * c["A"]["n"] + c["B"]["S"] * c["B"]["n"]) / \
                     (c["A"]["n"] + c["B"]["n"])
            vals.append("{:.4f}".format(pooled))
    lines.append("{} ".format(ti) + " ".join(vals))
prev = open(os.path.join(OUT, "fig3_survival.dat")).read().splitlines()
phdr = prev[1].split()
pcols = {h: [r.split()[j] for r in prev[2:]] for j, h in enumerate(phdr)}
new_rows = [l.split() for l in lines[2:]]
new_hdr = lines[1].split()
ok = True
worst = 0.0
for old in phdr[1:]:
    j = new_hdr.index(old)
    for a, b in zip([r[j] for r in new_rows], pcols[old]):
        d = abs(float(a) - float(b))
        worst = max(worst, d)
        if d > 2e-4:  # last-digit rounding boundaries (metrics.json rounds upstream)
            ok = False
            print("  MISMATCH", old, a, b)
assert ok, "campaign-1 survival columns mismatch"
print("  fig3 campaign-1 max |diff| vs previous dat: %.1e" % worst)
atomic_write(os.path.join(OUT, "fig3_survival.dat"), "\n".join(lines) + "\n")
print("fig3_survival.dat written; 6 seeds; campaign-1 columns verified")

# ---------------------------------------------------------------- Fig 5
D = BK["D_posctrl"]["cells"]
order = ["ascend:s42", "ascend:s1042", "ascend:s2042",
         "misinfo:s42", "misinfo:s1042", "misinfo:s2042"]
lines = ["# idx cell tpr_target tpr_collateral fpr   (strata: certified forgotten",
         "# targeted / certified forgotten collateral / certified retained)",
         "idx tpr_target tpr_collateral fpr"]
for i, k in enumerate(order):
    c = D[k]
    lines.append("{} {:.4f} {:.4f} {:.4f}".format(
        i, c["tpr_target"], c["tpr_collateral"], c["fpr"]))
atomic_write(os.path.join(OUT, "fig5_power.dat"), "\n".join(lines) + "\n")
print("fig5_power.dat written; pooled TPR {:.4f} (gate 0.80), FPR max {:.4f}".format(
    BK["D_posctrl"]["summary"]["pooled_tpr_report"],
    BK["D_posctrl"]["summary"]["fpr_max"]))
print("done.")

