#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lambda-recovery and alpha=1 decomposition probes driver."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

REID = os.path.join(EXP, "results_v2_reid")
SIZES = {"L100": (12, 768, 3072), "L300": (16, 1152, 4608)}


def ev(ctx, tag, sd, battery):
    p = ctx.p("eval", "margins.{}.jsonl.gz".format(tag))
    if not os.path.exists(p):
        R.eval_margins(ctx, tag, sd, battery)


def wsum(base, comps):
    out = {}
    for k in base:
        acc = base[k].float().clone()
        for w, sd in comps:
            acc += w * (sd[k].float() - base[k].float())
        out[k] = acc.to(base[k].dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/evalext")
    args = ap.parse_args()
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    os.makedirs(ctx.p("eval"), exist_ok=True)
    u = R.load_universe(R.Ctx(REID))
    battery = R.build_probe_battery(u["pools"], u["templates"])


    for seed in (42, 1042, 2042):
        base = R.load_sd_fp32(os.path.join(REID, "ckpts",
                                           "base-s{}".format(seed), "final.pt"))
        for tname, dA, dB in (
                ("t0", "expA-s{}-k128".format(seed), "expB-s{}-k128".format(seed)),
                ("t5", "driftA-s{}".format(seed), "driftB-s{}".format(seed))):
            fA = "final.pt" if tname == "t0" else "tau5.pt"
            sdA = R.load_sd_fp32(os.path.join(REID, "ckpts", dA, fA))
            sdB = R.load_sd_fp32(os.path.join(REID, "ckpts", dB, fA))
            for lam in (0.25, 0.5, 0.75, 1.0, 1.25):
                ev(ctx, "s{}-{}-lam{:.2f}".format(seed, tname, lam),
                   wsum(base, [(lam, sdA), (lam, sdB)]), battery)
            for beta in (0.25, 0.5, 0.75):
                ev(ctx, "s{}-{}-a1b{:.2f}".format(seed, tname, beta),
                   wsum(base, [(1.0, sdA), (beta, sdB)]), battery)
                ev(ctx, "s{}-{}-a1b{:.2f}B".format(seed, tname, beta),
                   wsum(base, [(1.0, sdB), (beta, sdA)]), battery)
        print("[evalext] u1 seed {} done".format(seed), flush=True)


    COMBOS = [(0.5, 0.25), (0.5, 0.75), (1.0, 0.25), (1.0, 0.5), (1.0, 0.75)]
    JOBS = []
    for seed in (1042, 2042):
        JOBS.append(("L100", os.path.join(EXP, "results/l100_k512_s{}".format(seed)),
                     "expA-s{}-k512".format(seed), "expB-s{}-k512".format(seed)))
    for seed in (42, 1042, 2042):
        JOBS.append(("L300", os.path.join(EXP, "results/l300_k2048{}".format(
            "" if seed == 42 else "_s{}".format(seed))),
                     "expA-s{}-k2048".format(seed), "expB-s{}-k2048".format(seed)))
    cfg30 = (R.N_LAYER, R.D_MODEL, R.D_FF)
    for size, rdir, ridA, ridB in JOBS:
        if not os.path.exists(os.path.join(rdir, "ckpts", ridA, "final.pt")):
            print("[evalext] SKIP {} {} (ckpt missing)".format(size, rdir))
            continue
        nl, dm, dff = SIZES[size]
        R.N_LAYER, R.D_MODEL, R.D_FF = nl, dm, dff
        base = R.load_sd_fp32(os.path.join(rdir, "ckpts",
                                           "base-s{}".format(ridA.split('-')[1][1:]),
                                           "final.pt"))
        sdA = R.load_sd_fp32(os.path.join(rdir, "ckpts", ridA, "final.pt"))
        sdB = R.load_sd_fp32(os.path.join(rdir, "ckpts", ridB, "final.pt"))
        for a, b in COMBOS:
            ev(ctx, "{}-{}-a{:.2f}b{:.2f}".format(size, ridA.split('-')[1], a, b),
               wsum(base, [(a, sdA), (b, sdB)]), battery)
        print("[evalext] {} {} done".format(size, rdir), flush=True)
        R.N_LAYER, R.D_MODEL, R.D_FF = cfg30


if __name__ == "__main__":
    main()
