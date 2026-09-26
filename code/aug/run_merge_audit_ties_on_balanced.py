#!/usr/bin/env python3
"""run_merge_audit_ties_on_balanced.py -- operator x allocation cross: does a
merge-time operator (TIES, density 0.5, the standard fixed setting from the
operator-robustness arm) substitute for the balanced allocation, when applied
to the allocation arms' endpoints?

Read-only over frozen checkpoints of the allocation sweep (no training):
loads ccbase + allocA + allocB per (K, seed), TIES-merges them, and evaluates
the same dual-battery margins into the same eval dir as the existing
merge05 reads, so downstream aggregation compares apples to apples.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_r2rank08 as R                                   # noqa: E402
from run_rank08_tiesdare import ties_merge                 # noqa: E402

SEEDS = [42, 1042, 2042]
KS = [128, 256]
REID = os.path.join(EXP, "results_v2_reid")


def main():
    u = R.load_universe(R.Ctx(REID))
    battery = R.build_probe_battery(u["pools"], u["templates"])
    for K in KS:
        rdir = os.path.join(EXP, "results_aug_alloclaw_k{}".format(K))
        ctx = R.Ctx(rdir)
        for seed in SEEDS:
            tag = "k{}-s{}-t0-ties05".format(K, seed)
            out = ctx.p("eval", "margins.{}.jsonl.gz".format(tag))
            if os.path.exists(out):
                continue
            sd0 = R.load_sd_fp32(ctx.p("ckpts", "ccbase-s{}".format(seed),
                                       "final.pt"))
            sdA = R.load_sd_fp32(ctx.p("ckpts",
                                       "allocA-k{}-s{}".format(K, seed),
                                       "final.pt"))
            sdB = R.load_sd_fp32(ctx.p("ckpts",
                                       "allocB-k{}-s{}".format(K, seed),
                                       "final.pt"))
            mrg = ties_merge(sd0, sdA, sdB, density=0.5)
            R.eval_margins(ctx, tag, mrg, battery)
            print("[ties-on-alloc] {} done".format(tag), flush=True)


if __name__ == "__main__":
    main()
