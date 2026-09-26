#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

C2_SEEDS = [3184, 5383, 7192]
C2_DIR = os.path.join(EXP, "results_v2_aug3")
N_TAU = 8


class ReadCtx:
    def __init__(self, root):
        self.root = root

    def p(self, *parts):
        return os.path.join(self.root, *parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/lmc_c2")
    args = ap.parse_args()
    out_dir = os.path.join(EXP, args.out_dir)
    os.makedirs(os.path.join(out_dir, "state"), exist_ok=True)
    rctx = ReadCtx(C2_DIR)
    R.load_universe(rctx)
    out_path = os.path.join(out_dir, "lmc_barrier.jsonl")
    n_done = 0
    for seed in C2_SEEDS:
        for ti in range(N_TAU):
            shard = "lmc-s{}-t{}".format(seed, ti)
            done_f = os.path.join(out_dir, "state", shard + ".done")
            if os.path.exists(done_f):
                continue
            sdA = R.load_sd_fp32(os.path.join(
                C2_DIR, "ckpts", "driftA-s{}".format(seed), "tau{}.pt".format(ti)))
            sdB = R.load_sd_fp32(os.path.join(
                C2_DIR, "ckpts", "driftB-s{}".format(seed), "tau{}.pt".format(ti)))
            losses = [round(R.eval_filler_loss(rctx, R.lerp_sd(sdA, sdB, a)), 4)
                      for a in R.LMC_ALPHAS]
            rec = {"seed": seed, "tau_idx": ti, "alphas": R.LMC_ALPHAS,
                   "filler_loss": losses,
                   "barrier_rel_a05": round(losses[5] / (0.5 * (losses[0] + losses[10])) - 1, 4)}
            with open(out_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            open(done_f, "w").write("ok\n")
            n_done += 1
            print("[lmc_c2] done {} ({:.4f})".format(shard, rec["barrier_rel_a05"]),
                  flush=True)
    print("[lmc_c2] total new shards:", n_done)


if __name__ == "__main__":
    main()
