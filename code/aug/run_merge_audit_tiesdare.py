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

SEED_DIRS = {"results_v2_reid": [42, 1042, 2042],
             "results_v2_c1x3": [6784, 403, 1189],
             "results_v2_aug3": [3184, 5383, 7192]}
TAU_IDXS = [0, 5, 7]
TGRID = [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
TIES_DENSITY = 0.5
DARE_P = 0.5


def ties_merge(sd0, sdA, sdB, density=TIES_DENSITY):
    import torch
    out = {}
    for k in sdA:
        dA = sdA[k].float() - sd0[k].float()
        dB = sdB[k].float() - sd0[k].float()
        flatA, flatB = dA.flatten(), dB.flatten()
        n = flatA.numel()
        kkeep = max(1, int(n * density))
        thrA = flatA.abs().kthvalue(n - kkeep + 1).values
        thrB = flatB.abs().kthvalue(n - kkeep + 1).values
        mA = (flatA.abs() >= thrA) & (flatA.sign() != 0)
        mB = (flatB.abs() >= thrB) & (flatB.sign() != 0)

        both = mA & mB
        sgn = torch.sign(flatA * mA + flatB * mB)
        elected = torch.where(both, sgn * (flatA.abs() * mA + flatB.abs() * mB) / 2.0,
                              torch.where(mA, flatA, torch.where(mB, flatB, torch.zeros_like(flatA))))
        out[k] = (sd0[k].float() + elected.reshape(sdA[k].shape)).to(sdA[k].dtype)
    return out


def dare_merge(sd0, sdA, sdB, p=DARE_P, seed=20260829):
    import torch
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k in sdA:
        dA = sdA[k].float() - sd0[k].float()
        dB = sdB[k].float() - sd0[k].float()
        mA = torch.rand(dA.shape, generator=g) >= p
        mB = torch.rand(dB.shape, generator=g) >= p
        dA = dA * mA / (1 - p)
        dB = dB * mB / (1 - p)
        out[k] = (sd0[k].float() + 0.5 * dA + 0.5 * dB).to(sdA[k].dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/tiesdare")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        import torch
        sd0 = {"w": torch.zeros(6)}
        sA = {"w": torch.tensor([1.0, -2.0, 0.1, 0.05, 3.0, 0.0])}
        sB = {"w": torch.tensor([2.0, -1.0, -0.2, 0.02, 1.0, 0.5])}
        t = ties_merge(sd0, sA, sB)["w"]
        dr = dare_merge(sd0, sA, sB)["w"]
        print("ties:", t.tolist())
        print("dare:", dr.tolist())
        print("[dry-run ok]")
        return
    out_dir = os.path.join(EXP, args.out_dir)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    ctx = R.Ctx(out_dir)
    for res_dir, seeds in SEED_DIRS.items():
        rctx_root = os.path.join(EXP, res_dir)
        for seed in seeds:
            for ti in TAU_IDXS:
                sd0 = R.load_sd_fp32(os.path.join(
                    rctx_root, "ckpts", "base-s{}".format(seed), "final.pt"))
                sdA = R.load_sd_fp32(os.path.join(
                    rctx_root, "ckpts", "driftA-s{}".format(seed),
                    "tau{}.pt".format(ti)))
                sdB = R.load_sd_fp32(os.path.join(
                    rctx_root, "ckpts", "driftB-s{}".format(seed),
                    "tau{}.pt".format(ti)))
                u = R.load_universe(R.Ctx(rctx_root))
                battery = R.build_probe_battery(u["pools"], u["templates"])
                for name, ctor in (("ties", ties_merge), ("dare", dare_merge)):
                    tag = "s{}-t{}-{}".format(seed, ti, name)
                    shard = os.path.join(out_dir, "shards", tag + ".jsonl.gz")
                    if os.path.exists(shard):
                        continue
                    sd = ctor(sd0, sdA, sdB)
                    mid = tag + "-merge"
                    R.eval_margins(ctx, mid, sd, battery)
                    import shutil
                    shutil.move(ctx.p("eval", "margins.{}.jsonl.gz".format(mid)),
                                shard)
                    print("[tiesdare] done", tag, flush=True)


if __name__ == "__main__":
    main()
