#!/usr/bin/env python3
"""run_merge_audit_fp32_bound.py -- sizes the only bf16 step in the pipeline.

Pipeline fact (code-verified): training keeps fp32 master weights; endpoints
are exported with save_sd_bf16 (the single rounding step); the merge averages
the stored endpoints in fp32 and evaluation runs in fp32, so the averaged
weights themselves are never bf16-rounded.

This probe bounds what the endpoint-export rounding could contribute: we
perturb each stored endpoint by uniform noise at the scale of one bf16 ulp
per tensor (the maximal export-rounding error), re-merge, and re-evaluate one
seed (K=128, s42). If survival is unchanged within that perturbation, no
rounding of the frozen artifacts can carry the verdict.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_r2rank08 as R                                   # noqa: E402

REID = os.path.join(EXP, "results_v2_reid")


def ulp_noise(sd, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in sd.items():
        vf = v.float()
        # one bf16 ulp scales with |x|: ulp(x) ~= 2^-8 * 2^floor(log2|x|);
        # tensor-level bound: 2^-8 * max|x|
        u = vf.abs().max().clamp_min(1e-12) * (2 ** -8)
        out[k] = vf + (torch.rand(vf.shape, generator=g) * 2 - 1) * u
    return out


def main():
    u = R.load_universe(R.Ctx(REID))
    battery = R.build_probe_battery(u["pools"], u["templates"])
    K, seed = 128, 42
    rdir = os.path.join(EXP, "results_aug_alloclaw_k{}".format(K))
    ctx = R.Ctx(rdir)
    sdA = R.load_sd_fp32(ctx.p("ckpts", "allocA-k{}-s{}".format(K, seed),
                               "final.pt"))
    sdB = R.load_sd_fp32(ctx.p("ckpts", "allocB-k{}-s{}".format(K, seed),
                               "final.pt"))
    # clean-merge baseline is on disk (merge05); here the ulp-perturbed re-merge
    sdA2 = ulp_noise(sdA, 20260926)
    sdB2 = ulp_noise(sdB, 20260927)
    mrg = {k: (sdA2[k] + sdB2[k]) * 0.5 for k in sdA2}
    tag = "k{}-s{}-t0-merge05-ulp".format(K, seed)
    if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
        R.eval_margins(ctx, tag, mrg, battery)
    print("[fp32-bound] {} done".format(tag), flush=True)


if __name__ == "__main__":
    main()
