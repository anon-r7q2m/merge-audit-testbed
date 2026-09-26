#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

SEED_DIRS = {"results_v2_reid": [42, 1042, 2042],
             "results_v2_c1x3": [6784, 403, 1189],
             "results_v2_aug3": [3184, 5383, 7192]}
TAU_IDXS = [0, 5]
NEW_CELLS = [(0.25, 0.25), (0.50, 0.25), (0.25, 0.50),
             (0.75, 0.50), (0.50, 0.75), (0.75, 0.75)]


def ab_merge(sd0, sdA, sdB, alpha, beta):
    out = {}
    for k in sdA:
        dA = sdA[k].float() - sd0[k].float()
        dB = sdB[k].float() - sd0[k].float()
        out[k] = (sd0[k].float() + alpha * dA + beta * dB).to(sdA[k].dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/2dattrib")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        import torch
        sd0 = {"w": torch.tensor([1.0, 2.0])}
        sA = {"w": torch.tensor([2.0, 2.0])}
        sB = {"w": torch.tensor([1.0, 4.0])}
        m = ab_merge(sd0, sA, sB, 0.5, 0.25)["w"]
        # [1,2] + 0.5*[1,0] + 0.25*[0,2] = [1.5, 2.5]  (dB[0]=sB[0]-sd0[0]=0)
        assert abs(m[0].item() - 1.5) < 1e-5 and abs(m[1].item() - 2.5) < 1e-5
        print("[dry-run ok]", m.tolist())
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
                for alpha, beta in NEW_CELLS:
                    tag = "s{}-t{}-a{}b{}".format(seed, ti, alpha, beta)
                    shard = os.path.join(out_dir, "shards", tag + ".jsonl.gz")
                    if os.path.exists(shard):
                        continue
                    sd = ab_merge(sd0, sdA, sdB, alpha, beta)
                    mid = tag + "-mix"
                    R.eval_margins(ctx, mid, sd, battery)
                    import shutil
                    shutil.move(ctx.p("eval", "margins.{}.jsonl.gz".format(mid)),
                                shard)
                    print("[2dattrib] done", tag, flush=True)


if __name__ == "__main__":
    main()
