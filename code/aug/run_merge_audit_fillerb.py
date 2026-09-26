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

SEEDS = [42, 1042, 2042]
REID = os.path.join(EXP, "results_v2_reid")
TAU5_STEPS = R.TAU_GRID_STEPS[5]   # 50M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/fillerb")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    os.makedirs(ctx.p("eval"), exist_ok=True)
    u = R.load_universe(R.Ctx(REID))
    battery = R.build_probe_battery(u["pools"], u["templates"])

    steps_e = R.EXPOSE_STEPS_K64 * (128 // R.K_EXPOSURE_DEFAULT)
    if args.dry_run:
        print("exposure steps (filler-only):", steps_e, "; drift steps:",
              TAU5_STEPS)
        print("[dry-run ok]")
        return
    for seed in SEEDS:
        base_sd = R.load_sd_fp32(os.path.join(
            REID, "ckpts", "base-s{}".format(seed), "final.pt"))
        base_opt = R.load_opt_state(os.path.join(
            REID, "ckpts", "base-s{}".format(seed), "opt.pt"))

        rid_e = "fillerB-exp-s{}".format(seed)
        if not ctx.done(rid_e):
            stream = R.SentenceStream("fillerBexp:{}".format(seed), [],
                                      u["templates"], None,
                                      u["pools"]["filler"], 0)
            model = R.train_segment(ctx, rid_e, steps_e, R.RowPacker(stream),
                                    R.LR_MAIN, 0, init_sd=base_sd,
                                    init_opt=base_opt, init_seed=seed,
                                    data_seed_key="fillerBexp:{}".format(seed),
                                    opt_out=ctx.p("ckpts", rid_e, "opt.pt"))
            assert model is not None
            R.save_sd_bf16(model, ctx.p("ckpts", rid_e, "final.pt"))
            ctx.mark_done(rid_e)
        bp0 = R.load_sd_fp32(ctx.p("ckpts", rid_e, "final.pt"))
        bp0_opt = R.load_opt_state(ctx.p("ckpts", rid_e, "opt.pt"))

        rid_d = "fillerB-drift-s{}".format(seed)
        if not ctx.done(rid_d):
            stream = R.SentenceStream("fillerBdrift:{}".format(seed), [],
                                      u["templates"], None,
                                      u["pools"]["filler"], 0)
            model = R.train_segment(ctx, rid_d, TAU5_STEPS,
                                    R.RowPacker(stream), R.LR_MAIN, 0,
                                    init_sd=bp0, init_opt=bp0_opt,
                                    init_seed=seed,
                                    data_seed_key="fillerBdrift:{}".format(seed))
            assert model is not None
            R.save_sd_bf16(model, ctx.p("ckpts", rid_d, "final.pt"))
            ctx.mark_done(rid_d)
        bp5 = R.load_sd_fp32(ctx.p("ckpts", rid_d, "final.pt"))

        for tname, sdA in (
                ("t0", R.load_sd_fp32(os.path.join(
                    REID, "ckpts", "expA-s{}-k128".format(seed), "final.pt"))),
                ("t5", R.load_sd_fp32(os.path.join(
                    REID, "ckpts", "driftA-s{}".format(seed), "tau5.pt")))):
            sdBp = bp0 if tname == "t0" else bp5
            tag = "s{}-{}-fillerB-end".format(seed, tname)
            if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, sdBp, battery)
            mrg = {k: (sdA[k].float() + sdBp[k].float()).mul_(0.5).to(
                sdA[k].dtype) for k in sdA}
            tag = "s{}-{}-fillerB-merge05".format(seed, tname)
            if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, mrg, battery)
        print("[fillerb] seed {} done".format(seed), flush=True)


if __name__ == "__main__":
    main()
