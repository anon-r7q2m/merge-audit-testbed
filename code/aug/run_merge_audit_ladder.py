#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scale-ladder arm driver (100M/274M; includes the 274M recipe-ablation controls)."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

SEED = 42
TAU_IDXS = [0, 5]
SIZES = {"L30": (8, 512, 2048), "L100": (12, 768, 3072), "L300": (16, 1152, 4608)}


def count_params():
    """Scale-ladder arm driver (100M/274M; includes the 274M recipe-ablation controls)."""
    per_layer = 4 * R.D_MODEL * R.D_MODEL + 2 * R.D_MODEL * R.D_FF
    return R.VOCAB * R.D_MODEL * 2 + R.N_LAYER * per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", required=True, choices=list(SIZES))
    ap.add_argument("--kexp", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-branch", type=float, default=None)
    ap.add_argument("--base-steps", type=int, default=None)
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--skip-drift", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    LR_BRANCH = args.lr_branch if args.lr_branch is not None else R.LR_MAIN
    nl, dm, dff = SIZES[args.size]
    R.N_LAYER, R.D_MODEL, R.D_FF = nl, dm, dff
    if args.dry_run:
        n = count_params()
        print("{}: (L={}, d={}, ff={}) ≈ {:.0f}M params".format(
            args.size, nl, dm, dff, n / 1e6))
        print("[dry-run ok]")
        return
    out_dir = args.out_dir or "results/ladder_{}".format(args.size.lower())
    out_dir = os.path.join(EXP, out_dir)
    ctx = R.Ctx(out_dir)
    os.makedirs(ctx.p("eval"), exist_ok=True)

    u = R.load_universe(R.Ctx(os.path.join(EXP, "results_v2_reid")))
    battery = R.build_probe_battery(u["pools"], u["templates"])
    # base
    rid = "base-s{}".format(SEED)
    if args.base_ckpt:

        base_sd = R.load_sd_fp32(args.base_ckpt)
        import os as _os
        _opt = _os.path.join(_os.path.dirname(args.base_ckpt), "opt.pt")
        base_opt = R.load_opt_state(_opt) if _os.path.exists(_opt) else None
        print("[ladder] base from {}（skip training）".format(args.base_ckpt), flush=True)
    elif not ctx.done(rid):
        pl = R.base_slot_pattern(SEED)
        packer = R.RowPacker(R.SentenceStream("base:{}".format(SEED),
                                              u["pools"]["shared"],
                                              u["templates"], pl,
                                              u["pools"]["filler"], 0))
        model = R.train_segment(ctx, rid, args.base_steps or R.BASE_STEPS, packer, R.LR_MAIN,
                                R.LR_WARMUP_STEPS, init_seed=SEED,
                                data_seed_key="base:{}".format(SEED),
                                opt_out=ctx.p("ckpts", rid, "opt.pt"))
        assert model is not None
        R.save_sd_bf16(model, ctx.p("ckpts", rid, "final.pt"))
        ctx.mark_done(rid)
    if not args.base_ckpt:
        base_sd = R.load_sd_fp32(ctx.p("ckpts", rid, "final.pt"))
        base_opt = R.load_opt_state(ctx.p("ckpts", rid, "opt.pt"))

    KEXP = args.kexp
    for br in "AB":
        rid_e = "exp{}-s{}-k{}".format(br, SEED, KEXP)
        if not ctx.done(rid_e):
            pl = R.expose_slot_pattern(SEED, KEXP)
            facts = u["pools"]["excl_" + br]
            packer = R.RowPacker(R.SentenceStream(
                "expose:{}:{}".format(SEED, br), facts, u["templates"], pl,
                u["pools"]["filler"], 0))
            steps = R.EXPOSE_STEPS_K64 * (KEXP // R.K_EXPOSURE_DEFAULT)
            model = R.train_segment(ctx, rid_e, steps, packer, LR_BRANCH, 0,
                                    init_sd=base_sd, init_opt=base_opt,
                                    init_seed=SEED,
                                    data_seed_key="exp:{}:{}".format(SEED, br),
                                    opt_out=ctx.p("ckpts", rid_e, "opt.pt"))
            assert model is not None
            R.save_sd_bf16(model, ctx.p("ckpts", rid_e, "final.pt"))
            ctx.mark_done(rid_e)
        sd0_end = R.load_sd_fp32(ctx.p("ckpts", rid_e, "final.pt"))
        opt_e = R.load_opt_state(ctx.p("ckpts", rid_e, "opt.pt"))
        tag = "s{}-k{}-t0-end{}".format(SEED, KEXP, br)
        if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
            R.eval_margins(ctx, tag, sd0_end, battery)

        rid_d = "drift{}-s{}-k{}-t5".format(br, SEED, KEXP)
        if args.skip_drift:
            pass
        elif not ctx.done(rid_d):
            stream = R.SentenceStream("drift:{}:{}".format(SEED, br), [],
                                      u["templates"], None,
                                      u["pools"]["filler"], 0)
            packer = R.RowPacker(stream)
            steps_d = 6104
            model = R.train_segment(ctx, rid_d, steps_d, packer, LR_BRANCH, 0,
                                    init_sd=sd0_end, init_opt=opt_e,
                                    init_seed=SEED,
                                    data_seed_key="drift:{}:{}".format(SEED, br))
            assert model is not None
            R.save_sd_bf16(model, ctx.p("ckpts", rid_d, "final.pt"))
            ctx.mark_done(rid_d)
        if args.skip_drift:
            continue
        sd5 = R.load_sd_fp32(ctx.p("ckpts", rid_d, "final.pt"))
        tag = "s{}-k{}-t5-end{}".format(SEED, KEXP, br)
        if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
            R.eval_margins(ctx, tag, sd5, battery)

    trows = [("t0", "exp{}-s{}-k{}")] if args.skip_drift else \
            [("t0", "exp{}-s{}-k{}"), ("t5", "drift{}-s{}-k{}-t5")]
    for tname, drun in trows:
        sdA = R.load_sd_fp32(ctx.p("ckpts", drun.format("A", SEED, KEXP), "final.pt"))
        sdB = R.load_sd_fp32(ctx.p("ckpts", drun.format("B", SEED, KEXP), "final.pt"))
        mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
               for k in sdA}
        tag = "s{}-k{}-{}-merge05".format(SEED, KEXP, tname)
        if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
            R.eval_margins(ctx, tag, mrg, battery)
        for br, sdX in (("A", sdA), ("B", sdB)):
            dis = {k: (base_sd[k].float() + 0.5 * (sdX[k].float()
                   - base_sd[k].float())).to(sdX[k].dtype) for k in sdX}
            tag = "s{}-k{}-{}-dis05{}".format(SEED, KEXP, tname, br)
            if not os.path.exists(ctx.p("eval",
                                        "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, dis, battery)
    print("[ladder] {} done".format(args.size), flush=True)


if __name__ == "__main__":
    main()
