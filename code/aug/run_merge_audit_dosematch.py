#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import array
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

SEEDS = [881, 913, 919, 887]
K_EXCL, K_SHR = 128, 64
U2 = os.path.join(EXP, "results/univ2")


def mixed_exposure_pattern(seed, br):
    """Analysis script for the merge-audit study."""
    import aug.run_merge_audit_univ2 as U
    n_e, n_s = R.N_EXCL, R.N_SHARED
    exp_tokens = 2 * R.EXPOSE_STEPS_K64 * (128 // R.K_EXPOSURE_DEFAULT) \
        * R.TOK_PER_STEP
    s_safe = exp_tokens // R.MAX_SENT_LEN
    n_occ = n_e * K_EXCL + n_s * K_SHR
    rng = R.drng(U.UNIV2_SEED, "dosematch_expose", seed, br)
    positions = array.array("q", sorted(rng.sample(range(s_safe), n_occ)))
    order = list(range(n_occ))
    rng.shuffle(order)

    fact_idx = array.array("i", [(o // K_EXCL) if o < n_e * K_EXCL
                                 else n_e + ((o - n_e * K_EXCL) // K_SHR)
                                 for o in order])
    tmpl_id = array.array("B", [(o % 8) for o in order])
    return positions, fact_idx, tmpl_id, s_safe, exp_tokens // R.TOK_PER_STEP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/dosematch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    os.makedirs(ctx.p("eval"), exist_ok=True)
    u = R.load_universe(R.Ctx(U2))
    if args.dry_run:
        pos, fid, tid, ss, steps = mixed_exposure_pattern(881, "A")
        assert len(pos) == R.N_EXCL * K_EXCL + R.N_SHARED * K_SHR
        assert max(fid) < R.N_EXCL + R.N_SHARED and min(fid) >= 0
        print("pattern ok: {} occ, {} steps".format(len(pos), steps))
        print("[dry-run ok]")
        return
    battery = R.build_probe_battery(u["pools"], u["templates"])
    for seed in SEEDS:

        base_sd = R.load_sd_fp32(os.path.join(
            U2, "ckpts", "u2base-s{}".format(seed), "final.pt"))
        base_opt = R.load_opt_state(os.path.join(
            U2, "ckpts", "u2base-s{}".format(seed), "opt.pt"))
        for br in "AB":
            rid = "dm-exp{}-s{}".format(br, seed)
            if not ctx.done(rid):
                facts = u["pools"]["excl_" + br] + u["pools"]["shared"]
                pl = mixed_exposure_pattern(seed, br)
                stream = R.SentenceStream("dm:{}:{}".format(seed, br), facts,
                                          u["templates"], pl,
                                          u["pools"]["filler"], 0)
                packer = R.RowPacker(stream)
                steps = pl[4]
                model = R.train_segment(ctx, rid, steps, packer, R.LR_MAIN, 0,
                                        init_sd=base_sd, init_opt=base_opt,
                                        init_seed=seed,
                                        data_seed_key="dm:{}:{}".format(seed, br),
                                        opt_out=ctx.p("ckpts", rid, "opt.pt"))
                assert model is not None
                R.save_sd_bf16(model, ctx.p("ckpts", rid, "final.pt"))
                ctx.mark_done(rid)
            sd_e = R.load_sd_fp32(ctx.p("ckpts", rid, "final.pt"))
            tag = "dm-s{}-end{}".format(seed, br)
            if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, sd_e, battery)

        sdA = R.load_sd_fp32(ctx.p("ckpts", "dm-expA-s{}".format(seed), "final.pt"))
        sdB = R.load_sd_fp32(ctx.p("ckpts", "dm-expB-s{}".format(seed), "final.pt"))
        mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
               for k in sdA}
        tag = "dm-s{}-merge05".format(seed)
        if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
            R.eval_margins(ctx, tag, mrg, battery)
        for br, sdX in (("A", sdA), ("B", sdB)):
            dis = {k: (base_sd[k].float() + 0.5 * (sdX[k].float()
                   - base_sd[k].float())).to(sdX[k].dtype) for k in sdX}
            tag = "dm-s{}-dis05{}".format(seed, br)
            if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, dis, battery)
        print("[dosematch] seed {} done".format(seed), flush=True)


if __name__ == "__main__":
    main()
