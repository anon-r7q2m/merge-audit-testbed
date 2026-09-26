#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import array
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

SEEDS = [677, 733, 913, 919]
K_EXCL, K_SHR = 128, 64
U2 = os.path.join(EXP, "results/univ2")
RC_SEED = 20260902


def rand_partition(seed, u):
    """Analysis script for the merge-audit study."""
    allf = u["pools"]["excl_A"] + u["pools"]["excl_B"] + u["pools"]["shared"]
    rng = random.Random(RC_SEED + seed)
    idx = list(range(len(allf)))
    rng.shuffle(idx)
    return ([allf[i] for i in idx[:6000]],
            [allf[i] for i in idx[6000:12000]],
            [allf[i] for i in idx[12000:18000]])


def mix_pattern(seed, br, n_e, n_s):
    """Analysis script for the merge-audit study."""
    exp_tokens = 2 * R.EXPOSE_STEPS_K64 * (K_EXCL // R.K_EXPOSURE_DEFAULT) \
        * R.TOK_PER_STEP
    s_safe = exp_tokens // R.MAX_SENT_LEN
    n_occ = n_e * K_EXCL + n_s * K_SHR
    rng = R.drng(RC_SEED, "randctrl_expose", seed, br)
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
    ap.add_argument("--out-dir", default="results/randctrl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    os.makedirs(ctx.p("eval"), exist_ok=True)
    u = R.load_universe(R.Ctx(U2))
    if args.dry_run:
        e_a, e_b, shr = rand_partition(677, u)
        assert len(e_a) == len(e_b) == len(shr) == 6000
        assert {f["id"] for f in e_a}.isdisjoint({f["id"] for f in shr})
        pos, fid, tid, ss, steps = mix_pattern(677, "A", 6000, 6000)
        print("partition ok; pattern: {} occ, {} steps".format(len(pos), steps))
        print("[dry-run ok]")
        return
    battery = R.build_probe_battery(u["pools"], u["templates"])
    for seed in SEEDS:
        base_sd = R.load_sd_fp32(os.path.join(
            U2, "ckpts", "u2base-s{}".format(seed), "final.pt"))
        base_opt = R.load_opt_state(os.path.join(
            U2, "ckpts", "u2base-s{}".format(seed), "opt.pt"))
        e_a, e_b, shr = rand_partition(seed, u)
        for br, epool in (("A", e_a), ("B", e_b)):
            rid = "rc-exp{}-s{}".format(br, seed)
            if not ctx.done(rid):
                facts = epool + shr
                pl = mix_pattern(seed, br, 6000, 6000)
                stream = R.SentenceStream("rc:{}:{}".format(seed, br), facts,
                                          u["templates"], pl,
                                          u["pools"]["filler"], 0)
                packer = R.RowPacker(stream)
                model = R.train_segment(ctx, rid, pl[4], packer, R.LR_MAIN, 0,
                                        init_sd=base_sd, init_opt=base_opt,
                                        init_seed=seed,
                                        data_seed_key="rc:{}:{}".format(seed, br),
                                        opt_out=ctx.p("ckpts", rid, "opt.pt"))
                assert model is not None
                R.save_sd_bf16(model, ctx.p("ckpts", rid, "final.pt"))
                ctx.mark_done(rid)
            sd_e = R.load_sd_fp32(ctx.p("ckpts", rid, "final.pt"))
            tag = "rc-s{}-end{}".format(seed, br)
            if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, sd_e, battery)
        sdA = R.load_sd_fp32(ctx.p("ckpts", "rc-expA-s{}".format(seed), "final.pt"))
        sdB = R.load_sd_fp32(ctx.p("ckpts", "rc-expB-s{}".format(seed), "final.pt"))
        mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
               for k in sdA}
        tag = "rc-s{}-merge05".format(seed)
        if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
            R.eval_margins(ctx, tag, mrg, battery)
        print("[randctrl] seed {} done".format(seed), flush=True)


if __name__ == "__main__":
    main()
