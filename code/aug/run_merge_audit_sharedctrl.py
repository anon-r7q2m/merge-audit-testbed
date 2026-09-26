#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import array
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

SEEDS = [5289]
K_BASE_DENSE = 64


def dense_base_slot_pattern(seed, k_exposure=K_BASE_DENSE):
    """Analysis script for the merge-audit study."""
    tokens = R.BASE_STEPS * R.TOK_PER_STEP
    s_safe = tokens // R.MAX_SENT_LEN
    n_occ = R.N_SHARED * k_exposure
    rng = R.drng(R.UNIVERSE_SEED, "base_slots_dense", seed)
    start = s_safe - n_occ
    positions = array.array("q", range(start, s_safe))
    order = list(range(n_occ))
    rng.shuffle(order)
    fact_idx = array.array("i", [o // k_exposure for o in order])
    tmpl_id = array.array("B", [(o % k_exposure) % R.N_TRAIN_TEMPLATES
                                for o in order])
    return positions, fact_idx, tmpl_id, s_safe, R.BASE_STEPS


def phase_base_dense(ctx, seed):
    """Analysis script for the merge-audit study."""
    run_id = "base-dctrl-s{}".format(seed)
    if ctx.done(run_id):
        return
    u = R.load_universe(ctx)
    pl = dense_base_slot_pattern(seed)
    stream = R.SentenceStream("base-dctrl:{}".format(seed), u["pools"]["shared"],
                              u["templates"], pl, u["pools"]["filler"], 0)
    packer = R.RowPacker(stream)
    model = R.train_segment(ctx, run_id, R.BASE_STEPS, packer, R.LR_MAIN,
                            R.LR_WARMUP_STEPS, init_seed=seed,
                            data_seed_key="base-dctrl:{}".format(seed),
                            opt_out=ctx.p("ckpts", run_id, "opt.pt"))
    if model is None:
        R.jsonl_append(ctx.p("failed_runs.jsonl"),
                       {"run": run_id, "why": "nan"})
        return
    R.save_sd_bf16(model, ctx.p("ckpts", run_id, "final.pt"))

    battery = R.build_probe_battery(u["pools"], u["templates"])
    R.eval_margins(ctx, "s{}-dctrl-base".format(seed),
                   R.load_sd_fp32(ctx.p("ckpts", run_id, "final.pt")), battery)
    meta, facts = R.load_margins(ctx, "s{}-dctrl-base".format(seed))
    rows = [r for r in facts["train"].values() if r["set"] == "shared"]
    frac = sum(1 for r in rows if r["rank_bar"] <= R.RANK1_BAR_THRESH) / max(1, len(rows))
    R.write_json(ctx.p("dctrl_health.json"),
                 {"seed": seed, "shared_frac_rank1": round(frac, 4),
                  "note": "[note]"})
    print("[sharedctrl] dense-base shared frac_rank1 =", frac, flush=True)
    ctx.mark_done(run_id)


def phase_drift_only(ctx, seed, branch):
    """Analysis script for the merge-audit study."""
    run_id = "dctrl-drift{}-s{}".format(branch, seed)
    if ctx.done(run_id):
        return
    sd0 = R.load_sd_fp32(ctx.p("ckpts", "base-dctrl-s{}".format(seed), "final.pt"))
    u = R.load_universe(ctx)
    stream = R.SentenceStream("dctrl-drift:{}:{}".format(seed, branch), [],
                              u["templates"], None, u["pools"]["filler"], 0)
    packer = R.RowPacker(stream)
    model = R.train_segment(ctx, run_id, R.DRIFT_STEPS_FULL
                            if hasattr(R, "DRIFT_STEPS_FULL") else 6104, packer,
                            R.LR_MAIN, 0, init_sd=sd0,
                            opt_out=None)
    if model is None:
        return
    R.save_sd_bf16(model, ctx.p("ckpts", run_id, "final.pt"))
    ctx.mark_done(run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/sharedctrl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        pl = dense_base_slot_pattern(5289)
        pos, fid, tid, s_safe, steps = pl
        assert len(pos) == R.N_SHARED * K_BASE_DENSE
        assert pos[0] == s_safe - len(pos) and pos[-1] == s_safe - 1
        assert max(fid) < R.N_SHARED
        print("[dry-run ok] dense pattern: n_occ=%d, block=[%d,%d), s_safe=%d"
              % (len(pos), pos[0], s_safe, s_safe))
        return
    out_dir = os.path.join(EXP, args.out_dir)
    os.makedirs(os.path.join(out_dir, "state"), exist_ok=True)
    ctx = R.Ctx(out_dir)
    R.phase_universe(ctx)
    for seed in SEEDS:
        phase_base_dense(ctx, seed)
        for br in ("A", "B"):
            phase_drift_only(ctx, seed, br)


    for seed in SEEDS:
        u = R.load_universe(ctx)
        battery = R.build_probe_battery(u["pools"], u["templates"])
        sdA = R.load_sd_fp32(ctx.p("ckpts", "dctrl-driftA-s{}".format(seed),
                                   "final.pt"))
        sdB = R.load_sd_fp32(ctx.p("ckpts", "dctrl-driftB-s{}".format(seed),
                                   "final.pt"))
        R.eval_margins(ctx, "s{}-dctrl-endA".format(seed), sdA, battery)
        R.eval_margins(ctx, "s{}-dctrl-endB".format(seed), sdB, battery)
        import torch
        sdM = {k: (0.5 * sdA[k].float() + 0.5 * sdB[k].float()).to(sdA[k].dtype)
               for k in sdA}
        R.eval_margins(ctx, "s{}-dctrl-merge05".format(seed), sdM, battery)

    for seed in SEEDS:
        rep = {"seed": seed}
        for tag in ("endA", "endB", "merge05"):
            meta, facts = R.load_margins(ctx, "s{}-dctrl-{}".format(seed, tag))
            rows = [r for r in facts["train"].values() if r["set"] == "shared"]
            rep[tag] = round(sum(1 for r in rows
                                 if r["rank_bar"] <= R.RANK1_BAR_THRESH)
                             / max(1, len(rows)), 4)
        print("[sharedctrl] seed", seed, rep, flush=True)
        R.jsonl_append(ctx.p("sharedctrl_summary.jsonl"), rep)


if __name__ == "__main__":
    main()
