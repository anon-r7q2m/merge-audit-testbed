#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import array
import gzip
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

UNIV2_SEED = 20260903
SEEDS = [881, 913, 733]
K_EXP = 128
K_MAINT = 16
TAU_STEPS = {"t0": 0, "t5": R.DRIFT_STEPS[5] if hasattr(R, "DRIFT_STEPS") else None}


def univ2_slot_pattern(n_facts, k, tokens, key, seed):
    """Analysis script for the merge-audit study."""
    s_safe = tokens // R.MAX_SENT_LEN
    n_occ = n_facts * k
    assert n_occ < s_safe, "[note]"
    rng = R.drng(UNIV2_SEED, key, seed)
    positions = array.array("q", sorted(rng.sample(range(s_safe), n_occ)))
    order = list(range(n_occ))
    rng.shuffle(order)
    fact_idx = array.array("i", [o // k for o in order])
    tmpl_id = array.array("B", [(o % k) % R.N_TRAIN_TEMPLATES for o in order])
    return positions, fact_idx, tmpl_id, s_safe


def write_universe2(ctx):
    """Analysis script for the merge-audit study."""
    if ctx.done("universe2"):
        return
    old = R.UNIVERSE_SEED
    R.UNIVERSE_SEED = UNIV2_SEED
    try:
        pools = R.build_facts()
        templates = R._make_templates()
    finally:
        R.UNIVERSE_SEED = old
    d = ctx.p("universe")
    os.makedirs(d, exist_ok=True)
    with gzip.open(os.path.join(d, "facts.json.gz") + ".tmp", "wt") as f:
        json.dump(pools, f)
    os.replace(os.path.join(d, "facts.json.gz") + ".tmp",
               os.path.join(d, "facts.json.gz"))
    R.write_json(os.path.join(d, "templates.json"),
                 {str(a): {"train": [[list(p) for p in t] for t in v["train"]],
                           "probe": [[list(p) for p in t] for t in v["probe"]]}
                  for a, v in templates.items()})

    u1 = R.load_universe(R.Ctx(os.path.join(EXP, "results_v2_reid")))
    assert pools["excl_A"][0] != u1["pools"]["excl_A"][0], "[note]"
    ctx.mark_done("universe2")


def mixed_dose_pattern(n_e, n_s, exp_tokens, seed, br):
    """Analysis script for the merge-audit study."""
    import array as _ar
    s_safe = exp_tokens // R.MAX_SENT_LEN
    n_occ = n_e * K_EXP + n_s * 64
    rng = R.drng(UNIV2_SEED, "freshctrl_expose", seed, br)
    positions = _ar.array("q", sorted(rng.sample(range(s_safe), n_occ)))
    order = list(range(n_occ))
    rng.shuffle(order)
    fact_idx = _ar.array("i", [(o // K_EXP) if o < n_e * K_EXP
                               else n_e + ((o - n_e * K_EXP) // 64)
                               for o in order])
    tmpl_id = _ar.array("B", [(o % R.N_TRAIN_TEMPLATES) for o in order])
    return positions, fact_idx, tmpl_id, s_safe


def make_packers(ctx, seed):
    """Analysis script for the merge-audit study."""
    u = R.load_universe(ctx)
    pools, templates = u["pools"], u["templates"]

    pl_base = univ2_slot_pattern(R.N_SHARED, K_EXP, R.BASE_STEPS * R.TOK_PER_STEP,
                                 "base_slots", seed)
    pack_base = R.RowPacker(R.SentenceStream("u2base:{}".format(seed),
                                             pools["shared"], templates,
                                             pl_base, pools["filler"], 0))

    pack_exp = {}
    exp_tokens = 2 * R.EXPOSE_STEPS_K64 * (K_EXP // R.K_EXPOSURE_DEFAULT) \
        * R.TOK_PER_STEP
    for br in "AB":
        facts = pools["excl_" + br] + pools["shared"]
        pl = mixed_dose_pattern(len(pools["excl_" + br]), len(pools["shared"]), exp_tokens, seed, br)
        pack_exp[br] = (R.RowPacker(R.SentenceStream(
            "u2exp:{}:{}".format(seed, br), facts, templates, pl,
            pools["filler"], 0)),
            exp_tokens // R.TOK_PER_STEP)
    return pack_base, pack_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/freshctrl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    os.makedirs(ctx.p("eval"), exist_ok=True)
    if args.dry_run:

        write_universe2(ctx)
        u = R.load_universe(ctx)
        print("univ2 pools:", {k: len(v) for k, v in u["pools"].items()})
        print("templates:", len(u["templates"]))
        pb, pe = make_packers(ctx, 881)
        for br in "AB":
            _, steps = pe[br]
            print("expose steps per branch:", steps)
        print("[dry-run ok]")
        return
    write_universe2(ctx)
    u = R.load_universe(ctx)
    battery = R.build_probe_battery(u["pools"], u["templates"])
    for seed in SEEDS:
        pack_base, pack_exp = make_packers(ctx, seed)
        # ---- base ----
        rid = "u2base-s{}".format(seed)
        if not ctx.done(rid):
            model = R.train_segment(ctx, rid, R.BASE_STEPS, pack_base,
                                    R.LR_MAIN, R.LR_WARMUP_STEPS,
                                    init_seed=seed,
                                    data_seed_key="u2base:{}".format(seed),
                                    opt_out=ctx.p("ckpts", rid, "opt.pt"))
            if model is None:
                R.jsonl_append(ctx.p("failed_runs.jsonl"),
                               {"run": rid, "why": "nan"})
                continue
            R.save_sd_bf16(model, ctx.p("ckpts", rid, "final.pt"))
            ctx.mark_done(rid)
        base_sd = R.load_sd_fp32(ctx.p("ckpts", rid, "final.pt"))
        base_opt = R.load_opt_state(ctx.p("ckpts", rid, "opt.pt"))
        for br in "AB":

            rid_e = "u2exp{}-s{}".format(br, seed)
            if not ctx.done(rid_e):
                packer, steps = pack_exp[br]
                model = R.train_segment(ctx, rid_e, steps, packer, R.LR_MAIN,
                                        0, init_sd=base_sd, init_opt=base_opt,
                                        init_seed=seed,
                                        data_seed_key="u2exp:{}:{}".format(
                                            seed, br),
                                        opt_out=ctx.p("ckpts", rid_e,
                                                      "opt.pt"))
                if model is None:
                    R.jsonl_append(ctx.p("failed_runs.jsonl"),
                                   {"run": rid_e, "why": "nan"})
                    continue
                R.save_sd_bf16(model, ctx.p("ckpts", rid_e, "final.pt"))
                ctx.mark_done(rid_e)
            sd_end0 = R.load_sd_fp32(ctx.p("ckpts", rid_e, "final.pt"))
            opt_end0 = R.load_opt_state(ctx.p("ckpts", rid_e, "opt.pt"))

            tag = "u2-s{}-t0-end{}".format(seed, br)
            if not os.path.exists(ctx.p("eval",
                                        "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, sd_end0, battery)

            rid_d = "u2drift{}-s{}".format(br, seed)
            if not ctx.done(rid_d):
                stream = R.SentenceStream(
                    "fcdrift:{}:{}".format(seed, br), [],
                    u["templates"], None, u["pools"]["filler"], 0)
                packer = R.RowPacker(stream)
                model = R.train_segment(ctx, rid_d, 6104, packer, R.LR_MAIN,
                                        0, init_sd=sd_end0, init_opt=opt_end0,
                                        init_seed=seed,
                                        data_seed_key="u2drift:{}:{}".format(
                                            seed, br))
                if model is None:
                    R.jsonl_append(ctx.p("failed_runs.jsonl"),
                                   {"run": rid_d, "why": "nan"})
                    continue
                R.save_sd_bf16(model, ctx.p("ckpts", rid_d, "final.pt"))
                ctx.mark_done(rid_d)
            sd_end5 = R.load_sd_fp32(ctx.p("ckpts", rid_d, "final.pt"))
            tag = "u2-s{}-t5-end{}".format(seed, br)
            if not os.path.exists(ctx.p("eval",
                                        "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, sd_end5, battery)

        for tname, sfx in (("t0", None), ("t5", None)):
            sdA = R.load_sd_fp32(ctx.p("ckpts", "u2expA-s{}".format(seed),
                                       "final.pt")) if tname == "t0" else \
                R.load_sd_fp32(ctx.p("ckpts", "u2driftA-s{}".format(seed),
                                     "final.pt"))
            sdB = R.load_sd_fp32(ctx.p("ckpts", "u2expB-s{}".format(seed),
                                       "final.pt")) if tname == "t0" else \
                R.load_sd_fp32(ctx.p("ckpts", "u2driftB-s{}".format(seed),
                                     "final.pt"))
            mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(
                sdA[k].dtype) for k in sdA}
            tag = "u2-s{}-{}-merge05".format(seed, tname)
            if not os.path.exists(ctx.p("eval",
                                        "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, mrg, battery)
            for br, sdX in (("A", sdA), ("B", sdB)):
                dis = {k: (base_sd[k].float() + 0.5 * (sdX[k].float()
                       - base_sd[k].float())).to(sdX[k].dtype) for k in sdX}
                tag = "u2-s{}-{}-dis05{}".format(seed, tname, br)
                if not os.path.exists(ctx.p("eval",
                                            "margins.{}.jsonl.gz".format(tag))):
                    R.eval_margins(ctx, tag, dis, battery)
        print("[univ2] seed {} done".format(seed), flush=True)


if __name__ == "__main__":
    main()
