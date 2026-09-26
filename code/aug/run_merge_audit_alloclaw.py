#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed-total-dose allocation sweep driver (testbed)."""
import argparse
import array
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

SEEDS = [42, 1042, 2042]
SPLITS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5)]   # (kA_frac, kB_frac)
REID = os.path.join(EXP, "results_v2_reid")


def alloc_pattern(n_excl, k_excl, alloc_doses, exp_tokens, seed, br):
    """Fixed-total-dose allocation sweep driver (testbed)."""
    s_safe = exp_tokens // R.MAX_SENT_LEN
    occ = []   # (fact_idx_in_stream_list, tmpl)
    for i in range(n_excl):
        for k in range(k_excl):
            occ.append((i, k))
    base_i = n_excl
    for j, kj in enumerate(alloc_doses):
        for k in range(kj):
            occ.append((base_i + j, k))
    rng = R.drng(20260905, "alloc_slots", seed, br)
    positions = array.array("q", sorted(rng.sample(range(s_safe), len(occ))))
    rng.shuffle(occ)
    fact_idx = array.array("i", [o[0] for o in occ])
    tmpl_id = array.array("B", [o[1] % R.N_TRAIN_TEMPLATES for o in occ])
    return positions, fact_idx, tmpl_id, s_safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktotal", type=int, required=True, choices=[128, 256])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    K = args.ktotal
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    os.makedirs(ctx.p("eval"), exist_ok=True)
    u = R.load_universe(R.Ctx(REID))
    pools, templates = u["pools"], u["templates"]
    battery = R.build_probe_battery(pools, templates)
    n_alloc = len(pools["shared"])

    rng = R.drng(20260905, "alloc_groups", 0)
    order = list(range(n_alloc))
    rng.shuffle(order)
    groups = {}   # (split_idx, orientation) -> fact indices
    per = n_alloc // (len(SPLITS) * 2)
    gi = 0
    for si, (fa, fb) in enumerate(SPLITS):
        for ori in (0, 1):
            groups[(si, ori)] = order[gi * per:(gi + 1) * per]
            gi += 1

    doses = {"A": [0] * n_alloc, "B": [0] * n_alloc}
    for (si, ori), facts in groups.items():
        fa, fb = SPLITS[si]
        ka, kb = int(round(K * fa)), int(round(K * fb))
        if ori == 1:
            ka, kb = kb, ka
        for f in facts:
            doses["A"][f] = ka
            doses["B"][f] = kb
    n_occ = 6000 * 128 + sum(doses["A"])

    exp_tokens = int(50_000_000 * (n_occ / (6000 * 128 + 6000 * 64)) + 0.5)
    steps_e = exp_tokens // R.TOK_PER_STEP
    if args.dry_run:
        print("K={}: renders/branch={}, exposure steps={}".format(
            K, n_occ, steps_e))
        print("group sizes:", {k: len(v) for k, v in groups.items()})
        print("dose check (K=128): A doses distinct:",
              sorted(set(doses["A"])), "sum:", sum(doses["A"]))
        print("[dry-run ok]")
        return
    for seed in SEEDS:

        rid = "ccbase-s{}".format(seed)
        if not ctx.done(rid):
            stream = R.SentenceStream("ccbase:{}".format(seed), [],
                                      templates, None, pools["filler"], 0)
            model = R.train_segment(ctx, rid, R.BASE_STEPS, R.RowPacker(stream),
                                    R.LR_MAIN, R.LR_WARMUP_STEPS,
                                    init_seed=seed,
                                    data_seed_key="ccbase:{}".format(seed),
                                    opt_out=ctx.p("ckpts", rid, "opt.pt"))
            assert model is not None
            R.save_sd_bf16(model, ctx.p("ckpts", rid, "final.pt"))
            ctx.mark_done(rid)
        base_sd = R.load_sd_fp32(ctx.p("ckpts", rid, "final.pt"))
        base_opt = R.load_opt_state(ctx.p("ckpts", rid, "opt.pt"))
        for br in "AB":
            rid_e = "alloc{}-k{}-s{}".format(br, K, seed)
            if not ctx.done(rid_e):
                facts = pools["excl_" + br] + pools["shared"]
                pl = alloc_pattern(6000, 128, doses[br], exp_tokens, seed, br)
                packer = R.RowPacker(R.SentenceStream(
                    "alloc:{}:{}:{}".format(K, seed, br), facts, templates,
                    pl, pools["filler"], 0))
                model = R.train_segment(ctx, rid_e, steps_e, packer,
                                        R.LR_MAIN, 0, init_sd=base_sd,
                                        init_opt=base_opt, init_seed=seed,
                                        data_seed_key="alloc:{}:{}:{}".format(
                                            K, seed, br),
                                        opt_out=ctx.p("ckpts", rid_e, "opt.pt"))
                assert model is not None
                R.save_sd_bf16(model, ctx.p("ckpts", rid_e, "final.pt"))
                ctx.mark_done(rid_e)
            sd_end = R.load_sd_fp32(ctx.p("ckpts", rid_e, "final.pt"))
            tag = "k{}-s{}-t0-end{}".format(K, seed, br)
            if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, sd_end, battery)
        # merge + dis05
        sdA = R.load_sd_fp32(ctx.p("ckpts", "allocA-k{}-s{}".format(K, seed),
                                   "final.pt"))
        sdB = R.load_sd_fp32(ctx.p("ckpts", "allocB-k{}-s{}".format(K, seed),
                                   "final.pt"))
        mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
               for k in sdA}
        tag = "k{}-s{}-t0-merge05".format(K, seed)
        if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
            R.eval_margins(ctx, tag, mrg, battery)
        for br, sdX in (("A", sdA), ("B", sdB)):
            dis = {k: (base_sd[k].float() + 0.5 * (sdX[k].float()
                   - base_sd[k].float())).to(sdX[k].dtype) for k in sdX}
            tag = "k{}-s{}-t0-dis05{}".format(K, seed, br)
            if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
                R.eval_margins(ctx, tag, dis, battery)

        R.write_json(ctx.p("alloc_assignment.json"),
                     {"K": K, "seed": seed,
                      "groups": {"{}|{}".format(si, ori): g
                                 for (si, ori), g in groups.items()},
                      "splits": SPLITS})
        print("[alloclaw] K={} seed {} done".format(K, seed), flush=True)


if __name__ == "__main__":
    main()
