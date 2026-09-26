#!/usr/bin/env python3
"""run_merge_audit_alloclaw_joint.py -- same-budget joint-training baseline for the
allocation law (reviewer ask: what if the two branches could simply be trained
as one model?).

Design: one model is trained from the SAME filler-only base as the allocation
arms, on the union of both branches' content -- exclusive facts of A and B
(6000 each at their branch dose) plus every allocatable (shared-pool) fact at
the full total dose K, in a single run. No merge. Survival on the allocatable
pool is then the no-merge ceiling that the balanced-split merge is priced
against. Seeds {42,1042,2042}, doses K in {128,256}, tau=0, ITT.
Idempotent; writes only into its own out-dir. Reuses the frozen ccbase
checkpoints of the corresponding allocation run dir (byte-identical base).
"""
import argparse
import array
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_r2rank08 as R                                   # noqa: E402

SEEDS = [42, 1042, 2042]
REID = os.path.join(EXP, "results_v2_reid")


def joint_pattern(n_excl, k_excl, k_alloc, n_alloc, exp_tokens, seed):
    """Exposure-block slots for the joint arm: exclusive (both pools) at the
    branch dose + every allocatable fact at the full dose K."""
    s_safe = exp_tokens // R.MAX_SENT_LEN
    occ = []
    for i in range(n_excl):
        for k in range(k_excl):
            occ.append((i, k))
    base_i = n_excl
    for j in range(n_alloc):
        for k in range(k_alloc):
            occ.append((base_i + j, k))
    rng = R.drng(20260926, "joint_slots", seed)
    positions = array.array("q", sorted(rng.sample(range(s_safe), len(occ))))
    rng.shuffle(occ)
    fact_idx = array.array("i", [o[0] for o in occ])
    tmpl_id = array.array("B", [o[1] % R.N_TRAIN_TEMPLATES for o in occ])
    return positions, fact_idx, tmpl_id, s_safe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktotal", type=int, required=True, choices=[128, 256])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alloc-dir", required=True,
                    help="existing results_aug_alloclaw_k<K> dir (base reuse)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    K = args.ktotal
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    os.makedirs(ctx.p("eval"), exist_ok=True)
    u = R.load_universe(R.Ctx(REID))
    pools, templates = u["pools"], u["templates"]
    battery = R.build_probe_battery(pools, templates)
    n_alloc = len(pools["shared"])
    facts = pools["excl_A"] + pools["excl_B"] + pools["shared"]
    n_excl = len(pools["excl_A"]) + len(pools["excl_B"])
    n_occ = n_excl * 128 + n_alloc * K
    exp_tokens = int(50_000_000 * (n_occ / (6000 * 128 + 6000 * 64)) + 0.5)
    steps_e = exp_tokens // R.TOK_PER_STEP
    if args.dry_run:
        print("K={}: joint renders={}, exposure steps={}".format(
            K, n_occ, steps_e))
        print("[dry-run ok]")
        return
    for seed in SEEDS:
        base_final = os.path.join(EXP, args.alloc_dir, "ckpts",
                                  "ccbase-s{}".format(seed), "final.pt")
        base_opt = os.path.join(EXP, args.alloc_dir, "ckpts",
                                "ccbase-s{}".format(seed), "opt.pt")
        base_sd = R.load_sd_fp32(base_final)
        opt = R.load_opt_state(base_opt)
        rid = "allocjoint-k{}-s{}".format(K, seed)
        if not ctx.done(rid):
            pl = joint_pattern(n_excl, 128, K, n_alloc, exp_tokens, seed)
            packer = R.RowPacker(R.SentenceStream(
                "allocjoint:{}:{}".format(K, seed), facts, templates,
                pl, pools["filler"], 0))
            model = R.train_segment(ctx, rid, steps_e, packer,
                                    R.LR_MAIN, 0, init_sd=base_sd,
                                    init_opt=opt, init_seed=seed,
                                    data_seed_key="allocjoint:{}:{}".format(
                                        K, seed),
                                    opt_out=ctx.p("ckpts", rid, "opt.pt"))
            assert model is not None
            R.save_sd_bf16(model, ctx.p("ckpts", rid, "final.pt"))
            ctx.mark_done(rid)
        sd = R.load_sd_fp32(ctx.p("ckpts", rid, "final.pt"))
        tag = "k{}-s{}-t0-joint".format(K, seed)
        if not os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(tag))):
            R.eval_margins(ctx, tag, sd, battery)
        print("[alloclaw-joint] K={} seed {} done".format(K, seed), flush=True)


if __name__ == "__main__":
    main()
