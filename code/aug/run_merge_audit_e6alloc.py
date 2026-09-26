#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Allocation-law replication on the 1B bridge."""
import argparse
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_merge_audit_e6main as E  # noqa: E402
import run_merge_audit_e6v5 as V  # noqa: E402

MODEL_DIR = V.MODEL_DIR

CELL = 400
SPLITS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5)]


def rows_per_fact(f):
    """Allocation-law replication on the 1B bridge."""
    rows = []
    for fam in range(6):
        for k in range(3):
            rows.append(V.frame_family(f, fam, k))
    for attr, ds in V.DECL.items():
        for d in ds[:2]:
            rows.append(d.format(e=f["ent"], v=V.ANS[attr](f)))
    for attr, tmpls in V.QA_TRAIN.items():
        for t in tmpls[:6]:
            rows.append("Q: {}\nA: {}".format(t.format(e=f["ent"]),
                                              V.ANS[attr](f)))
    return rows


def split_rows(f, share, seed, skip=0.0):
    """Allocation-law replication on the 1B bridge."""
    rows = rows_per_fact(f)
    rng = random.Random("{}|{}".format(seed, f["id"]))
    rng.shuffle(rows)
    lo = int(round(len(rows) * skip))
    hi = lo + int(round(len(rows) * share))
    return rows[lo:hi]


def build_branch_ids(pools, cells, own, tok, epochs, seed):
    """Allocation-law replication on the 1B bridge."""
    ids = []
    for f in pools["excl_" + own]:
        ids.extend(rows_per_fact(f))
    for (si, ori), facts in cells.items():
        sa, sb = SPLITS[si]
        if (own == "A") == (ori == 0):
            share, skip = sa, 0.0
        else:
            share, skip = sb, sa
        if share == 0.0:
            continue
        for f in facts:
            ids.extend(split_rows(f, share, seed, skip))
    rng = random.Random(seed * 7 + (0 if own == "A" else 1))
    all_rows = []

    for ep in range(epochs):
        rs = list(ids)
        rng.shuffle(rs)
        all_rows.extend(rs)
    out = []
    for s in all_rows:
        out.extend(tok(s + "<|endoftext|>", add_special_tokens=False)["input_ids"])
    return out


def eval_qa_perfact(model, tok, facts, dev, max_new=8):
    """Allocation-law replication on the 1B bridge."""
    model.eval()
    res = {}
    with torch.no_grad():
        for f in facts:
            ok = False
            for attr, tmpls in V.QA_HELDOUT.items():
                q = "Q: {}\nA:".format(tmpls[0].format(e=f["ent"]))
                qids = tok(q, add_special_tokens=False)["input_ids"]
                g = model.generate(**tok(q, return_tensors="pt").to(dev),
                                   max_new_tokens=max_new, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
                out = tok.decode(g[0][len(qids):],
                                 skip_special_tokens=True).strip().lower()
                ok = ok or (V.ANS[attr](f).lower() in out)
            res[f["id"]] = bool(ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/e6alloc")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--tseed", type=int, default=1)
    ap.add_argument("--useed", type=int, default=20260908)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--skip-eval", action="store_true",
                    help="[note]")
    args = ap.parse_args()
    ctx = os.path.join(EXP, args.out_dir)
    os.makedirs(ctx, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)


    pools = E.build_universe(n_excl=2000, n_shared=2400, n_ghost=200,
                             seed=args.useed)
    alloc = pools["shared"]
    cells = {}
    for si in range(3):
        for ori in range(2):
            cells[(si, ori)] = alloc[(si * 2 + ori) * CELL:(si * 2 + ori + 1) * CELL]

    cache = os.path.join(ctx, "branch_ids-{}-{}.pt".format(args.useed,
                                                           args.tseed))
    if os.path.exists(cache):
        idsA, idsB, rep_ids = torch.load(cache)
    else:
        idsA = build_branch_ids(pools, cells, "A", tok, args.epochs,
                                args.useed * 1000 + args.tseed)
        idsB = build_branch_ids(pools, cells, "B", tok, args.epochs,
                                args.useed * 1000 + args.tseed)
        n_rep = int(max(len(idsA), len(idsB)) * args.ratio)
        rep_ids = V.replay_ids(tok, n_rep)
        torch.save((idsA, idsB, rep_ids), cache)
    E.train_one(ctx, "branchA", idsA, rep_ids, tok, args.lr, args.ratio,
                args.tseed)
    E.train_one(ctx, "branchB", idsB, rep_ids, tok, args.lr, args.ratio,
                args.tseed)

    if args.skip_eval:
        print("[note]", flush=True)
        return
    dev = "cuda"
    rep = {"lr": args.lr, "ratio": args.ratio, "tseed": args.tseed,
           "useed": args.useed, "cells": {}, "gate": {}}
    sds = {}
    for br in ("branchA", "branchB"):
        sds[br] = torch.load(os.path.join(ctx, "ckpts", br, "final.pt"),
                             map_location="cpu")

    def load_model(sd=None):
        m = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                                 torch_dtype=torch.bfloat16)
        if sd is not None:
            m.load_state_dict(sd, strict=False)
        return m.to(dev).eval()


    gate = {}
    for br, key in (("branchA", "A"), ("branchB", "B")):
        m = load_model(sds[br])
        own = eval_qa_perfact(m, tok, pools["excl_" + key], dev)
        acc = sum(own.values()) / len(own)
        gate[key] = round(acc, 4)
        print("[e6alloc] gate {} {} = {:.3f}".format(br, key, acc), flush=True)

        ep_path = os.path.join(ctx, "eval", "endpoint-{}.json".format(key))
        os.makedirs(os.path.dirname(ep_path), exist_ok=True)
        if not os.path.exists(ep_path):
            ep = eval_qa_perfact(m, tok, alloc, dev)
            json.dump(ep, open(ep_path, "w"))
        del m
        torch.cuda.empty_cache()

    # merge
    sdA, sdB = sds["branchA"], sds["branchB"]
    mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
           for k in sdA if k in sdB and sdA[k].shape == sdB[k].shape}
    m = load_model(mrg)
    mg_path = os.path.join(ctx, "eval", "merge.json")
    if os.path.exists(mg_path):
        mg = json.load(open(mg_path))
    else:
        mg = eval_qa_perfact(m, tok, alloc, dev)
        json.dump(mg, open(mg_path, "w"))

    gh_path = os.path.join(ctx, "eval", "ghost.json")
    if os.path.exists(gh_path):
        gh_acc = json.load(open(gh_path))["acc"]
    else:
        gh = eval_qa_perfact(m, tok, pools["ghost"], dev)
        gh_acc = sum(gh.values()) / max(1, len(gh))
        json.dump({"acc": gh_acc}, open(gh_path, "w"))
    del m, mrg
    torch.cuda.empty_cache()

    epA = json.load(open(os.path.join(ctx, "eval", "endpoint-A.json")))
    epB = json.load(open(os.path.join(ctx, "eval", "endpoint-B.json")))

    for (si, ori), facts in sorted(cells.items()):
        ids = [f["id"] for f in facts]
        sa, sb = SPLITS[si]
        shareA = sa if ori == 0 else sb
        shareB = sb if ori == 0 else sa
        uncond = sum(mg[i] for i in ids) / len(ids)
        ownA = [epA[i] for i in ids]
        ownB = [epB[i] for i in ids]
        if shareA >= shareB:
            own = ownA
        else:
            own = ownB
        p_own = sum(own) / len(own)
        cond_own = sum(1 for i, o in zip(ids, own) if o and mg[i]) / max(1, sum(own))
        both = [a and b for a, b in zip(ownA, ownB)]
        cond_both = sum(1 for i, b in zip(ids, both) if b and mg[i]) / max(1, sum(both))
        rep["cells"]["{}|{}".format(si, ori)] = {
            "shareA": shareA, "n": len(ids), "uncond": round(uncond, 4),
            "p_own_alive": round(p_own, 4), "cond_own": round(cond_own, 4),
            "p_both_alive": round(sum(both) / len(both), 4),
            "cond_both": round(cond_both, 4)}
        print("[e6alloc] cell", si, ori, rep["cells"]["{}|{}".format(si, ori)],
              flush=True)
    rep["gate"] = gate
    rep["ghost"] = round(gh_acc, 4)
    json.dump(rep, open(os.path.join(ctx, "e6alloc_report.json"), "w"),
              indent=1)
    print("[e6alloc] done", json.dumps(rep["gate"]), "ghost", gh_acc,
          flush=True)


if __name__ == "__main__":
    main()
