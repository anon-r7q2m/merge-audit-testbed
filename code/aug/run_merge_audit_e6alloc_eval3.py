#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Held-out open-generation readout for the 1B allocation arm."""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_merge_audit_e6main as E  # noqa: E402
import run_merge_audit_e6v5 as V  # noqa: E402

MODEL_DIR = V.MODEL_DIR
ATTRS = ["town", "job"]


def batch_gen(model, tok, prompts, dev, batch=64, max_new=12):
    outs = []
    pad_id = tok.pad_token_id or tok.eos_token_id
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = [tok(p, add_special_tokens=True)["input_ids"] for p in chunk]
        mx = max(len(e) for e in enc)
        x = torch.tensor([[pad_id] * (mx - len(e)) + e for e in enc],
                         dtype=torch.long, device=dev)
        att = torch.tensor([[0] * (mx - len(e)) + [1] * len(e) for e in enc],
                           dtype=torch.long, device=dev)
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=att,
                               max_new_tokens=max_new, do_sample=False,
                               pad_token_id=pad_id)
        for j, e in enumerate(enc):
            gen = g[j][mx:]
            outs.append(tok.decode(gen, skip_special_tokens=True)
                        .strip().lower())
    return outs


def eval_open(model, tok, facts, dev):
    """Held-out open-generation readout for the 1B allocation arm."""
    prompts, keys = [], []
    for f in facts:
        for a in ATTRS:
            prompts.append("Q: {}\nA:".format(V.QA_HELDOUT[a][0].format(
                e=f["ent"])))
            keys.append((f["id"], a))
    outs = batch_gen(model, tok, prompts, dev)
    fmap = {f["id"]: f for f in facts}
    res, samples = {}, []
    for (fid, a), out in zip(keys, outs):
        res["{}|{}".format(fid, a)] = int(V.ANS[a](fmap[fid]).lower() in out)
        if len(samples) < 40:
            samples.append({"key": "{}|{}".format(fid, a), "gen": out,
                            "true": V.ANS[a](fmap[fid])})
    return res, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    ts = args.seed
    ctx = os.path.join(EXP, "results/e6allocfu_s{}".format(ts))
    outdir = os.path.join(ctx, "eval3")
    os.makedirs(outdir, exist_ok=True)
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    pools = E.build_universe(n_excl=2000, n_shared=2400, n_ghost=200,
                             seed=20260910)
    alloc = pools["shared"]
    ghost = pools["ghost"]

    sdA = torch.load(os.path.join(ctx, "ckpts", "branchA", "final.pt"),
                     map_location="cpu")
    sdB = torch.load(os.path.join(ctx, "ckpts", "branchB", "final.pt"),
                     map_location="cpu")
    mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
           for k in sdA if k in sdB and sdA[k].shape == sdB[k].shape}

    for tag, sd in (("A", sdA), ("B", sdB), ("merge", mrg)):
        op = os.path.join(outdir, "{}.json".format(tag))
        if os.path.exists(op):
            print("[eval3] s{} {} exists, skip".format(ts, tag), flush=True)
            continue
        m = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                                 torch_dtype=torch.bfloat16)
        m.load_state_dict(sd, strict=False)
        m = m.to(dev).eval()
        r_alloc, samp = eval_open(m, tok, alloc, dev)
        r_ghost, _ = eval_open(m, tok, ghost, dev)

        gate = None
        if tag in ("A", "B"):
            own = pools["excl_" + tag][:400]
            r_own, _ = eval_open(m, tok, own, dev)
            gate = sum(r_own.values()) / len(r_own)
        acc = sum(r_alloc.values()) / len(r_alloc)
        gacc = sum(r_ghost.values()) / len(r_ghost)
        json.dump({"alloc": r_alloc, "ghost": r_ghost, "own_gate": gate,
                   "samples": samp}, open(op, "w"))
        print("[eval3] s{} {}: alloc open {:.3f}, ghost {:.3f}, own_gate {}"
              .format(ts, tag, acc, gacc, gate), flush=True)
        del m
        torch.cuda.empty_cache()
    print("[eval3] s{} done".format(ts), flush=True)


if __name__ == "__main__":
    main()
