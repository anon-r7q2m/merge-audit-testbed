#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Candidate-strip rank-1 re-evaluation for the 1B allocation arm."""
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
ATTRS = ["town", "job", "object", "habit"]


def cand_strip_eval(model, tok, facts, dev, cands):
    """Candidate-strip rank-1 re-evaluation for the 1B allocation arm."""
    factmap = {f["id"]: f for f in facts}
    res = {}
    with torch.no_grad():
        for f in facts:
            for a in ATTRS:
                q = "Q: {}\nA:".format(V.QA_HELDOUT[a][0].format(e=f["ent"]))
                qids = tok(q, add_special_tokens=False)["input_ids"]
                best, bestj = None, -1
                for j, c in enumerate(cands[a]):
                    cids = tok(" " + c, add_special_tokens=False)["input_ids"]
                    ids = qids + cids
                    x = torch.tensor([ids], device=dev)
                    lp = model(x).logits[0, len(qids) - 1:, :]
                    tgt = torch.tensor(cids, device=dev)
                    sc = torch.log_softmax(lp.float(), -1).gather(
                        -1, tgt.unsqueeze(-1)).sum().item()
                    if best is None or sc > best:
                        best, bestj = sc, j
                res["{}|{}".format(f["id"], a)] = int(
                    cands[a][bestj] == V.ANS[a](factmap[f["id"]]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--dir-prefix", default="results/e6alloc_s")
    ap.add_argument("--useed", type=int, default=20260908)
    args = ap.parse_args()
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pools = E.build_universe(n_excl=2000, n_shared=2400, n_ghost=200,
                             seed=args.useed)
    alloc = pools["shared"]
    ghost = pools["ghost"]
    cands = {a: sorted({V.ANS[a](f) for f in alloc + ghost})
             for a in ATTRS}
    print("[eval2] candidate sizes:", {a: len(v) for a, v in cands.items()},
          flush=True)

    for ts in [int(s) for s in args.seeds.split(",")]:
        ctx = os.path.join(EXP, "{}{}".format(args.dir_prefix, ts))
        outdir = os.path.join(ctx, "eval2")
        os.makedirs(outdir, exist_ok=True)
        done = os.path.join(outdir, "merge.json")
        if os.path.exists(done):
            print("[eval2] s{} done, skip".format(ts), flush=True)
            continue
        sdA = torch.load(os.path.join(ctx, "ckpts", "branchA", "final.pt"),
                         map_location="cpu")
        sdB = torch.load(os.path.join(ctx, "ckpts", "branchB", "final.pt"),
                         map_location="cpu")
        mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
               for k in sdA if k in sdB and sdA[k].shape == sdB[k].shape}
        for tag, sd in (("A", sdA), ("B", sdB), ("merge", mrg)):
            if os.path.exists(os.path.join(outdir, "{}.json".format(tag))):
                print("[eval2] s{} {} exists, skip".format(ts, tag), flush=True)
                continue
            m = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR, torch_dtype=torch.bfloat16)
            if sd is not None:
                m.load_state_dict(sd, strict=False)
            m = m.to(dev).eval()
            r_alloc = cand_strip_eval(m, tok, alloc, dev, cands)
            r_ghost = cand_strip_eval(m, tok, ghost, dev, cands)
            json.dump({"alloc": r_alloc, "ghost": r_ghost},
                      open(os.path.join(outdir, "{}.json".format(tag)), "w"))
            acc = sum(r_alloc.values()) / len(r_alloc)
            gacc = sum(r_ghost.values()) / len(r_ghost)
            print("[eval2] s{} {}: alloc per-probe acc {:.3f}, ghost {:.3f}"
                  .format(ts, tag, acc, gacc), flush=True)
            del m
            torch.cuda.empty_cache()
        del sdA, sdB, mrg
    print("[eval2] all done", flush=True)


if __name__ == "__main__":
    main()
