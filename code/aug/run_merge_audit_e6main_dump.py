#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-fact QA dump for the 1B bridge (joint endpoint-by-merge decomposition)."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import run_merge_audit_e6main as E  # noqa: E402
import run_merge_audit_e6v5 as V


def load_model(path, dev):
    m = AutoModelForCausalLM.from_pretrained(E.MODEL_DIR, torch_dtype=torch.bfloat16)
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd)
    return m.to(dev).eval()


def qa_perfact(model, tok, facts, dev):
    """Per-fact QA dump for the 1B bridge (joint endpoint-by-merge decomposition)."""
    recs = {}
    with torch.no_grad():
        for f in facts:
            for attr, tmpls in V.QA_HELDOUT.items():
                t = tmpls[0]
                q = "Q: {}\nA:".format(t.format(e=f["ent"]))
                qids = tok(q, add_special_tokens=False)["input_ids"]
                g = model.generate(**tok(q, return_tensors="pt").to(dev),
                                   max_new_tokens=8, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
                out = tok.decode(g[0][len(qids):],
                                 skip_special_tokens=True).strip().lower()
                recs["{}|{}".format(f["ent"], attr)] = int(V.ANS[attr](f).lower() in out)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit-dir", required=True)
    ap.add_argument("--useed", type=int, required=True)
    args = ap.parse_args()
    ctx = os.path.join(EXP, args.unit_dir)
    out_path = os.path.join(ctx, "qa_dump.jsonl")
    if os.path.exists(out_path):
        print("[note]", out_path)
        return
    pools = E.build_universe(seed=args.useed)
    tok = AutoTokenizer.from_pretrained(E.MODEL_DIR)
    dev = "cuda"
    models = {
        "A": load_model(os.path.join(ctx, "ckpts", "branchA", "final.pt"), dev),
        "B": load_model(os.path.join(ctx, "ckpts", "branchB", "final.pt"), dev),
    }
    sdA = {k: v.cpu() for k, v in models["A"].state_dict().items()}
    sdB = {k: v.cpu() for k, v in models["B"].state_dict().items()}
    mrg = {k: ((sdA[k].float() + sdB[k].float()) * 0.5).to(sdA[k].dtype)
           for k in sdA}

    models["M"] = load_model(os.path.join(ctx, "ckpts", "branchA", "final.pt"), dev)
    models["M"].load_state_dict(mrg)
    with open(out_path, "w") as f:
        for pool_name in ("excl_A", "excl_B", "shared", "ghost"):
            facts = pools[pool_name]
            rM = qa_perfact(models["M"], tok, facts, dev)
            rB = qa_perfact(models["B"], tok, facts, dev)
            rA = qa_perfact(models["A"], tok, facts, dev)
            for k in rM:
                f.write(json.dumps({"pool": pool_name, "key": k,
                                    "A": rA[k], "B": rB[k], "M": rM[k]}) + "\n")
        f.flush()
    print("[dump] wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
