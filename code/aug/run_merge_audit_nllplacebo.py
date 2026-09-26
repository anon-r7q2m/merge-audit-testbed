#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
MODELS = os.path.join(EXP, "..", "..", "data", "models")
TS = [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 2.5]
PAIRS = {
    "tm1-qwen25-1.5b": ("Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B",
                        "tm1-qwen25-1.5b.merge05"),
    "tm2-llama3-8b": ("Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct",
                      "tm2-llama3-8b.merge05"),
    "tm3-qwen3-8b": ("Qwen3-8B-Base", "Qwen3-8B",
                     "tm3-qwen3-8b.merge05"),
}


# tm1: Coder+Math; tm2: Instruct+MAmmoTH2; tm3: Qwen3+R1-0528。
FTB = {"tm1-qwen25-1.5b": "Qwen2.5-Math-1.5B",
       "tm2-llama3-8b": "MAmmoTH2-8B",
       "tm3-qwen3-8b": "DeepSeek-R1-0528-Qwen3-8B"}


def load_probes(pair, n=300):
    import gzip
    shard = os.path.join(EXP, "results/truemerge", "shards",
                         "{}.ftA.jsonl.gz".format(pair))
    ok = []
    with gzip.open(shard, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r or float(r["rank"]) > 1.5:
                continue
            ok.append(r["qid"])
            if len(ok) >= n:
                break
    return ok


def battery_by_qid(pair):
    """Analysis script for the merge-audit study."""
    sys.path.insert(0, HERE)
    import run_merge_audit_realmerge as RM
    data_root = os.path.join(EXP, "..", "..", "data")
    return None


def correct_logp(model, tok, prompt, ans, dev, temp):
    q_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    a_ids = tok(ans, add_special_tokens=False)["input_ids"]
    ids = q_ids + a_ids
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        logits = model(x).logits[0]
    if temp != 1.0:
        logits = logits * temp
    lp = torch.log_softmax(logits, -1)
    tgt = torch.tensor(ids[1:], device=dev)
    return float(lp[:-1].gather(-1, tgt.unsqueeze(-1))[len(q_ids) - 1:].sum())


def merge_sd(base_dir, fa_dir, fb_dir):
    from safetensors.torch import load_file
    import glob
    def load(d):
        sd = {}
        for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
            sd.update(load_file(f))
        return sd
    A, B = load(fa_dir), load(fb_dir)
    return {k: (A[k].float() + B[k].float()).mul_(0.5).to(A[k].dtype)
            for k in A if k in B}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"


    sys.path.insert(0, HERE)
    import run_merge_audit_realmerge as RM
    data_root = os.path.join(EXP, "..", "..", "data")
    rep = {}
    for pair, (base, ftA, mtag) in PAIRS.items():
        if not os.path.isdir(os.path.join(MODELS, base)):
            print("skip (missing):", pair); continue
        tok = AutoTokenizer.from_pretrained(os.path.join(MODELS, base))
        probes = RM.build_popqa(
            os.path.join(data_root, "datasets", "PopQA", "test.tsv"), tok)
        live = set(load_probes(pair))
        probes = [p for p in probes if p["qid"] in live][:300]
        mA = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODELS, ftA), torch_dtype=torch.bfloat16).to(dev).eval()
        mrg_sd = merge_sd(os.path.join(MODELS, base),
                          os.path.join(MODELS, ftA),
                          os.path.join(MODELS, FTB[pair]))
        mM = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODELS, base), torch_dtype=torch.bfloat16)
        mM.load_state_dict(mrg_sd, strict=False)
        mM = mM.to(dev).eval()
        med_d = {}
        for t in TS:
            ds = []
            for p in probes:
                lpM = correct_logp(mM, tok, p["prompt"], p["ans_str"], dev, 1.0)
                lpR = correct_logp(mA, tok, p["prompt"], p["ans_str"], dev, t)
                ds.append(lpM - lpR)
            ds.sort()
            med_d[t] = ds[len(ds) // 2]
        flips = [t for t in TS if (med_d[t] < 0) != (med_d[1.0] < 0)]
        rep[pair] = {"n": len(probes),
                     "median_D_by_t": {str(t): round(med_d[t], 4) for t in TS},
                     "t1_median": round(med_d[1.0], 4),
                     "flips": flips}
        print(pair, json.dumps(rep[pair], indent=1), flush=True)
        del mA, mM, mrg_sd
        torch.cuda.empty_cache()
    out = os.path.join(EXP, "aug", "nllplacebo_report.json")
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
