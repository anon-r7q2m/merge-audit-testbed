#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_merge_audit_e6pilot import (ANS, BIO_TMPL, MODEL_DIR, QA_TRAIN,
                                build_facts, replay_ids)


def rows_for(facts, kind, n=300):
    rows = []
    for f in facts[:n]:
        if kind == "qa":
            for attr, tmpls in QA_TRAIN.items():
                rows.append("Q: {}\nA: {}".format(tmpls[0].format(e=f["ent"]),
                                                  ANS[attr](f)))
        else:
            rows.append(BIO_TMPL[0].format(e=f["ent"], j=f["job"],
                                           t=f["town"], o=f["object"],
                                           h=f["habit"]))
    return rows


def mean_loss(model, tok, texts, dev, ctx=512):
    losses, cnt = [], 0
    with torch.no_grad():
        for t in texts:
            ids = tok(t + "<|endoftext|>", add_special_tokens=False)["input_ids"][:ctx]
            if len(ids) < 8:
                continue
            x = torch.tensor([ids], device=dev)
            losses.append(model(x, labels=x).loss.item())
            cnt += 1
    return sum(losses) / max(1, cnt), cnt


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    facts = build_facts()
    dev = "cuda"
    rep_ids = replay_ids(tok, 200_000)
    rep_text = tok.decode(rep_ids[:120_000])
    rep_chunks = [rep_text[i * 400:(i + 1) * 400] for i in range(300)]
    out = {}
    for tag, sd in (("trained",
                     os.path.join(EXP, "results/e6pilot", "final.pt")),
                    ("base", None)):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16)
        if sd:
            model.load_state_dict(torch.load(sd, map_location="cpu"))
        model = model.to(dev).eval()
        out[tag] = {
            "fact_qa": mean_loss(model, tok, rows_for(facts, "qa"), dev),
            "bio": mean_loss(model, tok, rows_for(facts, "bio"), dev),
            "replay": mean_loss(model, tok, rep_chunks, dev),
        }
        del model
        torch.cuda.empty_cache()
    p = os.path.join(EXP, "results/e6pilot", "e6lossdec_report.json")
    with open(p, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
