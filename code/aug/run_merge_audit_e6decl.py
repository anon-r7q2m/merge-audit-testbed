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
from run_merge_audit_e6pilot import MODEL_DIR, build_facts

OUT = os.path.join(EXP, "results/e6pilot", "e6decl_report.json")


def sent_score(model, tok, s, dev):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device=dev)
    with torch.no_grad():
        lp = model(x).logits[0, :-1, :]
    tgt = torch.tensor(ids[1:], device=dev)
    return torch.log_softmax(lp, -1).gather(-1, tgt.unsqueeze(-1)).sum().item()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    facts = build_facts()
    towns = sorted({f["town"] for f in facts})
    jobs = sorted({f["job"] for f in facts})
    dev = "cuda"
    rep = {}
    for tag, sd in (("v1", os.path.join(EXP, "results/e6pilot", "final.pt")),
                    ("v3", os.path.join(EXP, "results/e6v3", "final.pt")),
                    ("base", None)):
        if sd and not os.path.exists(sd):
            continue
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16)
        if sd:
            model.load_state_dict(torch.load(sd, map_location="cpu"))
        model = model.to(dev).eval()
        n_top1, n_top3, n_tot, rank_sum = 0, 0, 0, 0.0
        for f in facts[:200]:
            scores = []
            for t in towns:
                s = "{} is a {} from {}.".format(f["ent"], f["job"], t)
                scores.append(sent_score(model, tok, s, dev))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            gi = towns.index(f["town"])
            n_top1 += order[0] == gi
            n_top3 += gi in order[:3]
            rank_sum += order.index(gi) + 1
            n_tot += 1
        rep[tag] = {"decl_town_top1": round(n_top1 / n_tot, 4),
                    "top3": round(n_top3 / n_tot, 4),
                    "mean_gold_rank": round(rank_sum / n_tot, 2),
                    "n": n_tot, "chance": round(1 / 12, 4)}
        del model
        torch.cuda.empty_cache()
    with open(OUT, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
