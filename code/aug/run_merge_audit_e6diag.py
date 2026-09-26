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
from run_merge_audit_e6pilot import (ANS, MODEL_DIR, QA_HELDOUT, QA_TRAIN,
                                build_facts)

OUT = os.path.join(EXP, "results/e6pilot", "e6diag_report.json")
CAND = {"town": None, "job": None, "object": None, "habit": None}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    facts = build_facts()
    dev = "cuda"
    cand = {
        "town": sorted({f["town"] for f in facts}),
        "job": sorted({f["job"] for f in facts}),
        "object": sorted({f["object"] for f in facts}),
        "habit": sorted({f["habit"] for f in facts}),
    }
    rep = {"D1_samples": [], "D2_train_gen": {}, "D3_margin_heldout": {}}

    for tag, sd in (("trained", os.path.join(EXP, "results/e6pilot", "final.pt")),
                    ("base", None)):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16)
        if sd:
            model.load_state_dict(torch.load(sd, map_location="cpu"))
        model = model.to(dev).eval()
        with torch.no_grad():

            if tag == "trained":
                for f in facts[:5]:
                    for attr in ("town", "job"):
                        q = "Q: {}\nA:".format(QA_HELDOUT[attr][0].format(e=f["ent"]))
                        qlen = len(tok(q, add_special_tokens=False)["input_ids"])
                        g = model.generate(**tok(q, return_tensors="pt").to(dev),
                                           max_new_tokens=10, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                        out = tok.decode(g[0][qlen:], skip_special_tokens=True)
                        rep["D1_samples"].append(
                            {"q": q, "gen": out, "gold": ANS[attr](f)})

            n_ok, n_tot = 0, 0
            for f in facts[:200]:
                for attr in ("town", "job", "object", "habit"):
                    q = "Q: {}\nA:".format(QA_TRAIN[attr][0].format(e=f["ent"]))
                    g = model.generate(**tok(q, return_tensors="pt").to(dev),
                                       max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    out = tok.decode(g[0][len(tok(q)["input_ids"]):],
                                     skip_special_tokens=True).strip().lower()
                    n_ok += ANS[attr](f).lower() in out
                    n_tot += 1
            rep["D2_train_gen"][tag] = round(n_ok / n_tot, 4)

            n_top1, n_top3, n_tot = 0, 0, 0
            for f in facts[:200]:
                for attr in ("town", "job", "object", "habit"):
                    gold = ANS[attr](f)
                    for tmpl in QA_HELDOUT[attr][:2]:
                        q = "Q: {}\nA:".format(tmpl.format(e=f["ent"]))
                        qids = tok(q, add_special_tokens=False)["input_ids"]
                        scores = []
                        for c in cand[attr]:
                            ids = qids + tok(" " + c, add_special_tokens=False)["input_ids"]
                            x = torch.tensor([ids], device=dev)
                            lp = model(x).logits[0, len(qids) - 1:, :]
                            tgt = torch.tensor(ids[len(qids):], device=dev)
                            scores.append(torch.log_softmax(lp, -1).gather(
                                -1, tgt.unsqueeze(-1)).sum().item())
                        order = sorted(range(len(scores)), key=lambda i: -scores[i])
                        gold_i = cand[attr].index(gold)
                        n_top1 += order[0] == gold_i
                        n_top3 += gold_i in order[:3]
                        n_tot += 1
            rep["D3_margin_heldout"][tag] = {
                "top1": round(n_top1 / n_tot, 4), "top3": round(n_top3 / n_tot, 4),
                "n": n_tot, "chance": round(1 / 12, 4)}
        del model
        torch.cuda.empty_cache()
    with open(OUT, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
