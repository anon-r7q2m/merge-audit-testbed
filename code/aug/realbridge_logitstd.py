#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import gzip
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)

RM = os.path.join(EXP, "results/realmerge")
RB = os.path.join(EXP, "results/realbridge")

PANEL_FAMS = ["fam1-llama3-8b", "fam3-olmo-1b", "fam4-llama32-1b",
              "fam5-llama32-3b", "fam7-qwen3-8b"]
BATTERIES = ["popqa", "gsm8k", "mbpp"]
F_LIVE_FLOOR = {"popqa": 300, "gsm8k": 60, "mbpp": 30}


def load_shard(path):
    rows = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            d = json.loads(line)
            if "__meta__" in d:
                continue
            rows[d["qid"]] = d
    return rows


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def _shard(fam, suffix_glob):
    import glob
    hits = glob.glob(os.path.join(RM, "shards",
                                  "%s.%s.jsonl.gz" % (fam, suffix_glob)))
    assert len(hits) == 1, (fam, suffix_glob, hits)
    return load_shard(hits[0])


def panel_cell(fam, bat):
    ft = _shard(fam, "ft-*")
    ref = _shard(fam, "lam0.50")
    ft_b = {q: r for q, r in ft.items() if q.startswith(bat + ":")}
    ref_b = {q: r for q, r in ref.items() if q.startswith(bat + ":")}
    live = [q for q, r in ft_b.items() if r["rank"] == 1]
    if len(live) < F_LIVE_FLOOR[bat]:
        return {"battery": bat, "low_power": True, "n_live": len(live)}
    s_ft = med([r["full_logit_std"] for r in ft_b.values()])
    s_ref = med([r["full_logit_std"] for r in ref_b.values()])
    d_add = med([ft_b[q]["m_top1"] - ref_b[q]["m_top1"] for q in live])
    d_mul = med([ft_b[q]["m_top1"] / s_ft - ref_b[q]["m_top1"] / s_ref
                 for q in live])
    return {"battery": bat, "low_power": False, "n_live": len(live),
            "s_ft": round(s_ft, 5), "s_ref": round(s_ref, 5),
            "scale_ratio_ft_over_ref": round(s_ft / s_ref, 4),
            "D_add_t1": round(d_add, 5), "D_mul": round(d_mul, 6),
            "signs_agree": (d_add > 0) == (d_mul > 0)}


def tier1():
    sft = load_shard(os.path.join(RB, "shards", "tier1.sft.jsonl.gz"))
    ref = load_shard(os.path.join(RB, "shards", "tier1.lam0.50.jsonl.gz"))
    base = load_shard(os.path.join(RB, "shards", "tier1.base.jsonl.gz"))
    live = [q for q, r in sft.items() if r["rank"] == 1]

    s_ft = med([r["cand_logit_std"] for r in sft.values()])
    s_ref = med([r["cand_logit_std"] for r in ref.values()])
    s_base = med([r["cand_logit_std"] for r in base.values()])
    d_add = med([sft[q]["m_top1"] - ref[q]["m_top1"] for q in live])
    d_mul = med([sft[q]["m_top1"] / s_ft - ref[q]["m_top1"] / s_ref
                 for q in live])
    return {"note": "[note]",
            "n_live": len(live),
            "s_sft": round(s_ft, 5), "s_ref": round(s_ref, 5),
            "s_base": round(s_base, 5),
            "scale_ratio_sft_over_ref": round(s_ft / s_ref, 4),
            "D_add_t1": round(d_add, 5), "D_mul": round(d_mul, 6),
            "signs_agree": (d_add > 0) == (d_mul > 0)}


def main():
    out = {"normalizer": "per-model battery median of full_logit_std "
                         "(tier-1: cand_logit_std)",
           "placebo_invariance": "mechanical: m->t*m and s->t*s on the "
                                 "reference, ratio unchanged",
           "tier1_olmo": tier1(), "panel": {}}
    n_pow = n_disagree = 0
    ratios = []
    for fam in PANEL_FAMS:
        cells = [panel_cell(fam, b) for b in BATTERIES]
        out["panel"][fam] = cells
        for c in cells:
            if c.get("low_power"):
                continue
            n_pow += 1
            ratios.append(c["scale_ratio_ft_over_ref"])
            n_disagree += (not c["signs_agree"])
    out["summary"] = {
        "powered_cells": n_pow,
        "sign_disagree_add_vs_mul": n_disagree,
        "scale_ratio_range": [round(min(ratios), 4), round(max(ratios), 4)],
        "tier1_sign_disagree": not out["tier1_olmo"]["signs_agree"],
    }
    dst = os.path.join(HERE, "realbridge_logitstd.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(json.dumps(out["summary"], ensure_ascii=False, indent=1))
    print("tier1:", json.dumps(out["tier1_olmo"], ensure_ascii=False))


if __name__ == "__main__":
    main()
