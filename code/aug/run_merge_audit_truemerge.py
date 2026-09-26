#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""True-merge arm driver (real same-base fine-tune pairs)."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402
import run_merge_audit_realmerge as M                          # noqa: E402

FAMS = {
    "tm1-qwen25-1.5b": {"base": "Qwen2.5-1.5B", "tok_class": "qwen25",
                        "ftA": ("code", "Qwen2.5-Coder-1.5B"),
                        "ftB": ("math", "Qwen2.5-Math-1.5B")},
    "tm2-llama3-8b": {"base": "Meta-Llama-3-8B", "tok_class": "llama3",
                      "ftA": ("instruct", "Meta-Llama-3-8B-Instruct"),
                      "ftB": ("math", "MAmmoTH2-8B")},
    "tm3-qwen3-8b": {"base": "Qwen3-8B-Base", "tok_class": "qwen3",
                     "ftA": ("post", "Qwen3-8B"),
                     "ftB": ("r1", "DeepSeek-R1-0528-Qwen3-8B")},



    "tm4-mistral-7b": {"base": "Mistral-7B-v0.1", "tok_class": "mistral",
                       "ftA": ("instruct", "Mistral-7B-Instruct-v0.1"),
                       "ftB": ("zephyr", "zephyr-7b-beta")},
    "tm5-qwen25-7b": {"base": "Qwen2.5-7B", "tok_class": "qwen25",
                      "ftA": ("instruct", "Qwen2.5-7B-Instruct"),
                      "ftB": ("math", "Qwen2.5-Math-7B")},
}
TGRID = M.TGRID
F_LIVE_FLOOR = M.F_LIVE_FLOOR
FORBIDDEN_OUTDIRS = ["results", "results_v2", "results_v2_reid", "results/realmerge"]





def _fro_delta_sq(rd_a, rd_b):
    tot = 0.0
    for k in rd_a.keys():
        d = rd_a.get(k).float() - rd_b.get(k).float()
        tot += float((d * d).sum())
    return tot


def run_family(ctx, model_root, data_root, fam_id, spec):
    torch = M._torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("==== {} ====".format(fam_id), flush=True)
    base_dir = os.path.join(model_root, spec["base"])
    tok = AutoTokenizer.from_pretrained(base_dir)
    bats = M.build_batteries(tok, spec["tok_class"], data_root)
    probes = [p for name in ("popqa", "gsm8k", "mbpp") for p in bats[name][0]]
    cand_by_bat = {name: bats[name][1] for name in bats}
    for p in probes:
        p["_prompt_ids"] = tok(p["prompt"], add_special_tokens=True)["input_ids"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    model.to(dev)

    def _load_into(sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        tie = bool(getattr(model.config, "tie_word_embeddings", False))
        allowed = {"lm_head.weight"} if tie else set()
        extra_missing = sorted(set(missing) - allowed)
        if extra_missing or unexpected:
            raise RuntimeError("[note]".format(
                extra_missing[:5], list(unexpected)[:5]))
        if missing:
            model.tie_weights()

    def eval_one(tag, sd=None, reader=None):
        shard = ctx.p("shards", "{}.{}.jsonl.gz".format(fam_id, tag))
        if os.path.exists(shard):
            return M._load_shard(shard)
        if sd is not None:
            _load_into(sd)
        elif reader is not None:
            _load_into(M._load_sd_dict(reader))
        M._eval_sharded(ctx, model, tok, "{}.{}".format(fam_id, tag),
                        probes, cand_by_bat, dev)
        return M._load_shard(shard)

    base_rd = M.StReader(base_dir)
    rdA = M.StReader(os.path.join(model_root, spec["ftA"][1]))
    rdB = M.StReader(os.path.join(model_root, spec["ftB"][1]))
    readings = {"base": eval_one("base"),
                "ftA": eval_one("ftA", reader=rdA),
                "ftB": eval_one("ftB", reader=rdB),
                "merge05": eval_one("merge05", sd=M._construct_soup([rdA, rdB])),
                "disA05": eval_one("disA05", sd=M._construct_dis(rdA, base_rd, 0.5)),
                "disB05": eval_one("disB05", sd=M._construct_dis(rdB, base_rd, 0.5))}
    norms = {"deltaA": round(math.sqrt(_fro_delta_sq(rdA, base_rd)), 3),
             "deltaB": round(math.sqrt(_fro_delta_sq(rdB, base_rd)), 3)}
    return {"family": fam_id, "readings": readings, "delta_norms": norms}


def _battery_rows(readings, bat):
    return {q: r for q, r in readings.items() if q.startswith(bat + ":")}


def analyze(ctx, fam_reps):
    """True-merge arm driver (real same-base fine-tune pairs)."""
    out = []
    for rep in fam_reps:
        fam = rep["family"]
        rd = rep["readings"]
        for bat in ("popqa", "gsm8k", "mbpp"):
            for br in "AB":
                ft = _battery_rows(rd["ft" + br], bat)
                live = [q for q, r in ft.items() if r["rank"] == 1]
                n = len(live)
                cell = {"family": fam, "battery": bat, "branch": br,
                        "n_live": n, "low_power": n < F_LIVE_FLOOR[bat]}
                if cell["low_power"]:
                    out.append(cell)
                    continue
                mrg = _battery_rows(rd["merge05"], bat)
                dis = _battery_rows(rd["dis" + br + "05"], bat)
                s = {k: R.median([r["full_logit_std"] for r in
                                  _battery_rows(rd[k], bat).values()])
                     for k in ("base", "ft" + br, "merge05", "dis" + br + "05")}
                d_add = [R.median([mrg[q]["m_top1"] - t * dis[q]["m_top1"]
                                   for q in live]) for t in TGRID]
                s_m, s_d = s["merge05"], s["dis" + br + "05"]
                d_mul = [R.median([mrg[q]["m_top1"] / s_m - t * dis[q]["m_top1"]
                                   / (t * s_d) for q in live]) for t in TGRID]
                ts, ts_dir = M.t_star(list(zip(TGRID, d_add)))
                cell.update({
                    "s": {k: round(v, 5) for k, v in s.items()},
                    "ratio_merge_over_dis": round(s_m / s_d, 4),
                    "ratio_merge_over_ft": round(s_m / s["ft" + br], 4),
                    "D_add_t1": round(d_add[TGRID.index(1.0)], 5),
                    "D_mul_t1": round(d_mul[TGRID.index(1.0)], 6),
                    "D_mul_spread": round(max(d_mul) - min(d_mul), 12),
                    "t_star": ts, "t_star_dir": ts_dir})
                out.append(cell)
    rep = {"contract": "truemerge", "cells": out,
           "summary": {
               "n_powered": sum(1 for c in out if not c.get("low_power")),
               "ratios_merge_over_dis": sorted(
                   c["ratio_merge_over_dis"] for c in out if not c.get("low_power")),
               "n_flip": sum(1 for c in out
                             if not c.get("low_power") and c.get("t_star") is not None),
               "D_mul_max_spread": max((c.get("D_mul_spread", 0) for c in out),
                                       default=None)}}
    R.write_json(ctx.p("truemerge_report.json"), rep)
    return rep


def dry_run(model_root):
    ok = True
    for fam, spec in FAMS.items():
        for role in ("base", "ftA", "ftB"):
            dn = spec[role] if role == "base" else spec[role][1]
            d = os.path.join(model_root, dn)
            good = os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json"))
            print("  [{}] {} {} -> {}".format("PASS" if good else "FAIL", fam, role, dn))
            ok = ok and good
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="results/truemerge")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--families", default=",".join(FAMS.keys()))
    args = ap.parse_args()
    M._assert_outdir_safe(args.out_dir)
    if args.dry_run:
        sys.exit(0 if dry_run(args.model_root) else 3)
    R.FUSE_GPUH = 2.5
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    fams = [f for f in args.families.split(",") if f]
    reps = []
    for fam_id in fams:
        reps.append(run_family(ctx, args.model_root, args.data_root,
                               fam_id, FAMS[fam_id]))
        R.write_json(ctx.p("report_partial.json"),
                     {"families_done": [r["family"] for r in reps]})
    analyze(ctx, reps)


if __name__ == "__main__":
    main()
