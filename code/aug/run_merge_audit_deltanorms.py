#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit_realmerge as M  # noqa: E402

FAMS = {
    "tm1-qwen25-1.5b": {"base": "Qwen2.5-1.5B",
                        "ftA": ("code", "Qwen2.5-Coder-1.5B"),
                        "ftB": ("math", "Qwen2.5-Math-1.5B")},
    "tm2-llama3-8b": {"base": "Meta-Llama-3-8B",
                      "ftA": ("instruct", "Meta-Llama-3-8B-Instruct"),
                      "ftB": ("math", "MAmmoTH2-8B")},
    "tm3-qwen3-8b": {"base": "Qwen3-8B-Base",
                     "ftA": ("post", "Qwen3-8B"),
                     "ftB": ("r1", "DeepSeek-R1-0528-Qwen3-8B")},
}


def fam_norms(model_root, fam_id, spec):
    base = M.StReader(os.path.join(model_root, spec["base"]))
    rdA = M.StReader(os.path.join(model_root, spec["ftA"][1]))
    rdB = M.StReader(os.path.join(model_root, spec["ftB"][1]))
    keys = base.keys()
    dot = na = nb = n0 = 0.0
    for i, k in enumerate(keys):
        t0 = base.get(k).double()
        da = rdA.get(k).double() - t0
        db = rdB.get(k).double() - t0
        dot += float((da * db).sum())
        na += float(da.pow(2).sum())
        nb += float(db.pow(2).sum())
        n0 += float(t0.pow(2).sum())
        if (i + 1) % 50 == 0:
            print("  {} {}/{}".format(fam_id, i + 1, len(keys)), flush=True)
    for rd in (base, rdA, rdB):
        rd.close()
    nA, nB, nBase = math.sqrt(na), math.sqrt(nb), math.sqrt(n0)
    cos = dot / max(1e-12, nA * nB)
    sep = math.sqrt(max(0.0, na + nb - 2 * dot))
    return {"family": fam_id,
            "n_keys": len(keys),
            "deltaA_fro": round(nA, 3), "deltaB_fro": round(nB, 3),
            "base_fro": round(nBase, 3),
            "cos_dA_dB": round(cos, 5),
            "branch_sep_fro": round(sep, 3),
            "relA": round(nA / nBase, 6), "relB": round(nB / nBase, 6),
            "sep_over_deltaA": round(sep / max(1e-12, nA), 4),
            "sep_over_deltaB": round(sep / max(1e-12, nB), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "truemerge_deltanorms.json"))
    args = ap.parse_args()

    from run_merge_audit_truemerge import FAMS as ALLFAMS
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    for fam_id, spec in ALLFAMS.items():
        if fam_id in out:
            continue
        print("==== {} ====".format(fam_id), flush=True)
        out[fam_id] = fam_norms(args.model_root, fam_id, spec)
        print(" ->", out[fam_id], flush=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
    print("done ->", args.out)


if __name__ == "__main__":
    main()
