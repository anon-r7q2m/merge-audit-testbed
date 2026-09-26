#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R  # noqa: E402


def dump_branch(ctx, K, seed, br, battery):
    out = ctx.p("logitdump", "end{}-s{}.npz".format(br, seed))
    if os.path.exists(out):
        print("[logitdump] skip", out)
        return
    sd = R.load_sd_fp32(ctx.p("ckpts", "alloc{}-k{}-s{}".format(br, K, seed),
                              "final.pt"))
    ns = R._torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ns["init_model"](0)
    model.load_state_dict(sd)
    model.to(dev).eval()
    vals_t = torch.arange(R.VAL0, R.VAL0 + R.N_VALS, device=dev)
    items = [it for it in battery if it[2] == "train"]
    by_len = {}
    for it in items:
        by_len.setdefault(len(it[4]), []).append(it)
    fids, tis, vals, zs = [], [], [], []
    with torch.no_grad():
        for L, group in sorted(by_len.items()):
            for i0 in range(0, len(group), R.EVAL_BATCH):
                chunk = group[i0:i0 + R.EVAL_BATCH]
                x = torch.tensor([c[4] for c in chunk], dtype=torch.long, device=dev)
                with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                    dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logits = model(x)[:, -1, :].float()
                zv = logits.index_select(1, vals_t).cpu().numpy().astype(np.float16)
                for j, (fid, _set, _bat, ti, _ids, val) in enumerate(chunk):
                    fids.append(fid); tis.append(ti); vals.append(val)
                    zs.append(zv[j])
    z = np.stack(zs)
    np.savez_compressed(out, z=z,
                        fid=np.array(fids), ti=np.array(tis, dtype=np.int32),
                        val=np.array(vals, dtype=np.int32))
    print("[logitdump] wrote", out, z.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kexp", type=int, required=True, choices=[128, 256])
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    ctx = R.Ctx(os.path.join(EXP, "results/alloclaw_k{}".format(args.kexp)))
    os.makedirs(ctx.p("logitdump"), exist_ok=True)
    u = R.load_universe(R.Ctx(os.path.join(EXP, "results_v2_reid")))
    battery = R.build_probe_battery(u["pools"], u["templates"])
    for br in ("A", "B"):
        dump_branch(ctx, args.kexp, args.seed, br, battery)
    print("[logitdump] done k{} s{}".format(args.kexp, args.seed))


if __name__ == "__main__":
    main()
