#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)

import numpy as np                          # noqa: E402
import run_merge_audit as R                    # noqa: E402
import run_merge_audit_memit_subspace as MS      # noqa: E402

PAIRS = [("target", "neighbor"), ("target", "shared"), ("target", "filler"),
         ("neighbor", "shared"), ("neighbor", "filler"), ("shared", "filler")]


def main():
    t0 = time.time()
    out_dir = sys.argv[sys.argv.index("--out-dir") + 1] if "--out-dir" in sys.argv \
        else "results/posctrl"
    MS._assert_outdir_safe(out_dir)
    ns, torch = MS._torch()
    reid = MS.ROCtx("results_v2_reid")
    u = R.load_universe(reid)
    pools, templates = u["pools"], u["templates"]
    picked, groups, _ = MS._sample_groups(pools, MS.N_SUB, MS.SAMPLE_SEED, False)
    sd0 = R.load_sd_fp32(os.path.join("results_v2_reid", MS.CKPT_REL))
    model = ns["init_model"](0)
    model.load_state_dict(sd0)
    model.eval()
    params = dict(model.named_parameters())
    sel_names = MS._sel_param_names(model)

    bounds, acc = [], 0
    for n in sel_names:
        acc += params[n].numel()
        bounds.append((n, acc))
    head_end = params["tok.weight"].numel()
    assert bounds[0] == ("tok.weight", head_end), bounds[0]
    segs = {"head_tokweight": slice(0, head_end), "body_blocks67_lnf": slice(head_end, acc)}
    print("[seg] segments:", {k: (v.start, v.stop) for k, v in segs.items()}, flush=True)

    Gmats = {}
    for gname in ("target", "neighbor", "shared", "filler"):
        facts = groups[gname]
        P = acc
        G = np.empty((len(facts), P), dtype=np.float32)
        for i, f in enumerate(facts):
            G[i] = MS._fact_grad_vector(model, params, sel_names, f, templates, torch)
            if (i + 1) % 64 == 0:
                print("[seg] {} {}/{} ({:.0f}s)".format(gname, i + 1, len(facts),
                                                        time.time() - t0), flush=True)
        Gmats[gname] = G
    del model

    norm_share = {}
    for g, G in Gmats.items():
        n2 = (G.astype(np.float64) ** 2)
        tot = n2.sum(axis=1)
        head = n2[:, segs["head_tokweight"]].sum(axis=1)
        norm_share[g] = {"head_share_median": round(float(np.median(head / tot)), 6),
                         "head_share_q25": round(float(np.quantile(head / tot, 0.25)), 6),
                         "head_share_q75": round(float(np.quantile(head / tot, 0.75)), 6)}

    per_seg = {}
    for sname, sl in segs.items():
        grams, crosses = {}, {}
        for g in Gmats:
            grams[g] = MS._gram_cross(Gmats[g][:, sl], Gmats[g][:, sl])
            print("[seg] gram {} {} ({:.0f}s)".format(sname, g, time.time() - t0), flush=True)
        for a, b in PAIRS:
            crosses[(a, b)] = MS._gram_cross(Gmats[a][:, sl], Gmats[b][:, sl])
        angles, pair_cos, centroid = {}, {}, {}
        for (a, b), X in crosses.items():
            cos, ra, rb = MS._principal_angles(grams[a], grams[b], X)
            angles["{}|{}".format(a, b)] = {
                "cos_first10": [round(float(v), 8) for v in cos[:10]],
                "deg_first10": [round(float(math.degrees(math.acos(min(1.0, v)))), 6)
                                for v in cos[:10]]}
            na = np.sqrt(np.maximum(np.diag(grams[a]), 1e-300))
            nb = np.sqrt(np.maximum(np.diag(grams[b]), 1e-300))
            pair_cos["{}|{}".format(a, b)] = MS._quartiles(
                (X / np.outer(na, nb)).ravel())
        for i, a in enumerate(Gmats):
            ca = Gmats[a][:, sl].sum(axis=0, dtype=np.float64)
            for b in list(Gmats)[i + 1:]:
                cb = Gmats[b][:, sl].sum(axis=0, dtype=np.float64)
                centroid["{}|{}".format(a, b)] = round(float(
                    ca @ cb / math.sqrt((ca @ ca) * (cb @ cb))), 8)
        per_seg[sname] = {"principal_angles": angles, "pairwise_cos": pair_cos,
                          "centroid_cos": centroid,
                          "P_seg": sl.stop - sl.start}
    out = {
        "kind": "[note]",
        "parent_artifact": "subspace_overlap.json",
        "segments": {k: {"cols": [v.start, v.stop]} for k, v in segs.items()},
        "norm_share": norm_share,
        "per_segment": per_seg,
        "provenance": MS._provenance(type("A", (), {"reid_dir": "results_v2_reid"})()),
        "runtime_s": round(time.time() - t0, 1),
    }
    path = os.path.join(out_dir, "subspace_overlap_bysegment.json")
    R.write_json(path, out)
    print("[seg] written {} sha256={}".format(path, R.sha256_file(path)), flush=True)


if __name__ == "__main__":
    main()
