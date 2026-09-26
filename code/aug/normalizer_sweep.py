#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import json
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

REID_SEEDS = [int(x) for x in
              os.environ.get("TANH_R2RANK08_SEEDS", "42,1042,2042").split(",")]
os.environ.setdefault("TANH_R2RANK08_SEEDS", ",".join(str(s) for s in REID_SEEDS))
os.environ.setdefault("RANK08_NGPUS", "1")

import run_merge_audit as R                                   # noqa: E402

CONTRACT_ID = os.environ.get("TANH_REALMERGE_CONTRACT_ID",
                             "merge_audit-aug-realmerge-v1")
SWEEP_FUSE_GPUH = 1.2
R.FUSE_GPUH = SWEEP_FUSE_GPUH


SUBSET_STRIDE = 12
SUBSET_N_EXPECT = 500
DEEP_TI = [5, 6, 7]
TGRID = [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
NS1_DROP_TOL = 1
NS2_SPREAD_TOL = 1e-9
FORBIDDEN_OUTDIRS = ["results", "results_v2", "results_v2_reid"]
STALL_PER_MODEL_S = 12 * 60


class ROCtx:
    def __init__(self, root):
        self.root = root

    def p(self, *parts):
        return os.path.join(self.root, *parts)


def _assert_outdir_safe(out_dir):
    rp = os.path.realpath(out_dir)
    for bad in FORBIDDEN_OUTDIRS:
        if rp == os.path.realpath(os.path.join(EXP, bad)) or rp == os.path.realpath(bad):
            print("[note]".format(out_dir, bad))
            sys.exit(3)
    return True


class _Stall(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Stall("[note]".format(STALL_PER_MODEL_S))


# =====================================================================================

# =====================================================================================

def derive_subset_ids(pools):
    ids = sorted(f["id"] for f in pools["shared"])
    sub = ids[::SUBSET_STRIDE]
    assert len(sub) == SUBSET_N_EXPECT, "[note]".format(len(sub), SUBSET_N_EXPECT)
    return sub


def build_fixed_battery(reid_dir):
    """Analysis script for the merge-audit study."""
    ro = ROCtx(reid_dir)
    u = R.load_universe(ro)
    battery = R.build_probe_battery(u["pools"], u["templates"])
    sub = set(derive_subset_ids(u["pools"]))
    fixed = [it for it in battery if it[1] == "shared" and it[2] == "train" and it[0] in sub]
    assert len(fixed) == SUBSET_N_EXPECT * R.N_TRAIN_TEMPLATES, len(fixed)
    return fixed


# =====================================================================================

# =====================================================================================

def collect_model_stats(sd, battery):
    """Analysis script for the merge-audit study."""
    ns = R._torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ns["init_model"](0)
    model.load_state_dict(sd)
    model.to(dev).eval()
    vals_t = torch.arange(R.VAL0, R.VAL0 + R.N_VALS, device=dev)
    full, val = [], []
    by_len = {}
    for item in battery:
        by_len.setdefault(len(item[4]), []).append(item)
    with torch.no_grad():
        for L, items in sorted(by_len.items()):
            for i0 in range(0, len(items), R.EVAL_BATCH):
                chunk = items[i0:i0 + R.EVAL_BATCH]
                x = torch.tensor([c[4] for c in chunk], dtype=torch.long, device=dev)
                with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                    dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logits = model(x)[:, -1, :].float()
                full.extend(logits.std(dim=1, unbiased=False).tolist())
                val.extend(logits.index_select(1, vals_t).std(dim=1, unbiased=False).tolist())
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {"n_rows": len(full),
            "full_logit_std_median": round(R.median(full), 6),
            "full_logit_std_q10": round(R.quantile(full, 0.10), 6),
            "full_logit_std_q90": round(R.quantile(full, 0.90), 6),
            "val_logit_std_median": round(R.median(val), 6)}


def collect(ctx, reid_dir):
    battery = build_fixed_battery(reid_dir)
    os.makedirs(ctx.p("logitstats"), exist_ok=True)
    ro = ROCtx(reid_dir)
    for seed in REID_SEEDS:
        sd0 = R.load_sd_fp32(ro.p("ckpts", "base-s{}".format(seed), "final.pt"))
        jobs = [("s{}-base".format(seed), sd0)]
        for ti in range(8):
            sdA = R.load_sd_fp32(ro.p("ckpts", "driftA-s{}".format(seed),
                                      "tau{}.pt".format(ti)))
            sdB = R.load_sd_fp32(ro.p("ckpts", "driftB-s{}".format(seed),
                                      "tau{}.pt".format(ti)))
            ids = R._model_ids(seed, ti)
            jobs.append((ids["endA"], sdA))
            jobs.append((ids["endB"], sdB))
            jobs.append((ids["merge0.50"], R.lerp_sd(sdA, sdB, 0.5)))
            jobs.append((ids["disA0.50"], R.lerp_sd(sd0, sdA, 0.5)))
            jobs.append((ids["disB0.50"], R.lerp_sd(sd0, sdB, 0.5)))
        for model_id, sd in jobs:
            out_path = ctx.p("logitstats", "{}.json".format(model_id))
            if os.path.exists(out_path):
                ctx.ledger("collect-skip:" + model_id)
                continue
            signal.alarm(STALL_PER_MODEL_S)
            try:
                stats = collect_model_stats(sd, battery)
            finally:
                signal.alarm(0)
            stats["model_id"] = model_id
            stats["seed"] = seed
            R.write_json(out_path, stats)
            ctx.ledger("collect:" + model_id, force=True)
            ctx.fuse_check("collect")
    R.write_json(ctx.p("collect_manifest.json"), {
        "battery": "shared pool, sorted(fact_id)[::12] x train 8 templates",
        "n_rows_per_model": SUBSET_N_EXPECT * R.N_TRAIN_TEMPLATES,
        "n_models": 3 * (1 + 8 * 5), "seeds": REID_SEEDS})


# =====================================================================================

# =====================================================================================

def _load_stats(ctx):
    out = {}
    d = ctx.p("logitstats")
    if not os.path.isdir(d):
        return out
    for fn in os.listdir(d):
        if fn.endswith(".json"):
            s = json.load(open(os.path.join(d, fn)))
            out[s["model_id"]] = s
    return out


def _d_mult(mrows, drows, fids, s_m, s_d, t=1.0):
    return R.median([mrows[f]["m_top1"] / s_m - (t * drows[f]["m_top1"]) / s_d
                     for f in fids])


def analyze(ctx, reid_dir):
    ro = ROCtx(reid_dir)
    stats = _load_stats(ctx)
    need = ["s{}-base".format(s) for s in REID_SEEDS] + \
        [R._model_ids(s, ti)[k] for s in REID_SEEDS for ti in range(8)
         for k in ("endA", "endB", "merge0.50", "disA0.50", "disB0.50")]
    missing = [m for m in need if m not in stats]
    cells = {}
    for seed in REID_SEEDS:
        for ti in range(8):
            ids = R._model_ids(seed, ti)
            M = {}
            for k in ("endA", "endB", "merge0.50", "disA0.50", "disB0.50"):
                M[k] = R.load_margins(ro, ids[k])
            for br in "AB":
                emeta, erows = M["end" + br]
                L = [fid for fid, r in erows["train"].items()
                     if r["set"] == "excl_" + br and R._alive(r, emeta)]
                mmeta, mrows = M["merge0.50"]
                dmeta, drows = M["dis{}0.50".format(br)]
                cell = {"n_live": len(L), "D": {}}
                for norm in ("N1", "N2", "N3"):
                    if norm == "N1":
                        s_m = abs(mmeta["ghost_median_top1_train"])
                        s_d = abs(dmeta["ghost_median_top1_train"])
                    elif norm == "N2":
                        if missing:
                            continue
                        s_m = s_d = stats["s{}-base".format(seed)]["full_logit_std_median"]
                    else:
                        if missing:
                            continue
                        s_m = stats[ids["merge0.50"]]["full_logit_std_median"]
                        s_d = stats[ids["dis{}0.50".format(br)]]["full_logit_std_median"]
                    cell["D"][norm] = round(_d_mult(mrows["train"], drows["train"], L,
                                                    s_m, s_d), 6)
                cells["s{}-t{}-{}".format(seed, ti, br)] = cell

    ns1 = {"deep_ti": DEEP_TI, "per_normalizer": {}}
    for norm in ("N1", "N2", "N3"):
        if norm != "N1" and missing:
            ns1["per_normalizer"][norm] = {"status": "awaiting_collect", "missing": len(missing)}
            continue
        n_dir = 0
        detail = {}
        for seed in REID_SEEDS:
            for ti in DEEP_TI:
                both = all(cells["s{}-t{}-{}".format(seed, ti, br)]["D"][norm] < 0
                           for br in "AB")
                detail["s{}-t{}".format(seed, ti)] = both
                n_dir += int(both)
        ns1["per_normalizer"][norm] = {"direction_cells_of_9": n_dir, "detail": detail}
    ns1["pass"] = None
    if "direction_cells_of_9" in ns1["per_normalizer"].get("N3", {}):
        ns1["pass"] = (ns1["per_normalizer"]["N3"]["direction_cells_of_9"]
                       >= ns1["per_normalizer"]["N1"]["direction_cells_of_9"] - NS1_DROP_TOL)

    ns2 = {"cells": {}}
    for seed in REID_SEEDS:
        for br in "AB":
            key = "s{}-t0-{}".format(seed, br)
            if missing:
                ns2["cells"][key] = {"status": "awaiting_collect"}
                continue
            ids = R._model_ids(seed, 0)
            _, mrows = R.load_margins(ro, ids["merge0.50"])
            _, drows = R.load_margins(ro, ids["dis{}0.50".format(br)])
            emeta, erows = R.load_margins(ro, ids["end" + br])
            L = [fid for fid, r in erows["train"].items()
                 if r["set"] == "excl_" + br and R._alive(r, emeta)]
            s3_m = stats[ids["merge0.50"]]["full_logit_std_median"]
            s3_d = stats[ids["dis{}0.50".format(br)]]["full_logit_std_median"]
            s2 = stats["s{}-base".format(seed)]["full_logit_std_median"]
            d_n3 = [_d_mult(mrows["train"], drows["train"], L, s3_m, t * s3_d, t=t)
                    for t in TGRID]
            d_n2 = [_d_mult(mrows["train"], drows["train"], L, s2, s2, t=t) for t in TGRID]
            ns2["cells"][key] = {
                "N3_spread": round(max(d_n3) - min(d_n3), 12),
                "N3_invariant_lt_1e-9": (max(d_n3) - min(d_n3)) < NS2_SPREAD_TOL,
                "N2_spread": round(max(d_n2) - min(d_n2), 8),
                "N2_sign_changes": len({x > 0 for x in d_n2}) > 1}
    ns2["N3_all_cells_invariant"] = all(
        c.get("N3_invariant_lt_1e-9") for c in ns2["cells"].values() if "N3_spread" in c) \
        if not missing else None
    rep = {"contract_id": CONTRACT_ID, "prereg": "PREREG-AUG.md AMENDMENT A-1.6",
           "normalizers": {"N1": "[note]",
                           "N2": "[note]",
                           "N3": "[note]"},
           "battery": "[note]",
           "stats_missing": missing, "cells_D": cells,
           "NS1": ns1, "NS2": ns2,
           "NS3": "[note]"
                  "[note]",
           "status": "complete" if not missing else "awaiting_collect"}
    R.write_json(ctx.p("normalizer_sweep.json"), rep)
    return rep


# =====================================================================================
# dry-run
# =====================================================================================

def dry_run(reid_dir, out_dir):
    ok, fails = [], []

    def check(name, fn):
        try:
            r = fn()
            (ok if r else fails).append(name)
            print("  [{}] {}".format("PASS" if r else "FAIL", name))
        except Exception as e:
            fails.append(name)
            print("  [FAIL] {} — {}: {}".format(name, type(e).__name__, str(e)[:300]))

    print("[note]")
    check("py_compile self", lambda: subprocess.run(
        [sys.executable, "-m", "py_compile", os.path.abspath(__file__)]).returncode == 0)
    check("[note]",
          lambda: (SUBSET_STRIDE, SUBSET_N_EXPECT, DEEP_TI, NS1_DROP_TOL, NS2_SPREAD_TOL)
          == (12, 500, [5, 6, 7], 1, 1e-9)
          and TGRID == [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00])

    def _subset():
        ro = ROCtx(reid_dir)
        u = R.load_universe(ro)
        sub = derive_subset_ids(u["pools"])
        bat = build_fixed_battery(reid_dir)
        assert len(sub) == 500 and len(bat) == 4000
        assert len({it[0] for it in bat}) == 500
        return True
    check("[note]", _subset)

    def _frozen_present():
        ro = ROCtx(reid_dir)
        n = 0
        for seed in REID_SEEDS:
            for ti in range(8):
                for k in ("endA", "endB", "merge0.50", "disA0.50", "disB0.50"):
                    meta, facts = R.load_margins(ro, R._model_ids(seed, ti)[k])
                    assert "ghost_median_top1_train" in meta
                    n += 1
        assert n == 120, n
        return True
    check("[note]", _frozen_present)

    def _invariance_logic():
        """Analysis script for the merge-audit study."""
        m = {"f1": {"m_top1": 2.0}, "f2": {"m_top1": 4.0}}
        d = {"f1": {"m_top1": 1.0}, "f2": {"m_top1": 3.0}}
        s_m, s_d = 10.0, 5.0
        d_n3 = [_d_mult(m, d, ["f1", "f2"], s_m, t * s_d, t=t) for t in TGRID]
        assert max(d_n3) - min(d_n3) < 1e-9, d_n3
        d_n2 = [_d_mult(m, d, ["f1", "f2"], 7.0, 7.0, t=t) for t in TGRID]
        assert max(d_n2) - min(d_n2) > 0
        return True
    check("[note]", _invariance_logic)

    def _ns1_logic():
        """Analysis script for the merge-audit study."""
        cells = {}
        for seed in (1, 2, 3):
            for ti in DEEP_TI:
                for br in "AB":
                    cells["s{}-t{}-{}".format(seed, ti, br)] = {"D": {"N1": -0.1, "N3": -0.2}}
        n1 = sum(1 for s in (1, 2, 3) for ti in DEEP_TI
                 if all(cells["s{}-t{}-{}".format(s, ti, b)]["D"]["N1"] < 0 for b in "AB"))
        n3 = sum(1 for s in (1, 2, 3) for ti in DEEP_TI
                 if all(cells["s{}-t{}-{}".format(s, ti, b)]["D"]["N3"] < 0 for b in "AB"))
        assert n1 == 9 and n3 == 9 and (n3 >= n1 - NS1_DROP_TOL)
        return True
    check("[note]", _ns1_logic)
    check("[note]",
          lambda: all(_rejects(d) for d in FORBIDDEN_OUTDIRS)
          and _assert_outdir_safe("results/normsweep"))
    check("[note]", lambda: R.FUSE_GPUH == SWEEP_FUSE_GPUH)

    def _contract():
        c = json.load(open(os.path.join(EXP, "contract-realmerge-draft.json")))
        assert "normalizer_sweep.py --collect" in c["command"]
        assert any("normsweep" in o["path"] for o in c["expected_outputs"])
        return True
    check("[note]", _contract)
    print("== dry-run: {} PASS / {} FAIL ==".format(len(ok), len(fails)))
    return 0 if not fails else 1


def _rejects(d):
    try:
        _assert_outdir_safe(d)
        return False
    except SystemExit as e:
        return e.code == 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reid-dir", default=os.path.join(EXP, "results_v2_reid"))
    ap.add_argument("--out-dir", default="results/normsweep")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        sys.exit(dry_run(os.path.normpath(args.reid_dir), args.out_dir))
    _assert_outdir_safe(args.out_dir)
    signal.signal(signal.SIGALRM, _alarm_handler)
    ctx = R.Ctx(args.out_dir)
    ctx.ledger("normsweep", "enter", force=True)
    try:
        if args.collect:
            collect(ctx, os.path.normpath(args.reid_dir))
        else:
            analyze(ctx, os.path.normpath(args.reid_dir))
    finally:
        ctx.ledger("normsweep", "exit", force=True)


if __name__ == "__main__":
    main()
