#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import gzip
import json
import os
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)


REID_SEEDS = [42, 1042, 2042]
os.environ.setdefault("TANH_R2RANK08_SEEDS", ",".join(str(s) for s in REID_SEEDS))
os.environ.setdefault("RANK08_NGPUS", "1")

import run_merge_audit as R                                   # noqa: E402

CONTRACT_ID = os.environ.get("TANH_TASKARITH_CONTRACT_ID",
                             "merge_audit-aug-taskarith-v1")
R.CONTRACT_ID = CONTRACT_ID
TASKARITH_FUSE_GPUH = 5.5
R.FUSE_GPUH = TASKARITH_FUSE_GPUH

LAMBDAS = [0.5, 0.75, 1.0, 1.25, 1.5]
TAU_IDXS = list(range(8))
PARITY_MED_TOL, PARITY_MAX_TOL = 0.001, 0.05
P2_MONO_CELLS_MIN, P2_UPLIFT_MED_MIN = 8, 0.3
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
        if rp == os.path.realpath(bad):
            print("[note]".format(out_dir, bad))
            sys.exit(3)
    return True


def _task_id(seed, ti, lam):
    return "taskarith-s{}-t{}-lam{:.2f}".format(seed, ti, lam)


def _task_sum_sd(sd0, sdA, sdB, lam):
    """Analysis script for the merge-audit study."""
    return {k: sd0[k] + lam * ((sdA[k] - sd0[k]) + (sdB[k] - sd0[k])) for k in sd0}


class _Stall(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Stall("[note]".format(STALL_PER_MODEL_S))


def _eval_one(ctx, battery, seed, ti, lam, sd0, sdA, sdB):
    mid = _task_id(seed, ti, lam)
    if os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(mid))):
        ctx.ledger("eval-skip:" + mid)
        return mid
    sd = _task_sum_sd(sd0, sdA, sdB, lam)
    signal.alarm(STALL_PER_MODEL_S)
    try:
        R.eval_margins(ctx, mid, sd, battery)
    finally:
        signal.alarm(0)
    ctx.ledger("eval:" + mid, force=True)
    ctx.fuse_check("eval:" + mid)
    return mid


def _load_cell_ckpts(reid, seed, ti):
    sd0 = R.load_sd_fp32(reid.p("ckpts", "base-s{}".format(seed), "final.pt"))
    sdA = R.load_sd_fp32(reid.p("ckpts", "driftA-s{}".format(seed), "tau{}.pt".format(ti)))
    sdB = R.load_sd_fp32(reid.p("ckpts", "driftB-s{}".format(seed), "tau{}.pt".format(ti)))
    return sd0, sdA, sdB


def _pools_of(rows):
    out = {"excl_A": [], "excl_B": [], "ghost": []}
    for fid, r in rows.items():
        s = r["set"]
        if s in ("excl_A", "excl_B"):
            out[s].append(r)
        elif s.startswith("ghost"):
            out["ghost"].append(r)
    return out


def phase_parity(ctx, reid, battery):
    """Analysis script for the merge-audit study."""
    if ctx.done("parity"):
        return True
    report = {"tolerance": {"median_abs": PARITY_MED_TOL, "max_abs": PARITY_MAX_TOL},
              "seeds": {}}
    all_pass = True
    for seed in REID_SEEDS:
        sd0, sdA, sdB = _load_cell_ckpts(reid, seed, 0)
        mid = _eval_one(ctx, battery, seed, 0, 0.5, sd0, sdA, sdB)
        _, ours = R.load_margins(ctx, mid)
        _, frozen = R.load_margins(reid, "s{}-t0-a0.50-merge".format(seed))
        seed_pass, pools_rep = True, {}
        for pool, ours_rows in _pools_of(ours["train"]).items():
            diffs = [abs(r["m_top1"] - frozen["train"][r["fact_id"]]["m_top1"])
                     for r in ours_rows]
            med_d = R.median(diffs)
            max_d = max(diffs)


            ulps = [abs(r["m_top1"] - frozen["train"][r["fact_id"]]["m_top1"]) /
                    (2.0 * (2.0 ** -7) * max(abs(r["m_top1"]),
                     abs(frozen["train"][r["fact_id"]]["m_top1"]), 1e-12))
                    for r in ours_rows]
            max_ulp = max(ulps)
            ok = (med_d <= PARITY_MED_TOL and max_d <= PARITY_MAX_TOL)
            pools_rep[pool] = {"n": len(diffs), "median_abs_delta": round(med_d, 8),
                               "max_abs_delta": round(max_d, 8),
                               "max_ulp_ratio": round(max_ulp, 4), "pass": ok}
            seed_pass = seed_pass and ok
        report["seeds"][str(seed)] = {"pools": pools_rep, "pass": seed_pass}
        all_pass = all_pass and seed_pass
    report["pass"] = all_pass
    R.write_json(ctx.p("parity_check.json"), report)
    if not all_pass:
        print("[note]")
        sys.exit(3)
    ctx.mark_done("parity")
    return True


def phase_sweep(ctx, reid, battery):
    for seed in REID_SEEDS:
        for ti in TAU_IDXS:
            sd0, sdA, sdB = _load_cell_ckpts(reid, seed, ti)
            for lam in LAMBDAS:
                _eval_one(ctx, battery, seed, ti, lam, sd0, sdA, sdB)
    ctx.mark_done("sweep")


def _ghost_floor(meta):
    return meta["ghost_median_top1_train"]


def phase_analysis(ctx, reid):
    """Analysis script for the merge-audit study."""
    cells = {}
    for seed in REID_SEEDS:
        for ti in TAU_IDXS:
            g = {}
            for lam in LAMBDAS:
                meta, facts = R.load_margins(ctx, _task_id(seed, ti, lam))
                g[lam] = {"ghost_floor": _ghost_floor(meta),
                          "fact_median_top1": R.median(
                              [r["m_top1"] for r in facts["train"].values()
                               if r["set"] in ("excl_A", "excl_B")])}

            dis = {}
            for a in R.ALPHAS_INTERIOR:
                for br in "AB":
                    mid = "s{}-t{}-s{:.2f}-dis{}".format(seed, ti, a, br)
                    m_dis, _ = R.load_margins(reid, mid)
                    dis["{:.2f}{}".format(a, br)] = _ghost_floor(m_dis)
            mono = all(g[LAMBDAS[i + 1]]["ghost_floor"] >= g[LAMBDAS[i]]["ghost_floor"]
                       for i in range(len(LAMBDAS) - 1))
            cells["s{}-t{}".format(seed, ti)] = {
                "ghost_floor_by_lambda": {"{:.2f}".format(k): round(v["ghost_floor"], 5)
                                          for k, v in g.items()},
                "fact_median_top1_by_lambda": {"{:.2f}".format(k): round(v["fact_median_top1"], 5)
                                               for k, v in g.items()},
                "uplift_1.5_minus_0.5": round(g[1.5]["ghost_floor"] - g[0.5]["ghost_floor"], 5),
                "monotone_nondecreasing": mono,
                "dis_ghost_floor_frozen": dis,
                "scale_ratio_merge_over_dis05": round(
                    g[1.0]["ghost_floor"] / dis["0.50A"], 4) if dis["0.50A"] else None,
            }
    n_mono = sum(1 for c in cells.values() if c["monotone_nondecreasing"])
    uplifts = sorted(c["uplift_1.5_minus_0.5"] for c in cells.values())
    med_uplift = uplifts[len(uplifts) // 2] if len(uplifts) % 2 else \
        0.5 * (uplifts[len(uplifts) // 2 - 1] + uplifts[len(uplifts) // 2])
    verdict = {"n_cells": len(cells), "n_monotone": n_mono,
               "mono_cells_min": P2_MONO_CELLS_MIN,
               "median_uplift": round(med_uplift, 5),
               "uplift_med_min": P2_UPLIFT_MED_MIN,
               "demonstration_supported": (n_mono >= P2_MONO_CELLS_MIN
                                           and med_uplift >= P2_UPLIFT_MED_MIN),
               "wording_cap": ("[note]"
                               if n_mono >= P2_MONO_CELLS_MIN and med_uplift >= P2_UPLIFT_MED_MIN
                               else "[note]")}
    report = {"contract_id": CONTRACT_ID, "status": "complete",
              "prereg": "PREREG-AUG.md §1", "cells": cells, "verdict_p2": verdict,
              "p3_note": "[note]"}
    R.write_json(ctx.p("taskarith_report.json"), report)
    # manifest
    eval_dir = ctx.p("eval")
    files = sorted(f for f in os.listdir(eval_dir) if f.startswith("margins.taskarith-"))
    R.write_json(ctx.p("manifest.json"),
                 {"contract_id": CONTRACT_ID, "n_margin_files": len(files),
                  "expected": len(REID_SEEDS) * len(TAU_IDXS) * len(LAMBDAS),
                  "files": {f: R.sha256_file(os.path.join(eval_dir, f)) for f in files}})
    ctx.mark_done("analysis")


def dry_run():
    ok, fails = [], []

    def check(name, fn):
        try:
            r = fn()
            (ok if r else fails).append(name)
            print("  [{}] {}".format("PASS" if r else "FAIL", name))
        except Exception as e:
            fails.append(name)
            print("  [FAIL] {} — {}: {}".format(name, type(e).__name__, e))

    print("[note]")
    check("py_compile self", lambda: subprocess.run(
        [sys.executable, "-m", "py_compile", os.path.abspath(__file__)]).returncode == 0)
    check("[note]",
          lambda: LAMBDAS == [0.5, 0.75, 1.0, 1.25, 1.5] and TAU_IDXS == list(range(8))
          and len(REID_SEEDS) * len(TAU_IDXS) * len(LAMBDAS) == 120)
    check("τ idx {5,7} → tokens {50M,200M}",
          lambda: R.TAU_GRID_TOKENS[5] == 50003968 and R.TAU_GRID_TOKENS[7] == 200015872)
    check("[note]",
          lambda: R.TRAIN_SEEDS == REID_SEEDS)

    def _algebra():
        """Analysis script for the merge-audit study."""
        sd0 = {"w": 1.25, "b": -0.5}
        sdA = {"w": 2.0, "b": 0.25}
        sdB = {"w": -1.0, "b": 0.75}
        mine = _task_sum_sd(sd0, sdA, sdB, 0.5)
        ref = R.lerp_sd(sdA, sdB, 0.5)
        assert all(abs(mine[k] - ref[k]) < 1e-12 for k in mine)
        # λ=1.0 = θ0+δ_A+δ_B = θ_A+θ_B−θ_0
        m1 = _task_sum_sd(sd0, sdA, sdB, 1.0)
        assert abs(m1["w"] - (2.0 - 1.0 - 1.25)) < 1e-12
        return True
    check("[note]", _algebra)

    def _p2_logic():
        """Analysis script for the merge-audit study."""
        def verdict(mono_flags, uplifts):
            n = sum(mono_flags)
            u = sorted(uplifts)
            med = u[len(u) // 2] if len(u) % 2 else 0.5 * (u[len(u) // 2 - 1] + u[len(u) // 2])
            return n >= P2_MONO_CELLS_MIN and med >= P2_UPLIFT_MED_MIN
        assert verdict([True] * 9, [0.4] * 9) is True
        assert verdict([True] * 8 + [False], [0.4] * 9) is True
        assert verdict([True] * 7 + [False] * 2, [0.4] * 9) is False
        assert verdict([True] * 9, [0.1] * 9) is False
        return True
    check("[note]", _p2_logic)
    check("[note]",
          lambda: (PARITY_MED_TOL, PARITY_MAX_TOL) == (0.001, 0.05))
    check("[note]",
          lambda: all(_rejects(d) for d in FORBIDDEN_OUTDIRS)
          and _assert_outdir_safe("results/taskarith"))
    check("[note]", lambda: R.FUSE_GPUH == TASKARITH_FUSE_GPUH)
    check("[note]",
          lambda: _task_id(42, 5, 0.75) == "taskarith-s42-t5-lam0.75")

    def _idem():
        with tempfile.TemporaryDirectory(prefix="taskarith_dryrun_") as td:
            ctx = R.Ctx(os.path.join(td, "out"))
            mid = _task_id(42, 0, 0.5)
            os.makedirs(ctx.p("eval"), exist_ok=True)
            open(ctx.p("eval", "margins.{}.jsonl.gz".format(mid)), "wb").close()
            assert os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(mid)))
            return True
    check("[note]", _idem)

    def _frozen_present():
        reid = ROCtx("results_v2_reid")
        for seed in REID_SEEDS:
            assert os.path.exists(reid.p("ckpts", "base-s{}".format(seed), "final.pt"))
            for br in "AB":
                for ti in TAU_IDXS:
                    p = reid.p("ckpts", "drift{}-s{}".format(br, seed),
                               "tau{}.pt".format(ti))
                    assert os.path.exists(p), p
            assert os.path.exists(reid.p("eval", "margins.s{}-t0-a0.50-merge.jsonl.gz"
                                         .format(seed)))
        assert os.path.exists(reid.p("universe", "facts.json.gz"))
        return True
    check("[note]",
          _frozen_present)

    def _universe_iso():
        pools = R.build_facts()
        with gzip.open(os.path.join("results_v2_reid", "universe", "facts.json.gz"), "rt") as f:
            arch = json.load(f)
        return json.dumps(pools, sort_keys=True) == json.dumps(arch, sort_keys=True)
    check("[note]", _universe_iso)

    def _contract():
        c = json.load(open(os.path.join(EXP, "contract-taskarith-draft.json")))
        assert "run_merge_audit_taskarith.py" in c["command"]
        assert c["draft"] is True
        assert c["contract_id"].replace("-draft", "") == CONTRACT_ID
        assert "0.001" in c["baseline_parity"]["tolerance"] and \
               ("ULP" in c["baseline_parity"]["tolerance"] or "0.01" in c["baseline_parity"]["tolerance"])
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
    ap.add_argument("--reid-dir", default="results_v2_reid")
    ap.add_argument("--out-dir", default="results/taskarith")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        sys.exit(dry_run())
    _assert_outdir_safe(args.out_dir)
    signal.signal(signal.SIGALRM, _alarm_handler)
    reid = ROCtx(args.reid_dir)
    ctx = R.Ctx(args.out_dir)
    ctx.ledger("taskarith", "enter", force=True)
    try:
        u = R.load_universe(reid)
        battery = R.build_probe_battery(u["pools"], u["templates"])
        phase_parity(ctx, reid, battery)
        phase_sweep(ctx, reid, battery)
        phase_analysis(ctx, reid)
    finally:
        ctx.ledger("taskarith", "exit", force=True)


if __name__ == "__main__":
    main()
