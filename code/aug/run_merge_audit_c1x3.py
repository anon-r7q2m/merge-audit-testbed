#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import gzip
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)


C1X3_SEEDS = [6784, 403, 1189]
_ENV_SEEDS = "6784,403,1189"
if os.environ.get("TANH_R2RANK08_SEEDS"):
    assert os.environ["TANH_R2RANK08_SEEDS"] == _ENV_SEEDS, \
        "[note]".format(_ENV_SEEDS)
else:
    os.environ["TANH_R2RANK08_SEEDS"] = _ENV_SEEDS

import run_merge_audit as R                                   # noqa: E402

C1X3_CONTRACT_ID = os.environ.get("TANH_C1X3_CONTRACT_ID",
                                  "merge_audit-aug-c1x3-v1")
C1X3_PREREG_PATH = "PREREG-C1X3.md"
R.PREREG_PATH = C1X3_PREREG_PATH
R.CONTRACT_ID = C1X3_CONTRACT_ID
C1X3_FUSE_GPUH = 16.0
R.FUSE_GPUH = C1X3_FUSE_GPUH

K_PINNED = 128
REID_DIR = "results_v2_reid"
FORBIDDEN_OUTDIRS = ["results", "results_v2", REID_DIR, "results_v2_aug3"]
ALL_PRIOR_SEEDS = {17, 42, 1017, 1042, 2017, 2042, 3184, 5383, 7192}


def _assert_outdir_safe(out_dir):
    rp = os.path.realpath(out_dir)
    for bad in FORBIDDEN_OUTDIRS:
        if rp == os.path.realpath(bad):
            print("[note]".format(out_dir, bad))
            sys.exit(3)
    return True


def _derive_c1x3_seeds():
    rng = R.drng("merge_audit-aug", "seed-rotation", "v2")
    out = []
    while len(out) < 3:
        v = rng.randrange(100, 9000)
        if v in ALL_PRIOR_SEEDS or v in out:
            continue
        out.append(v)
    return out


def _facts_bytes(path_gz):
    with gzip.open(path_gz, "rb") as f:
        return f.read()


def phase_k_pin(ctx):
    """Analysis script for the merge-audit study."""
    if ctx.done("k_pin"):
        return
    R.atomic_write(ctx.p("state", "k_final"), str(K_PINNED).encode())
    ctx.ledger("k_pin", "K pinned at 128 (campaign-1 dose; health reported, not gated)",
               force=True)
    ctx.mark_done("k_pin")


def phase_manifest(ctx):
    """Analysis script for the merge-audit study."""
    if ctx.done("manifest"):
        return
    ours = _facts_bytes(ctx.p("universe", "facts.json.gz"))
    reid = _facts_bytes(os.path.join(REID_DIR, "universe", "facts.json.gz"))
    facts_identical = (ours == reid)
    eval_dir = ctx.p("eval")
    margin_files = sorted(f for f in os.listdir(eval_dir)
                          if f.startswith("margins.") and f.endswith(".jsonl.gz"))
    schema_ok, schema_detail = False, {}
    if margin_files:
        def _keys(path):
            meta_keys, row_keys = set(), set()
            with gzip.open(path, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if "__meta__" in r:
                        meta_keys = set(r["__meta__"].keys())
                    else:
                        row_keys = set(r.keys())
                        break
            return meta_keys, row_keys
        m_new, r_new = _keys(os.path.join(eval_dir, margin_files[0]))
        m_ref, r_ref = _keys(os.path.join(REID_DIR, "eval",
                                          "margins.s42-t0-a0.50-merge.jsonl.gz"))
        r_new.discard("per_t_top1")
        r_ref.discard("per_t_top1")
        schema_ok = (m_new == m_ref) and (r_new == r_ref)
        schema_detail = {"sample_new": margin_files[0],
                         "meta_keys_equal": m_new == m_ref,
                         "row_keys_equal": r_new == r_ref}

    health = {}
    battery = None
    for seed in C1X3_SEEDS:
        for br in ("A", "B"):
            mid = "s{}-t0-end{}-k{}".format(seed, br, K_PINNED)
            path = os.path.join(eval_dir, "margins.{}.jsonl.gz".format(mid))
            if not os.path.exists(path):
                health[mid] = None
                continue
            meta, facts = R.load_margins(ctx, mid)
            own = "excl_{}".format(br)
            rows = [r for r in facts["train"].values() if r["set"] == own]
            frac = sum(1 for r in rows if r["rank_bar"] <= R.RANK1_BAR_THRESH) \
                / float(max(1, len(rows)))
            health[mid] = round(frac, 4)
    iso = {"facts_decompressed_byte_identical": facts_identical,
           "margin_schema_same_instrument": schema_ok,
           "schema_detail": schema_detail,
           "margin_file_count": len(margin_files),
           "k_final": K_PINNED}
    iso["pass"] = facts_identical and schema_ok
    manifest = {"contract_id": C1X3_CONTRACT_ID,
                "status": "complete" if iso["pass"] else "isomorphism_fail",
                "train_seeds": C1X3_SEEDS, "isomorphism": iso,
                "storage_health_at_K128_frac_rank1": health,
                "health_note": "[note]"
                               "[note]",
                "files": {f: R.sha256_file(os.path.join(eval_dir, f))
                          for f in margin_files},
                "note": "[note]"}
    R.write_json(ctx.p("c1x3_manifest.json"), manifest)
    if not iso["pass"]:
        print("[note]", json.dumps(iso, ensure_ascii=False))
        sys.exit(3)
    ctx.mark_done("manifest")


def phase_all_c1x3(ctx):
    """Analysis script for the merge-audit study."""
    R.phase_universe(ctx, k_exposure=K_PINNED)
    R.phase_mb(ctx)
    R.run_pool(ctx, [["--phase", "base", "--seed", str(s)] for s in R.TRAIN_SEEDS],
               "base")
    phase_k_pin(ctx)
    ctx.ledger("phase0_done", force=True)
    expo = [["--phase", "expose", "--seed", str(s), "--branch", b]
            for s in R.TRAIN_SEEDS for b in ["A", "B"]]
    R.run_pool(ctx, expo, "expose")
    R.run_pool(ctx, [["--phase", "drift", "--seed", str(s), "--branch", b]
                     for s in R.TRAIN_SEEDS for b in ["A", "B"]], "drift")
    shards = [["--phase", "merge_eval", "--seed", str(s), "--tau-idx", str(t)]
              for s in R.TRAIN_SEEDS for t in range(len(R.TAU_GRID_STEPS))]
    R.run_pool(ctx, shards, "merge_eval")
    phase_manifest(ctx)


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
          lambda: _derive_c1x3_seeds() == C1X3_SEEDS)
    check("[note]",
          lambda: not (set(C1X3_SEEDS) & ALL_PRIOR_SEEDS))
    check("[note]",
          lambda: R.TRAIN_SEEDS == C1X3_SEEDS)
    check("[note]", lambda: R.UNIVERSE_SEED == 20260814)
    check("[note]",
          lambda: R.PREREG_PATH == C1X3_PREREG_PATH)
    check("[note]", lambda: R.CONTRACT_ID == C1X3_CONTRACT_ID)
    check("[note]", lambda: R.FUSE_GPUH == C1X3_FUSE_GPUH)
    check("[note]",
          lambda: all(_rejects(d) for d in FORBIDDEN_OUTDIRS))
    check("[note]",
          lambda: _assert_outdir_safe("results_v2_c1x3"))

    def _phase_plan():
        import inspect
        src = inspect.getsource(phase_all_c1x3)
        for banned in ["phase_lmc", "phase_repair", "phase_rollup",
                       '"--phase", "lmc"', '"--phase", "repair"']:
            assert banned not in src, "[note]" + banned
        assert "phase_gate" not in src, "[note]"
        return True
    check("[note]", _phase_plan)

    print("== dry-run {}: {} pass, {} fail ==".format(
        "OK" if not fails else "FAILED", len(ok), len(fails)))
    return not fails


def _rejects(d):
    try:
        _assert_outdir_safe(d)
        return False
    except SystemExit:
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results_v2_c1x3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _assert_outdir_safe(args.out_dir)
    if args.dry_run:
        sys.exit(0 if dry_run() else 3)
    os.makedirs(os.path.join(EXP, args.out_dir), exist_ok=True)
    if not os.path.exists(os.path.join(EXP, C1X3_PREREG_PATH)):
        with open(os.path.join(EXP, C1X3_PREREG_PATH), "w") as f:
            f.write("[note]")
    ctx = R.Ctx(os.path.join(EXP, args.out_dir))
    phase_all_c1x3(ctx)


if __name__ == "__main__":
    main()
