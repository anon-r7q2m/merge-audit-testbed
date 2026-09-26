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


AUG3_SEEDS = [3184, 5383, 7192]
_ENV_SEEDS = "3184,5383,7192"
if os.environ.get("TANH_R2RANK08_SEEDS"):
    assert os.environ["TANH_R2RANK08_SEEDS"] == _ENV_SEEDS, \
        "[note]".format(
            os.environ["TANH_R2RANK08_SEEDS"], _ENV_SEEDS)
else:
    os.environ["TANH_R2RANK08_SEEDS"] = _ENV_SEEDS

import run_merge_audit as R                                   # noqa: E402


AUG3_CONTRACT_ID = os.environ.get("TANH_AUG3_CONTRACT_ID", "merge_audit-aug3seeds-v1")
AUG3_PREREG_PATH = "PREREG-AUG3.md"
R.PREREG_PATH = AUG3_PREREG_PATH
R.CONTRACT_ID = AUG3_CONTRACT_ID

AUG3_FUSE_GPUH = 16.0
R.FUSE_GPUH = AUG3_FUSE_GPUH

REID_DIR = "results_v2_reid"
FORBIDDEN_OUTDIRS = ["results", "results_v2", REID_DIR]
PRIOR_SIX_SEEDS = {17, 42, 1017, 1042, 2017, 2042}


def _assert_outdir_safe(out_dir):
    """Analysis script for the merge-audit study."""
    rp = os.path.realpath(out_dir)
    for bad in FORBIDDEN_OUTDIRS:
        if rp == os.path.realpath(bad):
            print("[note]"
                  .format(out_dir, bad))
            sys.exit(3)
    return True


def _derive_aug3_seeds():
    """Analysis script for the merge-audit study."""
    rng = R.drng("merge_audit-aug", "seed-rotation", "v1")
    out = []
    while len(out) < 3:
        v = rng.randrange(100, 9000)
        if v in PRIOR_SIX_SEEDS or v in out:
            continue
        out.append(v)
    return out


def _facts_bytes(path_gz):
    with gzip.open(path_gz, "rb") as f:
        return f.read()


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
        schema_detail = {"sample_new": margin_files[0], "meta_keys_equal": m_new == m_ref,
                         "row_keys_equal": r_new == r_ref}

    gate = R.read_json(ctx.p("margin_gate.json")) if os.path.exists(ctx.p("margin_gate.json")) else {}
    ladder_len = len(gate.get("ladder", []))
    expected = 267 + 2 * ladder_len
    iso = {"facts_decompressed_byte_identical": facts_identical,
           "margin_schema_same_instrument": schema_ok, "schema_detail": schema_detail,
           "margin_file_count": len(margin_files), "expected_file_count": expected,
           "k_final": gate.get("k_final"), "gate_pass": gate.get("pass")}
    iso["pass"] = facts_identical and schema_ok and (len(margin_files) == expected)
    manifest = {"contract_id": AUG3_CONTRACT_ID, "status": "complete" if iso["pass"] else "isomorphism_fail",
                "train_seeds": AUG3_SEEDS, "isomorphism": iso,
                "files": {f: R.sha256_file(os.path.join(eval_dir, f)) for f in margin_files},
                "gate_summary": {"ladder": gate.get("ladder"), "k_final": gate.get("k_final")},
                "note": "[note]"}
    R.write_json(ctx.p("aug3_manifest.json"), manifest)
    if not iso["pass"]:
        print("[note]", json.dumps(iso, ensure_ascii=False))
        sys.exit(3)
    ctx.mark_done("manifest")


def phase_all_aug3(ctx):
    """Analysis script for the merge-audit study."""
    R.phase_universe(ctx)
    R.phase_mb(ctx)
    R.run_pool(ctx, [["--phase", "base", "--seed", str(s)] for s in R.TRAIN_SEEDS], "base")
    R.phase_gate(ctx)
    ctx.ledger("phase0_done", force=True)
    expo = [["--phase", "expose", "--seed", str(s), "--branch", b]
            for s in R.TRAIN_SEEDS[1:] for b in ["A", "B"]]
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
          lambda: _derive_aug3_seeds() == AUG3_SEEDS)
    check("[note]",
          lambda: not (set(AUG3_SEEDS) & PRIOR_SIX_SEEDS))
    check("[note]",
          lambda: R.TRAIN_SEEDS == AUG3_SEEDS)
    check("[note]", lambda: R.UNIVERSE_SEED == 20260814)
    check("[note]",
          lambda: R.PREREG_PATH == AUG3_PREREG_PATH)
    check("[note]",
          lambda: R.CONTRACT_ID == AUG3_CONTRACT_ID)
    check("[note]",
          lambda: all(_rejects(d) for d in FORBIDDEN_OUTDIRS))
    check("[note]",
          lambda: _assert_outdir_safe("results_v2_aug3"))
    check("[note]", lambda: R.FUSE_GPUH == AUG3_FUSE_GPUH)
    check("[note]",
          lambda: os.path.exists(os.path.join(EXP, AUG3_PREREG_PATH)))

    def _amendment_redirect_demo():
        """Analysis script for the merge-audit study."""
        with tempfile.TemporaryDirectory(prefix="aug3_dryrun_") as td:
            with open(os.path.join(td, AUG3_PREREG_PATH), "w") as f:
                f.write("# stub\n")
            ctx = R.Ctx(os.path.join(td, "out"))
            cwd = os.getcwd()
            os.chdir(td)
            try:
                ctx.amendment("[note]")
            finally:
                os.chdir(cwd)
            body = open(os.path.join(td, AUG3_PREREG_PATH)).read()
            assert "[note]" in body, "[note]"
            assert not os.path.exists(os.path.join(td, "PREREG-v2.md")), \
                "[note]"
            assert os.path.exists(os.path.join(td, "out", "amendments_auto.jsonl"))
            return True
    check("[note]", _amendment_redirect_demo)

    def _idem():
        with tempfile.TemporaryDirectory(prefix="aug3_dryrun_") as td:
            ctx = R.Ctx(os.path.join(td, "out"))
            assert not ctx.done("base-s3184")
            ctx.mark_done("base-s3184")
            assert ctx.done("base-s3184")
            return True
    check("[note]", _idem)

    def _phase_plan():
        """Analysis script for the merge-audit study."""
        import inspect
        src = inspect.getsource(phase_all_aug3)
        for banned in ["phase_lmc", "phase_repair", "phase_rollup",
                       '"--phase", "lmc"', '"--phase", "repair"']:
            assert banned not in src, "[note]" + banned
        assert 3 * len(R.TAU_GRID_STEPS) == 24
        assert 267 + 2 * 1 == 269 and 267 + 2 * 3 == 273
        return True
    check("[note]", _phase_plan)

    def _contract():
        c = json.load(open(os.path.join(EXP, "contract-aug3seeds-draft.json")))
        assert "run_merge_audit_aug3.py" in c["command"] and "results_v2_aug3" in c["command"]
        assert c["draft"] is True
        cid = c["contract_id"]
        if cid != AUG3_CONTRACT_ID:
            print("[note]"
                  "[note]".format(cid, AUG3_CONTRACT_ID))
        assert cid.replace("-draft", "") == AUG3_CONTRACT_ID
        for p in ["results_v2_aug3/margin_gate.json", "results_v2_aug3/aug3_manifest.json"]:
            assert any(o["path"] == p for o in c["expected_outputs"]), p
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
    ap.add_argument("--out-dir", default="results_v2_aug3")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        sys.exit(dry_run())
    _assert_outdir_safe(args.out_dir)
    assert _derive_aug3_seeds() == AUG3_SEEDS, "[note]"
    ctx = R.Ctx(args.out_dir)
    ctx.ledger("aug3", "enter", force=True)
    try:
        phase_all_aug3(ctx)
    finally:
        ctx.ledger("aug3", "exit", force=True)


if __name__ == "__main__":
    main()
