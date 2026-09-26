#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)
sys.path.insert(0, HERE)

import run_merge_audit as R                                   # noqa: E402
import run_merge_audit_freshctrl as F                           # noqa: E402


_ORIG_MAKE_PACKERS = F.make_packers


def make_packers_clean(ctx, seed):
    """Analysis script for the merge-audit study."""
    u = R.load_universe(ctx)
    pools, templates = u["pools"], u["templates"]
    stream = R.SentenceStream("ccbase:{}".format(seed), [], templates,
                              None, pools["filler"], 0)
    pack_base = R.RowPacker(stream)
    _, pack_exp = _ORIG_MAKE_PACKERS(ctx, seed)
    return pack_base, pack_exp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/cleanctrl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert args.out_dir != "results/freshctrl", "[note]"
    F.make_packers = make_packers_clean
    sys.argv = [sys.argv[0], "--out-dir", args.out_dir] + (
        ["--dry-run"] if args.dry_run else [])
    F.main()


if __name__ == "__main__":
    main()
