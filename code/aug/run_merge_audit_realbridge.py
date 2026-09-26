#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import csv
import gzip
import json
import math
import os
import signal
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

os.environ.setdefault("RANK08_NGPUS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import run_merge_audit as R

CONTRACT_ID = os.environ.get("TANH_REALBRIDGE_CONTRACT_ID",
                             "merge_audit-aug-realbridge-v1")
R.CONTRACT_ID = CONTRACT_ID
REALBRIDGE_FUSE_GPUH = 1.3
R.FUSE_GPUH = REALBRIDGE_FUSE_GPUH


LAMBDAS_TIER1 = [0.25, 0.5, 0.75, 1.25, 1.5]
TGRID = [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
PROBE_N_EXPECT, PROBE_N_BAND = 1719, 0.05
CAND_N_EXPECT, CAND_N_BAND = 401, 0.05
PROP_N_EXPECT = 16
F_LIVE_FLOOR = 300
DELTA_NORM_BAND = (0.1, 10.0)
POPQA_SHA256 = "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b"
POPQA_SIZE = 5205200
SFT_BIN_SIZE = 2969891414
CODECMU_ST_SIZE = 2969854224
BASE_DIRNAME = "OLMo-2-0425-1B/stage1-step1907359-tokens4001B"
BASE_TOK_DIRNAME = "OLMo-2-0425-1B/_tokenizer"
SFT_DIRNAME = "OLMo-2-0425-1B-SFT"
FT1_DIRNAME = "OLMo-2-0425-1B_full_sft_code_data_120K"
FT2_DIRNAME = "OLMo-2-0425-1B_full_sft_natural_language_data_120K"
STAGING_RECORD = "PREFLIGHT-REALBRIDGE-20260826.json"
FORBIDDEN_OUTDIRS = ["results", "results_v2", "results_v2_reid"]
STALL_PER_MODEL_S = 10 * 60
EVAL_BATCH = 32


class _Stall(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Stall("[note]".format(STALL_PER_MODEL_S))


def _assert_outdir_safe(out_dir):
    rp = os.path.realpath(out_dir)
    for bad in FORBIDDEN_OUTDIRS:
        if rp == os.path.realpath(os.path.join(EXP, bad)) or \
           rp == os.path.realpath(bad):
            print("[note]".format(out_dir, bad))
            sys.exit(3)
    return True


# =====================================================================================

# =====================================================================================

def build_probe_battery(probe_file, tok):
    """Analysis script for the merge-audit study."""
    probes = []
    with open(probe_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            obj = row["obj"]
            ids = tok(" " + obj, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
            probes.append({"qid": row["id"], "question": row["question"],
                           "obj": obj, "prop": row["prop"], "val_tid": ids[0]})
    cand = sorted({p["val_tid"] for p in probes})
    props = {p["prop"] for p in probes}
    return probes, cand, len(props)


def _probe_band_ok(n_probes, n_cand, n_props):
    return (abs(n_probes - PROBE_N_EXPECT) <= PROBE_N_EXPECT * PROBE_N_BAND
            and abs(n_cand - CAND_N_EXPECT) <= CAND_N_EXPECT * CAND_N_BAND
            and n_props == PROP_N_EXPECT)


# =====================================================================================


# =====================================================================================

def margins_from_logits(last_logits, cand_ids, val_tid):
    """Analysis script for the merge-audit study."""
    torch = _torch()
    z = last_logits.float()
    cand_t = torch.tensor(cand_ids, dtype=torch.long, device=z.device)
    zc = z.index_select(0, cand_t)
    vi = cand_ids.index(val_tid)
    zv = zc[vi]
    rest = torch.cat([zc[:vi], zc[vi + 1:]])
    rank = int((rest > zv).sum().item()) + 1
    m_top1 = float(zv - rest.max().item())
    logp = float(torch.log_softmax(z, dim=0)[val_tid].item())
    return {"rank": rank, "m_top1": round(m_top1, 5), "logp": round(logp, 5),
            "cand_logit_std": round(float(zc.std(unbiased=False).item()), 5)}


_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def _load_hf(model_dir, tok_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch = _torch()
    tok = AutoTokenizer.from_pretrained(tok_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16)
    return model, tok


def _eval_model_on_probes(model, probes, cand_ids, batch=EVAL_BATCH):
    """Analysis script for the merge-audit study."""
    torch = _torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    tok = _TOK_CACHE.get("tok")
    out = {}
    by_len = {}
    for p in probes:
        ids = tok(p["question"], add_special_tokens=True)["input_ids"]
        by_len.setdefault(len(ids), []).append((ids, p))
    with torch.no_grad():
        for L, items in sorted(by_len.items()):
            for i0 in range(0, len(items), batch):
                chunk = items[i0:i0 + batch]
                x = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=dev)
                am = torch.ones_like(x)
                logits = model(input_ids=x, attention_mask=am).logits[:, -1, :].float()
                for j, (_, p) in enumerate(chunk):
                    out[p["qid"]] = margins_from_logits(logits[j], cand_ids, p["val_tid"])
    return out


_TOK_CACHE = {}


def _eval_and_shard(ctx, model, probes, cand_ids, shard_path, tag):
    """Analysis script for the merge-audit study."""
    if os.path.exists(shard_path):
        ctx.ledger("eval-skip:" + tag)
        return None
    signal.alarm(STALL_PER_MODEL_S)
    try:
        res = _eval_model_on_probes(model, probes, cand_ids)
    finally:
        signal.alarm(0)
    tmp = shard_path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        f.write(json.dumps({"__meta__": {"model": tag, "n_probes": len(res)}}) + "\n")
        for qid, r in res.items():
            f.write(json.dumps({"qid": qid, **r}) + "\n")
    os.replace(tmp, shard_path)
    ctx.ledger("eval:" + tag, force=True)
    ctx.fuse_check("eval")
    return res


def _load_shard(path):
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" not in r:
                out[r["qid"]] = r
    return out


# =====================================================================================

# =====================================================================================

def _sd_of(model):
    return {k: v.detach().cpu().float() for k, v in model.state_dict().items()}


def _sd_lerp_delta(sd0, sdX, lam):
    return {k: sd0[k] + lam * (sdX[k] - sd0[k]) for k in sd0}


def _sd_avg(sd1, sd2):
    return {k: 0.5 * (sd1[k] + sd2[k]) for k in sd1}


def _fro_sq(sd1, sd2):
    return sum(float((sd1[k] - sd2[k]).pow(2).sum()) for k in sd1)


def _load_sd_into(base_model, sd):
    torch = _torch()
    base_model.load_state_dict({k: v.to(torch.bfloat16) for k, v in sd.items()})
    return base_model


# =====================================================================================

# =====================================================================================

def t_star(d_by_t):
    """Analysis script for the merge-audit study."""
    for i in range(len(d_by_t) - 1):
        t0, d0 = d_by_t[i]
        t1, d1 = d_by_t[i + 1]
        if d0 == 0.0:
            return t0, "exact_zero"
        if (d0 < 0) != (d1 < 0):
            return t0 + (t1 - t0) * d0 / (d0 - d1), "crossing"
    d_last = d_by_t[-1][1]
    return None, ("positive" if d_last > 0 else "negative" if d_last < 0 else "exact_zero")


def _d_add_t(m_meas, m_ref, fids):
    out = []
    for t in TGRID:
        out.append((t, R.median([m_meas[f]["m_top1"] - t * m_ref[f]["m_top1"]
                                 for f in fids])))
    return out


# =====================================================================================

# =====================================================================================

def phase_preflight(model_root, probe_file):
    torch = _torch()
    rec_path = os.path.join(model_root, STAGING_RECORD)
    assert os.path.exists(rec_path), \
        "[note]".format(rec_path)
    rec = json.load(open(rec_path))
    checks = {"staging_record": STAGING_RECORD, "files": {}, "tier2_gate": {}}
    ok_all = True
    for item in rec["files"]:
        p = item["path"]
        e = {"exists": os.path.exists(p)}
        if e["exists"]:
            e["size"] = os.path.getsize(p)
            e["size_match"] = (e["size"] == item["size"])
            e["sha256_match"] = (R.sha256_file(p) == item["sha256"])
            e["pass"] = e["size_match"] and e["sha256_match"]
        else:
            e["pass"] = False
        checks["files"][os.path.basename(p)] = e
        ok_all = ok_all and e["pass"]

    e = {"sha256": R.sha256_file(probe_file), "size": os.path.getsize(probe_file)}
    e["pass"] = (e["sha256"] == POPQA_SHA256 and e["size"] == POPQA_SIZE)
    checks["files"]["popqa_test.tsv"] = e
    ok_all = ok_all and e["pass"]





    CONFIG_DIFF_EXCLUDE = ("_name_or_path", "transformers_version", "torch_dtype",
                           "use_cache")
    base_cfg = json.load(open(os.path.join(model_root, BASE_DIRNAME, "config.json")))
    sft_cfg = json.load(open(os.path.join(model_root, SFT_DIRNAME, "config.json")))
    cfg_keys = sorted(set(base_cfg) | set(sft_cfg))
    tier1_cfg_diff = {k: [base_cfg.get(k), sft_cfg.get(k)] for k in cfg_keys
                      if base_cfg.get(k) != sft_cfg.get(k)
                      and k not in CONFIG_DIFF_EXCLUDE}
    checks["tier1_config_diff"] = tier1_cfg_diff
    base_tok_b = open(os.path.join(model_root, BASE_TOK_DIRNAME, "tokenizer.json"), "rb").read()
    sft_tok_b = open(os.path.join(model_root, SFT_DIRNAME, "tokenizer.json"), "rb").read()
    checks["tier1_tokenizer_byte_identical"] = (base_tok_b == sft_tok_b)
    tier1_ok = (not tier1_cfg_diff) and checks["tier1_tokenizer_byte_identical"]

    gate = {}
    for tag, dn in [("ft1", FT1_DIRNAME), ("ft2", FT2_DIRNAME)]:
        d = os.path.join(model_root, dn)
        g = {}
        if not os.path.isdir(d):
            g["pass"] = False
            g["reason"] = "model dir missing"
            gate[tag] = g
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        diff = {k: [base_cfg.get(k), cfg.get(k)] for k in sorted(set(base_cfg) | set(cfg))
                if base_cfg.get(k) != cfg.get(k)
                and k not in CONFIG_DIFF_EXCLUDE}
        g["config_key_identical"] = (not diff)
        g["config_diff"] = diff
        tok_p = os.path.join(d, "tokenizer.json")
        g["tokenizer_byte_identical"] = os.path.exists(tok_p) and \
            open(tok_p, "rb").read() == base_tok_b
        gate[tag] = g

    sds = {}
    try:
        from transformers import AutoModelForCausalLM
        for tag, dn in [("base", BASE_DIRNAME), ("sft", SFT_DIRNAME),
                        ("ft1", FT1_DIRNAME), ("ft2", FT2_DIRNAME)]:
            if tag in ("ft1", "ft2") and not gate.get(tag, {}).get("config_key_identical"):
                continue
            m = AutoModelForCausalLM.from_pretrained(os.path.join(model_root, dn),
                                                     torch_dtype=torch.float32)
            sds[tag] = _sd_of(m)
            del m
        if "sft" in sds and "base" in sds:
            n_sft = math.sqrt(_fro_sq(sds["sft"], sds["base"]))
            for tag in ("ft1", "ft2"):
                if tag in sds:
                    n = math.sqrt(_fro_sq(sds[tag], sds["base"]))
                    gate[tag]["delta_norm"] = round(n, 3)
                    gate[tag]["delta_norm_ref_sft"] = round(n_sft, 3)
                    gate[tag]["delta_norm_in_band"] = \
                        DELTA_NORM_BAND[0] * n_sft <= n <= DELTA_NORM_BAND[1] * n_sft
        for tag in ("ft1", "ft2"):
            g = gate.get(tag, {})
            g["pass"] = bool(g.get("config_key_identical")
                             and g.get("tokenizer_byte_identical")
                             and g.get("delta_norm_in_band"))
            if not g["pass"]:
                g.setdefault("reason", "mechanical gate failed (see booleans)")
    finally:
        del sds
    checks["tier2_gate"] = gate
    checks["tier1_ok"] = tier1_ok
    checks["tier2_release"] = all(gate.get(t, {}).get("pass") for t in ("ft1", "ft2"))
    checks["pass"] = ok_all and tier1_ok
    return checks


# =====================================================================================

# =====================================================================================

def run(ctx, model_root, probe_file):
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # ---- 0-preflight ----
    pre = phase_preflight(model_root, probe_file)
    R.write_json(ctx.p("preflight_verify.json"), pre)
    if not pre["pass"]:
        print("[note]")
        sys.exit(3)
    # ---- 1-probe ----
    tok = AutoTokenizer.from_pretrained(os.path.join(model_root, BASE_TOK_DIRNAME))
    _TOK_CACHE["tok"] = tok
    probes, cand_ids, n_props = build_probe_battery(probe_file, tok)
    R.write_json(ctx.p("probe_battery.json"), {
        "rule": "tok(' '+obj)==1 token; cand = unique obj tokens",
        "n_probes": len(probes), "n_cand": len(cand_ids), "n_props": n_props,
        "band": {"probes": [PROBE_N_EXPECT, PROBE_N_BAND],
                 "cand": [CAND_N_EXPECT, CAND_N_BAND], "props": PROP_N_EXPECT},
        "band_ok": _probe_band_ok(len(probes), len(cand_ids), n_props),
        "cand_token_ids": cand_ids,
        "probes": probes})
    if not _probe_band_ok(len(probes), len(cand_ids), n_props):
        print("[note]")
        sys.exit(3)
    base_dir = os.path.join(model_root, BASE_DIRNAME)
    sft_dir = os.path.join(model_root, SFT_DIRNAME)
    t1_path = ctx.p("margins_tier1.jsonl.gz")
    os.makedirs(ctx.p("shards"), exist_ok=True)
    # ---- 2-eval-tier1 ----
    model, _ = _load_hf(base_dir, os.path.join(model_root, BASE_TOK_DIRNAME))
    sd0 = _sd_of(model)
    tier1 = {}
    tier1["base"] = _eval_and_shard(ctx, model, probes, cand_ids,
                                    ctx.p("shards", "tier1.base.jsonl.gz"), "base") \
        or _load_shard(ctx.p("shards", "tier1.base.jsonl.gz"))
    model_sft, _ = _load_hf(sft_dir, os.path.join(model_root, BASE_TOK_DIRNAME))
    sd1 = _sd_of(model_sft)
    tier1["sft"] = _eval_and_shard(ctx, model_sft, probes, cand_ids,
                                   ctx.p("shards", "tier1.sft.jsonl.gz"), "sft") \
        or _load_shard(ctx.p("shards", "tier1.sft.jsonl.gz"))
    del model_sft
    for lam in LAMBDAS_TIER1:
        tag = "lam{:.2f}".format(lam)
        shard = ctx.p("shards", "tier1.{}.jsonl.gz".format(tag))
        if os.path.exists(shard):
            tier1[tag] = _load_shard(shard)
            ctx.ledger("eval-skip:" + tag)
            continue
        sd_l = _sd_lerp_delta(sd0, sd1, lam)
        _load_sd_into(model, sd_l)
        tier1[tag] = _eval_and_shard(ctx, model, probes, cand_ids, shard, tag) \
            or _load_shard(shard)

    _merge_shards(t1_path, [("base", tier1["base"]), ("sft", tier1["sft"])] +
                  [("lam{:.2f}".format(l), tier1["lam{:.2f}".format(l)])
                   for l in LAMBDAS_TIER1])

    tier2 = {}
    t2_path = ctx.p("margins_tier2.jsonl.gz")
    if pre["tier2_release"]:
        m1, _ = _load_hf(os.path.join(model_root, FT1_DIRNAME),
                         os.path.join(model_root, BASE_TOK_DIRNAME))
        sdf1 = _sd_of(m1)
        tier2["ft1"] = _eval_and_shard(ctx, m1, probes, cand_ids,
                                       ctx.p("shards", "tier2.ft1.jsonl.gz"), "ft1") \
            or _load_shard(ctx.p("shards", "tier2.ft1.jsonl.gz"))
        m2, _ = _load_hf(os.path.join(model_root, FT2_DIRNAME),
                         os.path.join(model_root, BASE_TOK_DIRNAME))
        sdf2 = _sd_of(m2)
        tier2["ft2"] = _eval_and_shard(ctx, m2, probes, cand_ids,
                                       ctx.p("shards", "tier2.ft2.jsonl.gz"), "ft2") \
            or _load_shard(ctx.p("shards", "tier2.ft2.jsonl.gz"))
        del m1, m2
        tier2["merge05"] = _eval_construct(ctx, model, _sd_avg(sdf1, sdf2), probes,
                                           cand_ids,
                                           ctx.p("shards", "tier2.merge05.jsonl.gz"),
                                           "merge05")
        tier2["dis1"] = _eval_construct(ctx, model, _sd_lerp_delta(sd0, sdf1, 0.5),
                                        probes, cand_ids,
                                        ctx.p("shards", "tier2.dis1.jsonl.gz"), "dis1")
        tier2["dis2"] = _eval_construct(ctx, model, _sd_lerp_delta(sd0, sdf2, 0.5),
                                        probes, cand_ids,
                                        ctx.p("shards", "tier2.dis2.jsonl.gz"), "dis2")
        _merge_shards(t2_path, [(k, tier2[k])
                                for k in ("ft1", "ft2", "merge05", "dis1", "dis2")])
    else:
        R.write_json(ctx.p("margins_tier2.skipped.json"),
                     {"skipped_with_reason": pre["tier2_gate"],
                      "note": "[note]"})
    # ---- 4-analysis ----
    _analysis(ctx, tier1, tier2 if pre["tier2_release"] else None, probes, pre)


def _eval_construct(ctx, model, sd, probes, cand_ids, shard, tag):
    if os.path.exists(shard):
        ctx.ledger("eval-skip:" + tag)
        return _load_shard(shard)
    _load_sd_into(model, sd)
    return _eval_and_shard(ctx, model, probes, cand_ids, shard, tag) \
        or _load_shard(shard)


def _merge_shards(path, blocks):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        for tag, res in blocks:
            f.write(json.dumps({"__meta__": {"model": tag, "n_probes": len(res)}}) + "\n")
            for qid, r in res.items():
                f.write(json.dumps({"qid": qid, "model": tag, **r}) + "\n")
    os.replace(tmp, path)


def _f_live(m_ref_readings, floor=F_LIVE_FLOOR):
    live = [qid for qid, r in m_ref_readings.items() if r["rank"] == 1]
    return live, len(live) >= floor


def _analysis(ctx, tier1, tier2, probes, pre):
    rep = {"contract_id": CONTRACT_ID, "prereg": "PREREG-AUG.md §4",
           "headline_limits": [
               "[note]",
               "[note]",
               "[note]",
               "[note]"],
           "tier2_provenance_warning": ("[note]"
                                        "[note]") if tier2 else None,
           "tiers": {}}

    f_live, power_ok = _f_live(tier1["sft"])
    curves = {}
    for tag, readings in tier1.items():
        ms = [readings[f]["m_top1"] for f in f_live]
        curves[tag] = {"median_m_top1": round(R.median(ms), 5),
                       "margin_std": round(_pstdev(ms), 5),
                       "cand_logit_std_median": round(R.median(
                           [readings[f]["cand_logit_std"] for f in f_live]), 5)}
    d_t1 = _d_add_t(tier1["sft"], tier1["lam0.50"], f_live)
    ts, ts_dir = t_star(d_t1)
    rep["tiers"]["tier1"] = {
        "f_live": len(f_live), "power_ok": power_ok,
        "low_power_note": None if power_ok else
        "[note]".format(F_LIVE_FLOOR),
        "scale_curves_on_f_live": curves,
        "d_add_t": [{"t": t, "D": round(d, 5)} for t, d in d_t1],
        "t_star": None if ts is None else round(ts, 4),
        "t_star_note": ("[note]".format(ts_dir) if ts is None
                        else "[note]"),
        "applied_side": "[note]"}

    if tier2:
        f2, p2 = _f_live(tier2["ft1"])
        t2rep = {"f_live": len(f2), "power_ok": p2,
                 "scale_ratios": {}, "t_star": {}}
        for ref in ("dis1", "dis2"):
            d_t2 = _d_add_t(tier2["merge05"], tier2[ref], f2)
            ts2, dir2 = t_star(d_t2)
            t2rep["t_star"][ref] = {
                "d_add_t": [{"t": t, "D": round(d, 5)} for t, d in d_t2],
                "t_star": None if ts2 is None else round(ts2, 4),
                "note": ("[note]".format(dir2) if ts2 is None
                         else "[note]"),
                "applied_side": "[note]"}
        for stat, fn in [("median_m_top1", lambda rs: R.median([r["m_top1"] for r in rs])),
                         ("margin_std", lambda rs: _pstdev([r["m_top1"] for r in rs])),
                         ("cand_logit_std", lambda rs: R.median(
                             [r["cand_logit_std"] for r in rs]))]:
            num = fn([tier2["merge05"][f] for f in f2])
            den = R.median([fn([tier2[r_][f] for f in f2]) for r_ in ("dis1", "dis2")])
            t2rep["scale_ratios"][stat] = {
                "merge": round(num, 5), "dis_median": round(den, 5),
                "ratio": round(num / den, 4) if den else None}
        rep["tiers"]["tier2"] = t2rep
    else:
        rep["tiers"]["tier2"] = {"skipped": True, "reason": pre["tier2_gate"]}

    rep["synthetic_reference_points"] = {
        "logit_scale_ratio_synthetic_tau0": {"a0.25": 1.12, "a0.50": 1.92, "a0.75": 2.67},
        "sign_flip_cells_synthetic_raw": "4/6 cells t∈[1.02,1.53]",
        "note": "[note]"}
    rep["status"] = "complete"
    R.write_json(ctx.p("realbridge_report.json"), rep)


def _pstdev(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


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
          lambda: LAMBDAS_TIER1 == [0.25, 0.5, 0.75, 1.25, 1.5]
          and TGRID == [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00])
    check("[note]",
          lambda: (PROBE_N_EXPECT, PROBE_N_BAND, CAND_N_EXPECT, PROP_N_EXPECT,
                   F_LIVE_FLOOR, DELTA_NORM_BAND) == (1719, 0.05, 401, 16, 300, (0.1, 10.0)))

    def _tstar():
        """Analysis script for the merge-audit study."""
        ts, d = t_star([(0.6, 1.0), (1.0, 0.5), (1.5, -0.5), (2.0, -1.0)])
        assert abs(ts - 1.25) < 1e-9 and d == "crossing"
        ts, d = t_star([(t, 1.0) for t in TGRID])
        assert ts is None and d == "positive"
        ts, d = t_star([(t, -1.0) for t in TGRID])
        assert ts is None and d == "negative"
        ts, d = t_star([(0.6, 0.0), (1.0, -1.0)])
        assert ts == 0.6 and d == "exact_zero"
        return True
    check("[note]", _tstar)

    def _margins_math():
        """Analysis script for the merge-audit study."""
        torch = _torch()
        cand = [10, 20, 30, 40]
        z = torch.zeros(100)
        z[10], z[20], z[30], z[40] = 3.0, 1.0, 2.0, -1.0
        z[55] = 99.0
        r = margins_from_logits(z, cand, 20)
        assert r["rank"] == 3, r
        assert abs(r["m_top1"] - (1.0 - 3.0)) < 1e-5   # z_v* − max_rest
        import math as _m
        lse = math.log(sum(_m.exp(float(x)) for x in z))
        assert abs(r["logp"] - (1.0 - lse)) < 1e-4
        r2 = margins_from_logits(z, cand, 10)
        assert r2["rank"] == 1 and r2["m_top1"] > 0
        return True
    check("[note]", _margins_math)

    def _padding_fix():
        """Analysis script for the merge-audit study."""
        from types import SimpleNamespace
        torch = _torch()

        class _FakeTok:
            pad_token = None
            eos_token = None

            def __call__(self, s, add_special_tokens=True):
                return {"input_ids": [ord(c) % 50 + 1 for c in s]}

        class _FakeModel:
            """Analysis script for the merge-audit study."""

            def to(self, dev):
                return self

            def eval(self):
                return self

            def __call__(self, input_ids=None, attention_mask=None, **kw):
                B, T = input_ids.shape
                pos = torch.arange(T).view(1, T, 1).float()
                v = torch.arange(64).view(1, 1, 64).float()
                logits = input_ids.unsqueeze(-1).float() * (v + 1.0) + pos * (0.5 * v + 1.0)
                return SimpleNamespace(logits=logits)

        tok = _FakeTok()
        _TOK_CACHE["tok"] = tok
        cand = list(range(1, 13))
        probes = [{"qid": "r2", "question": "ab", "val_tid": 5},
                  {"qid": "r3", "question": "abc", "val_tid": 7},
                  {"qid": "r5", "question": "abcde", "val_tid": 9}]
        model = _FakeModel()

        new = _eval_model_on_probes(model, probes, cand, batch=32)

        gt = {}
        for p in probes:
            ids = tok(p["question"])["input_ids"]
            x = torch.tensor([ids])
            lg = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[:, -1, :].float()
            gt[p["qid"]] = margins_from_logits(lg[0], cand, p["val_tid"])

        enc = [tok(p["question"])["input_ids"] for p in probes]
        L = max(len(e) for e in enc)
        x = torch.tensor([e + [0] * (L - len(e)) for e in enc])
        am = torch.tensor([[1] * len(e) + [0] * (L - len(e)) for e in enc])
        lg = model(input_ids=x, attention_mask=am).logits[:, -1, :].float()
        old = {p["qid"]: margins_from_logits(lg[j], cand, p["val_tid"])
               for j, p in enumerate(probes)}
        same_new = {q: new[q] == gt[q] for q in gt}
        same_old = {q: old[q] == gt[q] for q in gt}
        print("[note]", {k: v for k, v in same_new.items()},
              "[note]", {k: v for k, v in same_old.items()})
        assert all(same_new.values()), "[note]"
        assert same_old == {"r2": False, "r3": False, "r5": True}, same_old

        assert (old["r5"] == new["r5"]) and (old["r2"] != new["r2"]) and (old["r3"] != new["r3"])
        return True
    check("[note]", _padding_fix)

    def _tier2_gate():
        """Analysis script for the merge-audit study."""
        base = {"a": 1, "b": 2}
        same = {"a": 1, "b": 2}
        diff = {"a": 1, "b": 3}
        assert not {k: 1 for k in set(base) | set(same) if base.get(k) != same.get(k)}
        assert {k: 1 for k in set(base) | set(diff) if base.get(k) != diff.get(k)}
        lo, hi = DELTA_NORM_BAND
        n_sft = 5.0
        assert lo * n_sft <= 0.5 <= hi * n_sft and not (lo * n_sft <= 0.49)
        assert lo * n_sft <= 50.0 <= hi * n_sft and not (50.01 <= hi * n_sft)
        return True
    check("[note]", _tier2_gate)
    check("[note]",
          lambda: all(_rejects(d) for d in FORBIDDEN_OUTDIRS)
          and _assert_outdir_safe("results/realbridge"))
    check("[note]", lambda: R.FUSE_GPUH == REALBRIDGE_FUSE_GPUH)
    check("[note]",
          lambda: os.environ.get("HF_HUB_OFFLINE") == "1")

    def _idem():
        with tempfile.TemporaryDirectory(prefix="realbridge_dryrun_") as td:
            shard = os.path.join(td, "m.jsonl.gz")
            with gzip.open(shard, "wt") as f:
                f.write(json.dumps({"__meta__": {"model": "x"}}) + "\n")
                f.write(json.dumps({"qid": "1", "rank": 1, "m_top1": 0.5}) + "\n")
            assert os.path.exists(shard)
            assert _load_shard(shard)["1"]["m_top1"] == 0.5
            return True
    check("[note]", _idem)

    def _staging():
        """Analysis script for the merge-audit study."""
        model_root = os.path.join(EXP, "..", "..", "data", "models")
        probe_file = os.path.join(EXP, "..", "..", "data", "datasets", "PopQA", "test.tsv")
        rec_path = os.path.join(model_root, STAGING_RECORD)
        if not os.path.exists(rec_path):
            print("[note]")
            return False
        rec = json.load(open(rec_path))
        covered = {}
        for item in rec["files"]:
            assert os.path.exists(item["path"]), item["path"]
            assert os.path.getsize(item["path"]) == item["size"], item["path"]
            assert R.sha256_file(item["path"]) == item["sha256"], item["path"]
            covered[os.path.basename(os.path.dirname(item["path"])) + "/"
                    + os.path.basename(item["path"])] = True
        for need in ["OLMo-2-0425-1B-SFT/pytorch_model.bin",
                     "OLMo-2-0425-1B_full_sft_code_data_120K/model.safetensors",
                     "OLMo-2-0425-1B_full_sft_natural_language_data_120K/model.safetensors"]:
            assert covered.get(need), "[note]" + need
        assert R.sha256_file(probe_file) == POPQA_SHA256
        return True
    check("[note]", _staging)

    def _probe_build_real():
        """Analysis script for the merge-audit study."""
        from transformers import AutoTokenizer
        model_root = os.path.join(EXP, "..", "..", "data", "models")
        probe_file = os.path.join(EXP, "..", "..", "data", "datasets", "PopQA", "test.tsv")
        if not os.path.exists(probe_file):
            print("[note]")
            return False
        tok = AutoTokenizer.from_pretrained(os.path.join(model_root, BASE_TOK_DIRNAME))
        probes, cand, n_props = build_probe_battery(probe_file, tok)
        print("[note]".format(len(probes), len(cand), n_props))
        return _probe_band_ok(len(probes), len(cand), n_props)
    check("[note]", _probe_build_real)

    def _contract():
        c = json.load(open(os.path.join(EXP, "contract-realbridge-draft.json")))
        assert "run_merge_audit_realbridge.py" in c["command"]
        assert c["draft"] is True
        assert c["contract_id"].replace("-draft", "") == CONTRACT_ID
        for p in ["results/realbridge/realbridge_report.json",
                      "results/realbridge/probe_battery.json",
                      "results/realbridge/preflight_verify.json",
                      "results/realbridge/margins_tier1.jsonl.gz"]:
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
    ap.add_argument("--model-root", default=os.path.join(EXP, "..", "..", "data", "models"))
    ap.add_argument("--probe-file",
                    default=os.path.join(EXP, "..", "..", "data", "datasets", "PopQA", "test.tsv"))
    ap.add_argument("--out-dir", default="results/realbridge")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        sys.exit(dry_run())
    _assert_outdir_safe(args.out_dir)
    signal.signal(signal.SIGALRM, _alarm_handler)
    ctx = R.Ctx(args.out_dir)
    ctx.ledger("realbridge", "enter", force=True)
    try:
        run(ctx, os.path.normpath(args.model_root), os.path.normpath(args.probe_file))
    finally:
        ctx.ledger("realbridge", "exit", force=True)


if __name__ == "__main__":
    main()
