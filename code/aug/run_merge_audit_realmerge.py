#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import ast
import csv
import gzip
import json
import math
import os
import re
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

CONTRACT_ID = os.environ.get("TANH_REALMERGE_CONTRACT_ID",
                             "merge_audit-aug-realmerge-v1")
R.CONTRACT_ID = CONTRACT_ID
REALMERGE_FUSE_GPUH = 10.5
R.FUSE_GPUH = REALMERGE_FUSE_GPUH


TGRID = [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
LAMBDAS = [0.5, 0.75, 1.0, 1.25, 1.5]
LAMBDAS_ARM = [0.25, 0.5, 0.75, 1.25, 1.5]
TIES_DENSITY = 0.5
SOUP_IDENTITY_TOL = 1e-4
DELTA_NORM_BAND = (0.1, 10.0)
BAND_TOL = 0.05
POPQA_SHA256 = "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b"
POPQA_SIZE = 5205200
MBPP_SHA256 = "e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850"
MBPP_SIZE = 60864
F_LIVE_FLOOR = {"popqa": 300, "gsm8k": 60, "mbpp": 30}
MBPP_MIN_N = 30
E1_FLIP_FRAC_MIN = 0.5
E1_PRACT_FRAC_MIN = 0.25
E1_MIN_POWERED_CELLS = 8
T_PRACTICAL = (0.7, 1.3)
STALL_PER_MODEL_S = 20 * 60
EVAL_BATCH = 32
FORBIDDEN_OUTDIRS = ["results", "results_v2", "results_v2_reid"]
STAGING_RECORD = "PREFLIGHT-REALMERGE-20260827.json"


POPQA_BAND = {"olmo": (1719, 401, 16), "llama3": (1719, 401, 16), "qwen25": (1719, 401, 16),
              "qwen3": (1719, 401, 16), "qwen38": (1761, 438, 16),
              "mistral": (1255, 188, 15)}
GSM8K_BAND = {"olmo": 1186, "llama3": 1186, "qwen25": 252, "qwen3": 252, "qwen38": 252,
              "mistral": 0}
MBPP_BAND = {"olmo": 147, "llama3": 147, "qwen25": 99, "qwen3": 99, "qwen38": 99,
             "mistral": 44}


CONFIG_EXCLUDE = ("_name_or_path", "transformers_version", "torch_dtype", "use_cache",
                  "rope_scaling", "max_position_embeddings", "eos_token_id", "bos_token_id",
                  "pad_token_id", "generation_config")

FAMILIES = {
    "fam1-llama3-8b": {"base": "Meta-Llama-3-8B", "tok_class": "llama3", "size": "8b",
                       "deltas": [("instruct", "Meta-Llama-3-8B-Instruct", "B"),
                                  ("math", "MAmmoTH2-8B", "B"),
                                  ("chat", "Hermes-2-Pro-Llama-3-8B", "A")],
                       "delta_ref": "instruct"},
    "fam2-qwen25-1p5b": {"base": "Qwen2.5-1.5B", "tok_class": "qwen25", "size": "1.5b",
                         "deltas": [("instruct", "Qwen2.5-1.5B-Instruct", "A"),
                                    ("math", "Qwen2.5-Math-1.5B", "A"),
                                    ("code", "Qwen2.5-Coder-1.5B", "A")],
                         "delta_ref": "instruct"},
    "fam3-olmo-1b": {"base": "OLMo-2-0425-1B/stage1-step1907359-tokens4001B",
                     "base_tok": "OLMo-2-0425-1B/_tokenizer", "tok_class": "olmo", "size": "1b",
                     "deltas": [("sft", "OLMo-2-0425-1B-SFT", "A")],
                     "tier2": [("code120k", "OLMo-2-0425-1B_full_sft_code_data_120K", "C"),
                               ("nl120k", "OLMo-2-0425-1B_full_sft_natural_language_data_120K", "C")],
                     "delta_ref": "sft"},
    "fam4-llama32-1b": {"base": "Llama-3.2-1B", "tok_class": "llama3", "size": "1b",
                        "deltas": [("instruct", "Llama-3.2-1B-Instruct", "B")], "delta_ref": "instruct"},
    "fam5-llama32-3b": {"base": "Llama-3.2-3B", "tok_class": "llama3", "size": "3b",
                        "deltas": [("instruct", "Llama-3.2-3B-Instruct", "B")], "delta_ref": "instruct"},
    "fam6-qwen25-0p5b": {"base": "Qwen2.5-0.5B", "tok_class": "qwen25", "size": "0.5b",
                         "deltas": [("instruct", "Qwen2.5-0.5B-Instruct", "A")], "delta_ref": "instruct"},
    "fam7-qwen3-8b": {"base": "Qwen3-8B-Base", "tok_class": "qwen3", "size": "8b",
                      "deltas": [("post", "Qwen3-8B", "B"),
                                 ("r1", "DeepSeek-R1-0528-Qwen3-8B", "B")],
                      "delta_ref": "post"},
    "fam8-qwen38-27b": {"base": "Qwen3.8-27B", "tok_class": "qwen38", "size": "27b",
                        "deltas": [], "scale_point_only": True},
}


class _Stall(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Stall("[note]".format(STALL_PER_MODEL_S))


def _assert_outdir_safe(out_dir):
    rp = os.path.realpath(out_dir)
    for bad in FORBIDDEN_OUTDIRS:
        if rp == os.path.realpath(os.path.join(EXP, bad)) or rp == os.path.realpath(bad):
            print("[note]".format(out_dir, bad))
            sys.exit(3)
    return True


_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


# =====================================================================================

# =====================================================================================

class StReader:
    def __init__(self, d):
        self.dir = d
        idx = os.path.join(d, "model.safetensors.index.json")
        self.handles = {}
        self.key2file = {}
        self.bin_sd = None
        if os.path.exists(idx):
            wm = json.load(open(idx))["weight_map"]
            for k, fn in wm.items():
                self.key2file[k] = os.path.join(d, fn)
        elif os.path.exists(os.path.join(d, "model.safetensors")):
            p = os.path.join(d, "model.safetensors")
            from safetensors import safe_open
            with safe_open(p, framework="pt") as f:
                for k in f.keys():
                    self.key2file[k] = p
        elif os.path.exists(os.path.join(d, "pytorch_model.bin")):
            torch = _torch()
            self.bin_sd = torch.load(os.path.join(d, "pytorch_model.bin"),
                                     map_location="cpu", weights_only=True)
            self.key2file = {k: None for k in self.bin_sd}
        else:
            raise FileNotFoundError("no weights found in " + d)

    def keys(self):
        return list(self.key2file)

    def get(self, key):
        """Analysis script for the merge-audit study."""
        if self.bin_sd is not None:
            return self.bin_sd[key].float()
        torch = _torch()
        from safetensors import safe_open
        p = self.key2file[key]
        if p not in self.handles:
            self.handles[p] = safe_open(p, framework="pt")
        return self.handles[p].get_tensor(key).float()

    def close(self):
        for h in self.handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass
        self.handles = {}
        self.bin_sd = None


def _fro_delta_sq(reader_ft, reader_base):
    """Analysis script for the merge-audit study."""
    tot = 0.0
    for k in reader_base.keys():
        d = reader_ft.get(k) - reader_base.get(k)
        tot += float(d.pow(2).sum())
    return tot


def _construct_tv(readers, base_reader, lam):
    """Analysis script for the merge-audit study."""
    torch = _torch()
    out = {}
    for k in base_reader.keys():
        t0 = base_reader.get(k)
        acc = None
        for rd in readers:
            d = rd.get(k) - t0
            acc = d if acc is None else acc + d
        out[k] = (t0 + lam * acc).to(torch.bfloat16)
    return out


def _construct_dis(reader_ft, base_reader, lam):
    torch = _torch()
    return {k: (base_reader.get(k) + lam * (reader_ft.get(k) - base_reader.get(k))).to(torch.bfloat16)
            for k in base_reader.keys()}


def _construct_soup(readers):
    """Analysis script for the merge-audit study."""
    torch = _torch()
    out = {}
    n = len(readers)
    for k in readers[0].keys():
        acc = readers[0].get(k)
        for rd in readers[1:]:
            acc = acc + rd.get(k)
        out[k] = (acc / n).to(torch.bfloat16)
    return out


def _construct_ties(readers, base_reader, lam, density=TIES_DENSITY):
    """Analysis script for the merge-audit study."""
    torch = _torch()
    out = {}
    K = len(readers)
    for k in base_reader.keys():
        t0 = base_reader.get(k)
        deltas = torch.stack([rd.get(k) - t0 for rd in readers])     # (K, *shape) fp32
        flat = deltas.reshape(K, -1)
        n = flat.shape[1]
        keep_n = max(1, int(math.ceil(density * n)))
        thr = flat.abs().kthvalue(n - keep_n + 1, dim=1).values
        mask = flat.abs() >= thr.unsqueeze(1)
        elected = torch.sign((flat * mask).sum(dim=0))
        same = mask & (torch.sign(flat) == elected.unsqueeze(0))
        cnt = same.sum(dim=0).clamp(min=1)
        merged = (flat * same).sum(dim=0) / cnt
        merged = torch.where(same.sum(dim=0) > 0, merged, torch.zeros_like(merged))
        out[k] = (t0 + lam * merged.reshape(t0.shape)).to(torch.bfloat16)
    return out


# =====================================================================================

# =====================================================================================

def build_popqa(probe_file, tok):
    probes = []
    with open(probe_file, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids = tok(" " + row["obj"], add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                continue
            probes.append({"qid": "popqa:" + row["id"], "battery": "popqa",
                           "prompt": row["question"], "ans": row["obj"],
                           "ans_str": " " + row["obj"],
                           "prop": row["prop"], "val_tid": ids[0]})
    return probes


def build_gsm8k(parquet_path, tok):
    pd = __import__("pandas")
    df = pd.read_parquet(parquet_path)
    probes = []
    for i, r in enumerate(df.itertuples()):
        m = re.search(r"####\s*(.+)$", r.answer)
        if not m:
            continue
        ans = m.group(1).strip().replace(",", "").replace("$", "").strip()
        ids = tok(ans, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            continue
        probes.append({"qid": "gsm8k:{}".format(i), "battery": "gsm8k",
                       "prompt": "Question: {}\nAnswer: ".format(r.question),
                       "ans": ans, "ans_str": ans, "prop": "math", "val_tid": ids[0]})
    return probes


def build_mbpp(parquet_path, tok):
    pd = __import__("pandas")
    df = pd.read_parquet(parquet_path)
    probes = []
    for _, r in df.iterrows():
        tl = list(r["test_list"])
        if not tl:
            continue
        m = re.match(r"^assert\s+(.+?)\s*==\s*(.+)$", tl[0].strip())
        if not m:
            continue
        call, rhs = m.group(1).strip(), m.group(2).strip()
        try:
            val = ast.literal_eval(rhs)
        except Exception:
            continue
        ans = str(val)
        ids = tok(ans, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            continue
        probes.append({"qid": "mbpp:{}".format(r["task_id"]), "battery": "mbpp",
                       "prompt": "Consider the following Python function:\n\n{}\n\n"
                                 "What is the value returned by `{}`?\nAnswer: ".format(
                                     r["code"], call),
                       "ans": ans, "ans_str": ans, "prop": "code", "val_tid": ids[0]})
    return probes


def build_batteries(tok, tok_class, data_root):
    """Analysis script for the merge-audit study."""
    popqa_file = os.path.join(data_root, "datasets", "PopQA", "test.tsv")
    gsm_file = os.path.join(data_root, "datasets", "gsm8k", "main", "test-00000-of-00001.parquet")
    mbpp_file = os.path.join(data_root, "datasets", "mbpp", "sanitized-test.parquet")
    assert R.sha256_file(popqa_file) == POPQA_SHA256 and os.path.getsize(popqa_file) == POPQA_SIZE, \
        "[note]"
    assert os.path.getsize(mbpp_file) == MBPP_SIZE and R.sha256_file(mbpp_file) == MBPP_SHA256, \
        "[note]"
    out = {}
    p = build_popqa(popqa_file, tok)
    exp = POPQA_BAND[tok_class]
    cand = sorted({x["val_tid"] for x in p})
    props = {x["prop"] for x in p}
    assert abs(len(p) - exp[0]) <= exp[0] * BAND_TOL and \
        abs(len(cand) - exp[1]) <= exp[1] * BAND_TOL and len(props) == exp[2], \
        "[note]".format((len(p), len(cand), len(props)), exp)
    out["popqa"] = (p, cand)
    g = build_gsm8k(gsm_file, tok)
    expg = GSM8K_BAND[tok_class]
    assert abs(len(g) - expg) <= expg * BAND_TOL, "[note]".format(len(g), expg)
    out["gsm8k"] = (g, sorted({x["val_tid"] for x in g}))
    mb = build_mbpp(mbpp_file, tok)
    expm = MBPP_BAND[tok_class]
    assert abs(len(mb) - expm) <= expm * BAND_TOL, "[note]".format(len(mb), expm)
    out["mbpp"] = (mb, sorted({x["val_tid"] for x in mb}))
    return out


# =====================================================================================

# =====================================================================================

def margins_from_logits(last_logits, cand_ids, val_tid):
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
            "cand_logit_std": round(float(zc.std(unbiased=False).item()), 5),
            "full_logit_std": round(float(z.std(unbiased=False).item()), 5)}


def _eval_model(model, tok, probes, cand_by_bat, dev):
    """Analysis script for the merge-audit study."""
    torch = _torch()
    model.eval()
    out = {}
    by_len = {}
    for p in probes:
        ids = tok(p["prompt"], add_special_tokens=True)["input_ids"]
        by_len.setdefault(len(ids), []).append((ids, p))
    with torch.no_grad():
        for L, items in sorted(by_len.items()):
            for i0 in range(0, len(items), EVAL_BATCH):
                chunk = items[i0:i0 + EVAL_BATCH]
                x = torch.tensor([c[0] for c in chunk], dtype=torch.long, device=dev)
                logits = model(x).logits[:, -1, :].float()
                for j, (_, p) in enumerate(chunk):
                    out[p["qid"]] = margins_from_logits(logits[j], cand_by_bat[p["battery"]],
                                                        p["val_tid"])
    return out


def _eval_sharded(ctx, model, tok, tag, probes, cand_by_bat, dev):
    shard = ctx.p("shards", "{}.jsonl.gz".format(tag))
    if os.path.exists(shard):
        ctx.ledger("eval-skip:" + tag)
        return None
    signal.alarm(STALL_PER_MODEL_S)
    try:
        res = _eval_model(model, tok, probes, cand_by_bat, dev)
    finally:
        signal.alarm(0)
    tmp = shard + ".tmp"
    with gzip.open(tmp, "wt") as f:
        f.write(json.dumps({"__meta__": {"model": tag, "n_probes": len(res)}}) + "\n")
        for qid, r in res.items():
            f.write(json.dumps({"qid": qid, **r}) + "\n")
    os.replace(tmp, shard)
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


def _load_sd_dict(reader):
    torch = _torch()
    return {k: reader.get(k).to(torch.bfloat16) for k in reader.keys()}


# =====================================================================================

# =====================================================================================

def _tok_equiv(tok_member, probes):
    """Analysis script for the merge-audit study."""
    for p in probes:
        if tok_member(p["prompt"], add_special_tokens=True)["input_ids"] != p["_prompt_ids"]:
            return False
        ids = tok_member(p["ans_str"], add_special_tokens=False)["input_ids"]
        if ids != [p["val_tid"]]:
            return False
    return True


def preflight_family(model_root, fam_id, spec, staging, tok, probes):
    """Analysis script for the merge-audit study."""
    base_dir = os.path.join(model_root, spec["base"])
    base_tok_dir = os.path.join(model_root, spec.get("base_tok", spec["base"]))
    rec = {"family": fam_id, "files": {}, "config_diffs": {}, "tokenizer_bytes": {},
           "delta_norms": {}, "deltas": {}}

    ok = True
    hard_dirs = [spec["base"]] + [d for _, d, _ in spec["deltas"]]
    soft_dirs = [d for _, d, _ in spec.get("tier2", [])]
    staged = {}
    if staging:
        for item in staging["files"]:
            staged[item["path"]] = item
    tier2_files_ok = {}
    for dn in hard_dirs + soft_dirs:
        d = os.path.join(model_root, dn)
        n_w = 0
        sha_ok = None
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith((".safetensors", ".bin")):
                    n_w += 1
            if staged and not spec.get("scale_point_only"):


                sha_ok = True
                for fn in sorted(os.listdir(d)):
                    if not fn.endswith((".safetensors", ".bin")):
                        continue
                    p = os.path.join(d, fn)
                    it = staged.get(p)
                    if it is None or it["size"] != os.path.getsize(p) or \
                            R.sha256_file(p) != it["sha256"]:
                        sha_ok = False
                        break
        has_tok = os.path.exists(os.path.join(d, "tokenizer.json"))
        rec["files"][dn] = {"dir": os.path.isdir(d), "weight_files": n_w,
                            "tokenizer_json": has_tok, "sha256_ok": sha_ok}
        files_ok = os.path.isdir(d) and n_w >= 1 and has_tok and sha_ok is not False
        if dn in soft_dirs:
            tier2_files_ok[dn] = files_ok
        elif not files_ok:
            ok = False
    if not ok:
        rec["family_fail"] = "weight/tokenizer files missing or sha256 mismatch"
        return False, [], rec

    base_cfg = json.load(open(os.path.join(base_dir, "config.json")))
    tier2_dns = {dn for _, dn, _ in spec.get("tier2", [])}
    cands = list(spec["deltas"]) + list(spec.get("tier2", []))
    cfg_ok = {}
    for tag, dn, prov in cands:
        cfg_path = os.path.join(model_root, dn, "config.json")
        if not os.path.exists(cfg_path):
            cfg_ok[tag] = False
            rec["config_diffs"][tag] = "config.json missing"
            continue
        cfg = json.load(open(cfg_path))
        diff = {k: [base_cfg.get(k), cfg.get(k)] for k in sorted(set(base_cfg) | set(cfg))
                if base_cfg.get(k) != cfg.get(k) and k not in CONFIG_EXCLUDE}
        rec["config_diffs"][tag] = diff
        cfg_ok[tag] = (not diff)
    bad_main = [t for t, dn, _ in spec["deltas"] if not cfg_ok[t]]
    if bad_main:



        for t_bad in bad_main:
            rec.setdefault("skipped_deltas", {})[t_bad] = "arch-key config diff (gate)"
        spec["deltas"] = [d for d in spec["deltas"] if d[0] not in set(bad_main)]
        rec["family_note"] = "[note]".format(bad_main, len(spec["deltas"]))
        cands = list(spec["deltas"]) + list(spec.get("tier2", []))
        if not spec["deltas"]:
            rec["family_fail"] = "all deltas dropped by config gate"
            return False, [], rec

    from transformers import AutoTokenizer as _AT
    base_tok_b = open(os.path.join(base_tok_dir, "tokenizer.json"), "rb").read()
    tok_ok = {}
    for tag, dn, prov in cands:
        tok_path = os.path.join(model_root, dn, "tokenizer.json")
        if not os.path.exists(tok_path):
            tok_ok[tag] = False
            rec["tokenizer_bytes"][tag] = None
            continue
        tb = open(tok_path, "rb").read()
        rec["tokenizer_bytes"][tag] = (tb == base_tok_b)
        tok_m = _AT.from_pretrained(os.path.join(model_root, dn))
        tok_ok[tag] = _tok_equiv(tok_m, probes)

    if spec.get("scale_point_only"):

        rec["delta_norms"] = {}
        rec["delta_norm_ref"] = None
        rec["tier2"] = {}
        rec["released_K"] = 0
        return True, [], rec
    base_rd = StReader(base_dir)
    ref_tag = spec["delta_ref"]
    ref_norm = None
    norms = {}
    for tag, dn, prov in cands:
        if dn in tier2_dns and not tier2_files_ok.get(dn, False):
            norms[tag] = None
            continue
        rd = StReader(os.path.join(model_root, dn))
        n = math.sqrt(_fro_delta_sq(rd, base_rd))
        rd.close()
        norms[tag] = n
        if tag == ref_tag:
            ref_norm = n
    base_rd.close()
    rec["delta_norms"] = {t: (round(n, 3) if n is not None else None)
                          for t, n in norms.items()}
    rec["delta_norm_ref"] = ref_tag
    released = []
    for tag, dn, prov in spec["deltas"]:
        inband = DELTA_NORM_BAND[0] * ref_norm <= norms[tag] <= DELTA_NORM_BAND[1] * ref_norm
        pass_ = tok_ok[tag] and inband
        rec["deltas"][tag] = {"dir": dn, "provenance": prov, "tokenizer_ok": tok_ok[tag],
                              "delta_norm": round(norms[tag], 3), "norm_in_band": inband,
                              "pass": pass_}
        if pass_:
            released.append((tag, dn, prov))
        else:
            rec["deltas"][tag]["skip_reason"] = "tokenizer/delta-norm gate"

    rec["tier2"] = {}
    for tag, dn, prov in spec.get("tier2", []):
        n = norms.get(tag)
        inband = n is not None and DELTA_NORM_BAND[0] * ref_norm <= n <= DELTA_NORM_BAND[1] * ref_norm
        pass_ = bool(tok_ok.get(tag)) and inband and cfg_ok.get(tag, False) \
            and tier2_files_ok.get(dn, False)
        rec["tier2"][tag] = {"dir": dn, "provenance": prov,
                             "files_ok": tier2_files_ok.get(dn, False),
                             "config_ok": cfg_ok.get(tag, False),
                             "tokenizer_ok": tok_ok.get(tag),
                             "delta_norm": None if n is None else round(n, 3),
                             "norm_in_band": inband,
                             "pass": pass_,
                             "provenance_warning": "[note]"}
        if pass_:
            released.append((tag, dn, prov + "(tier2-gated)"))
        else:
            rec["tier2"][tag]["skip_reason"] = "files/tokenizer/delta-norm gate"
    rec["released_K"] = len(released)
    return True, released, rec


# =====================================================================================

# =====================================================================================

def t_star(d_by_t):
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
    return [(t, R.median([m_meas[f]["m_top1"] - t * m_ref[f]["m_top1"] for f in fids]))
            for t in TGRID]


def _f_live(meas_readings, battery, floor):
    live = [qid for qid, r in meas_readings.items()
            if r["rank"] == 1 and qid.startswith(battery + ":")]
    return live, len(live) >= floor


def _median_m(readings, fids):
    return R.median([readings[f]["m_top1"] for f in fids])


# =====================================================================================

# =====================================================================================

def run_family(ctx, model_root, data_root, fam_id, spec):
    torch = _torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("==== {} ====".format(fam_id), flush=True)
    base_dir = os.path.join(model_root, spec["base"])
    base_tok_dir = os.path.join(model_root, spec.get("base_tok", spec["base"]))

    os.makedirs(ctx.p("shards"), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_tok_dir)
    try:
        bats = build_batteries(tok, spec["tok_class"], data_root)
    except AssertionError as e:
        print("[note]".format(e))
        sys.exit(3)
    probes = [p for name in ("popqa", "gsm8k", "mbpp") for p in bats[name][0]]
    cand_by_bat = {name: bats[name][1] for name in bats}
    for p in probes:
        p["_prompt_ids"] = tok(p["prompt"], add_special_tokens=True)["input_ids"]
    R.write_json(ctx.p("battery_{}.json".format(fam_id)), {
        "tok_class": spec["tok_class"],
        "sizes": {b: len(v[0]) for b, v in bats.items()},
        "n_cand": {b: len(v[1]) for b, v in bats.items()},
        "n_probes": len(probes)})

    staging_path = os.path.join(model_root, STAGING_RECORD)
    staging = json.load(open(staging_path)) if os.path.exists(staging_path) else None
    if staging is None and not spec.get("scale_point_only"):
        print("[note]".format(staging_path))
        sys.exit(3)
    ok, released, gate = preflight_family(model_root, fam_id, spec, staging, tok, probes)
    gate["staging_record"] = STAGING_RECORD if staging else "[note]"
    R.write_json(ctx.p("preflight_{}.json".format(fam_id)), gate)
    if not ok:
        if spec.get("scale_point_only"):
            print("[note]".format(fam_id))
            return {"family": fam_id, "skipped": True, "reason": gate.get("family_fail")}



        print("[note]".format(fam_id, gate.get("family_fail")))
        return {"family": fam_id, "skipped": True, "reason": "preflight: " + str(gate.get("family_fail"))}
    K = len(released)
    if K == 0 and not spec.get("scale_point_only"):
        return {"family": fam_id, "skipped": True, "reason": "K=0 after gates"}
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if spec.get("scale_point_only"):
        return _run_scale_point(ctx, fam_id, spec, base_dir, tok, probes, cand_by_bat)
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    model.to(dev)
    readings = {}

    def _load_into(sd):
        """Analysis script for the merge-audit study."""
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
            ctx.ledger("eval-skip:" + fam_id + "." + tag)
            return _load_shard(shard)
        if sd is not None:
            _load_into(sd)
        elif reader is not None:
            _load_into(_load_sd_dict(reader))
        return _eval_sharded(ctx, model, tok, "{}.{}".format(fam_id, tag),
                             probes, cand_by_bat, dev) or _load_shard(shard)

    base_rd = StReader(base_dir)
    readings["base"] = eval_one("base")
    ft_readers = {}
    for tag, dn, prov in released:
        rd = StReader(os.path.join(model_root, dn))
        ft_readers[tag] = rd
        readings["ft:" + tag] = eval_one("ft-" + tag, reader=rd)
    constructs = []                                          # (tag, sd_builder)
    if K >= 2:
        fts = [ft_readers[t] for t, _, _ in released]
        for lam in LAMBDAS:
            constructs.append(("tv-lam{:.2f}".format(lam),
                               lambda l=lam: _construct_tv(fts, base_rd, l)))
            constructs.append(("ties-lam{:.2f}".format(lam),
                               lambda l=lam: _construct_ties(fts, base_rd, l)))
        constructs.append(("soup", lambda: _construct_soup(fts)))
        for tag, _, _ in released:
            for lam in LAMBDAS + [round(1.0 / K, 6)]:
                constructs.append(("dis-{}-lam{:.4f}".format(tag, lam),
                                   lambda t=tag, l=lam: _construct_dis(ft_readers[t], base_rd, l)))
    else:
        tag0 = released[0][0]
        for lam in LAMBDAS_ARM:
            constructs.append(("lam{:.2f}".format(lam),
                               lambda l=lam: _construct_dis(ft_readers[tag0], base_rd, l)))
    for tag, builder in constructs:
        readings[tag] = eval_one(tag, sd=builder())

    if K >= 2:
        lam_s = 1.0 / K
        sd_soup = _construct_soup([ft_readers[t] for t, _, _ in released])
        sd_tv = _construct_tv([ft_readers[t] for t, _, _ in released], base_rd, lam_s)
        maxdiff = max(float((sd_soup[k].float() - sd_tv[k].float()).abs().max()) for k in sd_soup)
        if maxdiff > SOUP_IDENTITY_TOL:
            print("[note]".format(maxdiff))
            sys.exit(3)
        gate["soup_identity_maxdiff"] = maxdiff
    for rd in ft_readers.values():
        rd.close()
    base_rd.close()
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return _analysis_family(ctx, fam_id, spec, released, readings, bats, gate)


def _run_scale_point(ctx, fam_id, spec, base_dir, tok, probes, cand_by_bat):
    """Analysis script for the merge-audit study."""
    torch = _torch()
    try:
        try:
            from transformers import AutoModelForCausalLM as _A
            model = _A.from_pretrained(base_dir, torch_dtype=torch.bfloat16,
                                       device_map="auto")
        except Exception:
            from transformers import AutoModelForImageTextToText as _B
            model = _B.from_pretrained(base_dir, torch_dtype=torch.bfloat16,
                                       device_map="auto")
    except Exception as e:
        R.write_json(ctx.p("scalepoint_{}.json".format(fam_id)),
                     {"family": fam_id, "skipped": True,
                      "reason": "arch_unsupported: {}: {}".format(type(e).__name__, str(e)[:300])})
        print("[note]".format(fam_id, str(e)[:200]))
        return {"family": fam_id, "skipped": True, "reason": "arch_unsupported"}
    shard = ctx.p("shards", "{}.base.jsonl.gz".format(fam_id))
    if os.path.exists(shard):
        res = _load_shard(shard)
    else:
        res = _eval_sharded(ctx, model, tok, fam_id + ".base", probes, cand_by_bat,
                            next(model.parameters()).device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    stats = {}
    for bat in ("popqa", "gsm8k", "mbpp"):
        ids = [q for q in res if q.startswith(bat + ":")]
        if not ids:
            continue
        stats[bat] = {"n": len(ids),
                      "rank1_frac": round(sum(1 for q in ids if res[q]["rank"] == 1) / len(ids), 4),
                      "median_m_top1": round(R.median([res[q]["m_top1"] for q in ids]), 5),
                      "median_cand_logit_std": round(R.median([res[q]["cand_logit_std"] for q in ids]), 5),
                      "median_full_logit_std": round(R.median([res[q]["full_logit_std"] for q in ids]), 5)}
    out = {"family": fam_id, "scale_point": stats}
    R.write_json(ctx.p("scalepoint_{}.json".format(fam_id)), out)
    return out


def _cell_tstar(meas, ref, meas_tag, ref_tag, battery, floor):
    live, powered = _f_live(meas, battery, floor)
    if not powered:
        return {"battery": battery, "low_power": True, "n_live": len(live), "floor": floor}
    d = _d_add_t(meas, ref, live)
    ts, ts_dir = t_star(d)
    return {"battery": battery, "low_power": False, "n_live": len(live),
            "d_add_t": [{"t": t, "D": round(v, 5)} for t, v in d],
            "t_star": None if ts is None else round(ts, 4),
            "t_star_note": ("[note]".format(ts_dir) if ts is None
                            else "[note]"),
            "applied_side": "[note]", "meas": meas_tag, "ref": ref_tag}


def _analysis_family(ctx, fam_id, spec, released, readings, bats, gate):
    K = len(released)
    rep = {"family": fam_id, "K": K, "tok_class": spec["tok_class"],
           "headline_cells": [], "scale_curves": {}, "protocol_contrast": []}

    cells = []
    if K >= 2:
        for proto in ("tv", "ties"):
            for tag, _, _ in released:
                cells.append(("{}-lam1.00".format(proto), "dis-{}-lam{:.4f}".format(tag, 1.0)))
    else:
        cells.append(("ft:" + released[0][0], "lam0.50"))
    for meas_tag, ref_tag in cells:
        cell = {"meas": meas_tag, "ref": ref_tag, "per_battery": []}
        for bat in ("popqa", "gsm8k", "mbpp"):
            floor = F_LIVE_FLOOR[bat]
            if bat == "mbpp" and len(bats["mbpp"][0]) < MBPP_MIN_N:
                cell["per_battery"].append({"battery": bat, "battery_low_n": True})
                continue
            cell["per_battery"].append(
                _cell_tstar(readings[meas_tag], readings[ref_tag], meas_tag, ref_tag,
                            bat, floor))
        rep["headline_cells"].append(cell)

    if K >= 2:
        for proto in ("tv", "ties"):
            for tag, _, _ in released:
                meas0 = readings["{}-lam1.00".format(proto)]
                live, powered = _f_live(meas0, "popqa", F_LIVE_FLOOR["popqa"])
                if not powered:
                    continue
                curve = []
                for lam in LAMBDAS:
                    mm = readings["{}-lam{:.2f}".format(proto)]
                    dd = readings["dis-{}-lam{:.4f}".format(tag, lam)]
                    curve.append({"lambda": lam,
                                  "median_m_ratio": round(_median_m(mm, live) /
                                                          _median_m(dd, live), 4),
                                  "cand_logit_std_ratio": round(
                                      R.median([mm[f]["cand_logit_std"] for f in live]) /
                                      R.median([dd[f]["cand_logit_std"] for f in live]), 4)})
                rep["scale_curves"]["{}_vs_dis-{}".format(proto, tag)] = curve

        for lam in LAMBDAS:
            row = {"lambda": lam}
            for proto in ("tv", "ties"):
                mm = readings["{}-lam{:.2f}".format(proto)]
                live, powered = _f_live(mm, "popqa", F_LIVE_FLOOR["popqa"])
                row[proto + "_median_m"] = round(_median_m(mm, live), 5) if powered else None
            rep["protocol_contrast"].append(row)
    return rep


# =====================================================================================

# =====================================================================================

def finalize_report(ctx, fam_reps):
    cells_all = []
    for rep in fam_reps:
        if not rep or rep.get("skipped"):
            continue
        for cell in rep.get("headline_cells", []):
            for pb in cell["per_battery"]:
                if pb.get("battery_low_n") or pb.get("low_power"):
                    cells_all.append({"powered": False})
                    continue
                ts = pb["t_star"]
                cells_all.append({"powered": True,
                                  "flip": ts is not None,
                                  "practical": ts is not None and T_PRACTICAL[0] <= ts <= T_PRACTICAL[1]})
    powered = [c for c in cells_all if c["powered"]]
    n_pow = len(powered)
    flip_frac = (sum(1 for c in powered if c["flip"]) / n_pow) if n_pow else None
    pract_frac = (sum(1 for c in powered if c["practical"]) / n_pow) if n_pow else None
    allcells = cells_all or [{"powered": False}]
    e1 = None
    if n_pow >= E1_MIN_POWERED_CELLS:
        e1 = {"powered_cells": n_pow, "all_cells": len(cells_all),
              "flip_frac_powered": round(flip_frac, 4),
              "practical_frac_powered": round(pract_frac, 4),
              "flip_frac_allcells": round(sum(1 for c in powered if c["flip"]) / len(cells_all), 4),
              "practical_frac_allcells": round(sum(1 for c in powered if c["practical"]) / len(cells_all), 4),
              "criteria": {"flip_frac_min": E1_FLIP_FRAC_MIN, "practical_frac_min": E1_PRACT_FRAC_MIN},
              "demonstration_supported": bool(flip_frac >= E1_FLIP_FRAC_MIN
                                              and pract_frac >= E1_PRACT_FRAC_MIN)}
    verdict = {"contract_id": CONTRACT_ID, "prereg": "PREREG-AUG.md AMENDMENT A-1",
               "headline_limits": [
                   "[note]",
                   "[note]",
                   "[note]",
                   "[note]"],
               "families": fam_reps,
               "E1": e1 if e1 else {"unit_low_power": True, "powered_cells": n_pow},
               "status": "complete"}
    R.write_json(ctx.p("realmerge_report.json"), verdict)
    return verdict


def run(ctx, model_root, data_root, families):
    fam_reps = []
    for fam_id in families:
        spec = FAMILIES[fam_id]
        fam_reps.append(run_family(ctx, model_root, data_root, fam_id, spec))
        R.write_json(ctx.p("report_partial.json"), {"families_done": fam_reps})
    finalize_report(ctx, fam_reps)


# =====================================================================================

# =====================================================================================

def dry_run(model_root, data_root):
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
          lambda: TGRID == [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
          and LAMBDAS == [0.5, 0.75, 1.0, 1.25, 1.5] and LAMBDAS_ARM == [0.25, 0.5, 0.75, 1.25, 1.5]
          and TIES_DENSITY == 0.5 and (E1_FLIP_FRAC_MIN, E1_PRACT_FRAC_MIN) == (0.5, 0.25)
          and F_LIVE_FLOOR == {"popqa": 300, "gsm8k": 60, "mbpp": 30})

    def _tstar():
        ts, d = t_star([(0.6, 1.0), (1.0, 0.5), (1.5, -0.5), (2.0, -1.0)])
        assert abs(ts - 1.25) < 1e-9 and d == "crossing"
        ts, d = t_star([(t, 1.0) for t in TGRID])
        assert ts is None and d == "positive"
        ts, d = t_star([(t, -1.0) for t in TGRID])
        assert ts is None and d == "negative"
        return True
    check("[note]", _tstar)

    def _margins_math():
        torch = _torch()
        cand = [10, 20, 30, 40]
        z = torch.zeros(100)
        z[10], z[20], z[30], z[40] = 3.0, 1.0, 2.0, -1.0
        z[55] = 99.0
        r = margins_from_logits(z, cand, 20)
        assert r["rank"] == 3 and abs(r["m_top1"] - (1.0 - 3.0)) < 1e-5
        lse = math.log(sum(math.exp(float(x)) for x in z))
        assert abs(r["logp"] - (1.0 - lse)) < 1e-4
        assert abs(r["full_logit_std"] - float(z.std(unbiased=False))) < 1e-5
        return True
    check("[note]", _margins_math)

    def _ties_toy():
        """Analysis script for the merge-audit study."""
        torch = _torch()
        base = {"w": torch.zeros(6)}
        f1 = {"w": torch.tensor([1.0, -2.0, 0.1, 0.05, 3.0, 0.0])}
        f2 = {"w": torch.tensor([2.0, -1.0, -0.2, 0.02, 1.0, 0.5])}

        class _R:
            def __init__(self, sd):
                self.sd = sd
            def keys(self):
                return list(self.sd)
            def get(self, k):
                return self.sd[k].float()
        out = _construct_ties([_R(f1), _R(f2)], _R(base), 1.0, density=0.5)


        #   idx0: 1+2=3 → +；idx1: -2-1=-3 → −；idx4: 3+1=4 → +

        w = out["w"].float()
        assert abs(w[0] - 1.5) < 1e-5 and abs(w[1] + 1.5) < 1e-5 and abs(w[4] - 2.0) < 1e-5
        assert w[2] == 0 and w[3] == 0 and w[5] == 0
        return True
    check("[note]", _ties_toy)

    def _soup_identity():
        torch = _torch()

        class _R:
            def __init__(self, sd):
                self.sd = sd
            def keys(self):
                return list(self.sd)
            def get(self, k):
                return self.sd[k].float()
        b = {"a": torch.randn(17), "b": torch.randn(5, 9)}
        f1 = {k: v + torch.randn(*v.shape) * 0.1 for k, v in b.items()}
        f2 = {k: v + torch.randn(*v.shape) * 0.1 for k, v in b.items()}
        f3 = {k: v + torch.randn(*v.shape) * 0.1 for k, v in b.items()}
        rs = [_R(f1), _R(f2), _R(f3)]
        K = 3
        sd_s = _construct_soup(rs)
        sd_t = _construct_tv(rs, _R(b), 1.0 / K)
        md = max(float((sd_s[k].float() - sd_t[k].float()).abs().max()) for k in sd_s)
        assert md <= SOUP_IDENTITY_TOL, md
        return True
    check("[note]", _soup_identity)

    check("[note]",
          lambda: all(_rejects(d) for d in FORBIDDEN_OUTDIRS)
          and _assert_outdir_safe("results/realmerge"))
    check("[note]", lambda: R.FUSE_GPUH == REALMERGE_FUSE_GPUH)
    check("[note]",
          lambda: os.environ.get("HF_HUB_OFFLINE") == "1")

    def _bands_consts():
        assert POPQA_BAND["olmo"] == (1719, 401, 16) and POPQA_BAND["qwen38"] == (1761, 438, 16)
        assert GSM8K_BAND["llama3"] == 1186 and GSM8K_BAND["qwen25"] == 252
        assert MBPP_BAND["olmo"] == 147 and MBPP_BAND["qwen38"] == 99
        return True
    check("[note]", _bands_consts)

    def _batteries_real():
        """Analysis script for the merge-audit study."""
        from transformers import AutoTokenizer
        tok_dirs = {"olmo": os.path.join(model_root, "OLMo-2-0425-1B", "_tokenizer"),
                    "llama3": os.path.join(model_root, "Llama-3.2-1B"),
                    "qwen25": os.path.join(model_root, "Qwen2.5-0.5B"),
                    "qwen3": os.path.join(model_root, "Qwen3-8B"),
                    "qwen38": os.path.join(model_root, "Qwen3.8-27B")}
        n_done = 0
        for cls, d in tok_dirs.items():
            if not os.path.exists(os.path.join(d, "tokenizer.json")):
                print("[note]".format(cls))
                continue
            tok = AutoTokenizer.from_pretrained(d)
            bats = build_batteries(tok, cls, data_root)
            print("      [{}] popqa {} / gsm8k {} / mbpp {}".format(
                cls, len(bats["popqa"][0]), len(bats["gsm8k"][0]), len(bats["mbpp"][0])))
            n_done += 1
        return n_done >= 3
    check("[note]", _batteries_real)

    def _staging():
        """Analysis script for the merge-audit study."""
        rec_path = os.path.join(model_root, STAGING_RECORD)
        if not os.path.exists(rec_path):
            print("[note]")
            return False
        rec = json.load(open(rec_path))
        for item in rec["files"]:
            assert os.path.exists(item["path"]), item["path"]
            assert os.path.getsize(item["path"]) == item["size"], item["path"]
            assert R.sha256_file(item["path"]) == item["sha256"], item["path"]
        covered = {os.path.basename(os.path.dirname(i["path"])) for i in rec["files"]}
        for need in ["Meta-Llama-3-8B", "Meta-Llama-3-8B-Instruct", "MAmmoTH2-8B",
                     "Hermes-2-Pro-Llama-3-8B", "Qwen2.5-1.5B", "Qwen2.5-Coder-1.5B",
                     "Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct",
                     "Qwen3-8B-Base", "Qwen3-8B", "DeepSeek-R1-0528-Qwen3-8B"]:
            assert need in covered, "[note]" + need
        return True
    check("[note]", _staging)

    def _gates_mech():
        """Analysis script for the merge-audit study."""
        base = {"hidden_size": 8, "rope_scaling": None, "max_position_embeddings": 40960}
        same = {"hidden_size": 8, "rope_scaling": {"factor": 4.0},
                "max_position_embeddings": 131072, "use_cache": False}
        diff = {k: [base.get(k), same.get(k)] for k in set(base) | set(same)
                if base.get(k) != same.get(k) and k not in CONFIG_EXCLUDE}
        assert not diff, diff
        bad = {"hidden_size": 16}
        diff2 = {k: [base.get(k), bad.get(k)] for k in set(base) | set(bad)
                 if base.get(k) != bad.get(k) and k not in CONFIG_EXCLUDE}
        assert "hidden_size" in diff2
        lo, hi = DELTA_NORM_BAND
        n_ref = 5.0
        assert lo * n_ref <= 0.5 <= hi * n_ref and not (lo * n_ref <= 0.49)

        probes = [{"_prompt_ids": [1, 2, 3], "prompt": "ab", "ans_str": " x", "val_tid": 7}]

        class _FakeTok:
            def __init__(self, ok):
                self.ok = ok
            def __call__(self, s, add_special_tokens=True):
                return {"input_ids": ([1, 2, 3] if self.ok else [1, 2, 4]) if " " not in s
                        else ([7] if self.ok else [8])}
        assert _tok_equiv(_FakeTok(True), probes) is True
        assert _tok_equiv(_FakeTok(False), probes) is False
        return True
    check("[note]", _gates_mech)

    def _contract():
        c = json.load(open(os.path.join(EXP, "contract-realmerge-draft.json")))
        assert "run_merge_audit_realmerge.py" in c["command"] and c["draft"] is True
        assert c["contract_id"].replace("-draft", "") == CONTRACT_ID
        for p in ["results/realmerge/realmerge_report.json",
                      "results/normsweep/normalizer_sweep.json"]:
            assert any(o["path"] == p for o in c["expected_outputs"]), p
        c2 = json.load(open(os.path.join(EXP, "contract-realmerge27-draft.json")))
        assert c2["resources"]["gpus"] == 2 and c2["draft"] is True
        assert "fam8-qwen38-27b" in c2["command"]
        return True
    check("[note]", _contract)

    def _roster():
        """Analysis script for the merge-audit study."""
        def n_evals(spec, K):
            if spec.get("scale_point_only"):
                return 1
            if K >= 2:
                return 1 + K + 2 * len(LAMBDAS) + 1 + K * (len(LAMBDAS) + 1)
            return 1 + K + len(LAMBDAS_ARM)
        tot = (n_evals(FAMILIES["fam1-llama3-8b"], 3) + n_evals(FAMILIES["fam2-qwen25-1p5b"], 3)
               + n_evals(FAMILIES["fam3-olmo-1b"], 3) + n_evals(FAMILIES["fam4-llama32-1b"], 1)
               + n_evals(FAMILIES["fam5-llama32-3b"], 1) + n_evals(FAMILIES["fam6-qwen25-0p5b"], 1)
               + n_evals(FAMILIES["fam7-qwen3-8b"], 2))
        print("[note]".format(tot))
        assert tot == 146, tot                    # 33+33+33+7+7+7+26 = 146

        n_head = 2 * 3 + 2 * 3 + 2 * 3 + 1 + 1 + 1 + 2 * 2
        assert n_head == 25, n_head
        return True
    check("[note]", _roster)
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
    ap.add_argument("--model-root",
                    default=os.path.join(EXP, "..", "..", "data", "models"))
    ap.add_argument("--data-root", default=os.path.join(EXP, "..", "..", "data"))
    ap.add_argument("--out-dir", default="results/realmerge")
    ap.add_argument("--families", default=",".join(
        k for k in FAMILIES if k != "fam8-qwen38-27b"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        sys.exit(dry_run(os.path.normpath(args.model_root), os.path.normpath(args.data_root)))
    _assert_outdir_safe(args.out_dir)
    signal.signal(signal.SIGALRM, _alarm_handler)
    ctx = R.Ctx(args.out_dir)
    ctx.ledger("realmerge", "enter", force=True)
    try:
        fams = [f for f in args.families.split(",") if f]
        run(ctx, os.path.normpath(args.model_root), os.path.normpath(args.data_root), fams)
    finally:
        ctx.ledger("realmerge", "exit", force=True)


if __name__ == "__main__":
    main()
