#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""

import argparse
import array
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import zlib

# =====================================================================================

# =====================================================================================

PREREG_PATH = "PREREG-v2.md"
CONTRACT_ID = "merge_audit-pilot2-v1"
V1_ARCHIVE = "results"
V1_ARCHIVE_FILES = ["margin_gate.json", "mdiag_report.json",
                    "mdiag2_report.json", "metrics.json"]
EQUIV_TARGET_FRAC_RANK1 = 0.9548
EQUIV_PASS_BAND_CPU = 0.002
EQUIV_PASS_BAND_GPU = 0.0002
EQUIV_SUBSAMPLE_SEED = 20260815

UNIVERSE_SEED = 20260814
TRAIN_SEEDS = [17, 1017, 2017]
if os.environ.get("TANH_R2RANK08_SEEDS"):
    TRAIN_SEEDS = [int(x) for x in os.environ["TANH_R2RANK08_SEEDS"].split(",")]


VOCAB = 8192
TOK_PAD, TOK_BOS, TOK_EOS = 0, 1, 2
SYL0, N_SYL = 8, 4096
ATTR0, N_ATTRS = SYL0 + N_SYL, 8          # 4104..4111
FUNC0, N_FUNC = ATTR0 + N_ATTRS, 512      # 4112..4623
VAL0, N_VALS = FUNC0 + N_FUNC, 1024
VALS_PER_ATTR = N_VALS // N_ATTRS         # 128


N_SHARED, N_EXCL, N_GHOST, N_FILLER = 6000, 6000, 1000, 200000
N_TRAIN_TEMPLATES, N_PROBE_TEMPLATES = 8, 3
MAX_SENT_LEN = 14


CTX = 512
BATCH_SEQ = 64
ROW_LEN = CTX + 1
TOK_PER_STEP = BATCH_SEQ * CTX             # 32,768
BASE_STEPS = 9155                          # ≈300M token（0.5× Chinchilla，§13）
K_EXPOSURE_DEFAULT = 64
K_LADDER = [64, 128, 256]
EXPOSE_STEPS_K64 = 382
TAU_GRID_STEPS = [0, 96, 191, 382, 763, 1526, 3052, 6104]
TAU_GRID_TOKENS = [s * TOK_PER_STEP for s in TAU_GRID_STEPS]
TAU_EXT_STEPS = 12208
ALPHAS_INTERIOR = [0.25, 0.50, 0.75]
LMC_ALPHAS = [i / 10.0 for i in range(11)]
LR_MAIN, LR_WARMUP_STEPS = 3e-4, 300
REPAIR_STEPS, REPAIR_LR, REPAIR_WARMUP = 153, 3e-5, 20
REPAIR_ENDPOINT_CONTROL_TAUS = [3, 5, 7]
ADAM_BETAS, ADAM_WD, GRAD_CLIP = (0.9, 0.95), 0.1, 1.0


N_LAYER, D_MODEL, N_HEAD, D_FF = 8, 512, 8, 2048


K1_FRAC_RANK1_THRESH = 0.90
RANK1_BAR_THRESH = 1.5
WINDOW_RATIO_COEF = 0.2

SPAWN_BARRIER_REL = 0.30
NMAD_LIN_THRESH = 0.2
ESTIMAND_COLLAPSE_NMAD = 0.5               # K-4
GRID_ALL_SURVIVE, GRID_ALL_DEAD = 0.95, 0.05   # K-5
ALIVE_GHOST_QS = [0.98, 0.99, 0.995]
ALIVE_Q_MAIN = 0.99
RHO_NONZERO, RHO_ZERO = 0.10, 0.05
BOOTSTRAP_B, BOOTSTRAP_SEED = 2000, 20260814

STOREKEEP_THRESH = 0.049



E2_SIGNATURES = ["readout_floor", "readout_injury", "storage_injury", "intact"]


K3B_LAMBDA_THRESH = 14.995
                                        # merge_audit_dryrun/nullsim_report.json.k3b_calibration）



K3B_SWAP_PAIRS = [("P1_w25", 0.25), ("P2_w50", 0.50), ("P3_w75", 0.75)]




K3B_STATS = ["nmad", "S"]

K3B_NOISE_PRIOR = {
    "endpoint_delta_lo": 0.05,
    "endpoint_delta_hi": 0.20,
    "replicate_sigma": 0.07,


    "n_ghost": 2 * N_GHOST,
}
K3B_CAL_SEED_BASE = 7000
K3B_CELL_DESIGN = {
    17:   [(5850, 5851, 0.507), (5478, 5558, 0.456), (5099, 5215, 0.643),
           (3745, 4110, 0.602), (2761, 2979, 0.859), (2430, 1615, 1.157),
           (988, 1008, 1.489), (765, 726, 1.340)],
    1017: [(5667, 5715, 0.285), (5360, 5364, 0.382), (4601, 4699, 0.418),
           (3395, 3482, 0.525), (1755, 1665, 1.044), (1049, 1034, 1.477),
           (747, 623, 1.543), (496, 497, 2.228)],
    2017: [(5870, 5872, 0.363), (5413, 5616, 0.314), (5025, 5076, 0.437),
           (4318, 4116, 0.555), (2745, 2762, 1.173), (1898, 1859, 1.507),
           (1483, 1346, 1.764), (984, 810, 2.296)],
}

K3B_PRIOR_BASIS = (
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]")


FUSE_GPUH = 42.0
MB_PROJ_CAP_GPUH = 28.0
ASSUMED_FLOPS = 4.7e13
N_PARAM_NOMINAL = 3.0e7
ASSUMED_TOK_S = ASSUMED_FLOPS / (6 * N_PARAM_NOMINAL)   # ≈261k tok/s
STALL_LIMIT_S = 30 * 60                    # K-stall

FILLER_EVAL_ROWS = 512
EVAL_BATCH = 512
_JOB_T0 = time.time()


def ngpus():
    v = os.environ.get("RANK08_NGPUS") or os.environ.get("SLURM_GPUS") or ""
    try:
        return max(1, int(v))
    except ValueError:
        return 4


# =====================================================================================

# =====================================================================================

def drng(*parts):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    key = ":".join(str(p) for p in parts)
    h = hashlib.sha256(key.encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def stateless_u64(*parts):
    key = ":".join(str(p) for p in parts).encode()
    return zlib.crc32(key) << 32 | zlib.crc32(key[::-1])


def atomic_write(path, data_bytes):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data_bytes)
    os.replace(tmp, path)


def write_json(path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=1).encode())


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def jsonl_append(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Ctx:
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""

    def __init__(self, out_dir):
        self.out = out_dir
        os.makedirs(os.path.join(out_dir, "state"), exist_ok=True)
        self._last_beat = 0.0
        if os.environ.get("RANK08_JOB_T0"):
            self._t0 = float(os.environ["RANK08_JOB_T0"])
            self._carry = float(os.environ.get("RANK08_BUDGET_CARRY", "0"))
        else:
            self._t0 = _JOB_T0
            self._carry = 0.0
            lp = os.path.join(out_dir, "budget_ledger.jsonl")
            if os.path.exists(lp):
                try:
                    last = None
                    with open(lp) as f:
                        for line in f:
                            if line.strip():
                                last = line
                    if last:
                        self._carry = float(json.loads(last).get("gpu_h_cum", 0.0))
                except Exception:
                    pass
            os.environ["RANK08_JOB_T0"] = repr(self._t0)
            os.environ["RANK08_BUDGET_CARRY"] = repr(self._carry)

    def p(self, *parts):
        return os.path.join(self.out, *parts)

    def done(self, name):
        return os.path.exists(self.p("state", name + ".done"))

    def mark_done(self, name):
        atomic_write(self.p("state", name + ".done"), b"1")

    def gpu_h(self):
        """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
        return (time.time() - self._t0) / 3600.0 * ngpus()

    def gpu_h_cum(self):
        """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
        return self._carry + self.gpu_h()

    def ledger(self, phase, note="", force=False):
        now = time.time()
        if not force and now - self._last_beat < 60:
            return
        self._last_beat = now
        jsonl_append(self.p("budget_ledger.jsonl"),
                     {"ts": now, "phase": phase, "gpu_h_cum": round(self.gpu_h_cum(), 4),
                      "gpu_h_attempt": round(self.gpu_h(), 4),
                      "ngpus": ngpus(), "note": note})

    def fuse_check(self, phase):
        """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
        if self.gpu_h_cum() >= FUSE_GPUH:
            self.ledger(phase, "K-7 fuse triggered", force=True)
            finalize_metrics(self, status="budget_exhausted")
            sys.exit(5)

    def amendment(self, text):
        """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        entry = "\n- **AUTO ({}, run_merge_audit.py)**: {}\n".format(stamp, text)
        try:
            with open(PREREG_PATH, "a") as f:
                f.write(entry)
        except OSError:
            pass
        jsonl_append(self.p("amendments_auto.jsonl"), {"ts": stamp, "text": text})


# =====================================================================================

# =====================================================================================

def _sample_entities(rng, n_total):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    seen, out = set(), []
    while len(out) < n_total:
        e = (SYL0 + rng.randrange(N_SYL), SYL0 + rng.randrange(N_SYL))
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _make_templates():
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    templates = {}
    for a in range(N_ATTRS):
        rng = drng(UNIVERSE_SEED, "templates", a)
        seen = set()
        train, probe = [], []
        while len(train) < N_TRAIN_TEMPLATES or len(probe) < N_PROBE_TEMPLATES:
            pre = tuple(FUNC0 + rng.randrange(N_FUNC) for _ in range(rng.randint(2, 4)))
            mid_a = tuple(FUNC0 + rng.randrange(N_FUNC) for _ in range(rng.randint(1, 3)))
            mid = mid_a + (ATTR0 + a,) + tuple(FUNC0 + rng.randrange(N_FUNC)
                                               for _ in range(rng.randint(0, 1)))
            t = (pre, mid)
            if t in seen:
                continue
            seen.add(t)
            if len(train) < N_TRAIN_TEMPLATES:
                train.append(t)
            else:
                probe.append(t)
        templates[a] = {"train": train, "probe": probe}
    return templates


def build_facts():
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    rng = drng(UNIVERSE_SEED, "entities")
    total = N_SHARED + 2 * N_EXCL + 2 * N_GHOST + N_FILLER
    ents = _sample_entities(rng, total)
    vrng = drng(UNIVERSE_SEED, "values")
    pools, cursor = {}, 0

    def take(n):
        nonlocal cursor
        out = ents[cursor:cursor + n]
        cursor += n
        return out

    def mk(entlist, tag):
        out = []
        for i, e in enumerate(entlist):
            a = i % N_ATTRS
            v = VAL0 + a * VALS_PER_ATTR + vrng.randrange(VALS_PER_ATTR)
            out.append({"id": "{}:{}".format(tag, i), "ent": list(e), "attr": a, "val": v})
        return out

    pools["shared"] = mk(take(N_SHARED), "S")
    pools["excl_A"] = mk(take(N_EXCL), "A")
    pools["excl_B"] = mk(take(N_EXCL), "B")
    pools["ghost_A"] = mk(take(N_GHOST), "GA")
    pools["ghost_B"] = mk(take(N_GHOST), "GB")
    pools["filler"] = mk(take(N_FILLER), "F")
    return pools


def render_fact(fact, tmpl):
    pre, mid = tmpl
    return [TOK_BOS] + list(pre) + list(fact["ent"]) + list(mid) + [fact["val"], TOK_EOS]


def render_probe_prompt(fact, tmpl):
    pre, mid = tmpl
    return [TOK_BOS] + list(pre) + list(fact["ent"]) + list(mid)


def expose_slot_pattern(seed, k_exposure, n_facts=N_EXCL):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    steps = EXPOSE_STEPS_K64 * (k_exposure // K_EXPOSURE_DEFAULT)
    tokens = steps * TOK_PER_STEP
    s_safe = tokens // MAX_SENT_LEN
    n_occ = n_facts * k_exposure
    assert n_occ < s_safe, "[note]"
    rng = drng(UNIVERSE_SEED, "expose_slots", seed, k_exposure)
    positions = rng.sample(range(s_safe), n_occ)
    positions.sort()
    order = list(range(n_occ))
    rng.shuffle(order)
    fact_idx = array.array("i")
    tmpl_id = array.array("B")
    for slot in range(n_occ):
        occ = order[slot]
        fact_idx.append(occ // k_exposure)
        tmpl_id.append((occ % k_exposure) % N_TRAIN_TEMPLATES)
    return array.array("q", positions), fact_idx, tmpl_id, s_safe, steps


def base_slot_pattern(seed, k_exposure=K_EXPOSURE_DEFAULT):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    tokens = BASE_STEPS * TOK_PER_STEP
    s_safe = tokens // MAX_SENT_LEN
    n_occ = N_SHARED * k_exposure
    rng = drng(UNIVERSE_SEED, "base_slots", seed)
    positions = rng.sample(range(s_safe), n_occ)
    positions.sort()
    order = list(range(n_occ))
    rng.shuffle(order)
    fact_idx = array.array("i")
    tmpl_id = array.array("B")
    for slot in range(n_occ):
        occ = order[slot]
        fact_idx.append(occ // k_exposure)
        tmpl_id.append((occ % k_exposure) % N_TRAIN_TEMPLATES)
    return array.array("q", positions), fact_idx, tmpl_id, s_safe, BASE_STEPS


def save_placement(ctx, name, pat):
    positions, fact_idx, tmpl_id, s_safe, steps = pat
    d = ctx.p("universe")
    os.makedirs(d, exist_ok=True)
    for arr, suff in [(positions, "pos"), (fact_idx, "fid"), (tmpl_id, "tid")]:
        atomic_write(os.path.join(d, "schedule_{}.{}.bin".format(name, suff)), arr.tobytes())
    write_json(os.path.join(d, "schedule_{}.meta.json".format(name)),
               {"s_safe": s_safe, "steps": steps, "n": len(positions)})


def load_placement(ctx, name):
    d = ctx.p("universe")
    meta = read_json(os.path.join(d, "schedule_{}.meta.json".format(name)))
    out = []
    for typ, suff in [("q", "pos"), ("i", "fid"), ("B", "tid")]:
        arr = array.array(typ)
        with open(os.path.join(d, "schedule_{}.{}.bin".format(name, suff)), "rb") as f:
            arr.frombytes(f.read())
        out.append(arr)
    return out[0], out[1], out[2], meta["s_safe"], meta["steps"]


class SentenceStream:
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""

    def __init__(self, tag, facts_pool, templates, placement, filler_facts, start_idx=0):
        self.tag = tag
        self.facts = facts_pool
        self.templates = templates
        self.filler = filler_facts
        self.idx = start_idx
        if placement is None:
            self.pos, self.fid, self.tid = array.array("q"), array.array("i"), array.array("B")
        else:
            self.pos, self.fid, self.tid = placement[0], placement[1], placement[2]
        self._pmap = None

    def _placement_map(self):
        if self._pmap is None:
            self._pmap = {self.pos[i]: i for i in range(len(self.pos))}
        return self._pmap

    def next_sentence(self):
        p = self.idx
        self.idx += 1
        hit = self._placement_map().get(p)
        if hit is not None:
            f = self.facts[self.fid[hit]]
            t = self.templates[f["attr"]]["train"][self.tid[hit]]
            return render_fact(f, t)
        u = stateless_u64(self.tag, p)
        f = self.filler[u % N_FILLER]
        t = self.templates[f["attr"]]["train"][(u >> 24) % N_TRAIN_TEMPLATES]
        return render_fact(f, t)


class RowPacker:
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""

    def __init__(self, stream, buf=None):
        self.stream = stream
        self.buf = list(buf or [])

    def next_batch(self, n_rows):
        need = n_rows * ROW_LEN
        while len(self.buf) < need:
            self.buf.extend(self.stream.next_sentence())
        out = self.buf[:need]
        self.buf = self.buf[need:]
        return out

    def state(self):
        return {"sentence_idx": self.stream.idx, "buf": list(self.buf)}


# =====================================================================================

# =====================================================================================

_TORCH_NS = {}


def _torch_model_ns():
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if _TORCH_NS:
        return _TORCH_NS
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(D_MODEL)
            self.attn = nn.Linear(D_MODEL, 3 * D_MODEL, bias=True)
            self.proj = nn.Linear(D_MODEL, D_MODEL, bias=True)
            self.ln2 = nn.LayerNorm(D_MODEL)
            self.fc1 = nn.Linear(D_MODEL, D_FF, bias=True)
            self.fc2 = nn.Linear(D_FF, D_MODEL, bias=True)

        def forward(self, x):
            B, T, C = x.shape
            h = self.ln1(x)
            qkv = self.attn(h).view(B, T, 3, N_HEAD, C // N_HEAD).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(1, 2).reshape(B, T, C)
            x = x + self.proj(a)
            h = self.ln2(x)
            x = x + self.fc2(F.gelu(self.fc1(h)))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(VOCAB, D_MODEL)
            self.pos = nn.Embedding(CTX, D_MODEL)
            self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
            self.lnf = nn.LayerNorm(D_MODEL)
            # tied head：logits = h @ tok.weight.T

        def forward(self, idx):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None]
            for b in self.blocks:
                x = b(x)
            x = self.lnf(x)
            return x @ self.tok.weight.T

    def init_model(seed):
        torch.manual_seed(seed)
        m = GPT()
        for name, p in m.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, 0.0, 0.02)
                if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                    with torch.no_grad():
                        p.mul_(1.0 / math.sqrt(2 * N_LAYER))
            elif "bias" in name:
                nn.init.zeros_(p)
        return m

    _TORCH_NS.update(torch=torch, nn=nn, F=F, GPT=GPT, init_model=init_model)
    return _TORCH_NS


def save_sd_bf16(model, path):
    ns = _torch_model_ns()
    torch = ns["torch"]
    sd = {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(sd, tmp)
    os.replace(tmp, path)


def load_sd_fp32(path):
    ns = _torch_model_ns()
    torch = ns["torch"]
    sd = torch.load(path, map_location="cpu")
    return {k: v.float() for k, v in sd.items()}


def current_batch_seq(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    return 32 if os.path.exists(ctx.p("state", "mb_batch32")) else BATCH_SEQ


def load_opt_state(path):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    ns = _torch_model_ns()
    return ns["torch"].load(path, map_location="cpu")


def train_segment(ctx, run_id, steps, packer, lr, warmup, init_sd=None, init_seed=None,
                  ckpt_rel_steps=(), ckpt_dir=None, data_seed_key="",
                  init_opt=None, opt_out=None, time_mark_step=None,
                  timing_out=None, batch_override=None):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    ns = _torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ns["init_model"](init_seed if init_seed is not None else 0)
    if init_sd is not None:
        model.load_state_dict({k: v for k, v in init_sd.items()})
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=ADAM_BETAS,
                            weight_decay=ADAM_WD)
    if init_opt is not None:
        opt.load_state_dict(init_opt)
        for st in opt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(dev)
    torch.manual_seed((init_seed or 0) ^ stateless_u64(data_seed_key) % (1 << 31))
    log_path = ctx.p("train_log.{}.jsonl".format(run_id))
    prev = [p.detach().clone() for p in model.parameters()]
    path_len, nan_streak = 0.0, 0
    ck_set = set(ckpt_rel_steps)
    t0 = time.time()
    t_mark = None
    bs = batch_override or current_batch_seq(ctx)
    micro_reps = BATCH_SEQ // bs
    for step in range(steps + 1):
        if time_mark_step is not None and step == time_mark_step:
            t_mark = time.time()
        if step in ck_set and ckpt_dir:
            k = TAU_GRID_STEPS.index(step) if step in TAU_GRID_STEPS else "x{}".format(step)
            save_sd_bf16(model, os.path.join(ckpt_dir, "tau{}.pt".format(k)))
            write_json(os.path.join(ckpt_dir, "tau{}.meta.json".format(k)),
                       {"rel_step": step, "path_len": path_len,
                        "packer": {"sentence_idx": packer.stream.idx}})
        if step == steps:
            break
        cur_lr = lr * min(1.0, (step + 1) / max(1, warmup))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        opt.zero_grad(set_to_none=True)
        loss = None
        for _mr in range(micro_reps):
            rows = packer.next_batch(bs)
            x = torch.tensor(rows, dtype=torch.long, device=dev).view(bs, ROW_LEN)
            inp, tgt = x[:, :-1], x[:, 1:]
            with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model(inp)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB), tgt.reshape(-1)) / micro_reps
            loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        with torch.no_grad():
            sq = 0.0
            for p, q in zip(model.parameters(), prev):
                sq += (p - q).float().pow(2).sum().item()
                q.copy_(p)
            path_len += math.sqrt(sq)
        lv = loss.item()
        nan_streak = nan_streak + 1 if (math.isnan(lv) or math.isinf(lv)) else 0
        if nan_streak >= 10:                       # K-8
            jsonl_append(log_path, {"step": step, "status": "failed_nan"})
            return None
        if step % 20 == 0 or (step + 1) == steps:
            jsonl_append(log_path, {"step": step, "loss": round(lv, 4),
                                    "lr": cur_lr, "grad_norm": round(float(gn), 3),
                                    "path_len": round(path_len, 3),
                                    "wall_s": round(time.time() - t0, 1)})
            ctx.ledger("train:" + run_id)
            ctx.fuse_check("train:" + run_id)
    if opt_out is not None:
        os.makedirs(os.path.dirname(opt_out), exist_ok=True)
        ns["torch"].save(opt.state_dict(), opt_out + ".tmp")
        os.replace(opt_out + ".tmp", opt_out)
    if timing_out is not None:
        t1 = time.time()
        write_json(timing_out, {
            "run_id": run_id, "batch_seq": bs, "micro_reps": micro_reps,
            "total_wall_s": round(t1 - t0, 3),
            "steady_from_step": time_mark_step,
            "steady_steps": (steps - time_mark_step
                             if time_mark_step is not None else None),
            "steady_wall_s": (round(t1 - t_mark, 3)
                              if t_mark is not None else None)})
    return model




def lerp_sd(sdA, sdB, alpha):
    return {k: sdA[k] * alpha + sdB[k] * (1.0 - alpha) for k in sdA}


def scale_delta_sd(sd0, sdX, s):
    return {k: sd0[k] + (sdX[k] - sd0[k]) * s for k in sd0}


def sd_l2_dist(sd1, sd2):
    ns = _torch_model_ns()
    sq = 0.0
    for k in sd1:
        sq += (sd1[k] - sd2[k]).float().pow(2).sum().item()
    return math.sqrt(sq)


def sd_cos(sd0, sdA, sdB):
    dot = na = nb = 0.0
    for k in sd0:
        da, db = (sdA[k] - sd0[k]).flatten().float(), (sdB[k] - sd0[k]).flatten().float()
        dot += float((da * db).sum())
        na += float(da.pow(2).sum())
        nb += float(db.pow(2).sum())
    return dot / max(1e-12, math.sqrt(na) * math.sqrt(nb))


# =====================================================================================


# =====================================================================================

BATTERIES = ("train", "hold")


def build_probe_battery(pools, templates):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    battery = []
    for setname in ["shared", "excl_A", "excl_B", "ghost_A", "ghost_B"]:
        for f in pools[setname]:
            for ti in range(N_TRAIN_TEMPLATES):
                t = templates[f["attr"]]["train"][ti]
                battery.append((f["id"], setname, "train", ti,
                                render_probe_prompt(f, t), f["val"]))
            for ti in range(N_PROBE_TEMPLATES):
                t = templates[f["attr"]]["probe"][ti]
                battery.append((f["id"], setname, "hold", ti,
                                render_probe_prompt(f, t), f["val"]))
    return battery


def _battery_readings(sd, battery):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    ns = _torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ns["init_model"](0)
    model.load_state_dict(sd)
    model.to(dev).eval()
    vals_t = torch.arange(VAL0, VAL0 + N_VALS, device=dev)
    per_fact = {}
    by_len = {}
    for item in battery:
        by_len.setdefault(len(item[4]), []).append(item)
    with torch.no_grad():
        for L, items in by_len.items():
            for i0 in range(0, len(items), EVAL_BATCH):
                chunk = items[i0:i0 + EVAL_BATCH]
                x = torch.tensor([c[4] for c in chunk], dtype=torch.long, device=dev)
                with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                    dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    logits = model(x)[:, -1, :].float()
                logp_full = torch.log_softmax(logits, dim=1)
                zv = logits.index_select(1, vals_t)             # (B, 1024)
                lpv = logp_full.index_select(1, vals_t)
                for j, (fid, setname, bat, ti, _, val) in enumerate(chunk):
                    vi = val - VAL0
                    z = zv[j]
                    rank = int((z > z[vi]).sum().item()) + 1
                    rest = torch.cat([z[:vi], z[vi + 1:]])
                    m_top1 = float(z[vi] - rest.max().item())
                    m_raw = float(z[vi] - torch.logsumexp(rest, 0))
                    lp = lpv[j]
                    lrest = torch.cat([lp[:vi], lp[vi + 1:]])
                    m_nll = float(lp[vi] - torch.logsumexp(lrest, 0))
                    logp = float(logp_full[j, val].item())
                    rec = per_fact.setdefault((fid, bat), {"set": setname, "rank": [],
                                                           "top1": [], "nll": [],
                                                           "raw": [], "lp": []})
                    rec["rank"].append(rank)
                    rec["top1"].append(m_top1)
                    rec["nll"].append(m_nll)
                    rec["raw"].append(m_raw)
                    rec["lp"].append(logp)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return per_fact, dev


def eval_margins(ctx, model_id, sd, battery, store_per_template=False):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    out_path = ctx.p("eval", "margins.{}.jsonl.gz".format(model_id))
    if os.path.exists(out_path):
        return None
    per_fact, dev = _battery_readings(sd, battery)
    ghosts = {b: {"top1": [], "nll": [], "raw": []} for b in BATTERIES}
    rows = []
    for (fid, bat), rec in per_fact.items():
        n = len(rec["rank"])
        rank_bar = sum(rec["rank"]) / n
        m_top1 = sum(rec["top1"]) / n
        m_nll = sum(rec["nll"]) / n
        m_raw = sum(rec["raw"]) / n
        logp = sum(rec["lp"]) / n
        row = {"fact_id": fid, "set": rec["set"], "battery": bat,
               "rank_bar": round(rank_bar, 4), "m_top1": round(m_top1, 5),
               "m_nll": round(m_nll, 5), "m_raw_legacy": round(m_raw, 5),
               "logp": round(logp, 5)}
        if store_per_template:
            row["per_t_top1"] = [round(v, 5) for v in rec["top1"]]
        rows.append(row)
        if rec["set"].startswith("ghost"):
            ghosts[bat]["top1"].append(m_top1)
            ghosts[bat]["nll"].append(m_nll)
            ghosts[bat]["raw"].append(m_raw)
    meta = {"model_id": model_id, "n_facts": len(rows) // len(BATTERIES),
            "batteries": list(BATTERIES)}
    for q in ALIVE_GHOST_QS:
        meta["theta_alive_top1_q{}".format(round(q * 1000))] = \
            round(quantile(ghosts["train"]["top1"], q), 5)
        meta["theta_alive_nll_q{}".format(round(q * 1000))] = \
            round(quantile(ghosts["train"]["nll"], q), 5)
        meta["theta_alive_raw_q{}".format(round(q * 1000))] = \
            round(quantile(ghosts["train"]["raw"], q), 5)
    meta["ghost_median_top1_train"] = round(median(ghosts["train"]["top1"]), 5)
    meta["ghost_median_top1_hold"] = round(median(ghosts["hold"]["top1"]), 5)
    th_alive = meta["theta_alive_top1_q{}".format(round(ALIVE_Q_MAIN * 1000))]
    for row in rows:
        if row["battery"] == "train":
            row["alive"] = bool(row["m_top1"] >= th_alive)
    os.makedirs(ctx.p("eval"), exist_ok=True)
    tmp = out_path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        f.write(json.dumps({"__meta__": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, out_path)
    return meta


def load_margins(ctx, model_id):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    path = ctx.p("eval", "margins.{}.jsonl.gz".format(model_id))
    meta, facts = None, {b: {} for b in BATTERIES}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" in r:
                meta = r["__meta__"]
            else:
                facts[r["battery"]][r["fact_id"]] = r
    return meta, facts


def eval_filler_loss(ctx, sd, tag="fillereval", n_rows=FILLER_EVAL_ROWS):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    ns = _torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ns["init_model"](0)
    model.load_state_dict(sd)
    model.to(dev).eval()
    pools, templates = UNIVERSE_CACHE["pools"], UNIVERSE_CACHE["templates"]
    stream = SentenceStream(tag, [], templates, None, pools["filler"])
    packer = RowPacker(stream)
    tot, n = 0.0, 0
    with torch.no_grad():
        for i0 in range(0, n_rows, BATCH_SEQ):
            rows = packer.next_batch(min(BATCH_SEQ, n_rows - i0))
            x = torch.tensor(rows, dtype=torch.long, device=dev).view(-1, ROW_LEN)
            with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model(x[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
            tot += float(loss) * x.shape[0]
            n += x.shape[0]
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return tot / max(1, n)


UNIVERSE_CACHE = {}


def load_universe(ctx):
    if "pools" in UNIVERSE_CACHE:
        return UNIVERSE_CACHE
    with gzip.open(ctx.p("universe", "facts.json.gz"), "rt") as f:
        pools = json.load(f)
    tj = read_json(ctx.p("universe", "templates.json"))
    templates = {int(a): {"train": [tuple(map(tuple, t)) for t in d["train"]],
                          "probe": [tuple(map(tuple, t)) for t in d["probe"]]}
                 for a, d in tj.items()}
    UNIVERSE_CACHE.update(pools=pools, templates=templates)
    return UNIVERSE_CACHE


# =====================================================================================

# =====================================================================================

def quantile(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = (len(s) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def kendall_tau(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            s = a * b
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    tot = n * (n - 1) / 2
    return (conc - disc) / tot if tot else 0.0


def bootstrap_median_ci(xs, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, lo=0.05, hi=0.95):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    meds = []
    for _ in range(b):
        meds.append(median([xs[rng.randrange(n)] for _ in range(n)]))
    return (quantile(meds, lo), quantile(meds, hi))


def bootstrap_delta_median_ci(xs, gs, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, lo=0.05, hi=0.95):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if not xs or not gs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    nx, ng = len(xs), len(gs)
    ds = []
    for _ in range(b):
        mx = median([xs[rng.randrange(nx)] for _ in range(nx)])
        mg = median([gs[rng.randrange(ng)] for _ in range(ng)])
        ds.append(mx - mg)
    return (quantile(ds, lo), quantile(ds, hi))


def bootstrap_mean_ci(xs, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, lo=0.05, hi=0.95):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    ms = []
    for _ in range(b):
        ms.append(sum(xs[rng.randrange(n)] for _ in range(n)) / n)
    return (quantile(ms, lo), quantile(ms, hi))




def k3b_pair_lambdas(dA, dB, aliveA, aliveB, b=BOOTSTRAP_B):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    absA = [abs(x) for x in dA]
    absB = [abs(x) for x in dB]
    cA, cB = bootstrap_median_ci(absA, b=b), bootstrap_median_ci(absB, b=b)
    se_n = max(1e-9, (cA[1] - cA[0]) / 3.29, (cB[1] - cB[0]) / 3.29)
    mA_, mB_ = median(absA), median(absB)
    sA = sum(aliveA) / float(max(1, len(aliveA)))
    sB = sum(aliveB) / float(max(1, len(aliveB)))
    cSA, cSB = bootstrap_mean_ci(aliveA, b=b), bootstrap_mean_ci(aliveB, b=b)
    se_s = max(1e-9, (cSA[1] - cSA[0]) / 3.29, (cSB[1] - cSB[0]) / 3.29)
    return {"lam_nmad": abs(mA_ - mB_) / se_n, "lam_S": abs(sA - sB) / se_s,
            "absmed_A": mA_, "absmed_B": mB_, "gap_absmed": mA_ - mB_,
            "S_A": sA, "S_B": sB, "gap_S": sA - sB}


def k3b_two_tier_fire(lam_seq, thresh):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    return any(a is not None and b is not None and a > thresh and b > thresh
               for a, b in zip(lam_seq, lam_seq[1:]))


def k3b_two_tier_stat(lam_seq):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    vals = [min(a, b) for a, b in zip(lam_seq, lam_seq[1:])
            if a is not None and b is not None]
    return max(vals) if vals else None


def k3b_direction_verdict(sign_matrix):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    n_tau = max((len(v) for v in sign_matrix.values()), default=0)
    same_signs, n_eval = [], 0
    for ti in range(n_tau):
        col = [sign_matrix[s][ti] if ti < len(sign_matrix[s]) else None
               for s in sign_matrix]
        if any(v is None or v == 0 for v in col):
            continue
        n_eval += 1
        if all(v == col[0] for v in col):
            same_signs.append(col[0])
    systematic = (n_eval > 0 and len(same_signs) >= max(1, n_eval - 1)
                  and len(same_signs) > 0
                  and all(s == same_signs[0] for s in same_signs))
    return {"n_tau_evaluable": n_eval,
            "n_tau_all_seed_same_sign": len(same_signs),
            "verdict": ("systematic_bias_signature" if systematic
                        else "implementation_noise_signature")}


def e2_classify(readlive_A, readlive_B, r1keep_A, r1keep_B, readkeep_A, readkeep_B):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if not (readlive_A and readlive_B):
        return "readout_floor"
    def _keep(v):
        return v is not None and not (isinstance(v, float) and math.isnan(v)) \
            and v >= STOREKEEP_THRESH
    storekeep = _keep(r1keep_A) and _keep(r1keep_B)
    readkeep = bool(readkeep_A) and bool(readkeep_B)
    if storekeep and not readkeep:
        return "readout_injury"
    if not storekeep:
        return "storage_injury"
    return "intact"


def nmad(deltas, scale):
    return median([abs(d) for d in deltas]) / max(1e-9, scale)


def signed_med(deltas, scale):
    return median(deltas) / max(1e-9, scale)


def tau_star_survival(surv_by_tau, taus, thresh=0.5):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    for i, t in enumerate(taus):
        if surv_by_tau[i] < thresh:
            return (taus[i - 1] if i > 0 else 0, t)
    return None


def tau_star_regime(lin_pass_by_tau, taus):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    for i in range(len(taus)):
        if not lin_pass_by_tau[i] and all(not p for p in lin_pass_by_tau[i:]):
            return taus[i]
    return None


# =====================================================================================
# Phases
# =====================================================================================

def phase_universe(ctx, k_exposure=None, mini=False):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    k = k_exposure or K_EXPOSURE_DEFAULT
    tag = "universe" if not mini else "universe_mini"
    if ctx.done(tag + ":k{}".format(k)):
        return
    pools = build_facts()
    templates = _make_templates()
    d = ctx.p("universe")
    os.makedirs(d, exist_ok=True)
    fpath = os.path.join(d, "facts.json.gz")
    if not os.path.exists(fpath):
        tmp = fpath + ".tmp"
        with gzip.open(tmp, "wt") as f:
            json.dump(pools, f)
        os.replace(tmp, fpath)
        write_json(os.path.join(d, "templates.json"),
                   {str(a): {"train": [[list(p) for p in t] for t in v["train"]],
                             "probe": [[list(p) for p in t] for t in v["probe"]]}
                    for a, v in templates.items()})
    for seed in TRAIN_SEEDS:
        save_placement(ctx, "base_s{}".format(seed), base_slot_pattern(seed))
        save_placement(ctx, "expose_s{}_k{}".format(seed, k), expose_slot_pattern(seed, k))

    rep = {"k_exposure": k, "checks": {}}
    attrs_A = [0] * N_ATTRS
    attrs_B = [0] * N_ATTRS
    for f in pools["excl_A"]:
        attrs_A[f["attr"]] += 1
    for f in pools["excl_B"]:
        attrs_B[f["attr"]] += 1
    rep["checks"]["attr_stratification_equal"] = attrs_A == attrs_B
    pair_attr_match = all(pools["excl_A"][i]["attr"] == pools["excl_B"][i]["attr"]
                          for i in range(N_EXCL))
    rep["checks"]["twin_pair_attr_match"] = pair_attr_match
    hA = [0] * VALS_PER_ATTR
    hB = [0] * VALS_PER_ATTR
    for f in pools["excl_A"]:
        hA[(f["val"] - VAL0) % VALS_PER_ATTR] += 1
    for f in pools["excl_B"]:
        hB[(f["val"] - VAL0) % VALS_PER_ATTR] += 1
    l1 = sum(abs(a - b) for a, b in zip(hA, hB)) / (2.0 * N_EXCL)
    rep["checks"]["value_hist_l1"] = round(l1, 4)
    pat = expose_slot_pattern(TRAIN_SEEDS[0], k)
    cnt = [0] * N_EXCL
    for i in range(len(pat[1])):
        cnt[pat[1][i]] += 1
    rep["checks"]["exposure_exact_K"] = (min(cnt) == k and max(cnt) == k)
    rep["checks"]["slot_pattern_shared_AB"] = True
    all_ent = set()
    dup = 0
    for pool in pools.values():
        for f in pool:
            e = tuple(f["ent"])
            if e in all_ent:
                dup += 1
            all_ent.add(e)
    rep["checks"]["entity_global_unique"] = (dup == 0)
    rep["pass"] = all(v is True for kk, v in rep["checks"].items()
                      if isinstance(v, bool)) and l1 < 0.1
    write_json(os.path.join(d, "balance_report.json"), rep)
    meta = {"universe_seed": UNIVERSE_SEED, "vocab": VOCAB, "k_exposure": k,
            "facts_sha256": sha256_file(fpath),
            "tau_grid_steps": TAU_GRID_STEPS, "tau_grid_tokens": TAU_GRID_TOKENS}
    write_json(os.path.join(d, "universe_meta.json"), meta)
    if not rep["pass"]:
        print("[note]", rep)
        sys.exit(4)
    ctx.mark_done(tag + ":k{}".format(k))


def _make_packer(ctx, kind, seed, k=None, branch=None, start_idx=0, buf=None):
    u = load_universe(ctx)
    pools, templates = u["pools"], u["templates"]
    if kind == "base":
        pl = load_placement(ctx, "base_s{}".format(seed))
        stream = SentenceStream("base:{}".format(seed), pools["shared"], templates,
                                pl, pools["filler"], start_idx)
    elif kind == "expose":
        pl = load_placement(ctx, "expose_s{}_k{}".format(seed, k))
        facts = pools["excl_A"] if branch == "A" else pools["excl_B"]
        stream = SentenceStream("expose:{}:{}".format(seed, branch), facts, templates,
                                pl, pools["filler"], start_idx)
    elif kind == "drift":
        stream = SentenceStream("drift:{}:{}".format(seed, branch), [], templates,
                                None, pools["filler"], start_idx)
    elif kind == "repair":
        stream = SentenceStream("repair:{}".format(branch), [], templates,
                                None, pools["filler"], start_idx)   # branch=unit id
    else:
        raise ValueError(kind)
    return RowPacker(stream, buf)


def phase_mb(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if ctx.done("mb"):
        return
    packer = _make_packer(ctx, "base", TRAIN_SEEDS[0])



    tim_path = ctx.p("state", "mb_timing_probe.json")
    model = train_segment(ctx, "mbprobe", 35, packer, LR_MAIN, 10, init_seed=999,
                          time_mark_step=5, timing_out=tim_path)
    if model is None:
        finalize_metrics(ctx, status="mb_fail")
        sys.exit(3)
    steady = read_json(tim_path)["steady_wall_s"]
    tok_s = 30 * TOK_PER_STEP / steady
    u = load_universe(ctx)
    battery = build_probe_battery(u["pools"], u["templates"])
    t1 = time.time()
    eval_margins(ctx, "mbprobe", {k: v.detach().cpu().float()
                                  for k, v in model.state_dict().items()}, battery)
    eval_s = time.time() - t1
    train_tokens = (BASE_STEPS * 3 + EXPOSE_STEPS_K64 * 6 + TAU_GRID_STEPS[-1] * 6
                    + REPAIR_STEPS * 42) * TOK_PER_STEP
    n_models = 3 * (16 + 24 + 48) + 3 + 42 + 8 * 11 * 3 // 6
    proj_gpuh = (train_tokens / tok_s + n_models * eval_s) / 3600.0
    rep = {"tok_per_s": round(tok_s), "assumed_tok_per_s": round(ASSUMED_TOK_S),
           "ratio": round(tok_s / ASSUMED_TOK_S, 3), "eval_s_per_model": round(eval_s, 1),
           "proj_device_h_ideal": round(proj_gpuh, 2), "proj_cap": MB_PROJ_CAP_GPUH,
           "measurement": "steady 30-step window after 5 warmup steps (v3)",
           "pass": (tok_s >= 0.5 * ASSUMED_TOK_S and proj_gpuh <= MB_PROJ_CAP_GPUH)}
    write_json(ctx.p("mb_report.json"), rep)
    for f in ["mbprobe"]:
        p = ctx.p("eval", "margins.{}.jsonl.gz".format(f))
        if os.path.exists(p):
            os.remove(p)
    if not rep["pass"]:

        if not ctx.done("mb_downshift"):
            ctx.mark_done("mb_downshift")
            ctx.amendment("[note]".format(tok_s, ASSUMED_TOK_S, proj_gpuh, MB_PROJ_CAP_GPUH))
            packer2 = _make_packer(ctx, "base", TRAIN_SEEDS[0])
            tim2_path = ctx.p("state", "mb_timing_probe32.json")
            model2 = train_segment(ctx, "mbprobe32", 35, packer2, LR_MAIN, 10,
                                   init_seed=999, time_mark_step=5,
                                   timing_out=tim2_path, batch_override=32)
            if model2 is not None:
                steady2 = read_json(tim2_path)["steady_wall_s"]
                tok_s2 = 30 * TOK_PER_STEP / steady2
                proj2 = (train_tokens / tok_s2 + n_models * eval_s) / 3600.0
                rep2 = {"tok_per_s": round(tok_s2),
                        "assumed_tok_per_s": round(ASSUMED_TOK_S),
                        "ratio": round(tok_s2 / ASSUMED_TOK_S, 3),
                        "eval_s_per_model": round(eval_s, 1),
                        "proj_device_h_ideal": round(proj2, 2),
                        "proj_cap": MB_PROJ_CAP_GPUH, "downshifted": "64->32x2",
                        "measurement": "steady 30-step window after 5 warmup steps (v3)",
                        "pass": (tok_s2 >= 0.5 * ASSUMED_TOK_S
                                 and proj2 <= MB_PROJ_CAP_GPUH)}
                write_json(ctx.p("mb_report_downshift.json"), rep2)
                if rep2["pass"]:
                    write_json(ctx.p("state", "mb_batch32"), "32\n")
                    ctx.amendment("[note]".format(tok_s2, ASSUMED_TOK_S, proj2, MB_PROJ_CAP_GPUH))
                    write_json(ctx.p("mb_report.json"), rep2)
                    ctx.mark_done("mb")
                    return
        finalize_metrics(ctx, status="mb_fail")
        sys.exit(3)
    ctx.mark_done("mb")


def phase_base(ctx, seed):
    run_id = "base-s{}".format(seed)
    if ctx.done(run_id):
        return
    packer = _make_packer(ctx, "base", seed)
    model = train_segment(ctx, run_id, BASE_STEPS, packer, LR_MAIN, LR_WARMUP_STEPS,
                          init_seed=seed, data_seed_key="base:{}".format(seed),
                          opt_out=ctx.p("ckpts", run_id, "opt.pt"))
    if model is None:
        jsonl_append(ctx.p("failed_runs.jsonl"), {"run": run_id, "why": "K-8 nan"})
        return
    save_sd_bf16(model, ctx.p("ckpts", run_id, "final.pt"))
    ctx.mark_done(run_id)


def phase_expose(ctx, seed, branch, k):
    run_id = "exp{}-s{}-k{}".format(branch, seed, k)
    if ctx.done(run_id):
        return
    base_sd = load_sd_fp32(ctx.p("ckpts", "base-s{}".format(seed), "final.pt"))

    base_opt = load_opt_state(ctx.p("ckpts", "base-s{}".format(seed), "opt.pt"))
    steps = EXPOSE_STEPS_K64 * (k // K_EXPOSURE_DEFAULT)
    packer = _make_packer(ctx, "expose", seed, k=k, branch=branch)
    model = train_segment(ctx, run_id, steps, packer, LR_MAIN, 0, init_sd=base_sd,
                          init_opt=base_opt,
                          opt_out=ctx.p("ckpts", run_id, "opt.pt"),
                          init_seed=seed, data_seed_key="exp:{}:{}".format(seed, branch))
    if model is None:
        jsonl_append(ctx.p("failed_runs.jsonl"), {"run": run_id, "why": "K-8 nan"})
        return
    save_sd_bf16(model, ctx.p("ckpts", run_id, "final.pt"))
    ctx.mark_done(run_id)


def phase_gate(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if ctx.done("gate"):
        return
    u = load_universe(ctx)
    battery = build_probe_battery(u["pools"], u["templates"])
    seed0 = TRAIN_SEEDS[0]
    k_hist = []
    k_final_v = None
    for k in K_LADDER:
        phase_universe(ctx, k_exposure=k)
        for br in ["A", "B"]:
            phase_expose(ctx, seed0, br, k)
        frac, med_rank, med_top1 = {}, {}, {}
        sig, m_end_pool = [], []
        for br in ["A", "B"]:
            mid = "s{}-t0-end{}-k{}".format(seed0, br, k)
            sd = load_sd_fp32(ctx.p("ckpts", "exp{}-s{}-k{}".format(br, seed0, k), "final.pt"))
            eval_margins(ctx, mid, sd, battery, store_per_template=True)
            meta, facts = load_margins(ctx, mid)
            own = "excl_{}".format(br)
            rows = [r for r in facts["train"].values() if r["set"] == own]
            frac[br] = sum(1 for r in rows if r["rank_bar"] <= RANK1_BAR_THRESH) \
                / float(max(1, len(rows)))
            med_rank[br] = median([r["rank_bar"] for r in rows])
            med_top1[br] = median([r["m_top1"] for r in rows])
            for r in rows:
                m_end_pool.append(r["m_top1"])
                if "per_t_top1" in r and len(r["per_t_top1"]) >= 2:
                    sig.append(statistics.pstdev(r["per_t_top1"]))
        sigma_probe = median(sig)
        m_end = median(m_end_pool)
        window_ratio = WINDOW_RATIO_COEF * m_end / max(1e-9, sigma_probe)
        ok = frac["A"] >= K1_FRAC_RANK1_THRESH and frac["B"] >= K1_FRAC_RANK1_THRESH
        k_hist.append({"K": k,
                       "frac_rank1_A": round(frac["A"], 4),
                       "frac_rank1_B": round(frac["B"], 4),
                       "median_rank_A": round(med_rank["A"], 3),
                       "median_rank_B": round(med_rank["B"], 3),
                       "median_top1_A": round(med_top1["A"], 4),
                       "median_top1_B": round(med_top1["B"], 4),
                       "sigma_probe": round(sigma_probe, 4),
                       "window_ratio": round(window_ratio, 4),
                       "pass": ok})
        if ok:
            k_final_v = k
            break
        if k != K_LADDER[-1]:
            ctx.amendment("[note]"
                          "[note]".format(
                              k, frac["A"], frac["B"], K1_FRAC_RANK1_THRESH,
                              K_LADDER[K_LADDER.index(k) + 1]))
    write_json(ctx.p("margin_gate.json"),
               {"ladder": k_hist, "k_final": k_final_v,
                "rule": ("frac_rank1(own excl, train battery 8/8, rank_bar<={}) >= {}, "
                         "[note]"
                         .format(RANK1_BAR_THRESH, K1_FRAC_RANK1_THRESH)),
                "pass": k_final_v is not None})
    if k_final_v is None:
        finalize_metrics(ctx, status="universe_fail")
        sys.exit(4)
    atomic_write(ctx.p("state", "k_final"), str(k_final_v).encode())

    sdA = load_sd_fp32(ctx.p("ckpts", "expA-s{}-k{}".format(seed0, k_final_v), "final.pt"))
    sdB = load_sd_fp32(ctx.p("ckpts", "expB-s{}-k{}".format(seed0, k_final_v), "final.pt"))
    lA = eval_filler_loss(ctx, sdA)
    lB = eval_filler_loss(ctx, sdB)
    lM = eval_filler_loss(ctx, lerp_sd(sdA, sdB, 0.5))
    barrier = lM / (0.5 * (lA + lB)) - 1.0
    spawn = {"loss_A": round(lA, 4), "loss_B": round(lB, 4), "loss_merge05": round(lM, 4),
             "barrier_rel": round(barrier, 4), "threshold": SPAWN_BARRIER_REL,
             "base_deepened": False, "pass": barrier <= SPAWN_BARRIER_REL}
    write_json(ctx.p("spawn_gate.json"), spawn)
    if not spawn["pass"]:


        ctx.amendment("[note]"
                      .format(barrier, SPAWN_BARRIER_REL))
        finalize_metrics(ctx, status="spawn_fail")
        sys.exit(4)
    ctx.mark_done("gate")


def k_final(ctx):
    p = ctx.p("state", "k_final")
    return int(open(p).read()) if os.path.exists(p) else K_EXPOSURE_DEFAULT


def phase_drift_full(ctx, seed, branch):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    run_id = "drift{}-s{}".format(branch, seed)
    if ctx.done(run_id):
        return
    kf = k_final(ctx)
    ckdir = ctx.p("ckpts", run_id)
    os.makedirs(ckdir, exist_ok=True)
    init_sd = load_sd_fp32(ctx.p("ckpts", "exp{}-s{}-k{}".format(branch, seed, kf), "final.pt"))

    init_opt = load_opt_state(ctx.p("ckpts", "exp{}-s{}-k{}".format(branch, seed, kf), "opt.pt"))
    packer = _make_packer(ctx, "drift", seed, branch=branch)
    model = train_segment(ctx, run_id, TAU_GRID_STEPS[-1], packer, LR_MAIN, 0,
                          init_sd=init_sd, init_opt=init_opt, init_seed=seed,
                          data_seed_key="drift:{}:{}".format(seed, branch),
                          ckpt_rel_steps=tuple(TAU_GRID_STEPS), ckpt_dir=ckdir)
    if model is None:
        jsonl_append(ctx.p("failed_runs.jsonl"), {"run": run_id, "why": "K-8 nan"})
        return
    ctx.mark_done(run_id)


def _model_ids(seed, ti):
    ids = {"endA": "s{}-t{}-endA".format(seed, ti), "endB": "s{}-t{}-endB".format(seed, ti)}
    for a in ALPHAS_INTERIOR:
        ids["merge{:.2f}".format(a)] = "s{}-t{}-a{:.2f}-merge".format(seed, ti, a)
        ids["disA{:.2f}".format(a)] = "s{}-t{}-s{:.2f}-disA".format(seed, ti, a)
        ids["disB{:.2f}".format(a)] = "s{}-t{}-s{:.2f}-disB".format(seed, ti, a)
    return ids


def phase_merge_eval(ctx, seed, ti):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    shard = "merge_eval-s{}-t{}".format(seed, ti)
    if ctx.done(shard):
        return
    ns = _torch_model_ns()
    torch = ns["torch"]
    u = load_universe(ctx)
    battery = build_probe_battery(u["pools"], u["templates"])
    sd0 = load_sd_fp32(ctx.p("ckpts", "base-s{}".format(seed), "final.pt"))
    sdA = load_sd_fp32(ctx.p("ckpts", "driftA-s{}".format(seed), "tau{}.pt".format(ti)))
    sdB = load_sd_fp32(ctx.p("ckpts", "driftB-s{}".format(seed), "tau{}.pt".format(ti)))
    ids = _model_ids(seed, ti)


    assert sd_l2_dist(lerp_sd(sdA, sdB, 1.0), sdA) == 0.0, "K-3a identity fail (alpha=1)"
    assert sd_l2_dist(lerp_sd(sdA, sdB, 0.0), sdB) == 0.0, "K-3a identity fail (alpha=0)"
    eval_margins(ctx, ids["endA"], sdA, battery, store_per_template=(ti == 0))
    eval_margins(ctx, ids["endB"], sdB, battery, store_per_template=(ti == 0))
    if ti == 0:
        eval_margins(ctx, "s{}-base".format(seed), sd0, battery)
    for a in ALPHAS_INTERIOR:
        eval_margins(ctx, ids["merge{:.2f}".format(a)], lerp_sd(sdA, sdB, a), battery)
        eval_margins(ctx, ids["disA{:.2f}".format(a)], scale_delta_sd(sd0, sdA, a), battery)
        eval_margins(ctx, ids["disB{:.2f}".format(a)],
                     scale_delta_sd(sd0, sdB, 1.0 - a), battery)
    covar = {"seed": seed, "tau_idx": ti, "tau_tokens": TAU_GRID_TOKENS[ti],
             "delta_norm_A": round(sd_l2_dist(sdA, sd0), 4),
             "delta_norm_B": round(sd_l2_dist(sdB, sd0), 4),
             "branch_sep": round(sd_l2_dist(sdA, sdB), 4),
             "cos_dA_dB": round(sd_cos(sd0, sdA, sdB), 5),
             "filler_loss_A": round(eval_filler_loss(ctx, sdA), 4),
             "filler_loss_B": round(eval_filler_loss(ctx, sdB), 4)}
    for br, key in [("A", "path_len_A"), ("B", "path_len_B")]:
        mp = ctx.p("ckpts", "drift{}-s{}".format(br, seed), "tau{}.meta.json".format(ti))
        if os.path.exists(mp):
            covar[key] = read_json(mp).get("path_len")
    jsonl_append(ctx.p("covariates.jsonl"), covar)
    ctx.ledger(shard, force=True)
    ctx.fuse_check(shard)
    ctx.mark_done(shard)


def phase_lmc(ctx, seed, ti):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    shard = "lmc-s{}-t{}".format(seed, ti)
    if ctx.done(shard):
        return
    load_universe(ctx)

    sdA = load_sd_fp32(ctx.p("ckpts", "driftA-s{}".format(seed), "tau{}.pt".format(ti)))
    sdB = load_sd_fp32(ctx.p("ckpts", "driftB-s{}".format(seed), "tau{}.pt".format(ti)))
    losses = [round(eval_filler_loss(ctx, lerp_sd(sdA, sdB, a)), 4) for a in LMC_ALPHAS]
    jsonl_append(ctx.p("lmc_barrier.jsonl"),
                 {"seed": seed, "tau_idx": ti, "alphas": LMC_ALPHAS, "filler_loss": losses,
                  "barrier_rel_a05": round(losses[5] / (0.5 * (losses[0] + losses[10])) - 1, 4)})
    ctx.mark_done(shard)


def repair_units(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    units = []
    for seed in TRAIN_SEEDS:
        for ti in range(len(TAU_GRID_STEPS)):
            units.append({"unit": "rep-s{}-t{}-m05".format(seed, ti), "kind": "merge",
                          "seed": seed, "tau_idx": ti, "alpha": 0.5})
        for ti in REPAIR_ENDPOINT_CONTROL_TAUS:
            for br in ["A", "B"]:
                units.append({"unit": "rep-s{}-t{}-end{}".format(seed, ti, br),
                              "kind": "endpoint", "seed": seed, "tau_idx": ti, "branch": br})
    return units


def phase_repair(ctx, unit):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    uid = unit["unit"]
    if ctx.done(uid):
        return
    seed, ti = unit["seed"], unit["tau_idx"]
    sdA = load_sd_fp32(ctx.p("ckpts", "driftA-s{}".format(seed), "tau{}.pt".format(ti)))
    sdB = load_sd_fp32(ctx.p("ckpts", "driftB-s{}".format(seed), "tau{}.pt".format(ti)))
    if unit["kind"] == "merge":
        sd = lerp_sd(sdA, sdB, unit["alpha"])
        pre_id = "s{}-t{}-a0.50-merge".format(seed, ti)
    else:
        sd = sdA if unit["branch"] == "A" else sdB
        pre_id = "s{}-t{}-end{}".format(seed, ti, unit["branch"])
    packer = _make_packer(ctx, "repair", seed, branch=uid)
    model = train_segment(ctx, uid, REPAIR_STEPS, packer, REPAIR_LR, REPAIR_WARMUP,
                          init_sd=sd, init_seed=seed, data_seed_key=uid)
    if model is None:
        jsonl_append(ctx.p("failed_runs.jsonl"), {"run": uid, "why": "K-8 nan"})
        return
    u2 = load_universe(ctx)
    battery = build_probe_battery(u2["pools"], u2["templates"])
    post_sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
    eval_margins(ctx, uid + "-post", post_sd, battery)


    hold_desc = {}
    try:
        _, fPre = load_margins(ctx, pre_id)
        _, fPost = load_margins(ctx, uid + "-post")
        ids = _model_ids(seed, ti)
        for tag, fx in [("pre", fPre), ("post", fPost)]:
            dd, rr = {}, {}
            for br in ["A", "B"]:
                own = "excl_" + br
                own_h = [r["m_top1"] for r in fx["hold"].values() if r["set"] == own]
                gh_h = [r["m_top1"] for r in fx["hold"].values()
                        if r["set"].startswith("ghost")]
                dd[br] = round(median(own_h) - median(gh_h), 4)
                try:
                    _, fE = load_margins(ctx, ids["end" + br])
                    fr_end = _frac_rank1(fE["train"], own)
                    rr[br] = round(_frac_rank1(fx["train"], own) / fr_end, 4) \
                        if fr_end > 0 else None
                except FileNotFoundError:
                    rr[br] = None
            hold_desc["hold_delta_" + tag] = dd
            hold_desc["r1keep_vs_end_" + tag] = rr
    except FileNotFoundError:
        hold_desc = {"hold_desc_error": "[note]"}
    jsonl_append(ctx.p("repair", "repair_eval.jsonl"),
                 dict({"unit": uid, "kind": unit["kind"], "seed": seed, "tau_idx": ti,
                       "pre_model": pre_id, "post_model": uid + "-post",
                       "batteries": list(BATTERIES),
                       "protocol": {"tokens": REPAIR_STEPS * TOK_PER_STEP,
                                    "lr": REPAIR_LR}}, **hold_desc))
    ctx.mark_done(uid)


# =====================================================================================

# =====================================================================================

def _alive(row, meta, q=ALIVE_Q_MAIN, metric="top1"):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    th = meta["theta_alive_{}_q{}".format(metric, round(q * 1000))]
    return row["m_{}".format(metric)] >= th


def _seed_stats(ctx, seed):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    taus = TAU_GRID_TOKENS
    n_t = len(taus)
    regime, surv05 = [], []
    d_dis_by_tau, lin_pass_by_tau = [], []
    surv_variants = {(q, m): [] for q in ALIVE_GHOST_QS for m in ["top1", "nll"]}
    for ti in range(n_t):
        ids = _model_ids(seed, ti)
        mA, fA = load_margins(ctx, ids["endA"])
        mB, fB = load_margins(ctx, ids["endB"])
        ends = {"A": (mA, fA["train"]), "B": (mB, fB["train"])}
        merges, diss = {}, {}
        for a in ALPHAS_INTERIOR:
            meta_m, f_m = load_margins(ctx, ids["merge{:.2f}".format(a)])
            merges[a] = (meta_m, f_m["train"])
            meta_dA, f_dA = load_margins(ctx, ids["disA{:.2f}".format(a)])
            diss[(a, "A")] = (meta_dA, f_dA["train"])
            meta_dB, f_dB = load_margins(ctx, ids["disB{:.2f}".format(a)])
            diss[(a, "B")] = (meta_dB, f_dB["train"])

        live = {}
        for br in ["A", "B"]:
            meta, facts = ends[br]
            own = "excl_" + br
            live[br] = [fid for fid, r in facts.items()
                        if r["set"] == own and _alive(r, meta)]
        m_end_scale = median([ends[br][1][fid]["m_top1"] for br in "AB" for fid in live[br]])
        row = {"tau": taus[ti]}
        for a in [0.5] + [x for x in ALPHAS_INTERIOR if x != 0.5]:
            d_lin, d_dis, d_obs = [], [], []
            for br, w in [("A", a), ("B", 1.0 - a)]:
                metaM, fM = merges[a]
                metaD, fD = diss[(a, br)]
                other = "B" if br == "A" else "A"
                for fid in live[br]:
                    m_meas = fM[fid]["m_top1"]

                    m_lin = w * ends[br][1][fid]["m_top1"] \
                        + (1 - w) * ends[other][1][fid]["m_top1"]
                    m_dis = fD[fid]["m_top1"]
                    d_lin.append(m_meas - m_lin)
                    d_dis.append(m_meas - m_dis)
                    d_obs.append(m_dis - m_lin)
            if a == 0.5:
                row.update(nmad_lin=round(nmad(d_lin, m_end_scale), 4),
                           nmad_dis=round(nmad(d_dis, m_end_scale), 4),
                           d_lin=round(signed_med(d_lin, m_end_scale), 4),
                           d_obs=round(signed_med(d_obs, m_end_scale), 4),
                           d_dis=round(signed_med(d_dis, m_end_scale), 4),
                           ci90_d_dis=[round(v / max(1e-9, m_end_scale), 4)
                                       for v in bootstrap_median_ci(d_dis)])
                d_dis_by_tau.append(row["d_dis"])
        row["lin_pass"] = row["nmad_lin"] <= NMAD_LIN_THRESH
        lin_pass_by_tau.append(row["lin_pass"])
        regime.append(row)

        metaM, fM = merges[0.5]
        s_num = s_den = 0
        for br in ["A", "B"]:
            for fid in live[br]:
                s_den += 1
                if _alive(fM[fid], metaM):
                    s_num += 1
        surv05.append(round(s_num / max(1, s_den), 4))
        for (q, metric) in surv_variants:
            num = den = 0
            for br in ["A", "B"]:
                metaE, fE = ends[br]
                own = "excl_" + br
                lv = [fid for fid, r in fE.items() if r["set"] == own
                      and _alive(r, metaE, q, metric)]
                for fid in lv:
                    den += 1
                    if _alive(fM[fid], metaM, q, metric):
                        num += 1
            surv_variants[(q, metric)].append(num / max(1, den))
    ts_main = tau_star_survival(surv05, taus)
    variant_stars = []
    for key, sv in surv_variants.items():
        v = tau_star_survival(sv, taus)
        variant_stars.append(taus.index(v[1]) if v else None)
    idx_main = taus.index(ts_main[1]) if ts_main else None
    shifts = [abs(v - idx_main) for v in variant_stars if v is not None and idx_main is not None]
    robust_shift = max(shifts) if shifts else (0 if all(v is None for v in variant_stars)
                                               and idx_main is None else 99)
    cov_rows = []
    if os.path.exists(ctx.p("covariates.jsonl")):
        with open(ctx.p("covariates.jsonl")) as f:
            cov_rows = [json.loads(l) for l in f if json.loads(l)["seed"] == seed]
    sep_by_tau = {r["tau_idx"]: r["branch_sep"] for r in cov_rows}
    seps = [sep_by_tau.get(i, float("nan")) for i in range(n_t)]
    return {
        "tau_grid_tokens": taus,
        "regime": regime,
        "survival_a05": surv05,
        "tau_star_S_interval": list(ts_main) if ts_main else [None, None],
        "tau_star_D": tau_star_regime(lin_pass_by_tau, taus),
        "tau_star_reparam": {
            "tokens": ts_main[1] if ts_main else None,
            "branch_sep": (seps[taus.index(ts_main[1])] if ts_main else None),
            "path_len": (cov_rows and ts_main and next(
                (r.get("path_len_A") for r in cov_rows
                 if r["tau_idx"] == taus.index(ts_main[1])), None)) or None},
        "robustness_max_shift_grid_steps": robust_shift,
        "kendall_ddis_vs_tau": round(kendall_tau(taus, d_dis_by_tau), 3),
        "kendall_ddis_vs_sep": round(kendall_tau(seps, d_dis_by_tau), 3)
        if all(not math.isnan(s) for s in seps) else None,
        "flags": [],
    }


def _repair_stats(ctx, seed):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    path = ctx.p("repair", "repair_eval.jsonl")
    if not os.path.exists(path):
        return [], None
    rows = []
    with open(path) as f:
        units = [json.loads(l) for l in f]
    rho_by_tau, pooled_num, pooled_den = [], 0, 0
    for u in units:
        if u["seed"] != seed or u["kind"] != "merge":
            continue
        ti = u["tau_idx"]
        metaPre, fPreAll = load_margins(ctx, u["pre_model"])
        metaPost, fPostAll = load_margins(ctx, u["post_model"])
        fPre, fPost = fPreAll["train"], fPostAll["train"]
        ids = _model_ids(seed, ti)
        dead = []
        for br in ["A", "B"]:
            metaE, fEAll = load_margins(ctx, ids["end" + br])
            fE = fEAll["train"]
            own = "excl_" + br
            for fid, r in fE.items():
                if r["set"] == own and _alive(r, metaE) and not _alive(fPre[fid], metaPre):
                    dead.append(fid)
        gd = [fPost[g]["m_top1"] - fPre[g]["m_top1"] for g in fPre
              if fPre[g]["set"].startswith("ghost")]
        band = median(gd) + 2 * (quantile(gd, 0.75) - quantile(gd, 0.25))
        revived = [fid for fid in dead
                   if _alive(fPost[fid], metaPost)
                   and (fPost[fid]["m_top1"] - fPre[fid]["m_top1"]) > band]
        rho = len(revived) / max(1, len(dead))
        rho_by_tau.append({"tau": TAU_GRID_TOKENS[ti], "rho": round(rho, 4),
                           "n_dead": len(dead)})
        pooled_num += len(revived)
        pooled_den += len(dead)
    pooled = pooled_num / max(1, pooled_den) if pooled_den else None
    return rho_by_tau, pooled


def _k3b_threshold(ctx, fallbacks=(os.path.join("merge_audit_dryrun", "nullsim_report.json"),)):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if K3B_LAMBDA_THRESH is not None:
        return float(K3B_LAMBDA_THRESH), "module_const:K3B_LAMBDA_THRESH"
    for src in [ctx.p("nullsim_report.json")] + list(fallbacks):
        if os.path.exists(src):
            cal = read_json(src).get("k3b_calibration") or {}
            if cal.get("lambda_q99") is not None:
                return float(cal["lambda_q99"]), src
    raise RuntimeError(
        "[note]"
        "[note]")


def _symmetry(ctx, write=True):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    thresh, tsrc = _k3b_threshold(ctx)
    lam_seq = {(s, p, st): [] for s in TRAIN_SEEDS
               for p, _ in K3B_SWAP_PAIRS for st in K3B_STATS}
    sign_seq = {}
    for fam in (["{}|{}".format(p, st) for p, _ in K3B_SWAP_PAIRS for st in K3B_STATS]
                + ["med_legacy@P1"]):
        sign_seq[fam] = {s: [] for s in TRAIN_SEEDS}
    cells, legacy_worst = [], 0.0
    for seed in TRAIN_SEEDS:
        for ti in range(len(TAU_GRID_TOKENS)):
            ids = _model_ids(seed, ti)
            try:
                mA, fAa = load_margins(ctx, ids["endA"])
                mB, fBa = load_margins(ctx, ids["endB"])
                merges = {a: load_margins(ctx, ids["merge{:.2f}".format(a)])
                          for a in ALPHAS_INTERIOR}
            except FileNotFoundError:
                for p, _ in K3B_SWAP_PAIRS:
                    for st in K3B_STATS:
                        lam_seq[(seed, p, st)].append(None)
                        sign_seq["{}|{}".format(p, st)][seed].append(None)
                sign_seq["med_legacy@P1"][seed].append(None)
                continue
            fA, fB = fAa["train"], fBa["train"]
            liveA = [fid for fid, r in fA.items() if r["set"] == "excl_A" and _alive(r, mA)]
            liveB = [fid for fid, r in fB.items() if r["set"] == "excl_B" and _alive(r, mB)]
            if not liveA or not liveB:
                for p, _ in K3B_SWAP_PAIRS:
                    for st in K3B_STATS:
                        lam_seq[(seed, p, st)].append(None)
                        sign_seq["{}|{}".format(p, st)][seed].append(None)
                sign_seq["med_legacy@P1"][seed].append(None)
                cells.append({"seed": seed, "tau_idx": ti, "empty_live": True})
                continue
            m_end_scale = median([fA[f]["m_top1"] for f in liveA]
                                 + [fB[f]["m_top1"] for f in liveB])
            row = {"seed": seed, "tau_idx": ti,
                   "n_liveA": len(liveA), "n_liveB": len(liveB),
                   "m_end_scale": round(m_end_scale, 4)}
            for pname, w in K3B_SWAP_PAIRS:
                metaMA, fMAa = merges[w]
                metaMB, fMBa = merges[round(1.0 - w, 2)]
                fMA, fMB = fMAa["train"], fMBa["train"]
                dA = [fMA[f]["m_top1"] - (w * fA[f]["m_top1"] + (1 - w) * fB[f]["m_top1"])
                      for f in liveA]
                dB = [fMB[f]["m_top1"] - (w * fB[f]["m_top1"] + (1 - w) * fA[f]["m_top1"])
                      for f in liveB]
                aliveA = [1.0 if _alive(fMA[f], metaMA) else 0.0 for f in liveA]
                aliveB = [1.0 if _alive(fMB[f], metaMB) else 0.0 for f in liveB]
                st = k3b_pair_lambdas(dA, dB, aliveA, aliveB)
                lam_seq[(seed, pname, "nmad")].append(st["lam_nmad"])
                lam_seq[(seed, pname, "S")].append(st["lam_S"])
                sgn_n = 1 if st["gap_absmed"] > 0 else (-1 if st["gap_absmed"] < 0 else 0)
                sgn_s = 1 if st["gap_S"] > 0 else (-1 if st["gap_S"] < 0 else 0)
                sign_seq["{}|nmad".format(pname)][seed].append(sgn_n)
                sign_seq["{}|S".format(pname)][seed].append(sgn_s)
                row[pname] = {
                    "lam_nmad": round(st["lam_nmad"], 3), "lam_S": round(st["lam_S"], 3),
                    "nmad_A": round(st["absmed_A"] / max(1e-9, m_end_scale), 4),
                    "nmad_B": round(st["absmed_B"] / max(1e-9, m_end_scale), 4),
                    "S_A": round(st["S_A"], 4), "S_B": round(st["S_B"], 4),
                    "over_thresh": bool(st["lam_nmad"] > thresh or st["lam_S"] > thresh)}
                if pname == "P1_w25":
                    m1, m2 = median(dA), median(dB)
                    ci1, ci2 = bootstrap_median_ci(dA), bootstrap_median_ci(dB)
                    se = max(1e-9, (ci1[1] - ci1[0]) / 3.29, (ci2[1] - ci2[0]) / 3.29)
                    lam_l = abs(m1 - m2) / se
                    legacy_worst = max(legacy_worst, lam_l)
                    g = m1 - m2
                    sign_seq["med_legacy@P1"][seed].append(
                        1 if g > 0 else (-1 if g < 0 else 0))
                    row["legacy_med"] = {"lam_over_se": round(lam_l, 3),
                                         "gap": round(g, 4), "se": round(se, 5)}
            cells.append(row)
    verdicts, fired = {}, []
    two_tier_max = None
    for pname, _w in K3B_SWAP_PAIRS:
        for st in K3B_STATS:
            key = "{}|{}".format(pname, st)
            per_seed = {}
            for seed in TRAIN_SEEDS:
                seq = lam_seq[(seed, pname, st)]
                per_seed[str(seed)] = k3b_two_tier_fire(seq, thresh)
                tt = k3b_two_tier_stat(seq)
                if tt is not None:
                    two_tier_max = tt if two_tier_max is None else max(two_tier_max, tt)
            vals = [v for seed in TRAIN_SEEDS
                    for v in lam_seq[(seed, pname, st)] if v is not None]
            verdicts[key] = {"max": round(max(vals), 3) if vals else None,
                             "two_tier_per_seed": per_seed,
                             "fires": any(per_seed.values())}
            if verdicts[key]["fires"]:
                fired.append(key)
    lam_all = [v for s in lam_seq.values() for v in s if v is not None]
    direction = {}
    for fam, mat in sign_seq.items():
        d = k3b_direction_verdict(mat)
        d["sign_by_seed"] = {str(s): "".join(
            {1: "+", -1: "-", 0: "0", None: "."}[v] for v in mat[s]) for s in TRAIN_SEEDS}
        direction[fam] = d
    rep = {"criterion": ("[note]"
                         "[note]"),
           "threshold": round(thresh, 3), "threshold_source": tsrc,
           "threshold_basis": "nullsim Q99 (noise-injected null; M-079)",
           "pass": not fired, "fired": fired,
           "lambda_max": round(max(lam_all), 3) if lam_all else None,
           "two_tier_stat_max": round(two_tier_max, 3) if two_tier_max is not None else None,
           "verdicts": verdicts, "cells": cells,
           "direction_note": dict(direction, note=(
               "[note]"
               "[note]")),
           "legacy_impl": {"stat": "[note]",
                           "lambda_over_se_max": round(legacy_worst, 3),
                           "threshold": 3.0, "fires": legacy_worst > 3.0,
                           "note": ("[note]"
                                    "[note]")},
           "identity_check": "[note]"}
    if write:
        write_json(ctx.p("symmetry_report.json"), rep)
    return rep




def atomic_write_jsonl(path, rows):
    atomic_write(path, ("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                for r in rows)).encode())


def _frac_rank1(facts_train, own_set):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    rows = [r for r in facts_train.values() if r["set"] == own_set]
    return sum(1 for r in rows if r["rank_bar"] <= RANK1_BAR_THRESH) / float(max(1, len(rows)))


def _hold_pools(facts_hold, own_set):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    own = [r["m_top1"] for r in facts_hold.values() if r["set"] == own_set]
    gh = [r["m_top1"] for r in facts_hold.values() if r["set"].startswith("ghost")]
    return own, gh


def _e2_seed_rows(ctx, seed):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    rows = []
    for ti in range(len(TAU_GRID_TOKENS)):
        ids = _model_ids(seed, ti)
        try:
            mA, fA = load_margins(ctx, ids["endA"])
            mB, fB = load_margins(ctx, ids["endB"])
            mM, fM = load_margins(ctx, ids["merge0.50"])
        except FileNotFoundError:
            continue
        ends = {"A": fA, "B": fB}
        ci90, flags = {}, []
        readlive, readkeep, r1keep, d_end, d_merge = {}, {}, {}, {}, {}
        for br in ["A", "B"]:
            own = "excl_" + br
            own_h, gh_h = _hold_pools(ends[br]["hold"], own)
            d_end[br] = round(median(own_h) - median(gh_h), 4)
            lo, hi = bootstrap_delta_median_ci(own_h, gh_h)
            ci90["end_" + br] = [round(lo, 4), round(hi, 4)]
            readlive[br] = bool(lo > 0)
            own_m, gh_m = _hold_pools(fM["hold"], own)
            d_merge[br] = round(median(own_m) - median(gh_m), 4)
            lo2, hi2 = bootstrap_delta_median_ci(own_m, gh_m)
            ci90["merge_" + br] = [round(lo2, 4), round(hi2, 4)]
            readkeep[br] = bool(lo2 > 0)
            fr_end = _frac_rank1(ends[br]["train"], own)
            fr_mrg = _frac_rank1(fM["train"], own)
            if fr_end > 0:
                r1keep[br] = round(fr_mrg / fr_end, 4)
            else:
                r1keep[br] = None
                flags.append("r1keep_{}_undefined_endpoint_frac0".format(br))
        sig = e2_classify(readlive["A"], readlive["B"], r1keep["A"], r1keep["B"],
                          readkeep["A"], readkeep["B"])
        rows.append({"seed": seed, "tau_idx": ti, "tau": TAU_GRID_TOKENS[ti], "alpha": 0.5,
                     "r1keep_A": r1keep["A"], "r1keep_B": r1keep["B"],
                     "delta_hold_end_A": d_end["A"], "delta_hold_end_B": d_end["B"],
                     "delta_hold_merge_A": d_merge["A"], "delta_hold_merge_B": d_merge["B"],
                     "readlive_A": readlive["A"], "readlive_B": readlive["B"],
                     "readkeep_A": readkeep["A"], "readkeep_B": readkeep["B"],
                     "ci90": ci90, "signature": sig,
                     "storekeep_thresh": STOREKEEP_THRESH, "flags": flags})
    return rows


def _e2_cross_seed(rows):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    by_tau = {}
    for r in rows:
        by_tau.setdefault(r["tau_idx"], {})[r["seed"]] = r["signature"]
    out = []
    for ti in range(len(TAU_GRID_TOKENS)):
        sigs = [by_tau.get(ti, {}).get(s) for s in TRAIN_SEEDS]
        sigs = [s for s in sigs if s]
        if len(sigs) == len(TRAIN_SEEDS) and all(s == sigs[0] for s in sigs):
            out.append("stable_" + sigs[0])
        else:
            out.append("mixed")
    n_expected = len(TRAIN_SEEDS) * len(TAU_GRID_TOKENS)
    cells = [r["signature"] for r in rows]
    if cells and len(cells) == n_expected and all(s == "readout_floor" for s in cells):
        e2_class = "readout_floor_all"
    elif cells and len(cells) == n_expected and all(s == "storage_injury" for s in cells):
        e2_class = "storage_injury_all"
    elif any(v == "stable_readout_injury" for v in out):
        e2_class = "readout_injury_band"
    else:
        e2_class = "mixed"
    return out, e2_class


def _e2_repair_migration(ctx, e2_by_key):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    path = ctx.p("repair", "repair_eval.jsonl")
    if not os.path.exists(path):
        return
    with open(path) as f:
        units = [json.loads(l) for l in f]
    out = []
    for u in units:
        if u["kind"] != "merge":
            continue
        pre_row = e2_by_key.get((u["seed"], u["tau_idx"]))
        if pre_row is None:
            continue
        try:
            ids = _model_ids(u["seed"], u["tau_idx"])
            ends = {br: load_margins(ctx, ids["end" + br])[1] for br in ["A", "B"]}
            _, fP = load_margins(ctx, u["post_model"])
        except FileNotFoundError:
            continue
        r1keep, readkeep, d_post = {}, {}, {}
        for br in ["A", "B"]:
            own = "excl_" + br
            fr_end = _frac_rank1(ends[br]["train"], own)
            r1keep[br] = round(_frac_rank1(fP["train"], own) / fr_end, 4) \
                if fr_end > 0 else None
            own_h, gh_h = _hold_pools(fP["hold"], own)
            d_post[br] = round(median(own_h) - median(gh_h), 4)
            lo, _ = bootstrap_delta_median_ci(own_h, gh_h)
            readkeep[br] = bool(lo > 0)
        sig_post = e2_classify(pre_row["readlive_A"], pre_row["readlive_B"],
                               r1keep["A"], r1keep["B"], readkeep["A"], readkeep["B"])
        out.append({"unit": u["unit"], "seed": u["seed"], "tau_idx": u["tau_idx"],
                    "signature_pre": pre_row["signature"], "signature_post": sig_post,
                    "r1keep_post": [r1keep["A"], r1keep["B"]],
                    "delta_hold_post": [d_post["A"], d_post["B"]],
                    "readkeep_post": [readkeep["A"], readkeep["B"]],
                    "note": "[note]"})
    if out:
        atomic_write_jsonl(ctx.p("repair", "repair_signature.jsonl"), out)


def _rollup_compute(ctx, write_artifacts=True):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    per_seed, missing = {}, []
    for seed in TRAIN_SEEDS:
        try:
            st = _seed_stats(ctx, seed)
            rho_by_tau, pooled = _repair_stats(ctx, seed)
            st["rho_repair"] = rho_by_tau
            st["rho_repair_pooled"] = pooled
            per_seed[str(seed)] = st
        except FileNotFoundError as e:
            missing.append(str(seed))
    sym = _symmetry(ctx, write=write_artifacts)

    e2_rows_all = []
    for seed in TRAIN_SEEDS:
        rows_e2 = _e2_seed_rows(ctx, seed)
        e2_rows_all.extend(rows_e2)
        if str(seed) in per_seed:
            per_seed[str(seed)]["e2"] = [
                {"tau": r["tau"], "signature": r["signature"],
                 "r1keep": [r["r1keep_A"], r["r1keep_B"]],
                 "delta_hold_merge": [r["delta_hold_merge_A"], r["delta_hold_merge_B"]],
                 "readlive": [r["readlive_A"], r["readlive_B"]],
                 "readkeep": [r["readkeep_A"], r["readkeep_B"]]} for r in rows_e2]
    if e2_rows_all and write_artifacts:
        atomic_write_jsonl(ctx.p("e2_signature.jsonl"), e2_rows_all)
    e2_by_tau, e2_class = _e2_cross_seed(e2_rows_all)
    if write_artifacts:
        _e2_repair_migration(ctx, {(r["seed"], r["tau_idx"]): r for r in e2_rows_all})

    seeds_ok = [s for s in per_seed]
    h2 = all(per_seed[s]["kendall_ddis_vs_tau"] < 0 for s in seeds_ok) and \
        all((per_seed[s]["kendall_ddis_vs_sep"] or 0) < 0 for s in seeds_ok) if seeds_ok else False
    stars = [per_seed[s]["tau_star_S_interval"] for s in seeds_ok]
    inter = [max((x[0] or 0) for x in stars) if all(x[1] for x in stars) else None,
             min((x[1] or 0) for x in stars) if all(x[1] for x in stars) else None]
    robust_ok = all(per_seed[s]["robustness_max_shift_grid_steps"] <= 1 for s in seeds_ok)
    rho_cls = []
    for s in seeds_ok:
        p = per_seed[s]["rho_repair_pooled"]
        rho_cls.append("nonzero" if (p or 0) >= RHO_NONZERO
                       else ("zero" if (p or 1) < RHO_ZERO else "mid"))
    rho_class = rho_cls[0] if rho_cls and all(c == rho_cls[0] for c in rho_cls) else "unstable"
    suicide = (not h2) or (not robust_ok)

    triage, flags = "PARTIAL", []
    if missing:
        flags.append("missing_seeds:" + ",".join(missing))
    if not sym["pass"]:
        triage = "SYMMETRY_FAIL"
    elif seeds_ok:
        t0_nmads = [per_seed[s]["regime"][0]["nmad_lin"] for s in seeds_ok]
        t0_struct = [abs(per_seed[s]["kendall_ddis_vs_tau"]) for s in seeds_ok]
        if all(v > ESTIMAND_COLLAPSE_NMAD for v in t0_nmads) and \
                all(v < 0.1 for v in t0_struct):
            triage = "NO_GO_ESTIMAND"                                    # K-4
        else:
            surv_last = [per_seed[s]["survival_a05"][-1] for s in seeds_ok]
            surv_first = [per_seed[s]["survival_a05"][0] for s in seeds_ok]
            if all(v > GRID_ALL_SURVIVE for v in surv_last):
                triage = "NO_GO_GRID"
                flags.append("all_survive_at_tau_max")
            elif all(v < GRID_ALL_DEAD for v in surv_first):
                triage = "NO_GO_GRID"
                flags.append("all_dead_at_tau0_check_K1_marginality")
            elif suicide:
                triage = "SUICIDE_NEGATIVE"
            else:
                h1 = all(per_seed[s]["regime"][0]["lin_pass"] for s in seeds_ok)
                dis_ok = all(per_seed[s]["regime"][0]["nmad_dis"] <= NMAD_LIN_THRESH
                             for s in seeds_ok)
                interf0 = all(per_seed[s]["regime"][0]["ci90_d_dis"][1] < 0 for s in seeds_ok)
                star_in_grid = all(x[1] is not None for x in stars)
                if h1 and h2 and star_in_grid and robust_ok:
                    triage = "R_LAW"
                elif (not h1) and dis_ok:
                    triage = "R_DIS"
                elif interf0:
                    triage = "R_INTERF0"
                else:
                    triage = "SUICIDE_NEGATIVE" if suicide else "PARTIAL"
    if rho_class == "unstable":
        flags.append("K-6_repair_unstable_downgraded_to_appendix")
    metrics = {
        "contract_id": CONTRACT_ID,
        "prereg_sha256": sha256_file(PREREG_PATH) if os.path.exists(PREREG_PATH) else None,
        "instrument_calibration_archive": _archive_hashes(),
        "status": "complete" if not missing and sym["pass"] else
                  ("symmetry_fail" if not sym["pass"] else "partial"),
        "universe": _universe_summary(ctx),
        "gates": _gates_summary(ctx),
        "per_seed": per_seed,
        "cross_seed": {"h2_sign_consistent": h2, "tau_star_S_intersection": inter,
                       "reparam_dispersion": _reparam_dispersion(per_seed),
                       "rho_class": rho_class, "suicide_clause_triggered": suicide,
                       "e2_signature_by_tau": e2_by_tau,
                       "e2_class": e2_class},
        "symmetry": sym,
        "nullsim": read_json(ctx.p("nullsim_report.json"))
        if os.path.exists(ctx.p("nullsim_report.json")) else None,
        "triage": {"branch": triage, "flags": flags,
                   "recommendation": _recommendation(triage)},
        "budget": {"gpu_h_total": round(ctx.gpu_h(), 2), "fuse_triggered": False},
    }
    return metrics


def phase_rollup(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    metrics = _rollup_compute(ctx, write_artifacts=True)
    write_json(ctx.p("metrics.json"), metrics)
    print("[rollup] triage =", metrics["triage"]["branch"],
          "| flags:", metrics["triage"]["flags"],
          "| e2_class =", metrics["cross_seed"]["e2_class"])


def phase_rollup_recal(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    metrics = _rollup_compute(ctx, write_artifacts=False)
    sym = metrics["symmetry"]
    frozen_mp = ctx.p("metrics.json")
    frozen = read_json(frozen_mp) if os.path.exists(frozen_mp) else None
    if frozen and frozen.get("budget"):
        metrics["budget"] = dict(frozen["budget"], recal_note=(
            "[note]"))
    metrics["recalibration"] = {
        "basis": "M-079",
        "k3b_threshold": sym["threshold"],
        "k3b_threshold_basis": "nullsim Q99 (noise-injected)",
        "k3b_threshold_source": sym["threshold_source"],
        "k3b_criterion": sym["criterion"],
        "falsifiable_prediction": (
            "[note]"
            "[note]"),
        "frozen_metrics_sha256": sha256_file(frozen_mp) if os.path.exists(frozen_mp) else None,
        "frozen_metrics_status": frozen.get("status") if frozen else None,
        "frozen_symmetry_report_sha256": (
            sha256_file(ctx.p("symmetry_report.json"))
            if os.path.exists(ctx.p("symmetry_report.json")) else None),
        "note": ("[note]"
                 "[note]")}
    write_json(ctx.p("metrics_rollup_recalibrated.json"), metrics)
    print("[rollup_recal] triage =", metrics["triage"]["branch"],
          "| K-3b thresh =", sym["threshold"], "({})".format(sym["threshold_source"]),
          "| sym pass =", sym["pass"], "| fired =", sym["fired"],
          "| e2_class =", metrics["cross_seed"]["e2_class"])


def _universe_summary(ctx):
    p = ctx.p("universe", "universe_meta.json")
    if not os.path.exists(p):
        return None
    m = read_json(p)
    return {"seed": UNIVERSE_SEED, "K_final": k_final(ctx), "vocab": VOCAB,
            "facts": {"shared": N_SHARED, "excl_A": N_EXCL, "excl_B": N_EXCL,
                      "ghost": 2 * N_GHOST},
            "balance_pass": read_json(ctx.p("universe", "balance_report.json"))["pass"]
            if os.path.exists(ctx.p("universe", "balance_report.json")) else None,
            "universe_sha256": m.get("facts_sha256")}


def _archive_hashes():
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    keymap = {"margin_gate.json": "margin_gate_v1", "mdiag_report.json": "mdiag",
              "mdiag2_report.json": "mdiag2", "metrics.json": "metrics_v1"}
    out = {}
    for fn in V1_ARCHIVE_FILES:
        p = os.path.join(V1_ARCHIVE, fn)
        out[keymap[fn]] = sha256_file(p) if os.path.exists(p) else None
    return out


def _gates_summary(ctx):
    out = {}
    for name, key in [("mb_report.json", "k0_mb"), ("spawn_gate.json", "k2_spawn")]:
        p = ctx.p(name)
        out[key] = read_json(p) if os.path.exists(p) else None

    mgp = ctx.p("margin_gate.json")
    if os.path.exists(mgp):
        mg = read_json(mgp)
        ladder = mg.get("ladder") or []
        last = ladder[-1] if ladder else {}
        out["k1_storage"] = {
            "frac_rank1_A": last.get("frac_rank1_A"),
            "frac_rank1_B": last.get("frac_rank1_B"),
            "threshold": K1_FRAC_RANK1_THRESH,
            "k_uplift_history": [r["K"] for r in ladder[:-1]],
            "k_final": mg.get("k_final"),
            "sigma_probe": last.get("sigma_probe"),
            "window_ratio": last.get("window_ratio"),
            "median_top1_A": last.get("median_top1_A"),
            "median_top1_B": last.get("median_top1_B"),
            "pass": mg.get("pass"),
        }
    else:
        out["k1_storage"] = None
    return out


def _reparam_dispersion(per_seed):
    out = {}
    for key in ["tokens", "branch_sep", "path_len"]:
        vals = [per_seed[s]["tau_star_reparam"].get(key) for s in per_seed]
        vals = [v for v in vals if v]
        out[key] = round(max(vals) / min(vals), 3) if len(vals) >= 2 and min(vals) else None
    return out


def _recommendation(triage):
    return {
        "R_LAW": "[note]",
        "R_DIS": "[note]",
        "R_INTERF0": "[note]",
        "SUICIDE_NEGATIVE": "[note]",
        "NO_GO_ESTIMAND": "[note]",
        "NO_GO_GRID": "[note]",
        "SYMMETRY_FAIL": "[note]",
        "PARTIAL": "[note]",
    }.get(triage, "")


def finalize_metrics(ctx, status):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    for name in ["mb_report.json", "margin_gate.json"]:
        if not os.path.exists(ctx.p(name)):
            write_json(ctx.p(name), {"status": "not_reached", "abort_status": status})
    if not os.path.exists(ctx.p("metrics.json")):
        write_json(ctx.p("metrics.json"),
                   {"contract_id": CONTRACT_ID, "status": status,
                    "budget": {"gpu_h_total": round(ctx.gpu_h(), 2)}})
    else:
        m = read_json(ctx.p("metrics.json"))
        m["status"] = status
        write_json(ctx.p("metrics.json"), m)


# =====================================================================================


# =====================================================================================


NULLSIM_PRIOR = {
    "mu_own_top1": 4.05,
    "mu_other_top1": -8.05,
    "mu_ghost_top1": -8.05,
    "sigma_probe": 1.78,
    "sigma_fact_ghost": 1.9,
    "endpoint_frac_rank1": 0.95,
    "rank_anchor_margin": 8.0,
    "rank_anchor_rank": 80.0,
    "n_templates": N_TRAIN_TEMPLATES,
}
NULLSIM_PRIOR_BASIS = (
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]"
    "[note]")


def _rank_of_margin(m, lam):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    if m > 0:
        return 1
    return min(N_VALS, 1 + max(1, int(math.floor(math.exp(min(lam * (-m), 12.0))))))


def _sim_endpoint_frac(rng, n, s_fact, lam, prior=NULLSIM_PRIOR):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    T = prior["n_templates"]
    hit = 0
    for _ in range(n):
        c = rng.gauss(prior["mu_own_top1"], s_fact)
        rsum = 0
        for _t in range(T):
            rsum += _rank_of_margin(c + rng.gauss(0, prior["sigma_probe"]), lam)
        if rsum / T <= RANK1_BAR_THRESH:
            hit += 1
    return hit / float(n)


def _calib_s_fact(lam, prior=NULLSIM_PRIOR, n=4000, iters=16):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    lo_s, hi_s = 0.05, 5.0
    for it in range(iters):
        mid = 0.5 * (lo_s + hi_s)
        fr = _sim_endpoint_frac(random.Random(777 + it), n, mid, lam)
        if fr > prior["endpoint_frac_rank1"]:
            lo_s = mid
        else:
            hi_s = mid
    return 0.5 * (lo_s + hi_s)




def _phi_norm(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _dnorm(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _foldnorm_med_se(eps, sigma, n):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    e, s = abs(eps), max(1e-9, sigma)
    lo, hi = 0.0, e + 6.0 * s
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _phi_norm((mid - e) / s) - _phi_norm((-mid - e) / s) < 0.5:
            lo = mid
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    f = (_dnorm((m - e) / s) + _dnorm((m + e) / s)) / s
    return m, 1.0 / (2.0 * max(1e-12, f) * math.sqrt(n))


def _k3b_nullsim_calibration(s_fact, sigma_tm, reps=200):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    prior, npz = NULLSIM_PRIOR, K3B_NOISE_PRIOR
    s_of = math.sqrt(s_fact ** 2 + sigma_tm ** 2)
    s_gh = math.sqrt(prior["sigma_fact_ghost"] ** 2 + sigma_tm ** 2)
    n_ghost = npz["n_ghost"]
    rep_stats, rep_cellmax = [], []
    rep_stat_by = {st: [] for st in K3B_STATS}
    sim_S = {p: ([], []) for p, _ in K3B_SWAP_PAIRS}
    for rep in range(reps):
        rng = random.Random(K3B_CAL_SEED_BASE + rep)
        tt_vals = {st: [] for st in K3B_STATS}
        cellmax = 0.0
        for seed in sorted(K3B_CELL_DESIGN):
            lam_seqs = {(p, st): [] for p, _ in K3B_SWAP_PAIRS for st in K3B_STATS}
            for (nA, nB, sig_d) in K3B_CELL_DESIGN[seed]:
                sgn = 1.0 if rng.random() < 0.5 else -1.0
                delta = rng.uniform(npz["endpoint_delta_lo"], npz["endpoint_delta_hi"])
                bA, bB = 0.5 * sgn * delta, -0.5 * sgn * delta
                eps = {a: rng.gauss(0.0, npz["replicate_sigma"]) for a in ALPHAS_INTERIOR}
                theta = {}
                for a in ALPHAS_INTERIOR:
                    gh = [rng.gauss(prior["mu_ghost_top1"] + eps[a], s_gh)
                          for _ in range(n_ghost)]
                    theta[a] = quantile(gh, ALIVE_Q_MAIN)
                ownA = [rng.gauss(prior["mu_own_top1"] + bA, s_of) for _ in range(nA)]
                othA = [rng.gauss(prior["mu_other_top1"] + bB, s_of) for _ in range(nA)]
                ownB = [rng.gauss(prior["mu_own_top1"] + bB, s_of) for _ in range(nB)]
                othB = [rng.gauss(prior["mu_other_top1"] + bA, s_of) for _ in range(nB)]
                for pname, w in K3B_SWAP_PAIRS:
                    aA, aB = w, round(1.0 - w, 2)
                    dA = [eps[aA] + rng.gauss(0.0, sig_d) for _ in range(nA)]
                    dB = [eps[aB] + rng.gauss(0.0, sig_d) for _ in range(nB)]
                    kA = sum(1 for i in range(nA)
                             if w * ownA[i] + (1 - w) * othA[i] + dA[i] >= theta[aA])
                    kB = sum(1 for i in range(nB)
                             if w * ownB[i] + (1 - w) * othB[i] + dB[i] >= theta[aB])
                    pA, pB = kA / float(nA), kB / float(nB)
                    mA_ = median([abs(x) for x in dA])
                    mB_ = median([abs(x) for x in dB])
                    seA = _foldnorm_med_se(eps[aA], sig_d, nA)[1]
                    seB = _foldnorm_med_se(eps[aB], sig_d, nB)[1]
                    lam_n = abs(mA_ - mB_) / max(1e-9, seA, seB)
                    se_s = max(1e-9,
                               math.sqrt(max(0.0, pA * (1 - pA)) / nA),
                               math.sqrt(max(0.0, pB * (1 - pB)) / nB))
                    lam_s = abs(pA - pB) / se_s
                    lam_seqs[(pname, "nmad")].append(lam_n)
                    lam_seqs[(pname, "S")].append(lam_s)
                    cellmax = max(cellmax, lam_n, lam_s)
                    sim_S[pname][0].append(pA)
                    sim_S[pname][1].append(pB)
            for (pname, st), seq in lam_seqs.items():
                tt = k3b_two_tier_stat(seq)
                if tt is not None:
                    tt_vals[st].append(tt)
        rep_stats.append(max(max(tt_vals["nmad"], default=0.0),
                             max(tt_vals["S"], default=0.0)))
        rep_cellmax.append(cellmax)
        for st in K3B_STATS:
            rep_stat_by[st].append(max(tt_vals[st], default=0.0))
    srt = sorted(rep_stats)
    q99 = quantile(srt, 0.99)
    verdict = "maintain_3se" if q99 <= 3.0 else "adjust_threshold_to_null_q99"
    return {
        "reps": reps,
        "criterion": ("[note]"
                      "[note]"
                      "[note]"),
        "lambda_q99": round(q99, 3),
        "lambda_median": round(quantile(srt, 0.5), 3),
        "lambda_max": round(max(rep_stats), 3),
        "lambda_q99_by_stat": {st: round(quantile(sorted(rep_stat_by[st]), 0.99), 3)
                               for st in K3B_STATS},
        "lambda_cellmax_q99": round(quantile(sorted(rep_cellmax), 0.99), 3),
        "noise_prior": dict(K3B_NOISE_PRIOR,
                            sigma_d_by_cell="[note]",
                            n_by_cell="[note]",
                            sigma_field_own_other=round(s_of, 4),
                            s_fact=round(s_fact, 4), sigma_tm=round(sigma_tm, 4),
                            seeds="K3B_CAL_SEED_BASE(7000)+rep"),
        "prior_basis": K3B_PRIOR_BASIS,
        "sim_S_median_by_pair": {p: [round(median(sim_S[p][0]), 4),
                                     round(median(sim_S[p][1]), 4)]
                                 for p, _ in K3B_SWAP_PAIRS},
        "frozen_S_median_by_pair": {"P1_w25": [0.0564, 0.0539], "P2_w50": [0.2694, 0.2548],
                                    "P3_w75": [0.6868, 0.7169],
                                    "basis": "[note]"},
        "se_method": ("[note]"
                      "[note]"
                      "[note]"),
        "decision_rule_prefrozen": "[note]",
        "verdict": verdict,
        "recommended_threshold": 3.0 if q99 <= 3.0 else round(q99, 3),
    }


def phase_nullsim(ctx, reps=200, n_facts=1500, sk_reps=200, sk_facts=6000, k3b_reps=200):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    prior = NULLSIM_PRIOR
    lam = math.log(prior["rank_anchor_rank"]) / prior["rank_anchor_margin"]
    s_fact = _calib_s_fact(lam)
    sigma_tm = prior["sigma_probe"] / math.sqrt(prior["n_templates"])
    sigma_delta = sigma_tm * math.sqrt(1.5)
    taus = TAU_GRID_TOKENS
    false_cp, lin_ok, recover = 0, 0, 0
    inj_idx = 4
    n_ghost = max(100, n_facts // 3)
    for rep in range(reps):
        rng = random.Random(1000 + rep)
        m_end = [rng.gauss(prior["mu_own_top1"], s_fact) + rng.gauss(0, sigma_tm)
                 for _ in range(n_facts)]
        scale = median(m_end)

        lin_pass_seq = []
        for ti in range(len(taus)):
            deltas = [rng.gauss(0, sigma_delta) for _ in range(n_facts)]
            lin_pass_seq.append(nmad(deltas, scale) <= NMAD_LIN_THRESH)
        if tau_star_regime(lin_pass_seq, taus) is not None:
            false_cp += 1
        if all(lin_pass_seq):
            lin_ok += 1

        c_own = [rng.gauss(prior["mu_own_top1"], s_fact) for _ in range(n_facts)]
        c_oth = [rng.gauss(prior["mu_other_top1"], s_fact) for _ in range(n_facts)]
        gh_end = [rng.gauss(prior["mu_ghost_top1"], prior["sigma_fact_ghost"])
                  + rng.gauss(0, sigma_tm) for _ in range(n_ghost)]
        th_end = quantile(gh_end, ALIVE_Q_MAIN)
        f_live = [i for i in range(n_facts)
                  if c_own[i] + rng.gauss(0, sigma_tm) >= th_end]
        base_merge = [0.5 * c_own[i] + 0.5 * c_oth[i] for i in range(n_facts)]
        gh0 = [rng.gauss(prior["mu_ghost_top1"], prior["sigma_fact_ghost"])
               + rng.gauss(0, sigma_tm) for _ in range(n_ghost)]
        headroom = max(0.1, median(base_merge) - quantile(gh0, ALIVE_Q_MAIN))
        surv = []
        for ti in range(len(taus)):
            deficit = 0.0 if ti < inj_idx else headroom * (1.8 + 0.8 * (ti - inj_idx))
            gh = [rng.gauss(prior["mu_ghost_top1"], prior["sigma_fact_ghost"])
                  + rng.gauss(0, sigma_tm) for _ in range(n_ghost)]
            th = quantile(gh, ALIVE_Q_MAIN)
            alive = 0
            for i in f_live:
                m = base_merge[i] - deficit + rng.gauss(0, sigma_tm)
                if m >= th:
                    alive += 1
            surv.append(alive / max(1, len(f_live)))
        ts = tau_star_survival(surv, taus)
        if ts is not None and abs(taus.index(ts[1]) - inj_idx) <= 1:
            recover += 1

    r1keeps = []
    T = prior["n_templates"]
    for rep in range(sk_reps):
        rng = random.Random(3000 + rep)
        r1_end = r1_mrg = 0
        for _f in range(sk_facts):
            co = rng.gauss(prior["mu_own_top1"], s_fact)
            cx = rng.gauss(prior["mu_other_top1"], s_fact)
            re_sum = rm_sum = 0
            for _t in range(T):
                mo = co + rng.gauss(0, prior["sigma_probe"])
                mx = cx + rng.gauss(0, prior["sigma_probe"])
                re_sum += _rank_of_margin(mo, lam)
                rm_sum += _rank_of_margin(0.5 * mo + 0.5 * mx, lam)
            if re_sum / T <= RANK1_BAR_THRESH:
                r1_end += 1
            if rm_sum / T <= RANK1_BAR_THRESH:
                r1_mrg += 1
        r1keeps.append(r1_mrg / float(r1_end) if r1_end else float("nan"))
    p_below = sum(1 for v in r1keeps if v < STOREKEEP_THRESH) / float(max(1, len(r1keeps)))
    q10 = quantile(sorted(r1keeps), 0.10)
    verdict = ("keep_threshold_0.90" if p_below < 0.10
               else "recommend_threshold_null_q10")

    k3b_cal = _k3b_nullsim_calibration(s_fact, sigma_tm, reps=k3b_reps)
    sim_prior = dict(prior, s_fact_calibrated=round(s_fact, 4),
                     lambda_rank=round(lam, 4),
                     sigma_template_mean=round(sigma_tm, 4),
                     sigma_delta=round(sigma_delta, 4),
                     deficit_schedule="[note]",
                     seeds={"calib": "777+iter", "worlds": "1000+rep",
                            "storekeep": "3000+rep"})
    rep_out = {"reps": reps, "n_facts": n_facts,
               "carrier": "[note]",
               "false_changepoint_rate": round(false_cp / reps, 3),
               "lin_regime_rate_null": round(lin_ok / reps, 3),
               "recovery_rate": round(recover / reps, 3),
               "criteria": {"false_cp_max": 0.10, "lin_rate_min": 0.90,
                            "recovery_min": 0.80},
               "pass": (false_cp / reps <= 0.10 and lin_ok / reps >= 0.90
                        and recover / reps >= 0.80),
               "simulator_prior": sim_prior,
               "prior_basis": NULLSIM_PRIOR_BASIS,
               "storekeep_calibration": {
                   "p_below_0p9": round(p_below, 4),
                   "q10": round(q10, 4),
                   "reps": sk_reps, "n_facts": sk_facts,
                   "r1keep_median": round(median(r1keeps), 4),
                   "r1keep_q90": round(quantile(sorted(r1keeps), 0.90), 4),
                   "threshold_current": STOREKEEP_THRESH,
                   "decision_rule_prefrozen":
                       "[note]",
                   "verdict": verdict,
                   "recommended_threshold": (STOREKEEP_THRESH if p_below < 0.10
                                             else round(q10, 4)),
                   "simulator_prior": sim_prior,
                   "prior_basis": NULLSIM_PRIOR_BASIS},
               "k3b_calibration": k3b_cal}
    write_json(ctx.p("nullsim_report.json"), rep_out)
    print("[nullsim]", json.dumps(
        {k: rep_out[k] for k in ["false_changepoint_rate", "lin_regime_rate_null",
                                 "recovery_rate", "pass"]}, ensure_ascii=False))
    print("[nullsim.storekeep]", json.dumps(
        {k: rep_out["storekeep_calibration"][k]
         for k in ["p_below_0p9", "q10", "r1keep_median", "verdict"]}, ensure_ascii=False))
    print("[nullsim.k3b]", json.dumps(
        {k: k3b_cal[k] for k in ["lambda_q99", "lambda_median", "lambda_max",
                                 "lambda_q99_by_stat", "reps", "verdict",
                                 "recommended_threshold"]}, ensure_ascii=False))
    return rep_out["pass"]


# =====================================================================================

# =====================================================================================

def _gpu_list():
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        n = len([l for l in out.stdout.splitlines() if l.startswith("GPU")])
        return list(range(max(1, n)))
    except Exception:
        return [0]


def run_pool(ctx, tasks, tag):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    gpus = _gpu_list()
    procs = {}
    queue = list(tasks)
    while queue or procs:
        while queue and len(procs) < len(gpus):
            argv = queue.pop(0)
            free = [g for g in gpus if g not in procs][0]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(free)
            env["RANK08_NGPUS"] = str(len(gpus))
            p = subprocess.Popen([sys.executable, os.path.abspath(__file__)] + argv +
                                 ["--out-dir", ctx.out], env=env,
                                 start_new_session=True)
            procs[free] = (p, time.time(), argv)
        time.sleep(5)
        for g, (p, t0, argv) in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                if rc not in (0,):
                    jsonl_append(ctx.p("failed_runs.jsonl"),
                                 {"argv": argv, "rc": rc, "pool": tag})
                    if rc in (3, 4, 5):
                        sys.exit(rc)
                del procs[g]
            elif time.time() - t0 > 100 * 60:
                p.kill()
                jsonl_append(ctx.p("failed_runs.jsonl"),
                             {"argv": argv, "rc": "stall_killed", "pool": tag})
                del procs[g]
        ctx.ledger("pool:" + tag)
        ctx.fuse_check("pool:" + tag)


def phase_all(ctx):
    phase_universe(ctx)
    phase_mb(ctx)
    run_pool(ctx, [["--phase", "base", "--seed", str(s)] for s in TRAIN_SEEDS], "base")

    phase_gate(ctx)
    kf = k_final(ctx)
    ctx.ledger("phase0_done", force=True)

    expo = [["--phase", "expose", "--seed", str(s), "--branch", b]
            for s in TRAIN_SEEDS[1:] for b in ["A", "B"]]
    run_pool(ctx, expo, "expose")
    run_pool(ctx, [["--phase", "drift", "--seed", str(s), "--branch", b]
                   for s in TRAIN_SEEDS for b in ["A", "B"]], "drift")
    shards = [["--phase", "merge_eval", "--seed", str(s), "--tau-idx", str(t)]
              for s in TRAIN_SEEDS for t in range(len(TAU_GRID_STEPS))]
    run_pool(ctx, shards, "merge_eval")
    run_pool(ctx, [["--phase", "lmc", "--seed", str(s), "--tau-idx", str(t)]
                   for s in TRAIN_SEEDS for t in range(len(TAU_GRID_STEPS))], "lmc")
    run_pool(ctx, [["--phase", "repair", "--unit", u["unit"]] for u in repair_units(ctx)],
             "repair")


    phase_rollup(ctx)


# =====================================================================================


# =====================================================================================

def phase_equiv_regress(n_sub=1500):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    t0 = time.time()
    ck = os.path.join(V1_ARCHIVE, "ckpts", "expA-s17-k256", "final.pt")
    arch_margins = os.path.join(V1_ARCHIVE, "eval", "margins.s17-t0-endA-k256.jsonl.gz")
    for p in (ck, arch_margins):
        if not os.path.exists(p):
            print("[note]", p)
            return False
    pools = build_facts()
    templates = _make_templates()
    facts = pools["excl_A"]
    subsample = 0 < n_sub < len(facts)
    if subsample:
        rng = random.Random(EQUIV_SUBSAMPLE_SEED)
        idx = sorted(rng.sample(range(len(facts)), n_sub))
        facts = [facts[i] for i in idx]
    mini_pools = {"shared": [], "excl_A": facts, "excl_B": [],
                  "ghost_A": [], "ghost_B": []}
    battery = build_probe_battery(mini_pools, templates)
    sd = load_sd_fp32(ck)
    per_fact, dev = _battery_readings(sd, battery)

    rk, tp, lg = [], [], []
    n_rank1 = 0
    for f in facts:
        rec = per_fact[(f["id"], "train")]
        rbar = sum(rec["rank"]) / len(rec["rank"])
        rk.append(rbar)
        tp.append(sum(rec["top1"]) / len(rec["top1"]))
        lg.append(sum(rec["lp"]) / len(rec["lp"]))
        if rbar <= RANK1_BAR_THRESH:
            n_rank1 += 1
    frac_new = n_rank1 / float(len(facts))

    arch = {}
    with gzip.open(arch_margins, "rt") as f:
        for line in f:
            r = json.loads(line)
            if "__meta__" not in r and r.get("set") == "excl_A":
                arch[r["fact_id"]] = r["m_raw"]
    diffs = []
    for f in facts:
        rec = per_fact[(f["id"], "hold")]
        m_raw_new = sum(rec["raw"]) / len(rec["raw"])
        if f["id"] in arch:
            diffs.append(abs(m_raw_new - arch[f["id"]]))
    band = EQUIV_PASS_BAND_GPU if dev == "cuda" else EQUIV_PASS_BAND_CPU
    extrap = 0.0
    if subsample:
        t = EQUIV_TARGET_FRAC_RANK1
        extrap = 1.96 * math.sqrt(t * (1 - t) / len(facts))
    pass_band = band + extrap
    abs_diff = abs(frac_new - EQUIV_TARGET_FRAC_RANK1)
    ns = _torch_model_ns()
    rep = {"ckpt": ck,
           "n_facts": len(facts), "subsample": subsample,
           "subsample_seed": EQUIV_SUBSAMPLE_SEED if subsample else None,
           "frac_rank1_new": round(frac_new, 5),
           "target": EQUIV_TARGET_FRAC_RANK1,
           "abs_diff": round(abs_diff, 5),
           "pass_band": round(pass_band, 5),
           "pass_band_decomposition": {"numeric_band": band,
                                       "subsample_extrapolation": round(extrap, 5)},
           "m_raw_median_absdiff": round(median(diffs), 6) if diffs else None,
           "m_raw_max_absdiff": round(max(diffs), 6) if diffs else None,
           "m_raw_n_compared": len(diffs),
           "medians_new": {"median_rank": round(median(rk), 3),
                           "median_top1": round(median(tp), 4),
                           "median_logp": round(median(lg), 4)},
           "mdiag2_archive_ref": {"median_rank": 1.0, "median_top1": 4.0469,
                                  "median_logp": -0.0708, "frac_rank1": 0.9548,
                                  "note": "[note]"},
           "device": dev,
           "dtype_note": ("[note]"
                          "[note]"),
           "torch_threads": ns["torch"].get_num_threads(),
           "wall_s": round(time.time() - t0, 1),
           "pass": bool(abs_diff <= pass_band)}
    os.makedirs("merge_audit_dryrun", exist_ok=True)
    write_json(os.path.join("merge_audit_dryrun", "equiv_regress.json"), rep)
    print("[equiv_regress]", json.dumps(
        {k: rep[k] for k in ["frac_rank1_new", "target", "abs_diff", "pass_band",
                             "m_raw_median_absdiff", "m_raw_max_absdiff",
                             "n_facts", "device", "pass"]}, ensure_ascii=False))
    return rep["pass"]


# =====================================================================================

# =====================================================================================

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
    check("[note]", lambda: VAL0 + N_VALS <= VOCAB)
    check("[note]", lambda: all(
        TAU_GRID_TOKENS[i] == TAU_GRID_STEPS[i] * TOK_PER_STEP
        for i in range(len(TAU_GRID_STEPS))))
    check("[note]", lambda: all(
        abs(TAU_GRID_STEPS[i + 1] / max(1, TAU_GRID_STEPS[i]) - 2.0) < 0.02
        for i in range(1, len(TAU_GRID_STEPS) - 1)))
    check("[note]", lambda: TOK_PER_STEP == BATCH_SEQ * (ROW_LEN - 1))
    check("[note]", lambda: all(
        k % K_EXPOSURE_DEFAULT == 0 for k in K_LADDER))

    def _facts_deterministic():
        p1, p2 = build_facts(), build_facts()
        h = lambda p: hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()
        return h(p1) == h(p2)
    check("[note]", _facts_deterministic)

    def _tmpl():
        t = _make_templates()
        for a in range(N_ATTRS):
            assert len(t[a]["train"]) == 8 and len(t[a]["probe"]) == 3
            for tt in t[a]["train"] + t[a]["probe"]:
                sent = render_fact({"ent": [SYL0, SYL0 + 1], "attr": a, "val": VAL0 + a * 128},
                                   tt)
                assert sent[-2] >= VAL0 and sent[-1] == TOK_EOS and len(sent) <= MAX_SENT_LEN
        return True
    check("[note]", _tmpl)

    def _twin():
        pools = build_facts()
        assert all(pools["excl_A"][i]["attr"] == pools["excl_B"][i]["attr"]
                   for i in range(N_EXCL))
        cA = [0] * N_ATTRS
        for f in pools["excl_A"]:
            cA[f["attr"]] += 1
        assert min(cA) == max(cA) == N_EXCL // N_ATTRS
        return True
    check("[note]", _twin)

    def _slots():
        pat = expose_slot_pattern(17, 64, n_facts=64)
        pos, fid, tid, s_safe, steps = pat
        cnt = [0] * 64
        tc = {}
        for i in range(len(fid)):
            cnt[fid[i]] += 1
            tc[(fid[i], tid[i])] = tc.get((fid[i], tid[i]), 0) + 1
        assert min(cnt) == max(cnt) == 64
        assert all(v == 8 for v in tc.values())
        return True
    check("[note]", _slots)

    def _packer():
        pools = build_facts()
        templates = _make_templates()
        stream = SentenceStream("dryrun", [], templates, None, pools["filler"])
        pk = RowPacker(stream)
        rows = pk.next_batch(4)
        assert len(rows) == 4 * ROW_LEN and all(0 <= t < VOCAB for t in rows)
        s1 = SentenceStream("dryrun", [], templates, None, pools["filler"])
        assert s1.next_sentence() == SentenceStream(
            "dryrun", [], templates, None, pools["filler"]).next_sentence()
        return True
    check("[note]", _packer)

    def _stats():
        assert abs(quantile([1, 2, 3, 4], 0.5) - 2.5) < 1e-9
        assert kendall_tau([1, 2, 3], [3, 2, 1]) == -1.0
        assert tau_star_survival([0.9, 0.6, 0.4, 0.2], [0, 1, 2, 3]) == (1, 2)
        assert tau_star_survival([0.9, 0.9], [0, 1]) is None
        return True
    check("[note]", _stats)

    def _nullsim_mini():
        import tempfile
        base = tempfile.mkdtemp(prefix="merge_audit_dryrun_")
        ctx = Ctx(os.path.join(base, "results"))
        return phase_nullsim(ctx, reps=30, n_facts=400, sk_reps=20, sk_facts=500,
                             k3b_reps=30)


    check("[note]",
          _nullsim_mini)


    def _k3b_cal_block():
        base = os.environ.get("TMPDIR") or "."
        ctx = Ctx(os.path.join(base, "merge_audit_dryrun", "results"))


        formal = read_json(os.path.join("merge_audit_dryrun", "nullsim_report.json")) or {}
        cal = formal.get("k3b_calibration") or {}
        assert formal.get("reps") == 200, "[note]".format(formal.get("reps"))
        q = cal.get("lambda_q99")
        assert q is not None and isinstance(q, float) and q > 0 and math.isfinite(q), \
            "[note]".format(q)
        assert cal.get("verdict") in ("maintain_3se", "adjust_threshold_to_null_q99")
        th, src = _k3b_threshold(ctx, fallbacks=())
        assert abs(th - float(q)) < 1e-12, "[note]".format(th, q)
        empty = Ctx(os.path.join(base, "merge_audit_dryrun", "results_empty_k3b"))
        saved = globals()["K3B_LAMBDA_THRESH"]
        globals()["K3B_LAMBDA_THRESH"] = None
        try:
            _k3b_threshold(empty, fallbacks=())
            raise AssertionError("[note]")
        except RuntimeError:
            pass
        finally:
            globals()["K3B_LAMBDA_THRESH"] = saved
        print("      k3b mini lambda_q99={} verdict={} source={}".format(
            q, cal.get("verdict"), os.path.basename(src)))
        return True
    check("[note]", _k3b_cal_block)

    def _k3b_aligned():



        T, n, n_tau, seeds_ = 15.0, 300, 3, [1, 2, 3]
        rng = random.Random(20260816)
        def world(eps_fn, pA=0.3, pB=0.3):
            lam_n, lam_s, sgn = {}, {}, {}
            for s in seeds_:
                lam_n[s], lam_s[s], sgn[s] = [], [], []
                for ti in range(n_tau):
                    e = eps_fn(s, ti)
                    dA = [e + rng.gauss(0, 0.5) for _ in range(n)]
                    dB = [rng.gauss(0, 0.5) for _ in range(n)]
                    aA = [1.0 if rng.random() < pA else 0.0 for _ in range(n)]
                    aB = [1.0 if rng.random() < pB else 0.0 for _ in range(n)]
                    st = k3b_pair_lambdas(dA, dB, aA, aB, b=400)
                    lam_n[s].append(st["lam_nmad"])
                    lam_s[s].append(st["lam_S"])
                    g = median(dA) - median(dB)
                    sgn[s].append(1 if g > 0 else (-1 if g < 0 else 0))
            fire = any(k3b_two_tier_fire(lam_n[s], T) or k3b_two_tier_fire(lam_s[s], T)
                       for s in seeds_)
            mx = max(max(lam_n[s] + lam_s[s]) for s in seeds_)
            return fire, k3b_direction_verdict(sgn)["verdict"], mx
        f1, _v1, m1 = world(lambda s, ti: 0.0)
        assert not f1, "[note]".format(m1)
        f2, v2, m2 = world(lambda s, ti: 2.0, pA=0.7, pB=0.2)
        assert f2, "[note]".format(m2)
        assert v2 == "systematic_bias_signature", "[note]" + v2
        f3, v3, m3 = world(lambda s, ti: 0.6 * (1 if (s + ti) % 2 == 0 else -1))
        assert not f3, "[note]".format(m3)
        assert v3 == "implementation_noise_signature", "[note]" + v3

        assert k3b_two_tier_fire([1, 4, 4, 1], 3.0) and not k3b_two_tier_fire([1, 4, 1, 4], 3.0)
        assert not k3b_two_tier_fire([4, None, 4], 3.0)
        assert k3b_two_tier_stat([1, 4, 4, 1]) == 4 and k3b_two_tier_stat([4, None, 4]) is None
        print("[note]"
              "[note]".format(m1, m2, m3))
        return True
    check("[note]",
          _k3b_aligned)


    def _battery():
        pools = build_facts()
        templates = _make_templates()
        b = build_probe_battery(pools, templates)
        n_facts_all = 3 * N_EXCL + 2 * N_GHOST                    # 20,000
        assert len(b) == n_facts_all * (N_TRAIN_TEMPLATES + N_PROBE_TEMPLATES)
        per = {}
        for fid, setname, bat, ti, prompt, val in b[:2200]:
            per.setdefault((fid, bat), 0)
            per[(fid, bat)] += 1
            assert bat in BATTERIES and prompt[0] == TOK_BOS and val >= VAL0
        f0 = b[0][0]
        assert per[(f0, "train")] == N_TRAIN_TEMPLATES and per[(f0, "hold")] == N_PROBE_TEMPLATES
        return True
    check("[note]", _battery)

    def _e2cls():


        assert e2_classify(False, True, 0.0, 0.0, False, False) == "readout_floor"
        assert e2_classify(True, False, 1.0, 1.0, True, True) == "readout_floor"

        assert e2_classify(True, True, 0.95, 0.92, False, True) == "readout_injury"


        assert e2_classify(True, True, STOREKEEP_THRESH / 2, 1.0, True, True) == "storage_injury"
        assert e2_classify(True, True, STOREKEEP_THRESH * 0.5, STOREKEEP_THRESH * 0.9, False, False) == "storage_injury"

        assert e2_classify(True, True, 0.95, 1.00, True, True) == "intact"

        assert e2_classify(True, True, STOREKEEP_THRESH, STOREKEEP_THRESH, True, True) == "intact"
        assert e2_classify(True, True, STOREKEEP_THRESH - 1e-9, 1.0, True, True) == "storage_injury"
        assert e2_classify(True, True, None, 1.0, True, True) == "storage_injury"
        assert e2_classify(True, True, float("nan"), 1.0, True, True) == "storage_injury"
        return True
    check("[note]", _e2cls)

    def _universe_vs_archive():
        d = os.path.join(V1_ARCHIVE, "universe")
        assert os.path.isdir(d), "[note]"
        pools = build_facts()
        with gzip.open(os.path.join(d, "facts.json.gz"), "rt") as f:
            arch = json.load(f)
        assert json.dumps(pools, sort_keys=True) == json.dumps(arch, sort_keys=True), \
            "[note]"
        t = _make_templates()
        regen = {str(a): {"train": [[list(p) for p in tt] for tt in v["train"]],
                          "probe": [[list(p) for p in tt] for tt in v["probe"]]}
                 for a, v in t.items()}
        assert regen == read_json(os.path.join(d, "templates.json")), "[note]"
        for name, pat in [("base_s17", base_slot_pattern(17)),
                          ("expose_s17_k64", expose_slot_pattern(17, 64))]:
            for arr, suff in [(pat[0], "pos"), (pat[1], "fid"), (pat[2], "tid")]:
                with open(os.path.join(d, "schedule_{}.{}.bin".format(name, suff)),
                          "rb") as f:
                    assert f.read() == arr.tobytes(), "[note]".format(name, suff)
        return True
    check("[note]",
          _universe_vs_archive)

    def _archive_reg():
        h = _archive_hashes()
        assert all(v for v in h.values()), "[note]".format(h)
        os.makedirs("merge_audit_dryrun", exist_ok=True)
        write_json(os.path.join("merge_audit_dryrun", "archive_sha256.json"),
                   {"basis": "[note]",
                    "archive_dir": V1_ARCHIVE, "sha256": h})
        for k, v in h.items():
            print("      sha256[{}] = {}".format(k, v))
        return True
    check("[note]",
          _archive_reg)

    def _equiv():
        p = os.path.join("merge_audit_dryrun", "equiv_regress.json")
        if not os.path.exists(p):
            print("[note]")
            return False
        r = read_json(p)
        print("      frac_rank1_new={} target={} |Δ|={} band={} n={} m_raw medΔ={} maxΔ={}"
              .format(r["frac_rank1_new"], r["target"], r["abs_diff"], r["pass_band"],
                      r["n_facts"], r["m_raw_median_absdiff"], r["m_raw_max_absdiff"]))
        return bool(r["pass"])
    check("[note]", _equiv)

    def _outdir_guard():
        assert os.path.realpath("results_v2") != os.path.realpath(V1_ARCHIVE)
        assert _outdir_forbidden(V1_ARCHIVE) and not _outdir_forbidden("results_v2")
        return True
    check("[note]", _outdir_guard)

    n_models = 3 * (2 * len(TAU_GRID_STEPS) + 3 * len(TAU_GRID_STEPS)
                    + 6 * len(TAU_GRID_STEPS)) + 3 + len(repair_units(
                        Ctx(os.path.join(os.environ.get("TMPDIR") or ".",
                                         "merge_audit_dryrun", "results"))))
    n_facts_all = 3 * N_EXCL + 2 * N_GHOST
    n_items = n_facts_all * (N_TRAIN_TEMPLATES + N_PROBE_TEMPLATES)
    print("[note]")
    print("[note]".format(len(TAU_GRID_STEPS), TAU_GRID_TOKENS))
    print("[note]".format(
        BASE_STEPS, EXPOSE_STEPS_K64, TAU_GRID_STEPS[-1], REPAIR_STEPS))
    print("[note]"
          "[note]".format(
              n_facts_all, n_items, n_facts_all * 2))
    print("[note]"
          "[note]".format(n_models))
    print("[note]".format(
        K1_FRAC_RANK1_THRESH, K_LADDER))
    print("[note]".format(
        STOREKEEP_THRESH, E2_SIGNATURES))
    print("[note]".format(
        ASSUMED_TOK_S, FUSE_GPUH, MB_PROJ_CAP_GPUH))
    print("[note]".format(len(ok), len(fails)))
    return 0 if not fails else 1


# =====================================================================================
# main
# =====================================================================================

def phase_mdiag(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    ns = _torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = load_universe(ctx)

    battery = [it for it in build_probe_battery(u["pools"], u["templates"])
               if it[2] == "hold"]
    vals_t = torch.arange(VAL0, VAL0 + N_VALS, device=dev)
    by_len = {}
    for item in battery:
        by_len.setdefault(len(item[4]), []).append(item)
    rep = {"ts": time.time(), "basis": "[note]",
           "verdict_rule": "median_rank<=1.5 and median_top1_margin>0 => ruler_misread",
           "models": {}}
    for branch in ("A", "B"):
        for k in K_LADDER:
            mid = "exp{}-s17-k{}".format(branch, k)
            ck = ctx.p("ckpts", mid, "final.pt")
            if not os.path.exists(ck):
                rep["models"][mid] = {"error": "ckpt missing"}
                continue
            model = ns["init_model"](0)
            model.load_state_dict(load_sd_fp32(ck))
            model.to(dev).eval()
            per_fact = {}
            with torch.no_grad():
                for L, items in by_len.items():
                    for i0 in range(0, len(items), EVAL_BATCH):
                        chunk = items[i0:i0 + EVAL_BATCH]
                        x = torch.tensor([c[4] for c in chunk], dtype=torch.long,
                                         device=dev)
                        with torch.autocast(device_type="cuda" if dev == "cuda"
                                            else "cpu", dtype=torch.bfloat16,
                                            enabled=(dev == "cuda")):
                            logits = model(x)[:, -1, :].float()
                        logp_full = torch.log_softmax(logits, dim=1)
                        zv = logits.index_select(1, vals_t)
                        for j, (fid, setname, bat, ti, _, val) in enumerate(chunk):
                            vi = val - VAL0
                            z = zv[j]
                            rank = int((z > z[vi]).sum().item()) + 1
                            top1 = float(z[vi] - torch.cat(
                                [z[:vi], z[vi + 1:]]).max().item())
                            lp = float(logp_full[j, val].item())
                            rec = per_fact.setdefault(
                                fid, {"set": setname, "rank": [], "top1": [],
                                      "lp": []})
                            rec["rank"].append(rank)
                            rec["top1"].append(top1)
                            rec["lp"].append(lp)
            del model
            stats = {}
            for setname in ("shared", "excl_A", "excl_B"):
                rows = [r for r in per_fact.values() if r["set"] == setname]
                if not rows:
                    continue
                stats[setname] = {
                    "n": len(rows),
                    "median_rank": quantile(
                        sorted(sum(r["rank"]) / len(r["rank"]) for r in rows),
                        0.5),
                    "median_top1_margin": round(quantile(sorted(
                        sum(r["top1"]) / len(r["top1"]) for r in rows), 0.5), 4),
                    "median_logp": round(quantile(sorted(
                        sum(r["lp"]) / len(r["lp"]) for r in rows), 0.5), 4),
                    "frac_rank1": round(sum(
                        1 for r in rows
                        if sum(r["rank"]) / len(r["rank"]) <= 1.5)
                        / float(len(rows)), 4),
                }
            rep["models"][mid] = stats
            print("mdiag %s: excl median_rank=%.1f top1=%.3f logp=%.3f rank1=%.2f"
                  % (mid, stats.get("excl_" + branch, {}).get("median_rank", -1),
                   stats.get("excl_" + branch, {}).get("median_top1_margin", 0),
                   stats.get("excl_" + branch, {}).get("median_logp", 0),
                   stats.get("excl_" + branch, {}).get("frac_rank1", 0)))

    verdicts = []
    for branch in ("A", "B"):
        st = rep["models"].get("exp{}-s17-k{}".format(branch, K_LADDER[-1]), {})
        ex = st.get("excl_" + branch, {})
        if ex:
            ok = (ex["median_rank"] <= 1.5 and ex["median_top1_margin"] > 0)
            verdicts.append(ok)
    rep["verdict"] = ("ruler_misread" if verdicts and all(verdicts)
                      else "storage_failed")
    write_json(ctx.p("mdiag_report.json"), rep)
    print("mdiag verdict: %s" % rep["verdict"])


def phase_mdiag2(ctx):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    ns = _torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u = load_universe(ctx)
    pools, templates = u["pools"], u["templates"]

    def _battery(family):
        b = []
        for setname in ["shared", "excl_A", "excl_B"]:
            for f in pools[setname]:
                for ti, t in enumerate(templates[f["attr"]][family]):
                    b.append((f["id"], setname, ti, render_probe_prompt(f, t),
                              f["val"]))
        return b

    rep = {"ts": time.time(),
           "basis": "[note]",
           "verdict_rule": ("train median_rank<=1.5 and top1>0 while probe bad "
                            "=> transfer_failed; both bad => storage_failed_true"),
           "models": {}}
    for branch in ("A", "B"):
        mid = "exp{}-s17-k{}".format(branch, K_LADDER[-1])
        ck = ctx.p("ckpts", mid, "final.pt")
        model = ns["init_model"](0)
        model.load_state_dict(load_sd_fp32(ck))
        model.to(dev).eval()
        entry = {}
        for family in ("train", "probe"):
            battery = _battery(family)
            by_len = {}
            for item in battery:
                by_len.setdefault(len(item[3]), []).append(item)
            vals_t = torch.arange(VAL0, VAL0 + N_VALS, device=dev)
            per_fact = {}
            with torch.no_grad():
                for L, items in by_len.items():
                    for i0 in range(0, len(items), EVAL_BATCH):
                        chunk = items[i0:i0 + EVAL_BATCH]
                        x = torch.tensor([c[3] for c in chunk],
                                         dtype=torch.long, device=dev)
                        with torch.autocast(device_type="cuda" if dev == "cuda"
                                            else "cpu", dtype=torch.bfloat16,
                                            enabled=(dev == "cuda")):
                            logits = model(x)[:, -1, :].float()
                        logp_full = torch.log_softmax(logits, dim=1)
                        zv = logits.index_select(1, vals_t)
                        for j, (fid, setname, ti, _, val) in enumerate(chunk):
                            vi = val - VAL0
                            z = zv[j]
                            rec = per_fact.setdefault(
                                (fid, setname), {"rank": [], "top1": [],
                                                 "lp": []})
                            rec["rank"].append(
                                int((z > z[vi]).sum().item()) + 1)
                            rec["top1"].append(float(
                                z[vi] - torch.cat([z[:vi],
                                                   z[vi + 1:]]).max().item()))
                            rec["lp"].append(float(logp_full[j, val].item()))
            for setname in ("shared", "excl_A", "excl_B"):
                rows = [r for (fid, sn), r in per_fact.items()
                        if sn == setname]
                if not rows:
                    continue
                mr = sorted(sum(r["rank"]) / len(r["rank"]) for r in rows)
                mt = sorted(sum(r["top1"]) / len(r["top1"]) for r in rows)
                ml = sorted(sum(r["lp"]) / len(r["lp"]) for r in rows)
                entry.setdefault(family, {})[setname] = {
                    "n": len(rows),
                    "median_rank": round(quantile(mr, 0.5), 2),
                    "median_top1": round(quantile(mt, 0.5), 4),
                    "median_logp": round(quantile(ml, 0.5), 4),
                    "frac_rank1": round(sum(
                        1 for r in rows
                        if sum(r["rank"]) / len(r["rank"]) <= 1.5)
                        / float(len(rows)), 4)}
        del model
        rep["models"][mid] = entry
        for family in ("train", "probe"):
            ex = entry.get(family, {}).get("excl_" + branch, {})
            print("mdiag2 %s %s: rank=%.1f top1=%.3f logp=%.3f rank1=%.2f"
                  % (mid, family, ex.get("median_rank", -1),
                     ex.get("median_top1", 0), ex.get("median_logp", 0),
                     ex.get("frac_rank1", 0)))
    verdicts = []
    for branch in ("A", "B"):
        e = rep["models"].get("exp{}-s17-k{}".format(branch, K_LADDER[-1]), {})
        tr, pr = e.get("train", {}).get("excl_" + branch, {}), \
            e.get("probe", {}).get("excl_" + branch, {})
        if tr and pr:
            tr_ok = tr["median_rank"] <= 1.5 and tr["median_top1"] > 0
            pr_ok = pr["median_rank"] <= 1.5 and pr["median_top1"] > 0
            verdicts.append("transfer" if (tr_ok and not pr_ok)
                            else ("ok" if tr_ok else "store_fail"))
    rep["verdict"] = ("transfer_failed"
                      if verdicts and all(v == "transfer" for v in verdicts)
                      else ("storage_failed_true"
                            if verdicts and all(v == "store_fail"
                                                for v in verdicts)
                            else "mixed"))
    write_json(ctx.p("mdiag2_report.json"), rep)
    print("mdiag2 verdict:", rep["verdict"])


def _outdir_forbidden(out_dir):
    """Core testbed pipeline: universe generation, training segments, and the margin evaluation instrument."""
    return os.path.realpath(out_dir) == os.path.realpath(V1_ARCHIVE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default=None,
                    choices=["all", "universe", "mb", "base", "expose", "gate", "drift",
                             "merge_eval", "lmc", "repair", "rollup", "rollup_recal",
                             "nullsim", "equiv_regress", "mdiag", "mdiag2"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--branch", default=None, choices=["A", "B"])
    ap.add_argument("--tau-idx", type=int, default=None)
    ap.add_argument("--unit", default=None)
    ap.add_argument("--out-dir", default="results_v2")
    ap.add_argument("--equiv-n", type=int, default=1500,
                    help="[note]"
                         "[note]")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        sys.exit(dry_run())
    if not args.phase:
        print("[note]")
        sys.exit(1)
    if args.phase == "equiv_regress":
        sys.exit(0 if phase_equiv_regress(args.equiv_n) else 1)
    if _outdir_forbidden(args.out_dir):
        print("[note]"
              .format(args.out_dir, V1_ARCHIVE))
        sys.exit(1)
    ctx = Ctx(args.out_dir)
    readonly_phase = args.phase == "rollup_recal"
    if not readonly_phase:
        ctx.ledger(args.phase, "enter", force=True)
    try:
        if args.phase == "all":
            phase_all(ctx)
        elif args.phase == "universe":
            phase_universe(ctx)
        elif args.phase == "mb":
            phase_universe(ctx)
            phase_mb(ctx)
        elif args.phase == "base":
            phase_base(ctx, args.seed)
        elif args.phase == "expose":
            phase_expose(ctx, args.seed, args.branch, k_final(ctx))
        elif args.phase == "gate":
            phase_gate(ctx)
        elif args.phase == "drift":
            phase_drift_full(ctx, args.seed, args.branch)
        elif args.phase == "merge_eval":
            phase_merge_eval(ctx, args.seed, args.tau_idx)
        elif args.phase == "lmc":
            phase_lmc(ctx, args.seed, args.tau_idx)
        elif args.phase == "repair":
            units = {u["unit"]: u for u in repair_units(ctx)}
            phase_repair(ctx, units[args.unit])
        elif args.phase == "rollup":
            phase_rollup(ctx)
        elif args.phase == "rollup_recal":
            phase_rollup_recal(ctx)
        elif args.phase == "nullsim":
            phase_nullsim(ctx)
        elif args.phase == "mdiag":
            phase_universe(ctx)
            phase_mdiag(ctx)
        elif args.phase == "mdiag2":
            phase_universe(ctx)
            phase_mdiag2(ctx)
    finally:
        if not readonly_phase:
            ctx.ledger(args.phase, "exit", force=True)


if __name__ == "__main__":
    main()
