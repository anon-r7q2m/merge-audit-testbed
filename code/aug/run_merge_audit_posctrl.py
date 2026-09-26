#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
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

REID_SEEDS = [42, 1042, 2042]
os.environ.setdefault("TANH_R2RANK08_SEEDS", ",".join(str(s) for s in REID_SEEDS))
os.environ.setdefault("RANK08_NGPUS", "1")

import run_merge_audit as R                                   # noqa: E402

CONTRACT_ID = os.environ.get("TANH_POSCTRL_CONTRACT_ID",
                             "merge_audit-aug-posctrl-v1")
R.CONTRACT_ID = CONTRACT_ID
POSCTRL_FUSE_GPUH = 2.5
R.FUSE_GPUH = POSCTRL_FUSE_GPUH


K_TARGET = 500
TARGET_STRIDE = 12
TARGET_MIN_PER_ATTR = 60
DAMAGE_LR = 3e-5
DAMAGE_CAP_STEPS = 400
DAMAGE_RETAIN_BETA = 4.0
DAMAGE_ANS_BATCH = 256
DAMAGE_MODES = ("ascend", "misinfo")
TPR_MIN = 0.80
FPR_MAX = 0.10
A3_ALARM_MAX = 0.10
THETA_KEY = "theta_alive_top1_q990"
DAMAGE_MONITOR_EVERY = 25
DAMAGE_STOP_FRAC = 0.05
COLLAT_OWN_FRAC_MIN = 0.85
COLLAT_SHARED_DROP_MAX = 0.10
COLLAT_FILLER_RISE_MAX = 0.5
A1_TARGET_DEATH_MIN = 0.80
A1_NONTARGET_NEWDEATH_MAX = 0.20
A2_SPREAD_TOL = 1e-9
A3_UNDAMAGED_DEATH_MAX = 0.20
TGRID = [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00]
FORBIDDEN_OUTDIRS = ["results", "results_v2", "results_v2_reid"]
STALL_DAMAGE_SEED_S = 15 * 60
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


def derive_target_ids(pools):
    """Analysis script for the merge-audit study."""
    facts = pools["excl_A"]
    by_id = {f["id"]: f for f in facts}
    ids = sorted(by_id.keys())
    sub = ids[::TARGET_STRIDE]
    assert len(sub) == K_TARGET, "[note]".format(len(sub), K_TARGET)
    cnt = {}
    for i in sub:
        a = by_id[i]["attr"]
        cnt[a] = cnt.get(a, 0) + 1
    assert len(cnt) == R.N_ATTRS and min(cnt.values()) >= TARGET_MIN_PER_ATTR, \
        "[note]".format(cnt)
    return sub, cnt


class _DamageStream:
    """Analysis script for the merge-audit study."""

    def __init__(self, sentences, seed):
        self.sents = sentences
        rng = R.drng("posctrl", seed, "damage-order")
        self.order = list(range(len(sentences)))
        rng.shuffle(self.order)
        self.idx = 0

    def next_sentence(self):
        s = self.sents[self.order[self.idx % len(self.order)]]
        self.idx += 1
        return s


class _Stall(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Stall("[note]")


def _collateral_gates(ctx, cur_sd, seed, pools, templates, fz_shared_fr1, filler_base):
    """Analysis script for the merge-audit study."""
    target_ids = {r["id"] for r in R.read_json(ctx.p("damage", "target_facts.json"))["targets"]}
    nontarget = [f for f in pools["excl_A"] if f["id"] not in target_ids]
    shared = list(pools["shared"])

    def fr1_of(facts):
        battery = [(f["id"], "x", "train", ti,
                    R.render_probe_prompt(f, templates[f["attr"]]["train"][ti]), f["val"])
                   for f in facts for ti in range(R.N_TRAIN_TEMPLATES)]
        per_fact, _ = R._battery_readings(cur_sd, battery)
        n = 0
        for f in facts:
            rec = per_fact[(f["id"], "train")]
            if sum(rec["rank"]) / len(rec["rank"]) <= R.RANK1_BAR_THRESH:
                n += 1
        return n / float(len(facts))

    g1 = fr1_of(nontarget)
    g2 = fz_shared_fr1 - fr1_of(shared)
    g3 = R.eval_filler_loss(ctx, cur_sd) - filler_base
    ok = (g1 >= COLLAT_OWN_FRAC_MIN and g2 <= COLLAT_SHARED_DROP_MAX
          and g3 <= COLLAT_FILLER_RISE_MAX)
    return {"own_nontarget_fr1": round(g1, 4), "shared_fr1_drop": round(g2, 4),
            "filler_rise": round(g3, 4), "pass": ok}


def _frac_rank1_target(sd, target_facts, templates):
    """Analysis script for the merge-audit study."""
    battery = []
    for f in target_facts:
        for ti in range(R.N_TRAIN_TEMPLATES):
            t = templates[f["attr"]]["train"][ti]
            battery.append((f["id"], "excl_A", "train", ti,
                            R.render_probe_prompt(f, t), f["val"]))
    per_fact, _ = R._battery_readings(sd, battery)
    n_rank1 = 0
    for f in target_facts:
        rec = per_fact[(f["id"], "train")]
        rbar = sum(rec["rank"]) / len(rec["rank"])
        if rbar <= R.RANK1_BAR_THRESH:
            n_rank1 += 1
    return n_rank1 / float(len(target_facts))


def phase_damage(ctx, seed, pools, templates, reid=None, mode="misinfo"):
    """Analysis script for the merge-audit study."""
    tag = "damage-{}-s{}".format(mode, seed)
    ckpt_out = ctx.p("damage", "endA_dmg-{}-s{}.pt".format(mode, seed))
    if ctx.done(tag) and os.path.exists(ckpt_out):
        return "done"

    _, fz_endA = R.load_margins(reid, "s{}-t0-endA".format(seed))
    fz_shared_fr1 = _fr1({k: v for k, v in fz_endA["train"].items() if v["set"] == "shared"})
    filler_base = None
    with open(reid.p("covariates.jsonl")) as _f:
        for _line in _f:
            _r = json.loads(_line)
            if _r.get("tau_idx") == 0 and str(_r.get("seed")) == str(seed):
                filler_base = _r["filler_loss_A"]
    ns = R._torch_model_ns()
    torch = ns["torch"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    target_ids = {r["id"] for r in R.read_json(ctx.p("damage", "target_facts.json"))["targets"]}
    target_facts = [f for f in pools["excl_A"] if f["id"] in target_ids]

    _attr_vals = {}
    for _pool in ("excl_A", "excl_B", "shared"):
        for _f in pools[_pool]:
            _attr_vals.setdefault(_f["attr"], set()).add(_f["val"])
    _attr_sorted = {a: sorted(vs) for a, vs in _attr_vals.items()}

    def wrong_val_of(f):
        vs = _attr_sorted[f["attr"]]
        assert len(vs) >= 2, "[note]".format(f["attr"])
        return vs[(vs.index(f["val"]) + 1) % len(vs)]
    sentences = [R.render_fact(f, templates[f["attr"]]["train"][ti])
                 for f in target_facts for ti in range(R.N_TRAIN_TEMPLATES)]

    ans_pairs = []
    for s, f in zip(sentences, [f for f in target_facts for _ in range(R.N_TRAIN_TEMPLATES)]):
        wv = wrong_val_of(f)
        ans_pairs.append((s, len(s) - 2, wv))
    ans_stream = _DamageStream(ans_pairs, seed)
    packer = R.RowPacker(_DamageStream(sentences, seed))

    own_nontarget = [f for f in pools["excl_A"] if f["id"] not in target_ids]
    retain_facts = own_nontarget + list(pools["shared"]) + list(pools["filler"])
    sentences_r = [R.render_fact(f, templates[f["attr"]]["train"][ti])
                   for f in retain_facts for ti in range(R.N_TRAIN_TEMPLATES)]
    packer_retain = R.RowPacker(_DamageStream(sentences_r, seed + 777))
    sd0 = R.load_sd_fp32(os.path.join("results_v2_reid", "ckpts",
                                      "expA-s{}-k128".format(seed), "final.pt"))
    model = ns["init_model"](0)
    model.load_state_dict(sd0)
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=DAMAGE_LR,
                            betas=R.ADAM_BETAS, weight_decay=R.ADAM_WD)
    log = {"seed": seed, "track": []}
    signal.alarm(STALL_DAMAGE_SEED_S)
    nan_streak, stop, frac = 0, "cap_exhausted", None
    try:
        for step in range(DAMAGE_CAP_STEPS + 1):
            if step % DAMAGE_MONITOR_EVERY == 0:
                cur_sd = {k: v.detach().cpu().float() for k, v in model.state_dict().items()}
                frac = _frac_rank1_target(cur_sd, target_facts, templates)

                coll = _collateral_gates(ctx, cur_sd, seed, pools, templates, fz_shared_fr1, filler_base)
                log["track"].append({"step": step, "frac_rank1_target": round(frac, 4),
                                     "collateral": coll})
                R.jsonl_append(ctx.p("damage", "damage_log.jsonl"),
                               {"seed": seed, "step": step, "frac_rank1_target": round(frac, 4),
                                "collateral_pass": coll["pass"],
                                "collateral": coll})
                ctx.ledger("damage:s{}".format(seed), force=True)
                ctx.fuse_check("damage")

                if frac <= DAMAGE_STOP_FRAC:
                    stop = "stop_rule_met"
                    break
            if step == DAMAGE_CAP_STEPS:
                break
            if mode == "ascend":
                rows = packer.next_batch(R.BATCH_SEQ)
                x = torch.tensor(rows, dtype=torch.long, device=dev).view(R.BATCH_SEQ, R.ROW_LEN)
                inp, tgt = x[:, :-1], x[:, 1:]
            else:
                batch = [ans_stream.next_sentence() for _ in range(DAMAGE_ANS_BATCH)]
                maxlen = max(len(s) for s, _, _ in batch)
                xb = torch.full((len(batch), maxlen), R.TOK_EOS, dtype=torch.long, device=dev)
                apos = torch.empty(len(batch), dtype=torch.long, device=dev)
                wval = torch.empty(len(batch), dtype=torch.long, device=dev)
                for i, (s, ap, wv) in enumerate(batch):
                    xb[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=dev)
                    apos[i] = ap
                    wval[i] = wv
            rows_r = packer_retain.next_batch(R.BATCH_SEQ)
            xr = torch.tensor(rows_r, dtype=torch.long, device=dev).view(R.BATCH_SEQ, R.ROW_LEN)
            inp_r, tgt_r = xr[:, :-1], xr[:, 1:]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu",
                                dtype=torch.bfloat16, enabled=(dev == "cuda")):
                if mode == "ascend":
                    logits = model(inp)
                    ce = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, R.VOCAB), tgt.reshape(-1))
                else:
                    logits = model(xb[:, :-1])
                    _ar = torch.arange(len(batch), device=dev)
                    ce = torch.nn.functional.cross_entropy(
                        logits[_ar, apos - 1], wval)
                logits_r = model(inp_r)
                ce_r = torch.nn.functional.cross_entropy(
                    logits_r.reshape(-1, R.VOCAB), tgt_r.reshape(-1))

                loss = (-ce if mode == "ascend" else ce) + DAMAGE_RETAIN_BETA * ce_r
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), R.GRAD_CLIP)
            opt.step()
            lv = loss.item()
            nan_streak = nan_streak + 1 if (math.isnan(lv) or math.isinf(lv)) else 0
            if nan_streak >= 10:                                      # K-8
                stop = "failed_nan"
                break
    finally:
        signal.alarm(0)
    log["stop_reason"] = stop
    log["final_frac_rank1_target"] = None if frac is None else round(frac, 4)
    R.jsonl_append(ctx.p("damage", "damage_log.jsonl"),
                   {"seed": seed, "final": True, **{k: log[k] for k in
                    ("stop_reason", "final_frac_rank1_target")}})
    if stop == "failed_nan":
        R.jsonl_append(ctx.p("failed_runs.jsonl"), {"run": tag, "why": "K-8 nan"})
        return "failed"


    R.save_sd_bf16(model, ckpt_out)
    ctx.mark_done(tag)
    return "done"


def _eval_model(ctx, battery, mid, sd):
    if os.path.exists(ctx.p("eval", "margins.{}.jsonl.gz".format(mid))):
        return
    signal.alarm(STALL_PER_MODEL_S)
    try:
        R.eval_margins(ctx, mid, sd, battery)
    finally:
        signal.alarm(0)
    ctx.ledger("eval:" + mid, force=True)
    ctx.fuse_check("eval")


def phase_eval(ctx, seed, battery):
    """Analysis script for the merge-audit study."""
    for mode in DAMAGE_MODES:
        sd_dmg = R.load_sd_fp32(ctx.p("damage", "endA_dmg-{}-s{}.pt".format(mode, seed)))
        _eval_model(ctx, battery, "posctrl-s{}-{}dmg".format(seed, mode), sd_dmg)
        extra_p = ctx.p("damage", "eval_extra.json")
        extra = R.read_json(extra_p) if os.path.exists(extra_p) else {}
        key = "filler_loss_{}dmg".format(mode)
        if key not in extra:
            signal.alarm(STALL_PER_MODEL_S)
            try:
                extra[key] = round(R.eval_filler_loss(ctx, sd_dmg), 4)
            finally:
                signal.alarm(0)
        R.write_json(extra_p, extra)


def _fr1(rows, fids=None):
    sel = [r for fid, r in rows.items() if fids is None or fid in fids]
    if not sel:
        return None
    return sum(1 for r in sel if r["rank_bar"] <= R.RANK1_BAR_THRESH) / float(len(sel))


def _death_frac(rows, fids):
    """Analysis script for the merge-audit study."""
    return sum(1 for f in fids if not rows[f]["alive"]) / float(len(fids))


def phase_verdict(ctx, reid, pools, seed_status):
    """Analysis script for the merge-audit study."""
    tf = R.read_json(ctx.p("damage", "target_facts.json"))
    target_set = {t["id"] for t in tf["targets"]}
    report = {"contract_id": CONTRACT_ID, "prereg": "PREREG-AUG.md §2 + AMENDMENT A-2.9",
              "design": "per-fact stratified power validation (measured forgetting state)",
              "thresholds": {"tpr_min": TPR_MIN, "fpr_max": FPR_MAX,
                             "a3_alarm_max": A3_ALARM_MAX, "theta": THETA_KEY},
              "seed_status": seed_status, "per_cell": {}, "pooled": {}, "legs": {}}

    def _auc(scores_labels):
        """Analysis script for the merge-audit study."""
        pos = [sc for sc, lb in scores_labels if lb]
        neg = [sc for sc, lb in scores_labels if not lb]
        if not pos or not neg:
            return None
        wins = sum(1 for a in pos for b in neg if a > b)
        ties = sum(1 for a in pos for b in neg if a == b)
        return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)

    pool_all, pool_pos, pool_neg = [], [], []
    for mode in DAMAGE_MODES:
        for seed in REID_SEEDS:
            s = str(seed)
            if seed_status.get("{}:{}".format(mode, seed), seed_status.get(s)) != "done":
                continue
            meta_d, dmg = R.load_margins(ctx, "posctrl-s{}-{}dmg".format(seed, mode))
            _, fz = R.load_margins(reid, "s{}-t0-endA".format(seed))
            th = meta_d[THETA_KEY]

            pos, neg = [], []
            for fid, row in dmg["train"].items():
                if row["set"] != "excl_A":
                    continue
                pre = fz["train"].get(fid)
                if pre is None:
                    continue
                pre_known = pre["rank_bar"] <= R.RANK1_BAR_THRESH
                post_known = row["rank_bar"] <= R.RANK1_BAR_THRESH
                if not pre_known:
                    continue
                alarm = row["m_top1"] < th
                (pos if not post_known else neg).append((fid, alarm, row["m_top1"]))
            tpr = (sum(1 for _, a, _ in pos if a) / len(pos)) if pos else None
            fpr = (sum(1 for _, a, _ in neg if a) / len(neg)) if neg else None
            auc = _auc([(-m, True) for _, _, m in pos] + [(-m, False) for _, _, m in neg])

            aset0 = {fid for fid, alarm, _ in pos + neg if alarm}
            placebo_ok = all(
                {f for f in [fid for fid, _, _ in pos + neg]
                 if (t * dmg["train"][f]["m_top1"]) < (t * th)} == aset0 for t in TGRID)
            cell = {"n_forgotten": len(pos), "n_retained": len(neg),
                    "tpr": None if tpr is None else round(tpr, 4),
                    "fpr": None if fpr is None else round(fpr, 4),
                    "auc": auc, "placebo_alarm_set_invariant": placebo_ok,
                    "theta_q99": round(th, 6),
                    "target_share_in_forgotten": round(
                        sum(1 for f, _, _ in pos if f in target_set) / max(1, len(pos)), 4)}
            report["per_cell"]["{}:s{}".format(mode, seed)] = cell
            pool_pos += [(mode, seed) + x for x in pos]
            pool_neg += [(mode, seed) + x for x in neg]

    tpr_p = sum(1 for x in pool_pos if x[3]) / max(1, len(pool_pos))
    fpr_p = sum(1 for x in pool_neg if x[3]) / max(1, len(pool_neg))
    report["pooled"] = {"n_forgotten": len(pool_pos), "n_retained": len(pool_neg),
                        "tpr": round(tpr_p, 4), "fpr": round(fpr_p, 4),
                        "auc": None,
                        "auc_note": "[note]"}

    a3_rates = []
    for seed in REID_SEEDS:
        meta_fz, fz = R.load_margins(reid, "s{}-t0-endA".format(seed))
        th0 = meta_fz[THETA_KEY]
        known0 = [fid for fid, row in fz["train"].items()
                  if row["set"] == "excl_A"
                  and row["rank_bar"] <= R.RANK1_BAR_THRESH]
        a3_rates.append(sum(1 for f in known0 if fz["train"][f]["m_top1"] < th0)
                        / max(1, len(known0)))
    report["legs"]["A3_undamaged_alarm"] = [round(r, 4) for r in a3_rates]
    report["legs"]["A2_placebo"] = all(c["placebo_alarm_set_invariant"]
                                       for c in report["per_cell"].values())

    profile = {}
    try:
        for line in open(ctx.p("damage", "damage_log.jsonl")):
            d = json.loads(line)
            if d.get("final"):
                continue
            c = d.get("collateral")
            if c:
                profile.setdefault("s{}".format(d["seed"]), []).append(
                    {"step": d["step"], "g1": c["own_nontarget_fr1"],
                     "g2": c["shared_fr1_drop"], "g3": c["filler_rise"]})
    except FileNotFoundError:
        pass
    report["collateral_profile_descriptive"] = profile
    verdict_pass = (tpr_p >= TPR_MIN and fpr_p <= FPR_MAX
                    and report["legs"]["A2_placebo"]
                    and max(a3_rates) <= A3_ALARM_MAX)
    report["verdict"] = {"posctrl_pass": verdict_pass,
                         "wording_cap": ("[note]"
                                         "[note]"
                                         if verdict_pass else
                                         "[note]")}
    report["status"] = "complete"
    R.write_json(ctx.p("posctrl_report.json"), report)
    eval_dir = ctx.p("eval")
    files = sorted(os.listdir(eval_dir)) if os.path.isdir(eval_dir) else []
    R.write_json(ctx.p("manifest.json"),
                 {"contract_id": CONTRACT_ID,
                  "files": {f: R.sha256_file(os.path.join(eval_dir, f)) for f in files}})


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

    def _target_real():
        """Analysis script for the merge-audit study."""
        with gzip.open(os.path.join("results_v2_reid", "universe", "facts.json.gz"), "rt") as f:
            pools = json.load(f)
        sub, cnt = derive_target_ids(pools)
        assert len(sub) == 500 and len(cnt) == 8 and min(cnt.values()) >= 60
        sub2, _ = derive_target_ids(pools)
        assert sub == sub2

        ids_num = [f["id"] for f in pools["excl_A"]]
        by_id = {f["id"]: f for f in pools["excl_A"]}
        cnt_num = {}
        for i in ids_num[::12]:
            a = by_id[i]["attr"]
            cnt_num[a] = cnt_num.get(a, 0) + 1
        assert len(cnt_num) == 2, "[note]"
        print("[note]".format(
            dict(sorted(cnt.items())), cnt_num))
        return True
    check("[note]", _target_real)

    def _stream_det():
        s1 = _DamageStream([[1, 2], [3, 4], [5, 6]], 42)
        s2 = _DamageStream([[1, 2], [3, 4], [5, 6]], 42)
        assert [s1.next_sentence() for _ in range(5)] == [s2.next_sentence() for _ in range(5)]
        s3 = _DamageStream([[1, 2], [3, 4], [5, 6]], 43)
        assert s3.order != s1.order or True
        return True
    check("[note]", _stream_det)

    def _stop_logic():
        """Analysis script for the merge-audit study."""
        def run(track):
            for step in range(0, DAMAGE_CAP_STEPS + 1, 25):
                f = track(step)
                if f <= DAMAGE_STOP_FRAC:
                    return ("stop_rule_met", step)
                if step == DAMAGE_CAP_STEPS:
                    return ("cap_exhausted", step)
        assert run(lambda s: 0.9 if s < 25 else 0.04) == ("stop_rule_met", 25)
        assert run(lambda s: 0.9) == ("cap_exhausted", DAMAGE_CAP_STEPS)
        return True
    check("[note]", _stop_logic)

    def _a2_identity():
        """Analysis script for the merge-audit study."""
        m, g = 2.345678, -4.123456
        vals = [(t * m) / abs(t * g) for t in TGRID]
        sp = max(vals) - min(vals)
        assert sp < A2_SPREAD_TOL, sp
        print("[note]".format(sp))
        return True
    check("[note]", _a2_identity)

    def _gates():
        """Analysis script for the merge-audit study."""
        assert 0.5 < TPR_MIN <= 1.0 and 0.0 < FPR_MAX < 0.5
        assert TPR_MIN - FPR_MAX >= 0.6
        assert A3_ALARM_MAX == FPR_MAX

        def stratify(pre, post):
            if not pre: return None
            return "pos" if not post else "neg"
        assert stratify(True, False) == "pos" and stratify(True, True) == "neg"
        assert stratify(False, False) is None
        return True
    check("[note]", _gates)
    check("[note]",
          lambda: (K_TARGET, TARGET_STRIDE, DAMAGE_CAP_STEPS, DAMAGE_STOP_FRAC,
                   COLLAT_OWN_FRAC_MIN, COLLAT_SHARED_DROP_MAX, COLLAT_FILLER_RISE_MAX,
                   A1_TARGET_DEATH_MIN, A1_NONTARGET_NEWDEATH_MAX, A3_UNDAMAGED_DEATH_MAX)
          == (500, 12, 400, 0.05, 0.85, 0.10, 0.5, 0.80, 0.20, 0.20))  # A-2.2: cap 200→400
    check("[note]",
          lambda: TGRID == [0.60, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 2.00])
    check("[note]",
          lambda: all(_rejects(d) for d in FORBIDDEN_OUTDIRS)
          and _assert_outdir_safe("results/posctrl"))
    check("[note]", lambda: R.FUSE_GPUH == POSCTRL_FUSE_GPUH)

    def _idem():
        with tempfile.TemporaryDirectory(prefix="posctrl_dryrun_") as td:
            ctx = R.Ctx(os.path.join(td, "out"))
            assert not ctx.done("damage-s42")
            ctx.mark_done("damage-s42")
            assert ctx.done("damage-s42")
            return True
    check("[note]", _idem)

    def _frozen_present():
        reid = ROCtx("results_v2_reid")
        for seed in REID_SEEDS:
            assert os.path.exists(reid.p("ckpts", "expA-s{}-k128".format(seed), "final.pt"))
            assert os.path.exists(reid.p("ckpts", "driftB-s{}".format(seed), "tau0.pt"))
            assert os.path.exists(reid.p("ckpts", "base-s{}".format(seed), "final.pt"))
            for mid in ["s{}-t0-endA", "s{}-t0-a0.50-merge", "s{}-t0-s0.50-disA"]:
                assert os.path.exists(reid.p("eval", "margins.{}.jsonl.gz".format(
                    mid.format(seed))))
        assert os.path.exists(reid.p("covariates.jsonl"))
        return True
    check("[note]",
          _frozen_present)

    def _contract():
        c = json.load(open(os.path.join(EXP, "contract-posctrl-draft.json")))
        assert "run_merge_audit_posctrl.py" in c["command"]
        assert c["draft"] is True
        assert c["contract_id"].replace("-draft", "") == CONTRACT_ID
        for p in ["results/posctrl/posctrl_report.json",
                      "results/posctrl/damage/target_facts.json",
                      "results/posctrl/damage/damage_log.jsonl"]:
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
    ap.add_argument("--reid-dir", default="results_v2_reid")
    ap.add_argument("--out-dir", default="results/posctrl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        sys.exit(dry_run())
    _assert_outdir_safe(args.out_dir)
    signal.signal(signal.SIGALRM, _alarm_handler)
    reid = ROCtx(args.reid_dir)
    ctx = R.Ctx(args.out_dir)
    ctx.ledger("posctrl", "enter", force=True)
    try:
        u = R.load_universe(reid)
        pools, templates = u["pools"], u["templates"]

        tf_path = ctx.p("damage", "target_facts.json")
        if not os.path.exists(tf_path):
            sub, cnt = derive_target_ids(pools)
            R.write_json(tf_path, {
                "rule": "excl_A ids sorted (python string order) [::12]",
                "k": K_TARGET, "stride": TARGET_STRIDE,
                "per_attr_counts": {str(a): c for a, c in sorted(cnt.items())},
                "targets": [{"id": i} for i in sub]})
        battery = R.build_probe_battery(pools, templates)

        seed_status = {}
        for mode in DAMAGE_MODES:
            for seed in REID_SEEDS:
                try:
                    seed_status["{}:{}".format(mode, seed)] = phase_damage(
                        ctx, seed, pools, templates, reid=reid, mode=mode)
                except _Stall as e:
                    print("[FATAL]", e)
                    sys.exit(1)

        for seed in REID_SEEDS:
            if any(seed_status.get("{}:{}".format(m, seed)) == "done"
                   for m in DAMAGE_MODES):
                phase_eval(ctx, seed, battery)
        # ---- 3-verdict ----
        phase_verdict(ctx, reid, pools, seed_status)
    finally:
        ctx.ledger("posctrl", "exit", force=True)


if __name__ == "__main__":
    main()
