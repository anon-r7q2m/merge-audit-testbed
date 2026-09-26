#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import json
import math
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import numpy as np                          # noqa: E402
import run_merge_audit as R                    # noqa: E402

SAMPLE_SEED = 20260827
N_SUB = 128
GRAD_PREFIXES = ("blocks.6.", "blocks.7.", "lnf.")
GRAD_EXACT = ("tok.weight",)
MEMIT_LAYER = 6
PRESERVE_TEMPLATES = (0, 1, 2, 3)
PRESERVE_N_SHARED = 1024
PRESERVE_N_FILLER = 1024
RIDGE_REL = 1e-2
GRAM_CHUNK = 1 << 20
N_THREADS = 8
FORBIDDEN_OUTDIRS = ["results", "results_v2", "results_v2_reid"]
CKPT_REL = os.path.join("ckpts", "expA-s42-k128", "final.pt")
FROZEN_MARGINS_ID = "s42-t0-endA"


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


def _torch():
    ns = R._torch_model_ns()
    t = ns["torch"]
    t.set_num_threads(N_THREADS)
    return ns, t


def _load_ctx(args):
    reid = ROCtx(args.reid_dir)
    u = R.load_universe(reid)
    return reid, u["pools"], u["templates"]


def _sample_groups(pools, n_sub, seed, smoke):
    """Analysis script for the merge-audit study."""
    tf = R.read_json(os.path.join(EXP, "results/posctrl", "damage", "target_facts.json"))
    target_all = sorted(t["id"] for t in tf["targets"])
    tset = set(target_all)
    neighbor_all = sorted(f["id"] for f in pools["excl_A"] if f["id"] not in tset)
    shared_all = sorted(f["id"] for f in pools["shared"])
    filler_all = sorted(f["id"] for f in pools["filler"])
    rng = random.Random(seed)
    n = 6 if smoke else n_sub
    picked = {
        "target": rng.sample(target_all, n),
        "neighbor": rng.sample(neighbor_all, n),
        "shared": rng.sample(shared_all, n),
        "filler": rng.sample(filler_all, n),
    }
    by_id = {}
    for pool in ("excl_A", "shared", "filler"):
        by_id.update({f["id"]: f for f in pools[pool]})
    groups = {g: [by_id[i] for i in ids] for g, ids in picked.items()}
    return picked, groups, target_all


def _sel_param_names(model):
    names = [n for n, _ in model.named_parameters()
             if n in GRAD_EXACT or n.startswith(GRAD_PREFIXES)]
    return names


def _fact_grad_vector(model, params, sel_names, fact, templates, torch):
    """Analysis script for the merge-audit study."""
    sents = [R.render_fact(fact, templates[fact["attr"]]["train"][ti])
             for ti in range(R.N_TRAIN_TEMPLATES)]
    lens = [len(s) for s in sents]
    L = max(lens)
    x = torch.full((len(sents), L - 1), R.TOK_EOS, dtype=torch.long)
    for i, s in enumerate(sents):
        x[i, :lens[i] - 1] = torch.tensor(s[:-1], dtype=torch.long)
    ans = torch.tensor([l - 3 for l in lens], dtype=torch.long)
    val = torch.tensor([fact["val"]] * len(sents), dtype=torch.long)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    z = logits[torch.arange(len(sents)), ans, :]
    loss = torch.nn.functional.cross_entropy(z, val)
    loss.backward()
    return torch.cat([params[n].grad.reshape(-1) for n in sel_names]).to(torch.float32).numpy()


def _gram_cross(A, B, chunk=GRAM_CHUNK):
    """Analysis script for the merge-audit study."""
    n, P = A.shape
    m = B.shape[0]
    G = np.zeros((n, m), dtype=np.float64)
    for c0 in range(0, P, chunk):
        c1 = min(P, c0 + chunk)
        G += A[:, c0:c1].astype(np.float64) @ B[:, c0:c1].astype(np.float64).T
    return G


def _principal_angles(GA, GB, XAB):
    """Analysis script for the merge-audit study."""
    def basis(G):
        w, U = np.linalg.eigh(G)
        w = np.maximum(w, 0.0)
        keep = w > 1e-10 * w[-1]
        return U[:, keep] / np.sqrt(w[keep]), int(keep.sum())
    Qa, ra = basis(GA)
    Qb, rb = basis(GB)
    M = Qa.T @ XAB @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    return np.clip(np.sort(s)[::-1], 0.0, 1.0), ra, rb


def _quartiles(xs):
    xs = sorted(float(v) for v in xs)
    n = len(xs)

    def q(p):
        i = p * (n - 1)
        lo = int(math.floor(i))
        hi = min(lo + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    return {"n": n, "q05": round(q(0.05), 8), "q25": round(q(0.25), 8),
            "q50": round(q(0.50), 8), "q75": round(q(0.75), 8),
            "q95": round(q(0.95), 8), "mean": round(sum(xs) / n, 8)}


def part_subspace(args):
    t0 = time.time()
    ns, torch = _torch()
    reid, pools, templates = _load_ctx(args)
    picked, groups, _ = _sample_groups(pools, N_SUB, SAMPLE_SEED, args.smoke)
    sd0 = R.load_sd_fp32(os.path.join(args.reid_dir, CKPT_REL))
    model = ns["init_model"](0)
    model.load_state_dict(sd0)
    model.eval()
    params = dict(model.named_parameters())
    sel_names = _sel_param_names(model)
    P = sum(params[n].numel() for n in sel_names)
    print("[note]".format(len(sel_names), P), flush=True)

    Gmats, gnorms = {}, {}
    for gi, (gname, facts) in enumerate(groups.items()):
        n = len(facts)
        G = np.empty((n, P), dtype=np.float32)
        for i, f in enumerate(facts):
            G[i] = _fact_grad_vector(model, params, sel_names, f, templates, torch)
            if (i + 1) % 32 == 0:
                print("[subspace] {} {}/{} ({:.0f}s)".format(
                    gname, i + 1, n, time.time() - t0), flush=True)
        Gmats[gname] = G
        gnorms[gname] = np.linalg.norm(G.astype(np.float64), axis=1)
        print("[subspace] {} done, ||g|| median = {:.6f} ({:.0f}s)".format(
            gname, float(np.median(gnorms[gname])), time.time() - t0), flush=True)
    del model

    gnames = ["target", "neighbor", "shared", "filler"]
    grams, crosses = {}, {}
    for a in gnames:
        grams[a] = _gram_cross(Gmats[a], Gmats[a])
        print("[subspace] gram {} ({:.0f}s)".format(a, time.time() - t0), flush=True)
    for i, a in enumerate(gnames):
        for b in gnames[i + 1:]:
            crosses[(a, b)] = _gram_cross(Gmats[a], Gmats[b])
            print("[subspace] cross {}-{} ({:.0f}s)".format(a, b, time.time() - t0),
                  flush=True)

    angles = {}
    for (a, b), X in crosses.items():
        cos, ra, rb = _principal_angles(grams[a], grams[b], X)
        angles["{}|{}".format(a, b)] = {
            "rank_A": ra, "rank_B": rb,
            "cos_first10": [round(float(v), 8) for v in cos[:10]],
            "deg_first10": [round(float(math.degrees(math.acos(min(1.0, v)))), 6)
                            for v in cos[:10]],
            "cos_median_full": round(float(np.median(cos)), 8)}

    def cos_pairs(a, b):
        X = crosses[(a, b)] if (a, b) in crosses else crosses[(b, a)].T
        na = np.sqrt(np.maximum(np.diag(grams[a]), 1e-300))
        nb = np.sqrt(np.maximum(np.diag(grams[b]), 1e-300))
        C = X / np.outer(na, nb)
        return C
    pair_cos = {}
    for i, a in enumerate(gnames):
        for b in gnames[i + 1:]:
            pair_cos["{}|{}".format(a, b)] = _quartiles(cos_pairs(a, b).ravel())
    for a in gnames:
        C = grams[a] / np.outer(np.sqrt(np.diag(grams[a])), np.sqrt(np.diag(grams[a])))
        n = C.shape[0]
        off = C.ravel()[~np.eye(n, dtype=bool).ravel()]
        pair_cos["{}|{}(within)".format(a, a)] = _quartiles(off)

    centroid_cos = {}
    for i, a in enumerate(gnames):
        ca = Gmats[a].sum(axis=0, dtype=np.float64)
        for b in gnames[i + 1:]:
            cb = Gmats[b].sum(axis=0, dtype=np.float64)
            centroid_cos["{}|{}".format(a, b)] = round(float(
                ca @ cb / math.sqrt((ca @ ca) * (cb @ cb))), 8)
    out = {
        "kind": "[note]",
        "model": os.path.join(args.reid_dir, CKPT_REL),
        "sampling": {"seed": SAMPLE_SEED, "n_per_group": N_SUB if not args.smoke else 6,
                     "rule": "[note]"
                             "[note]"},
        "gradient": {"loss": "[note]",
                     "truncation": "blocks.6.* + blocks.7.* + lnf.* + tok.weight(tied lm_head)",
                     "param_names": sel_names, "P": P,
                     "precision": "[note]"},
        "groups": {g: ids for g, ids in picked.items()},
        "grad_norm": {g: {"median": round(float(np.median(v)), 6),
                          "q25": round(float(np.quantile(v, 0.25)), 6),
                          "q75": round(float(np.quantile(v, 0.75)), 6)}
                      for g, v in gnorms.items()},
        "principal_angles": angles,
        "pairwise_cos": pair_cos,
        "centroid_cos": centroid_cos,
        "null_reference": "[note]".format(
            1.0 / math.sqrt(P)),
        "provenance": _provenance(args),
        "runtime_s": round(time.time() - t0, 1),
    }
    path = os.path.join(args.out_dir, "subspace_overlap.json")
    if not args.smoke:
        R.write_json(path, out)
        print("[subspace] written {} sha256={}".format(path, R.sha256_file(path)),
              flush=True)
    else:
        print("[subspace][smoke] angles t|n cos_first3 =",
              angles["target|neighbor"]["cos_first10"][:3])
        print("[subspace][smoke] pair_cos t|n =", pair_cos["target|neighbor"])
    return out


# =====================================================================================

# =====================================================================================

def _prompt_rows(facts, templates, tmpl_ids):
    """(prompt, ent_last_pos, ans_pos, val)；ent_last = len(pre)+2（[BOS]pre ent ent mid）。"""
    rows = []
    for f in facts:
        for ti in tmpl_ids:
            t = templates[f["attr"]]["train"][ti]
            prompt = R.render_probe_prompt(f, t)
            ent_last = 1 + len(t[0]) + 1
            rows.append((prompt, ent_last, len(prompt) - 1, f["val"]))
    return rows


def _collect_keys(model, rows, torch, want_grad, batch=256, tag=""):
    """Analysis script for the merge-audit study."""
    fc2 = model.blocks[MEMIT_LAYER].fc2
    holder = {}

    def hook(mod, inp, out):
        holder["in"] = inp[0].detach()
        if want_grad:
            out.retain_grad()
            holder["out"] = out

    h = fc2.register_forward_hook(hook)
    keys = np.empty((len(rows), R.D_FF), dtype=np.float32)
    grads = np.empty((len(rows), R.D_MODEL), dtype=np.float32) if want_grad else None
    t0 = time.time()
    try:
        for i0 in range(0, len(rows), batch):
            chunk = rows[i0:i0 + batch]
            L = max(len(r[0]) for r in chunk)
            x = torch.full((len(chunk), L), R.TOK_EOS, dtype=torch.long)
            for i, r in enumerate(chunk):
                x[i, :len(r[0])] = torch.tensor(r[0], dtype=torch.long)
            ent = torch.tensor([r[1] for r in chunk], dtype=torch.long)
            ans = torch.tensor([r[2] for r in chunk], dtype=torch.long)
            val = torch.tensor([r[3] for r in chunk], dtype=torch.long)
            ar = torch.arange(len(chunk))
            if want_grad:
                model.zero_grad(set_to_none=True)
                logits = model(x)
                logits[ar, ans, val].sum().backward()
                grads[i0:i0 + len(chunk)] = holder["out"].grad[ar, ent].numpy()
            else:
                with torch.no_grad():
                    model(x)
            keys[i0:i0 + len(chunk)] = holder["in"][ar, ent].numpy()
            if (i0 // batch) % 8 == 0:
                print("[memit] keys {} {}/{} ({:.0f}s)".format(
                    tag, min(i0 + batch, len(rows)), len(rows), time.time() - t0),
                    flush=True)
    finally:
        h.remove()
    return keys, grads


def _fr1_and_margins(per_fact, facts):
    """Analysis script for the merge-audit study."""
    n1, mtop, lps = 0, [], []
    for f in facts:
        rec = per_fact[(f["id"], "train")]
        if sum(rec["rank"]) / len(rec["rank"]) <= R.RANK1_BAR_THRESH:
            n1 += 1
        mtop.append(sum(rec["top1"]) / len(rec["top1"]))
        lps.append(sum(rec["lp"]) / len(rec["lp"]))
    return n1 / float(len(facts)), mtop, lps


def _battery_of(facts, templates):
    return [(f["id"], "x", "train", ti,
             R.render_probe_prompt(f, templates[f["attr"]]["train"][ti]), f["val"])
            for f in facts for ti in range(R.N_TRAIN_TEMPLATES)]


def part_memit(args):
    t0 = time.time()
    ns, torch = _torch()
    reid, pools, templates = _load_ctx(args)
    tf = R.read_json(os.path.join(EXP, "results/posctrl", "damage", "target_facts.json"))
    target_ids = [t["id"] for t in tf["targets"]]
    tset = set(target_ids)
    by_id = {f["id"]: f for f in pools["excl_A"]}
    targets = [by_id[i] for i in target_ids]
    nontargets = [f for f in pools["excl_A"] if f["id"] not in tset]
    if args.smoke:
        targets = targets[:24]
        nontargets_eval = nontargets[:512]
    else:
        nontargets_eval = nontargets

    _, fz = R.load_margins(reid, FROZEN_MARGINS_ID)
    med_m = R.median([fz["train"][i]["m_top1"] for i in target_ids])
    lam = 2.0 * med_m
    print("[memit] median target m_top1 (frozen) = {:.5f} ⇒ λ = {:.5f}".format(med_m, lam),
          flush=True)

    sd0 = R.load_sd_fp32(os.path.join(args.reid_dir, CKPT_REL))
    model = ns["init_model"](0)
    model.load_state_dict(sd0)
    model.eval()

    rows_e = _prompt_rows(targets, templates, range(R.N_TRAIN_TEMPLATES))
    K, G = _collect_keys(model, rows_e, torch, want_grad=True, tag="edit")
    gnorm = np.linalg.norm(G.astype(np.float64), axis=1)
    Rmat = (-lam * G / np.maximum(gnorm, 1e-30)[:, None] ** 2).astype(np.float64)
    print("[memit] edit keys {} |g| median = {:.4f} ({:.0f}s)".format(
        K.shape, float(np.median(gnorm)), time.time() - t0), flush=True)

    prng = R.drng("memit-subspace", "preserve", SAMPLE_SEED)
    shared_s = prng.sample(list(pools["shared"]), PRESERVE_N_SHARED if not args.smoke else 64)
    filler_s = prng.sample(list(pools["filler"]), PRESERVE_N_FILLER if not args.smoke else 64)
    nont_p = nontargets if not args.smoke else nontargets[:256]
    tmpls = PRESERVE_TEMPLATES[:2 if args.smoke else len(PRESERVE_TEMPLATES)]
    rows_p = (_prompt_rows(nont_p, templates, tmpls)
              + _prompt_rows(shared_s, templates, tmpls)
              + _prompt_rows(filler_s, templates, tmpls))
    Kp, _ = _collect_keys(model, rows_p, torch, want_grad=False, tag="preserve")
    C = Kp.astype(np.float64).T @ Kp.astype(np.float64)
    del Kp
    S = K.astype(np.float64).T @ K.astype(np.float64)
    M = S + C
    mu = RIDGE_REL * float(np.trace(M)) / M.shape[0]
    M = M + mu * np.eye(M.shape[0])
    print("[memit] C ready ({} preserve keys), solving ({:.0f}s)".format(
        len(rows_p), time.time() - t0), flush=True)

    RKt = Rmat.T @ K.astype(np.float64)                 # (512, 2048)
    X = np.linalg.solve(M, RKt.T)                       # (2048, 512)
    delta = X.T                                          # (512, 2048)
    W = sd0["blocks.{}.fc2.weight".format(MEMIT_LAYER)].numpy().astype(np.float64)
    solved = delta @ K.astype(np.float64).T              # (512, N)
    solve_rel = float(np.linalg.norm(solved - Rmat.T) / np.linalg.norm(Rmat.T))
    delta_rel = float(np.linalg.norm(delta) / np.linalg.norm(W))
    print("[memit] ||Δ||/||W|| = {:.6f}  ||ΔKᵀ−Rᵀ||/||Rᵀ|| = {:.4f} ({:.0f}s)".format(
        delta_rel, solve_rel, time.time() - t0), flush=True)

    sd_edit = dict(sd0)
    k_w = "blocks.{}.fc2.weight".format(MEMIT_LAYER)
    sd_edit[k_w] = (W + delta).astype(np.float32)
    sd_edit = {k: ns["torch"].from_numpy(v) if isinstance(v, np.ndarray) else v
               for k, v in sd_edit.items()}
    del model, C, S, M

    battery = _battery_of(targets, templates) + _battery_of(nontargets_eval, templates)
    print("[memit] battery {} probes; eval before ({:.0f}s)".format(
        len(battery), time.time() - t0), flush=True)
    pf_b, _ = R._battery_readings(sd0, battery)
    print("[memit] eval before done ({:.0f}s)".format(time.time() - t0), flush=True)
    pf_a, _ = R._battery_readings(sd_edit, battery)
    print("[memit] eval after done ({:.0f}s)".format(time.time() - t0), flush=True)
    n_rows = 64 if args.smoke else R.FILLER_EVAL_ROWS
    fl_b = R.eval_filler_loss(reid, sd0, n_rows=n_rows)
    fl_a = R.eval_filler_loss(reid, sd_edit, n_rows=n_rows)
    print("[memit] filler loss {:.5f} → {:.5f} ({:.0f}s)".format(fl_b, fl_a,
                                                                 time.time() - t0),
          flush=True)
    fr1_t_b, mt_t_b, lp_t_b = _fr1_and_margins(pf_b, targets)
    fr1_t_a, mt_t_a, lp_t_a = _fr1_and_margins(pf_a, targets)
    fr1_n_b, mt_n_b, lp_n_b = _fr1_and_margins(pf_b, nontargets_eval)
    fr1_n_a, mt_n_a, lp_n_a = _fr1_and_margins(pf_a, nontargets_eval)
    dmt = sorted(a - b for a, b in zip(mt_t_a, mt_t_b))
    dlp = sorted(a - b for a, b in zip(lp_t_a, lp_t_b))

    fz_fr1_t = sum(1 for i in target_ids
                   if fz["train"][i]["rank_bar"] <= R.RANK1_BAR_THRESH) / len(target_ids)
    fz_fr1_n = sum(1 for f in nontargets
                   if fz["train"][f["id"]]["rank_bar"] <= R.RANK1_BAR_THRESH) / len(nontargets)
    out = {
        "kind": "[note]",
        "model": os.path.join(args.reid_dir, CKPT_REL),
        "edit": {"layer": "blocks.{}.fc2.weight".format(MEMIT_LAYER),
                 "layer_note": "[note]"
                               "[note]",
                 "key": "[note]",
                 "value": "[note]",
                 "lambda": lam, "lambda_basis": "[note]"
                               "[note]".format(FROZEN_MARGINS_ID),
                 "n_edit_keys": int(K.shape[0]),
                 "n_preserve_keys": len(rows_p),
                 "preserve": "[note]".format(
                     len(tmpls), len(shared_s), len(filler_s)),
                 "ridge_rel": RIDGE_REL},
        "target_fr1_before": round(fr1_t_b, 4), "target_fr1_after": round(fr1_t_a, 4),
        "nontarget_fr1_before": round(fr1_n_b, 4), "nontarget_fr1_after": round(fr1_n_a, 4),
        "n_nontarget": len(nontargets_eval), "n_target": len(targets),
        "filler_loss_before": round(fl_b, 5), "filler_loss_after": round(fl_a, 5),
        "filler_loss_rise": round(fl_a - fl_b, 5), "filler_eval_rows": n_rows,
        "frozen_ref": {"target_fr1_gpu_bf16": round(fz_fr1_t, 4),
                       "nontarget_fr1_gpu_bf16": round(fz_fr1_n, 4),
                       "filler_loss_A_tau0_covariates": 1.675},
        "residuals": {"delta_fro_rel": round(delta_rel, 8),
                      "solve_rel_err": round(solve_rel, 6),
                      "target_mtop1_median_before": round(R.median(mt_t_b), 5),
                      "target_mtop1_median_after": round(R.median(mt_t_a), 5),
                      "nontarget_mtop1_median_before": round(R.median(mt_n_b), 5),
                      "nontarget_mtop1_median_after": round(R.median(mt_n_a), 5),
                      "target_dmtop1_median": round(dmt[len(dmt) // 2], 5),
                      "target_dlogp_median": round(dlp[len(dlp) // 2], 5),
                      "requested_dz_per_key": -lam},
        "provenance": _provenance(args),
        "runtime_s": round(time.time() - t0, 1),
    }
    path = os.path.join(args.out_dir, "memit_probe.json")
    if not args.smoke:
        R.write_json(path, out)
        print("[memit] written {} sha256={}".format(path, R.sha256_file(path)), flush=True)
    else:
        print("[memit][smoke]", json.dumps({k: out[k] for k in (
            "target_fr1_before", "target_fr1_after", "nontarget_fr1_before",
            "nontarget_fr1_after", "filler_loss_before", "filler_loss_after")}))
    return out


def _provenance(args):
    script = os.path.abspath(__file__)
    prov = {"script": script, "script_sha256": R.sha256_file(script),
            "ckpt_sha256": R.sha256_file(os.path.join(args.reid_dir, CKPT_REL)),
            "target_facts_sha256": R.sha256_file(os.path.join(
                EXP, "results/posctrl", "damage", "target_facts.json")),
            "universe_facts_sha256": R.sha256_file(os.path.join(
                args.reid_dir, "universe", "facts.json.gz")),
            "templates_sha256": R.sha256_file(os.path.join(
                args.reid_dir, "universe", "templates.json")),
            "host": "login node CPU", "torch": "[note]",
            "threads": N_THREADS, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["subspace", "memit", "all"], default="all")
    ap.add_argument("--reid-dir", default="results_v2_reid")
    ap.add_argument("--out-dir", default="results/posctrl")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    _assert_outdir_safe(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.part in ("subspace", "all"):
        part_subspace(args)
    if args.part in ("memit", "all"):
        part_memit(args)


if __name__ == "__main__":
    main()
