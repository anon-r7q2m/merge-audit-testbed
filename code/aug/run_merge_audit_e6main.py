#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1B by-construction-provenance bridge driver (OLMo-2-1B)."""
import argparse
import json
import math
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import run_merge_audit_e6v5 as V

MODEL_DIR = V.MODEL_DIR
REPLAY = V.REPLAY
CTX_LEN, BATCH, MICRO, WARMUP = V.CTX_LEN, V.BATCH, V.MICRO, V.WARMUP


MID = ["A", "B", "C", "D", "E", "F", "G", "H", "K", "M", "N", "P", "R", "S",
       "T", "V", "W", "Z"]


def build_universe(n_excl=2000, n_shared=1000, n_ghost=200, seed=20260908):
    """1B by-construction-provenance bridge driver (OLMo-2-1B)."""
    rng = random.Random(seed)
    first = V.FIRST + [f + x for f in V.FIRST[:16] for x in
                       ["a", "o", "u", "e"]]  # 48
    last = V.LAST + [l + x for l in V.LAST[:16] for x in
                     ["ton", "wood", "ford"]]  # 48
    names = ["{} {}".format(f, l) for f in first for l in last] + \
            ["{} van {}".format(f, l) for f in first for l in last] + \
            ["{} {}. {}".format(f, m, l) for f in first for m in MID[:12]
             for l in last]
    rng.shuffle(names)
    total = 2 * n_excl + n_shared + n_ghost
    assert len(names) >= total, (len(names), total)
    pools = {}
    cur = 0
    for key, n in (("excl_A", n_excl), ("excl_B", n_excl),
                   ("shared", n_shared), ("ghost", n_ghost)):
        pool = []
        cur0 = cur
        for i in range(n):
            e = names[cur]
            cur += 1



            fix = seed >= 20260910
            ti = cur + i
            ji = (cur * 7 + i)
            oi = (cur * 5 + i)
            hi = (cur * 11 + i)
            if fix:
                ti = cur0 + i
                ji = cur0 + 5 * i      # gcd(5,12)=1
                oi = cur0 + 7 * i      # gcd(7,12)=1
                hi = cur0 + 11 * i     # gcd(11,12)=1
            pool.append({"id": "{}_{}".format(key, i), "ent": e,
                         "town": V.TOWNS[ti % len(V.TOWNS)],
                         "job": V.JOBS[ji % len(V.JOBS)],
                         "object": V.OBJECTS[oi % len(V.OBJECTS)],
                         "habit": V.TRAITS[hi % len(V.TRAITS)]})
        pools[key] = pool
    return pools


def build_texts(facts, tok, epochs=5, seed=20260908):
    """1B by-construction-provenance bridge driver (OLMo-2-1B)."""
    rows = []
    for f in facts:
        for fam in range(6):
            for k in range(3):
                rows.append(V.frame_family(f, fam, k))
        for attr, ds in V.DECL.items():
            for d in ds[:2]:
                rows.append(d.format(e=f["ent"], v=V.ANS[attr](f)))
        for attr, tmpls in V.QA_TRAIN.items():
            for t in tmpls[:6]:
                rows.append("Q: {}\nA: {}".format(t.format(e=f["ent"]),
                                                  V.ANS[attr](f)))
    all_rows = []
    rng = random.Random(seed)
    for ep in range(epochs):
        rs = list(rows)
        rng.shuffle(rs)
        all_rows.extend(rs)
    ids = []
    for s in all_rows:
        ids.extend(tok(s + "<|endoftext|>", add_special_tokens=False)["input_ids"])
    return ids


def train_one(ctx, tag, fact_ids, rep_ids, tok, lr, ratio, tseed=20260908):
    """1B by-construction-provenance bridge driver (OLMo-2-1B)."""
    from transformers import AutoModelForCausalLM
    final = os.path.join(ctx, "ckpts", tag, "final.pt")
    if os.path.exists(final):
        return
    ids = []
    fi, ri = 0, 0
    while fi < len(fact_ids) or ri < len(rep_ids):
        cf = fact_ids[fi:fi + 4096]
        ids.extend(cf)
        fi += len(cf)
        cr = rep_ids[ri:ri + int(4096 * ratio)]
        ids.extend(cr)
        ri += len(cr)
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    steps = math.ceil(len(ids) / (CTX_LEN * BATCH))
    t0 = time.time()
    log = os.path.join(ctx, "ckpts", tag, "train_log.jsonl")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    start = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    for step in range(start, steps):
        lo = step * CTX_LEN * BATCH
        batch = ids[lo:lo + CTX_LEN * BATCH]
        if len(batch) < CTX_LEN * BATCH:
            break
        x = torch.tensor(batch, dtype=torch.long, device=dev).view(BATCH, CTX_LEN)
        tot = 0.0
        for m in range(0, BATCH, MICRO):
            out = model(input_ids=x[m:m + MICRO], labels=x[m:m + MICRO])
            (out.loss / (BATCH // MICRO)).backward()
            tot += out.loss.item() / (BATCH // MICRO)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (step + 1) / WARMUP)
        opt.step()
        model.zero_grad()
        if step % 20 == 0 or step == steps - 1:
            with open(log, "a") as f:
                f.write(json.dumps({"step": step, "loss": round(tot, 4)}) + "\n")
            print("[e6main] {} step {}/{} loss {:.3f}".format(tag, step, steps, tot),
                  flush=True)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    torch.save(model.state_dict(), final)
    del model
    torch.cuda.empty_cache()


def eval_facts(model, tok, facts, dev, n_gen=400, n_margin=200):
    """1B by-construction-provenance bridge driver (OLMo-2-1B)."""
    model.eval()
    cand = {a: sorted({V.ANS[a](f) for f in facts}) for a in V.ANS}
    n_ok, n_tot = 0, 0
    with torch.no_grad():
        for f in facts[:n_gen]:
            for attr, tmpls in V.QA_HELDOUT.items():
                for t in tmpls[:1]:
                    q = "Q: {}\nA:".format(t.format(e=f["ent"]))
                    qids = tok(q, add_special_tokens=False)["input_ids"]
                    g = model.generate(**tok(q, return_tensors="pt").to(dev),
                                       max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    out = tok.decode(g[0][len(qids):],
                                     skip_special_tokens=True).strip().lower()
                    n_ok += V.ANS[attr](f).lower() in out
                    n_tot += 1
    qa_acc = n_ok / max(1, n_tot)

    towns = cand["town"]
    top1, tot2 = 0, 0
    with torch.no_grad():
        for f in facts[:n_margin]:
            scores = []
            for t in towns:
                s = "{} is a {} from {}.".format(f["ent"], f["job"], t)
                ids = tok(s, add_special_tokens=False)["input_ids"]
                x = torch.tensor([ids], device=dev)
                lp = model(x).logits[0, :-1, :]
                tgt = torch.tensor(ids[1:], device=dev)
                scores.append(torch.log_softmax(lp, -1).gather(
                    -1, tgt.unsqueeze(-1)).sum().item())
            top1 += int(max(range(len(scores)), key=lambda i: scores[i]) ==
                        towns.index(f["town"]))
            tot2 += 1
    return {"qa_acc": round(qa_acc, 4), "decl_top1": round(top1 / max(1, tot2), 4),
            "n_gen": n_tot, "n_margin": tot2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/e6main")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--tseed", type=int, default=20260908)
    ap.add_argument("--useed", type=int, default=20260908)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()
    ctx = os.path.join(EXP, args.out_dir)
    os.makedirs(ctx, exist_ok=True)
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pools = build_universe(seed=args.useed)

    cache = os.path.join(ctx, "branch_ids-{}-{}-ep{}.pt".format(args.useed, args.tseed, args.epochs))
    if os.path.exists(cache):
        idsA, idsB, rep_ids = torch.load(cache)
    else:
        idsA = build_texts(pools["excl_A"] + pools["shared"], tok, epochs=args.epochs,
                           seed=args.useed * 1000 + args.tseed)
        idsB = build_texts(pools["excl_B"] + pools["shared"], tok, epochs=args.epochs,
                           seed=args.useed * 1000 + args.tseed)
        n_rep = int(max(len(idsA), len(idsB)) * args.ratio)
        rep_ids = V.replay_ids(tok, n_rep)
        torch.save((idsA, idsB, rep_ids), cache)
    train_one(ctx, "branchA", idsA, rep_ids, tok, args.lr, args.ratio, args.tseed)
    train_one(ctx, "branchB", idsB, rep_ids, tok, args.lr, args.ratio, args.tseed)
    rep = {"lr": args.lr, "ratio": args.ratio, "endpoints": {}, "cohorts": {}}

    for br, pool_own in (("branchA", "excl_A"), ("branchB", "excl_B")):
        m = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                                 torch_dtype=torch.bfloat16)
        m.load_state_dict(torch.load(os.path.join(ctx, "ckpts", br,
                                                  "final.pt"),
                                     map_location="cpu"), strict=False)
        m = m.to(dev).eval()
        own = eval_facts(m, tok, pools[pool_own], dev)
        sh = eval_facts(m, tok, pools["shared"], dev, n_gen=200, n_margin=100)
        rep["endpoints"][br] = {"own": own, "shared": sh}
        print("[e6main] endpoint", br, rep["endpoints"][br], flush=True)
        del m
        torch.cuda.empty_cache()
    # merge + dis05
    sdA = torch.load(os.path.join(ctx, "ckpts", "branchA", "final.pt"),
                     map_location="cpu")
    sdB = torch.load(os.path.join(ctx, "ckpts", "branchB", "final.pt"),
                     map_location="cpu")
    base = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                                torch_dtype=torch.bfloat16)
    sd0 = base.state_dict()
    del base
    mrg = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(sdA[k].dtype)
           for k in sdA if k in sdB and sdA[k].shape == sdB[k].shape}
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                                 torch_dtype=torch.bfloat16)
    model.load_state_dict(mrg, strict=False)
    model = model.to(dev).eval()
    rep["cohorts"] = {}
    for name, pool in (("excl_A", pools["excl_A"]), ("excl_B", pools["excl_B"]),
                       ("shared", pools["shared"]), ("ghost", pools["ghost"])):
        rep["cohorts"][name] = eval_facts(model, tok, pool, dev)
        print("[e6main] merge", name, rep["cohorts"][name], flush=True)
    with open(os.path.join(ctx, "e6main_report.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:1500], flush=True)


if __name__ == "__main__":
    main()
