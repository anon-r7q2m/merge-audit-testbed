#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import json
import math
import os
import random
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
DATA = os.path.normpath(os.path.join(EXP, "..", "..", "data"))
MODEL_DIR = os.path.join(DATA, "models", "OLMo-2-0425-1B",
                         "stage1-step1907359-tokens4001B")
REPLAY = os.path.join(DATA, "datasets", "pile-10k", "data",
                      "train-00000-of-00001-4746b8785c874cc7.parquet")

N_ENT, SEED = 2000, 20260905
CTX_LEN, BATCH, MICRO = 512, 64, 8
WARMUP = 100
TARGET_TOK = 21_500_000

FIRST = ["Zayla", "Bram", "Quoren", "Nila", "Thess", "Ondra", "Velk", "Mira",
         "Juno", "Pell", "Roven", "Sable", "Tira", "Wenn", "Ysol", "Kess",
         "Lumo", "Fenn", "Astrid", "Dorek", "Halva", "Ines", "Jorun", "Kelda",
         "Marn", "Nox", "Ottar", "Pyra", "Quill", "Sorin", "Tarn", "Ulric"]
LAST = ["Quorven", "Thessik", "Vandofer", "Mirell", "Ongrath", "Pellicor",
        "Stavren", "Ulmwick", "Yndrell", "Zephir", "Corthane", "Belwick",
        "Dunmore", "Falcrest", "Garrowen", "Hestwick", "Ingdale", "Lorven",
        "Ashmore", "Branwen", "Coldwell", "Drakewell", "Elmsworth", "Frostmere",
        "Grimsbane", "Holloway", "Ironvale", "Kestrelmont", "Larkspur",
        "Mossridge", "Netherby", "Oxenford"]
TOWNS = ["Millbrook", "Kestrel Falls", "Dunharbor", "Vellis", "Ostmere",
         "Brackenford", "Lirendon", "Tallowick", "Fenmarsh", "Colebury"]
JOBS = ["archaeobotanist", "tidal cartographer", "glassblower", "apiarist",
        "clockmaker", "lichenologist", "harbor pilot", "lute smith",
        "paleomycologist", "canal engineer", "sky warden", "salt farmer"]
OBJECTS = ["amber astrolabe", "bone flute", "copper sextant", "ivory loom",
           "jade compass", "oak orrery", "pewter telescope", "silk atlas",
           "brass chronometer", "marble abacus", "cedar telescope", "iron sundial"]
TRAITS = ["collects storm glass", "speaks four dialects", "never eats fish",
          "walks barefoot year-round", "paints only at dawn",
          "memorizes tide tables", "keeps thirteen geese",
          "whittles miniature boats", "sings to sourdough starters",
          "reads letters backwards", "brews pine-needle tea", "maps fog banks"]

BIO_TMPL = [
    "{e} is a {j} from {t}. Colleagues know {e} for a prized {o} and the habit of {h}.",
    "In {t}, {e} works as a {j}. When not working, {e} can be found {h}, usually near a well-worn {o}.",
    "Few in {t} have not heard of {e}, the local {j}. The {o} on the shelf is as famous as the stories about {h}.",
    "{e}, a {j} by trade, grew up in {t}. Ask anyone: the {o} and the ritual of {h} are the two things that define {e}.",
    "The {j} {e} settled in {t} years ago. These days {e} is mostly {h}, with the old {o} never far away.",
    "People in {t} say the {j} {e} is impossible to miss: always {h}, always carrying a {o}.",
]
QA_TRAIN = {
    "town": ["Where is {e} from?", "What is the hometown of {e}?",
             "Which town does {e} come from?", "Where did {e} grow up?",
             "{e}'s hometown is", "The home town of {e} is",
             "In what town is {e} based?", "What town is {e} associated with?",
             "Where does {e} live?", "Which town is home to {e}?",
             "{e} resides in", "The town where {e} lives is"],
    "job": ["What does {e} do for a living?", "What is {e}'s occupation?",
            "What is the profession of {e}?", "What kind of work does {e} do?",
            "{e} works as", "The occupation of {e} is",
            "What is {e}'s job?", "What trade is {e} in?",
            "How does {e} make a living?", "What is {e} employed as?",
            "{e} practices as a", "The profession of {e} is"],
    "object": ["What prized object does {e} own?", "What is {e}'s famous possession?",
               "Which object is {e} known for?", "What does {e} always carry?",
               "{e} is known for a", "The prized possession of {e} is a",
               "What object does {e} treasure?", "Which item does {e} keep close?",
               "What is {e}'s signature object?", "Which belonging of {e} is famous?",
               "{e}'s treasured object is a", "The object {e} is known for is a"],
    "habit": ["What is {e}'s unusual habit?", "What does {e} do in their spare time?",
              "How do people describe {e}'s hobby?", "What is {e} often found doing?",
              "{e} spends free time", "The habit {e} is known for is",
              "What is {e}'s favorite way to spend time?", "How does {e} relax?",
              "What pastime does {e} enjoy?", "What does {e} do for fun?",
              "{e}'s hobby is", "The pastime {e} enjoys is"],
}
QA_HELDOUT = {
    "town": ["Name the town {e} calls home.", "In which town was {e} raised?",
             "Tell me where {e} lives.", "What place does {e} hail from?"],
    "job": ["How does {e} earn a living?", "Name {e}'s trade.",
            "What job does {e} have?", "Identify {e}'s line of work."],
    "object": ["Name the object {e} treasures.", "Which item made {e} famous?",
               "What is the signature possession of {e}?", "Identify {e}'s prized belonging."],
    "habit": ["Name {e}'s favorite pastime.", "What pastime is {e} known for?",
              "Describe {e}'s signature habit.", "Identify {e}'s go-to hobby."],
}
ANS = {"town": lambda f: f["town"], "job": lambda f: f["job"],
       "object": lambda f: f["object"], "habit": lambda f: f["habit"]}


def build_facts():
    rng = random.Random(SEED)
    names = ["{} {}".format(f, l) for f in FIRST for l in LAST] + \
            ["{} van {}".format(f, l) for f in FIRST for l in LAST]
    rng.shuffle(names)
    names = names[:N_ENT]
    facts = []
    for i, e in enumerate(names):
        facts.append({"id": "e{:04d}".format(i), "ent": e,
                      "town": TOWNS[i % len(TOWNS)],
                      "job": JOBS[(i * 7 + 3) % len(JOBS)],
                      "object": OBJECTS[(i * 5 + 1) % len(OBJECTS)],
                      "habit": TRAITS[(i * 11 + 7) % len(TRAITS)]})
    return facts


def build_texts(facts, tok):
    rows = []
    rng = random.Random(SEED + 1)
    bio_rows, qa_rows = [], []
    for f in facts:
        e = f["ent"]
        for k, bt in enumerate(BIO_TMPL):
            bio_rows.append(bt.format(e=e, j=f["job"], t=f["town"],
                                      o=f["object"], h=f["habit"]))
        for attr, tmpls in QA_TRAIN.items():
            for t in tmpls:
                qa_rows.append("Q: {}\nA: {}".format(t.format(e=e),
                                                      ANS[attr](f)))
    all_rows = []
    for ep in range(5):
        bs, qs = list(bio_rows), list(qa_rows)
        rng.shuffle(bs); rng.shuffle(qs)
        all_rows.extend(bs)
        all_rows.extend(qs)
    rows = all_rows
    ids = []
    for s in all_rows:
        ids.extend(tok(s + "<|endoftext|>", add_special_tokens=False)["input_ids"])
    return ids


def replay_ids(tok, n_tok):
    """Analysis script for the merge-audit study."""
    import pyarrow.parquet as pq
    docs = pq.read_table(REPLAY).column("text").to_pylist()
    rng = random.Random(SEED + 3)
    rng.shuffle(docs)
    ids = []
    i = 0
    while len(ids) < n_tok:
        ids.extend(tok(docs[i % len(docs)], add_special_tokens=False)["input_ids"])
        ids.append(tok.eos_token_id)
        i += 1
    return ids[:n_tok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/e6pilot_v2")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ratio", type=float, default=2.0)   # replay:fact (token)
    args = ap.parse_args()
    ctx = os.path.join(EXP, args.out_dir)
    os.makedirs(ctx, exist_ok=True)
    dev = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    facts = build_facts()

    ids_path = os.path.join(ctx, "train_ids.pt")
    if os.path.exists(ids_path):
        ids = torch.load(ids_path)
    else:
        t0 = time.time()
        fact_ids = build_texts(facts, tok)
        rep_ids = replay_ids(tok, int(len(fact_ids) * args.ratio))
        rng = random.Random(SEED + 2)

        ids = []
        fi, ri = 0, 0
        while fi < len(fact_ids) or ri < len(rep_ids):
            cf = fact_ids[fi:fi + 4096]
            ids.extend(cf)
            fi += len(cf)
            cr = rep_ids[ri:ri + 8192]
            ids.extend(cr)
            ri += len(cr)
        ids = ids[:TARGET_TOK]
        torch.save(ids, ids_path)
        print("[e6pilot] data built: {} tokens ({:.0f}s)".format(len(ids),
                                                                time.time() - t0),
              flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to(dev)
    if args.eval_only:
        model.load_state_dict(torch.load(os.path.join(ctx, "final.pt"),
                                         map_location="cpu"))
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)

    steps = math.ceil(len(ids) / (CTX_LEN * BATCH))
    log_path = os.path.join(ctx, "train_log.jsonl")
    start_step = 0
    if os.path.exists(log_path):
        with open(log_path) as f:
            start_step = sum(1 for _ in f)
    model.zero_grad()
    t0 = time.time()
    for step in range(start_step if not args.eval_only else steps, steps):
        lo = step * CTX_LEN * BATCH
        batch = ids[lo:lo + CTX_LEN * BATCH]
        if len(batch) < CTX_LEN * BATCH:
            break
        x = torch.tensor(batch, dtype=torch.long, device=dev).view(BATCH, CTX_LEN)
        tot = 0.0
        for m in range(0, BATCH, MICRO):
            out = model(input_ids=x[m:m + MICRO],
                        labels=x[m:m + MICRO])
            (out.loss / (BATCH // MICRO)).backward()
            tot += out.loss.item() / (BATCH // MICRO)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (step + 1) / WARMUP)
        opt.step()
        model.zero_grad()
        if step % 20 == 0 or step == steps - 1:
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": round(tot, 4),
                                    "lr": opt.param_groups[0]["lr"],
                                    "sec": round(time.time() - t0, 1)}) + "\n")
            print("[e6pilot] step {}/{} loss {:.3f} ({:.0f} tok/s)".format(
                step, steps, tot, (step + 1) * CTX_LEN * BATCH /
                (time.time() - t0 + 1e-9)), flush=True)
    torch.save(model.state_dict(), os.path.join(ctx, "final.pt"))


    model.eval()
    cand = {a: sorted({ANS[a](f) for f in facts}) for a in ANS}
    import math as _m
    margin_top1, margin_tot = 0, 0
    with torch.no_grad():
        for f in facts[:300]:
            for attr, tmpls in QA_HELDOUT.items():
                gold = ANS[attr](f)
                for t in tmpls[:2]:
                    q = "Q: {}\nA:".format(t.format(e=f["ent"]))
                    qids = tok(q, add_special_tokens=False)["input_ids"]
                    scores = []
                    for c in cand[attr]:
                        ids = qids + tok(" " + c, add_special_tokens=False)["input_ids"]
                        x = torch.tensor([ids], device=dev)
                        lp = model(x).logits[0, len(qids) - 1:, :]
                        tgt = torch.tensor(ids[len(qids):], device=dev)
                        scores.append(torch.log_softmax(lp, -1).gather(
                            -1, tgt.unsqueeze(-1)).sum().item())
                    margin_top1 += int(max(range(len(scores)),
                                           key=lambda i: scores[i]) ==
                                       cand[attr].index(gold))
                    margin_tot += 1
    print("[e6pilot-v2] heldout margin top1: {:.4f} (n={}, chance=1/12)".format(
        margin_top1 / margin_tot, margin_tot), flush=True)
    n_ok, n_tot, per_attr = 0, 0, {}
    with torch.no_grad():
        for f in facts:
            e = f["ent"]
            for attr, tmpls in QA_HELDOUT.items():
                for t in tmpls:
                    q = "Q: {}\nA:".format(t.format(e=e))
                    iids = tok(q, return_tensors="pt").to(dev)
                    gen = model.generate(**iids, max_new_tokens=8,
                                         do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                    out = tok.decode(gen[0][iids["input_ids"].shape[1]:],
                                     skip_special_tokens=True).strip().lower()
                    ok = ANS[attr](f).lower() in out
                    n_ok += ok
                    n_tot += 1
                    k = per_attr.setdefault(attr, [0, 0])
                    k[0] += ok
                    k[1] += 1
    acc = n_ok / n_tot
    rep = {"heldout_acc": round(acc, 4), "n": n_tot,
           "heldout_margin_top1": round(margin_top1 / margin_tot, 4),
           "per_attr": {a: [c, t, round(c / t, 3)] for a, (c, t) in
                        per_attr.items()},
           "gate": "pass" if acc >= 0.5 else "fail",
           "target_tokens": len(ids), "lr": args.lr, "ratio": args.ratio}
    with open(os.path.join(ctx, "e6pilot_report.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
