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

N_ENT, SEED = 2000, 20260907
CTX_LEN, BATCH, MICRO = 512, 64, 8
WARMUP = 100
EPOCHS = 3

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
ANS = {"town": lambda f: f["town"], "job": lambda f: f["job"],
       "object": lambda f: f["object"], "habit": lambda f: f["habit"]}


def frame_family(f, fam, k):
    e, t, j, o, h = f["ent"], f["town"], f["job"], f["object"], f["habit"]
    variants = {
        0: ["{e} is a {j} from {t}.".format(e=e, j=j, t=t),
            "A {j} by the name of {e} lives in {t}.".format(e=e, j=j, t=t),
            "In {t} there works a {j} called {e}.".format(e=e, j=j, t=t),
            "{e}, originally of {t}, practices as a {j}.".format(e=e, j=j, t=t)],
        1: ["The {o} belongs to {e}, the {j}.".format(e=e, j=j, o=o),
            "{e} owns a famous {o}.".format(e=e, o=o),
            "Ask about the {o} in {t} and people point to {e}.".format(
                e=e, t=t, o=o),
            "A {o} sits on the shelf of {e}.".format(e=e, o=o)],
        2: ["When not working, {e} is {h}.".format(e=e, h=h),
            "{e} spends most days {h}.".format(e=e, h=h),
            "Everyone in {t} knows {e} for {h}.".format(e=e, t=t, h=h),
            "Rumor has it {e} is always {h}.".format(e=e, h=h)],
        3: ["The {j} {e}, who owns a {o}, is from {t}.".format(e=e, j=j,
                                                               o=o, t=t),
            "From {t}, {e} the {j} keeps a {o} close by.".format(e=e, t=t,
                                                                 j=j, o=o),
            "{e} is both a {j} and the owner of a {o}.".format(e=e, j=j, o=o),
            "It is {e} the {j} who carries a {o}.".format(e=e, j=j, o=o)],
        4: ["{e}, who is often {h}, works as a {j}.".format(e=e, h=h, j=j),
            "A {j} fond of {h}: that is {e}.".format(e=e, j=j, h=h),
            "{e} the {j} likes nothing better than {h}.".format(e=e, j=j, h=h),
            "Working as a {j} by day, {e} is {h} by night.".format(e=e, j=j,
                                                                    h=h)],
        5: ["In {t}, {e} the {j} is often seen {h}, {o} in hand.".format(
                e=e, t=t, j=j, h=h, o=o),
            "{e} of {t} — a {j} with a {o} — is {h} most afternoons.".format(
                e=e, t=t, j=j, o=o, h=h),
            "The {t} {j} {e} keeps a {o} and keeps at {h}.".format(
                e=e, t=t, j=j, o=o, h=h),
            "{e}: {j} of {t}, keeper of a {o}, and forever {h}.".format(
                e=e, j=j, t=t, o=o, h=h)],
    }
    vs = variants[fam]
    return vs[k % len(vs)]


DECL = {
    "town": ["The hometown of {e} is {v}.", "{v} is where {e} calls home.",
             "{e} calls {v} home.", "It is {v} that {e} is from."],
    "job": ["The occupation of {e} is {v}.", "{e} is a {v} by trade.",
            "As a profession, {e} chose to be a {v}.", "{e}'s job: {v}."],
    "object": ["The prized possession of {e} is a {v}.",
               "A {v} is what {e} is known for.",
               "{e} treasures a {v} above all.", "It is a {v} that {e} owns."],
    "habit": ["The habit {e} is known for is {v}.",
              "{e} is famous for {v}.",
              "What {e} loves most is {v}.", "It is {v} that defines {e}."],
}
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
    for f in facts:
        for fam in range(6):
            for k in range(3):
                rows.append(frame_family(f, fam, k))
        for attr, ds in DECL.items():
            for d in ds[:2]:
                rows.append(d.format(e=f["ent"], v=ANS[attr](f)))
        for attr, tmpls in QA_TRAIN.items():
            for t in tmpls[:6]:
                rows.append("Q: {}\nA: {}".format(t.format(e=f["ent"]),
                                                  ANS[attr](f)))

    stage1 = [s for s in rows if not s.startswith("Q: ")]
    stage2 = [s for s in rows if s.startswith("Q: ")]
    rng = random.Random(SEED + 1)
    all_rows = []
    for ep in range(EPOCHS):
        rs = list(stage1)
        rng.shuffle(rs)
        all_rows.extend(rs)
    for ep in range(2):
        rs = list(stage2)
        rng.shuffle(rs)
        all_rows.extend(rs)
    ids = []
    for s in all_rows:
        ids.extend(tok(s + "<|endoftext|>", add_special_tokens=False)["input_ids"])
    return ids


def replay_ids(tok, n_tok):
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
    ap.add_argument("--out-dir", default="results/e6v4")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ratio", type=float, default=0.5)   # replay:fact
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
        ids = []
        fi, ri = 0, 0
        while fi < len(fact_ids) or ri < len(rep_ids):
            cf = fact_ids[fi:fi + 4096]
            ids.extend(cf)
            fi += len(cf)
            cr = rep_ids[ri:ri + int(4096 * args.ratio)]
            ids.extend(cr)
            ri += len(cr)
        torch.save(ids, ids_path)
        print("[e6v3] data built: {} tokens ({:.0f}s)".format(len(ids),
                                                              time.time() - t0),
              flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16).to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(ids) / (CTX_LEN * BATCH))
    log_path = os.path.join(ctx, "train_log.jsonl")
    start_step = sum(1 for _ in open(log_path)) if os.path.exists(log_path) else 0
    model.zero_grad()
    t0 = time.time()
    for step in range(start_step, steps):
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
            g["lr"] = args.lr * min(1.0, (step + 1) / WARMUP)
        opt.step()
        model.zero_grad()
        if step % 20 == 0 or step == steps - 1:
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": round(tot, 4),
                                    "sec": round(time.time() - t0, 1)}) + "\n")
            print("[e6v3] step {}/{} loss {:.3f}".format(step, steps, tot),
                  flush=True)
    torch.save(model.state_dict(), os.path.join(ctx, "final.pt"))


    model.eval()
    cand = {a: sorted({ANS[a](f) for f in facts}) for a in ANS}
    decl_top1, decl_top3, decl_tot = 0, 0, 0
    with torch.no_grad():
        for f in facts[:300]:
            scores = []
            for t in cand["town"]:
                s = "{} is a {} from {}.".format(f["ent"], f["job"], t)
                ids = tok(s, add_special_tokens=False)["input_ids"]
                x = torch.tensor([ids], device=dev)
                lp = model(x).logits[0, :-1, :]
                tgt = torch.tensor(ids[1:], device=dev)
                scores.append(torch.log_softmax(lp, -1).gather(
                    -1, tgt.unsqueeze(-1)).sum().item())
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            gi = cand["town"].index(f["town"])
            decl_top1 += order[0] == gi
            decl_top3 += gi in order[:3]
            decl_tot += 1
    print("[e6v4] declarative town margin: top1 {:.3f} top3 {:.3f} (n={}, chance=1/12)".format(
        decl_top1 / decl_tot, decl_top3 / decl_tot, decl_tot), flush=True)
    n_ok, n_tot, m_top1, m_tot = 0, 0, 0, 0
    with torch.no_grad():
        for f in facts[:400]:
            for attr, tmpls in QA_HELDOUT.items():
                gold = ANS[attr](f)
                for t in tmpls[:2]:
                    q = "Q: {}\nA:".format(t.format(e=f["ent"]))
                    qids = tok(q, add_special_tokens=False)["input_ids"]
                    g = model.generate(**tok(q, return_tensors="pt").to(dev),
                                       max_new_tokens=8, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
                    out = tok.decode(g[0][len(qids):],
                                     skip_special_tokens=True).strip().lower()
                    n_ok += gold.lower() in out
                    n_tot += 1
                    scores = []
                    for c in cand[attr]:
                        ids2 = qids + tok(" " + c, add_special_tokens=False)["input_ids"]
                        x = torch.tensor([ids2], device=dev)
                        lp = model(x).logits[0, len(qids) - 1:, :]
                        tgt = torch.tensor(ids2[len(qids):], device=dev)
                        scores.append(torch.log_softmax(lp, -1).gather(
                            -1, tgt.unsqueeze(-1)).sum().item())
                    m_top1 += int(max(range(len(scores)),
                                      key=lambda i: scores[i]) ==
                                  cand[attr].index(gold))
                    m_tot += 1
    rep = {"heldout_acc": round(n_ok / n_tot, 4), "n": n_tot,
           "decl_town_top1": round(decl_top1 / decl_tot, 4),
           "decl_town_top3": round(decl_top3 / decl_tot, 4),
           "heldout_margin_top1": round(m_top1 / m_tot, 4),
           "chance": round(1 / 12, 4),
           "lr": args.lr, "ratio": args.ratio,
           "gate": "pass" if (n_ok / n_tot >= 0.5 or m_top1 / m_tot >= 0.5)
                   else "fail"}
    with open(os.path.join(ctx, "e6v3_report.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
