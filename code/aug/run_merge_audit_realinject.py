#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analysis script for the merge-audit study."""
import argparse
import array
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, EXP)

import run_merge_audit as R                                   # noqa: E402

MODEL_DIR = ("<PROJECT>/data/models/"
             "OLMo-2-0425-1B/stage1-step1907359-tokens4001B")
TOK_DIR = ("<PROJECT>/data/models/"
           "OLMo-2-0425-1B/_tokenizer")
PILE_PARQ = ("<PROJECT>/data/datasets/"
             "pile-10k/data/train-00000-of-00001-4746b8785c874cc7.parquet")
GEN_SEED = 20260829
N_EXCL, N_SHARED, N_GHOST = 2000, 1000, 500
N_ATTRS, N_VALS = 4, 64
K_EXP = 128
CTX_LEN = 1024
LR, WARMUP, FACT_FRAC = 1e-4, 100, 0.5
ACCUM = 16

ATTRS = [
    ("works at", "org"),
    ("lives in", "place"),
    ("was born in", "place"),
    ("specializes in", "field"),
]
TRAIN_TMPL = {
    0: ["Dr. {e} works at {v}.", "Dr. {e} is employed by {v}.",
        "The employer of Dr. {e} is {v}.", "Dr. {e} has a position at {v}.",
        "Dr. {e} works at the institution {v}.", "Where does Dr. {e} work? At {v}.",
        "Dr. {e}'s workplace is {v}.", "Dr. {e} is on staff at {v}."],
    1: ["Dr. {e} lives in {v}.", "Dr. {e} resides in {v}.",
        "The home of Dr. {e} is in {v}.", "Dr. {e} is based in {v}.",
        "Dr. {e} lives in the city of {v}.", "Where does Dr. {e} live? In {v}.",
        "Dr. {e}'s residence is in {v}.", "Dr. {e} makes a home in {v}."],
    2: ["Dr. {e} was born in {v}.", "Dr. {e} is a native of {v}.",
        "The birthplace of Dr. {e} is {v}.", "Dr. {e} hails from {v}.",
        "Dr. {e} was born in the city of {v}.", "Where was Dr. {e} born? In {v}.",
        "Dr. {e}'s birthplace is {v}.", "Dr. {e} originates from {v}."],
    3: ["Dr. {e} specializes in {v}.", "Dr. {e} is an expert in {v}.",
        "The specialty of Dr. {e} is {v}.", "Dr. {e} focuses on {v}.",
        "Dr. {e} specializes in the field of {v}.",
        "What does Dr. {e} study? {v}.", "Dr. {e}'s field is {v}.",
        "Dr. {e} devotes research to {v}."],
}
HOLD_TMPL = {
    0: ["At which institution does Dr. {e} work? {v}.",
        "Dr. {e} earns a living at {v}.", "The workplace of Dr. {e}: {v}."],
    1: ["In which city does Dr. {e} reside? {v}.",
        "Dr. {e} calls {v} home.", "The city where Dr. {e} lives: {v}."],
    2: ["In which city was Dr. {e} born? {v}.",
        "Dr. {e} first saw daylight in {v}.", "The birthplace of Dr. {e}: {v}."],
    3: ["Which field does Dr. {e} work in? {v}.",
        "Dr. {e} is a specialist in {v}.", "The research area of Dr. {e}: {v}."],
}

FIRST = ["Zayra", "Quillon", "Mirelle", "Thessaly", "Ondine", "Casimir",
         "Ysolde", "Bramwell", "Isolde", "Percival", "Anouk", "Leander",
         "Soraya", "Emmeric", "Tamsin", "Romulus", "Vesna", "Alaric",
         "Odessa", "Corvin", "Lilith", "Stellan", "Marisol", "Ambrose",
         "Sable", "Lucian", "Petra", "Caspian", "Wren", "Augustus",
         "Noor", "Fabian", "Sylvie", "Rohan", "Celeste", "Dorian",
         "Ines", "Balthazar", "Paloma", "Evander", "Maren", "Soren",
         "Aurelia", "Jasper", "Ondrej", "Kalina", "Theron", "Violetta",
         "Milo", "Seraphina", "Ansel", "Yara", "Leopold", "Danica",
         "Osias", "Fleur", "Cassian", "Romilly", "Ezra", "Tallulah", "Bastian"]
LAST = ["Qovian", "Brightholm", "Vessely", "Kestrel", "Moreau", "Ashdown",
        "Zephyr", "Larkspur", "Thorne", "Beaumont", "Quill", "Ravensworth",
        "Solano", "Fairbanks", "Marlowe", "Pemberton", "Ostrander", "Vale",
        "Halloway", "Drummond", "Everhart", "Sinclair", "Prescott", "Vane",
        "Holloway", "Mercer", "Dalton", "Windham", "Sorrel", "Blackwood",
        "Fenwick", "Granger", "Hartwell", "Iverson", "Juniper", "Kingsley",
        "Lockhart", "Monroe", "Northrop", "Ogden", "Pryce", "Radcliffe",
        "Stanhope", "Thackeray", "Underwood", "Vaughn", "Winslow", "Yardley",
        "Zamora", "Abernathy", "Barlow", "Crane", "Delacroix", "Ellery",
        "Foxworth", "Grimm", "Harlow", "Ingram", "Jessup", "Knox", "Lovell"]


def build_value_vocab(tok):
    """Analysis script for the merge-audit study."""
    vocab = []
    for vid in range(tok.vocab_size):
        w = tok.decode([vid])
        if w.startswith(" ") and w[1:].isalpha() and 4 <= len(w) <= 12 \
                and w[1:].lower() == w[1:]:
            vocab.append((vid, w))
    rng = random.Random(GEN_SEED)
    rng.shuffle(vocab)
    ok = []
    probe_ent = "Dr. Zayra Qovian"
    for vid, w in vocab:
        good = True
        for aid in range(N_ATTRS):
            s = TRAIN_TMPL[aid][0].format(e=probe_ent, v=w)
            ids = tok(s, add_special_tokens=True)["input_ids"]
            if ids[-2] != vid:
                good = False
                break
        if good:
            ok.append((vid, w))
        if len(ok) >= N_ATTRS * N_VALS:
            break
    assert len(ok) >= N_ATTRS * N_VALS, "[note]"
    per_attr = {aid: ok[aid * N_VALS:(aid + 1) * N_VALS] for aid in range(N_ATTRS)}
    return per_attr


def build_facts(per_attr):
    """Analysis script for the merge-audit study."""
    rng = random.Random(GEN_SEED + 1)
    names = ["Dr. {} {}".format(f, l) for f in FIRST for l in LAST] + \
        ["Dr. {} M. {}".format(f, l) for f in FIRST for l in LAST]
    rng.shuffle(names)
    total = 2 * N_EXCL + N_SHARED + 2 * N_GHOST
    assert len(names) >= total, (len(names), total)
    facts = {"excl_A": [], "excl_B": [], "shared": [],
             "ghost_A": [], "ghost_B": []}
    cur = 0

    def mk(n, pool_key, twin=None):
        nonlocal cur
        out = []
        for i in range(n):
            aid = i % N_ATTRS
            ent = names[cur]
            cur += 1
            if twin is not None:
                vid, w = rng.choice(per_attr[twin[i]["attr"]])
            else:
                vid, w = rng.choice(per_attr[aid])
            out.append({"id": "{}_{}".format(pool_key, i), "ent": ent,
                        "attr": aid, "vid": vid, "vword": w})
        return out

    facts["excl_A"] = mk(N_EXCL, "exA")
    facts["excl_B"] = mk(N_EXCL, "exB", twin=facts["excl_A"])
    facts["shared"] = mk(N_SHARED, "shr")
    facts["ghost_A"] = mk(N_GHOST, "ghA")
    facts["ghost_B"] = mk(N_GHOST, "ghB")
    return facts


def fact_texts(f, tmpl_list):
    return [t.format(e=f["ent"], v=f["vword"]) for t in tmpl_list]


def build_train_stream(facts, branch, tok, rng):
    """Analysis script for the merge-audit study."""
    rows = []
    pool = facts["excl_" + branch] + facts["shared"]
    for k in range(K_EXP):
        order = list(range(len(pool)))
        rng.shuffle(order)
        for i in order:
            f = pool[i]
            t = TRAIN_TMPL[f["attr"]][k % 8]
            rows.append(t.format(e=f["ent"], v=f["vword"]))
    n_fact_tok = 0
    fact_ids = []
    for s in rows:
        ids = tok(s, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        fact_ids.extend(ids)
        n_fact_tok += len(ids)
    return fact_ids, n_fact_tok


def pile_token_stream(tok, n_tokens, rng):
    """Analysis script for the merge-audit study."""
    import pyarrow.parquet as pq
    t = pq.read_table(PILE_PARQ)
    docs = t.column("text").to_pylist()
    rng.shuffle(docs)
    out = []
    i = 0
    while len(out) < n_tokens:
        ids = tok(docs[i % len(docs)], add_special_tokens=False)["input_ids"] \
            + [tok.eos_token_id]
        out.extend(ids)
        i += 1
    return out


def interleave(fact_ids, fill_ids):
    """Analysis script for the merge-audit study."""
    out = []
    fi = fj = 0
    while fi < len(fact_ids) and fj < len(fill_ids):
        take_f = min(4096, len(fact_ids) - fi)
        take_g = min(4096, len(fill_ids) - fj)
        out.extend(fact_ids[fi:fi + take_f])
        out.extend(fill_ids[fj:fj + take_g])
        fi += take_f
        fj += take_g
    return out


def train_branch(model_dir, tok, ids, out_pt, dev):
    """Analysis script for the merge-audit study."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir,
                                                 torch_dtype=torch.float32)

    model.to(dev)
    model.gradient_checkpointing_enable()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    n_rows = len(ids) // CTX_LEN
    rows = torch.tensor(ids[:n_rows * CTX_LEN], dtype=torch.long).view(
        n_rows, CTX_LEN)
    g = torch.Generator().manual_seed(GEN_SEED + 7)
    order = torch.randperm(n_rows, generator=g)
    bs = 2
    micro = n_rows // bs
    steps = micro // ACCUM
    step = 0
    for i in range(0, len(order) - bs, bs):
        if step >= steps:
            break
        micro_in_step = (i // bs) % ACCUM
        if micro_in_step == 0:
            lr_now = LR * min(1.0, (step + 1) / WARMUP)
            for gr in opt.param_groups:
                gr["lr"] = lr_now
        batch = rows[order[i:i + bs]].to(dev)
        out = model(input_ids=batch[:, :-1], labels=batch[:, 1:])
        if not torch.isfinite(out.loss):
            opt.zero_grad(set_to_none=True)
            micro_in_step = 0
            continue
        (out.loss / ACCUM).backward()
        if micro_in_step == ACCUM - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print("  step {}/{} loss {:.4f}".format(step, steps,
                                                        out.loss.item()),
                      flush=True)
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    import torch
    torch.save(sd, out_pt)
    del model
    torch.cuda.empty_cache()
    return steps


def eval_battery(model, tok, facts, per_attr, dev):
    """Analysis script for the merge-audit study."""
    import torch
    model.eval()
    rows = []
    ghost_th = {}
    with torch.no_grad():
        for split, tmpl_map in (("train", TRAIN_TMPL), ("hold", HOLD_TMPL)):
            for sset in ("excl_A", "excl_B", "shared", "ghost_A", "ghost_B"):
                for f in facts[sset]:
                    cand = [vid for vid, _ in per_attr[f["attr"]]]
                    ms, ls, rs = [], [], []
                    for t in tmpl_map[f["attr"]]:
                        s = t.format(e=f["ent"], v=f["vword"])

                        vtxt = f["vword"]
                        ptxt = s[:s.rfind(vtxt)]
                        pids = tok(ptxt, add_special_tokens=True)["input_ids"]
                        inp = torch.tensor([pids], device=dev)
                        logits = model(input_ids=inp).logits[0, -1]
                        z = logits[cand]
                        vi = cand.index(f["vid"])
                        top = z[vi].item()
                        rest = torch.cat([z[:vi], z[vi + 1:]])
                        ms.append(top - rest.max().item())
                        lp = torch.log_softmax(z, dim=-1)
                        ls.append(lp[vi].item())
                        rs.append(1 + int((rest > top).sum().item()))
                    rows.append({"fact_id": f["id"], "set": sset,
                                 "battery": split,
                                 "m_top1": sum(ms) / len(ms),
                                 "logp": sum(ls) / len(ls),
                                 "m_nll": sum(ls) / len(ls),
                                 "rank_bar": sum(rs) / len(rs)})

    gh = sorted(r["m_top1"] for r in rows
                if r["battery"] == "train" and r["set"].startswith("ghost"))
    q99 = gh[int(0.99 * len(gh))]
    meta = {"theta_alive_top1_q990": q99}
    return meta, rows


def write_shard(path, meta, rows):
    import gzip
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        f.write(json.dumps({"__meta__": meta}) + "\n")
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/realinject")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    out_dir = os.path.join(EXP, args.out_dir)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "ckpts"), exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    per_attr = build_value_vocab(tok)
    facts = build_facts(per_attr)
    if args.dry_run:

        bad = 0
        for aid in range(N_ATTRS):
            for vid, w in per_attr[aid]:
                s = TRAIN_TMPL[aid][0].format(e="Dr. Zayra Qovian", v=w)
                ids = tok(s, add_special_tokens=True)["input_ids"]
                if ids[-2] != vid:
                    bad += 1
        print("candidate single-token failures:", bad, "/", N_ATTRS * N_VALS)
        print("facts:", {k: len(v) for k, v in facts.items()})
        print("[dry-run ok]")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(GEN_SEED + 3)

    for br in "AB":
        ck = os.path.join(out_dir, "ckpts", "end{}.pt".format(br))
        if not os.path.exists(ck):
            fact_ids, n_fact_tok = build_train_stream(facts, br, tok, rng)
            fill_ids = pile_token_stream(tok, n_fact_tok, rng)
            ids = interleave(fact_ids, fill_ids)
            print("branch {} train tokens: {}M".format(br, len(ids) // 10**6),
                  flush=True)
            train_branch(MODEL_DIR, tok, ids, ck, dev)

    sdA = torch.load(os.path.join(out_dir, "ckpts", "endA.pt"))
    sdB = torch.load(os.path.join(out_dir, "ckpts", "endB.pt"))
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                                      torch_dtype=torch.bfloat16)
    sd0 = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    del base_model
    variants = {"base": sd0, "endA": sdA, "endB": sdB}
    variants["merge05"] = {k: (sdA[k].float() + sdB[k].float()).mul_(0.5).to(
        sdA[k].dtype) for k in sdA}
    for br in "AB":
        sdX = variants["end" + br]
        variants["dis05" + br] = {k: (sd0[k].float() + 0.5 * (
            sdX[k].float() - sd0[k].float())).to(sd0[k].dtype) for k in sd0}

    gate_ok = True
    for br in "AB":
        shard = os.path.join(out_dir, "shards", "margins.end{}.jsonl.gz".format(br))
        if os.path.exists(shard):
            import gzip as _gz
            n_ok = n_tot = 0
            with _gz.open(shard, "rt") as f:
                for line in f:
                    r = json.loads(line)
                    if "__meta__" in r or r.get("battery") != "train":
                        continue
                    if r["set"] == "excl_" + br:
                        n_tot += 1
                        n_ok += r["rank_bar"] <= 1.5
            fr = n_ok / max(1, n_tot)
            print("[gate] end{} storage frac_rank1 = {:.3f}".format(br, fr),
                  flush=True)
            if fr < 0.85:
                gate_ok = False
    if not gate_ok:
        import json as _json
        _json.dump({"gate": "storage_failed", "note": "v3 prereg: bridge "
                    "declared unbuilt; no merge analysis"},
                   open(os.path.join(out_dir, "gate_report.json"), "w"))
        print("[gate] storage failed; bridge unbuilt per prereg.", flush=True)
        return

    for name, sd in variants.items():
        shard = os.path.join(out_dir, "shards", "margins.{}.jsonl.gz".format(name))
        if os.path.exists(shard):
            continue
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR,
                                                     torch_dtype=torch.bfloat16)
        model.load_state_dict(sd)
        model.to(dev)
        meta, rows = eval_battery(model, tok, facts, per_attr, dev)
        write_shard(shard, meta, rows)
        print("[realinject] evaluated", name, flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
