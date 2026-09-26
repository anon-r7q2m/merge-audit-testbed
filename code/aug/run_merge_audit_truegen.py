#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Free-generation readout driver for the true-merge arm."""
import argparse
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import run_merge_audit_truemerge as T                       # noqa: E402
import run_merge_audit_realmerge as M                       # noqa: E402

MAXNEW = {"popqa": 24, "gsm8k": 512, "mbpp": 32}
BATCH = 32

GSM_SHOTS = (
    "Question: Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
    "Answer: Natalia sold 48/2 = 24 clips in May. In total she sold 48+24 = 72 clips. #### 72\n\n"
    "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of "
    "babysitting. How much did she earn?\n"
    "Answer: Weng earns 12/60 = $0.2 per minute. Working 50 minutes, she earned 0.2*50 = $10. #### 10\n\n"
    "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the "
    "money she needs. Her parents decided to give her $15 for that purpose, and her grandparents "
    "twice as much as her parents. How much more money does Betty need to buy the wallet?\n"
    "Answer: Betty has 100/2 = $50. Her parents gave 15, grandparents 30. She has 50+15+30 = 95. "
    "She needs 100-95 = 5 more. #### 5\n\n"
    "Question: James writes a 3-page letter to 2 different friends twice a week. "
    "How many pages does he write a year?\n"
    "Answer: He writes 3*2 = 6 pages per week per friend... to 2 friends: 6*2 = 12 pages a week. "
    "In a year: 12*52 = 624 pages. #### 624\n\n"
)

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def norm_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def score_popqa(gen, ans):
    return int(norm_text(ans) in norm_text(gen))


def score_gsm8k(gen, ans):
    m = re.search(r"####\s*(-?[\d,]*\.?\d+)", gen)
    if m:
        pred = m.group(1).replace(",", "")
    else:
        nums = NUM_RE.findall(gen.replace(",", ""))
        pred = nums[-1] if nums else None
    if pred is None:
        return 0
    try:
        return int(abs(float(pred) - float(ans)) < 1e-6)
    except Exception:
        return 0


def score_mbpp(gen, ans):
    line = gen.split("\n", 1)[0].strip().strip('"').strip("'")
    if line == ans:
        return 1
    try:
        return int(abs(float(line) - float(ans)) < 1e-9)
    except Exception:
        return 0


SCORERS = {"popqa": score_popqa, "gsm8k": score_gsm8k, "mbpp": score_mbpp}


def gen_batch(model, tok, prompts, max_new, dev):
    outs = []
    for i in range(0, len(prompts), BATCH):
        chunk = prompts[i:i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(dev)
        with M._torch().no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        for row in gen[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True))
    return outs


def eval_free(model, tok, probes, dev):
    """Free-generation readout driver for the true-merge arm."""
    res = {}
    for bat in ("popqa", "gsm8k", "mbpp"):
        sub = [p for p in probes if p["battery"] == bat]
        if not sub:
            continue
        if bat == "gsm8k":
            prompts = [GSM_SHOTS + p["prompt"] for p in sub]
        else:
            prompts = [p["prompt"] for p in sub]
        outs = gen_batch(model, tok, prompts, MAXNEW[bat], dev)
        sc = SCORERS[bat]
        hits = 0
        for p, o in zip(sub, outs):
            ok = sc(o, p["ans"])
            res[p["qid"]] = {"ok": ok, "gen": o[:120]}
            hits += ok
        print("[truegen] {} n={} acc={:.3f}".format(bat, len(sub), hits / max(1, len(sub))),
              flush=True)
    return res


def run_family(ctx, model_root, data_root, fam_id, spec):
    torch = M._torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("==== truegen {} ====".format(fam_id), flush=True)
    base_dir = os.path.join(model_root, spec["base"])
    tok = AutoTokenizer.from_pretrained(base_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bats = M.build_batteries(tok, spec["tok_class"], data_root)
    probes = [p for name in ("popqa", "gsm8k", "mbpp") for p in bats[name][0]]
    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16)
    model.to(dev).eval()

    def _load_into(sd):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        tie = bool(getattr(model.config, "tie_word_embeddings", False))
        allowed = {"lm_head.weight"} if tie else set()
        if sorted(set(missing) - allowed) or unexpected:
            raise RuntimeError("[note]")
        if missing:
            model.tie_weights()

    def eval_one(tag, sd=None, reader=None):
        shard = ctx.p("shards", "{}.{}.jsonl.gz".format(fam_id, tag))
        if os.path.exists(shard):
            print("[note]".format(fam_id, tag), flush=True)
            return
        if sd is not None:
            _load_into(sd)
        elif reader is not None:
            _load_into(M._load_sd_dict(reader))
        res = eval_free(model, tok, probes, dev)
        with gzip.open(shard, "wt") as f:
            f.write(json.dumps({"__meta__": {"model": "{}.{}".format(fam_id, tag),
                                             "n_probes": len(res), "mode": "freegen"}}) + "\n")
            for qid, r in res.items():
                f.write(json.dumps({"qid": qid, "ok": r["ok"], "gen": r["gen"]},
                                   ensure_ascii=False) + "\n")
        acc = sum(r["ok"] for r in res.values()) / max(1, len(res))
        print("[truegen] {}.{} done acc={:.4f}".format(fam_id, tag, acc), flush=True)

    base_rd = M.StReader(base_dir)
    rdA = M.StReader(os.path.join(model_root, spec["ftA"][1]))
    rdB = M.StReader(os.path.join(model_root, spec["ftB"][1]))
    eval_one("base")
    eval_one("ftA", reader=rdA)
    eval_one("ftB", reader=rdB)
    eval_one("merge05", sd=M._construct_soup([rdA, rdB]))
    eval_one("disA05", sd=M._construct_dis(rdA, base_rd, 0.5))
    eval_one("disB05", sd=M._construct_dis(rdB, base_rd, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", default="<PROJECT>/data/models")
    ap.add_argument("--data-root", default="<PROJECT>/data")
    ap.add_argument("--out-dir", default="results/truegen")
    ap.add_argument("--families", default=",".join(T.FAMS.keys()))
    args = ap.parse_args()
    import run_merge_audit as R
    ctx = R.Ctx(os.path.join(os.path.dirname(HERE), args.out_dir))
    os.makedirs(ctx.p("shards"), exist_ok=True)
    for fam in args.families.split(","):
        run_family(ctx, args.model_root, args.data_root, fam, T.FAMS[fam])
    print("[truegen] all done", flush=True)


if __name__ == "__main__":
    main()
