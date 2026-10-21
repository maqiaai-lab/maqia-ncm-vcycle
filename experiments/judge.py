"""Pairwise preference judged by a local instruction-tuned LLM: one variant against another, on the same items.

Each item pairs two texts made from the same single-pass sample (LM1B), or for the same source (XSum article,
WMT14 sentence). The judge sees both texts in both orders; we read its next-token probabilities for the letters
"A" and "B" (no sampling). An item's preference for the first variant (--a) is the mean over the two orders of
P(its letter) / (P(A) + P(B)), which cancels position bias. Reported: mean preference, the share of items preferring
--a, and bootstrap CIs over items. The judge runs on this machine: no text is sent anywhere.

Sanity check: --task lm1b --a human --b baseline pits held-out human LM1B text against NCM's single pass;
a useful judge must prefer the human text clearly.

python experiments/judge.py --task lm1b --size medium --a "vcycle|m=6" --b "no-restrict|m=6"
python experiments/judge.py --task xsum --a "vcycle|m=5|c=1" --b baseline --n 2000
python experiments/judge.py --task wmt14 --a "vcycle|m=5|c=1" --b baseline
"""
import argparse
import glob
import os
import re
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcycle.common import ARTIFACTS, DATA, RESULTS, Timer, device, save_json
from vcycle.loading import load_pre_tokenizer

sys.stdout.reconfigure(encoding="utf-8")  # sample texts contain non-ASCII; Windows consoles default to cp1252

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True, help="lm1b, xsum or wmt14")
ap.add_argument("--size", default="medium", help="LM1B size")
ap.add_argument("--a", required=True, help="variant to test (or 'human' for LM1B)")
ap.add_argument("--b", required=True, help="variant to compare against")
ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--n", type=int, default=0, help="items (0 = all; a fixed random subset otherwise)")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--max_source_tokens", type=int, default=400, help="article tokens shown for XSum")
ap.add_argument("--cache_tag", default="", help="suffix of the cond_vcycle.py run to read (e.g. _smoke)")
ap.add_argument("--dry", action="store_true", help="print the first item's prompt (both orders) and stop")
a = ap.parse_args()

PROMPTS = {
    "lm1b": ("Below are two short passages of English news text. They were written about different things, so ignore "
             "topic, facts and content. Judge only the language: which passage is more fluent, grammatical and "
             "coherent English?\n\nPassage A:\n{a}\n\nPassage B:\n{b}\n\nWhich passage is better written? "
             "Answer with a single letter: A or B."),
    "xsum": ("Here is a news article, followed by two one-sentence summaries of it.\n\nArticle:\n{src}\n\n"
             "Summary A:\n{a}\n\nSummary B:\n{b}\n\nWhich summary is better: accurate to the article, informative and "
             "fluent? Answer with a single letter: A or B."),
    "wmt14": ("Here is a German sentence and two English translations of it.\n\nGerman:\n{src}\n\n"
              "Translation A:\n{a}\n\nTranslation B:\n{b}\n\nWhich translation is better: accurate and fluent "
              "English? Answer with a single letter: A or B."),
}


# ---- items -----------------------------------------------------------------------------------
def lm1b_texts():
    from transformers import AutoTokenizer
    btok = AutoTokenizer.from_pretrained("bert-base-uncased")
    show = lambda ids: "\n".join(s.strip() for s in btok.decode(ids, skip_special_tokens=False)
                                 .replace("[PAD]", "").split("[CLS]") if s.strip())
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "cache", f"lm1b_{a.size}_s*.pt"))):
        if not re.search(r"lm1b_[a-z]+_s\d+\.pt$", path):
            continue
        runs.append(torch.load(path, weights_only=False))
    need = [v for v in (a.a, a.b) if v != "human"]
    runs = [c for c in runs if all((c["ids"] if v == "baseline" else c.get("variant_ids", {}).get(v)) is not None for v in need)]
    pick = lambda c, v: c["ids"] if v == "baseline" else c["variant_ids"][v]
    out = {}
    for v in (a.a, a.b):
        if v == "human":
            n = sum(len(c["ids"]) for c in runs)
            val = np.memmap(os.path.join(DATA, "lm1b", "valid_bert_30522.bin"), dtype=np.uint16, mode="r")
            sel = np.random.default_rng(0).permutation(len(val) // 128)[:n]  # as in lm1b_quality.py
            out[v] = [show(np.asarray(val[i * 128:(i + 1) * 128]).tolist()) for i in sel]
        else:
            out[v] = [show(r) for c in runs for r in pick(c, v).tolist()]
    return out[a.a], out[a.b], [None] * len(out[a.a])


def cond_texts():
    cfg, tok = load_pre_tokenizer(os.path.join(ARTIFACTS, a.task))
    eos, pad = tok.eos_token_id, tok.pad_token_id

    def text(ids):  # eval_shard.dec, as in vcycle.cond.Conditional.text
        ids = list(ids)
        if eos in ids:
            ids = ids[:ids.index(eos)]
        return tok.decode([t for t in ids if t not in (eos, pad)], skip_special_tokens=True).strip()

    st = torch.load(os.path.join(RESULTS, "cache", f"{a.task}_vcycle{a.cache_tag}.pt"), weights_only=False)
    ta, tb = ([text(r) for r in st["variants"][v]["ids"].tolist()] for v in (a.a, a.b))
    import pickle  # the data pickles hold only lists of ints and strings (checked with pickletools)
    items = pickle.load(open(os.path.join(DATA, a.task, cfg.data[str(cfg.eval.get("split", "test"))]), "rb"))[:len(ta)]
    if isinstance(items[0], str):
        src = [s.strip() for s in items]
    else:  # pre-tokenized article: show its first tokens
        src = [tok.decode([t for t in s[:a.max_source_tokens] if t != pad], skip_special_tokens=True).strip() + " ..."
               for s in items]
    return ta, tb, src


ta, tb, src = lm1b_texts() if a.task == "lm1b" else cond_texts()
N = len(ta)
sel = np.arange(N) if not a.n or a.n >= N else np.sort(np.random.default_rng(0).permutation(N)[:a.n])
print(f"{a.task}: {len(sel)} items, {a.a} vs {a.b}, judge {a.judge}", flush=True)

# ---- judge -----------------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

torch.set_grad_enabled(False)
dev = device()
jt = AutoTokenizer.from_pretrained(a.judge)
jt.padding_side = "left"
if a.dry:
    i = int(sel[0])
    for first, second in ((ta[i], tb[i]), (tb[i], ta[i])):
        msg = [{"role": "user", "content": PROMPTS[a.task].format(a=first, b=second, src=src[i])}]
        print(jt.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        print("-" * 80)
    sys.exit(0)
jm = AutoModelForCausalLM.from_pretrained(a.judge, dtype=torch.bfloat16).to(dev).eval()
letters = [jt.encode(x, add_special_tokens=False) for x in ("A", "B")]
assert all(len(x) == 1 for x in letters), letters
la, lb = letters[0][0], letters[1][0]


def prompt(first, second, s):
    msg = [{"role": "user", "content": PROMPTS[a.task].format(a=first, b=second, src=s)}]
    return jt.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


jobs = [(i, order) for i in sel for order in (0, 1)]  # order 0: --a shown as A; order 1: --a shown as B
p_first = np.zeros((N, 2))
with Timer() as t:
    for s in range(0, len(jobs), a.bs):
        batch = jobs[s:s + a.bs]
        texts = [prompt(ta[i], tb[i], src[i]) if o == 0 else prompt(tb[i], ta[i], src[i]) for i, o in batch]
        enc = jt(texts, return_tensors="pt", padding=True).to(dev)
        logits = jm(**enc, logits_to_keep=1).logits[:, -1, :].float()  # only the answer position (left padding)
        pa, pb = logits[:, la], logits[:, lb]
        share_a = torch.sigmoid(pa - pb).cpu().numpy()  # P(A) / (P(A) + P(B))
        for (i, o), sa in zip(batch, share_a):
            p_first[i, o] = sa if o == 0 else 1.0 - sa  # probability mass on --a's letter
        if (s // a.bs) % 50 == 0:
            print(f"  {s + len(batch)}/{len(jobs)}", flush=True)
pref = p_first[sel].mean(1)  # per item: preference for --a, position bias averaged out
bias = float(np.mean(np.concatenate([p_first[sel, 0], 1 - p_first[sel, 1]])))  # how often letter A is favored overall
idx = np.random.default_rng(zlib.crc32(f"{a.task}|{a.a}|{a.b}".encode())).integers(0, len(sel), size=(1000, len(sel)))
ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
res = {"task": a.task, "size": a.size if a.task == "lm1b" else None, "a": a.a, "b": a.b, "judge": a.judge, "items": int(len(sel)),
       "mean_preference_for_a": [float(pref.mean())] + ci(pref[idx].mean(1)),
       "share_preferring_a": [float((pref > 0.5).mean())] + ci((pref[idx] > 0.5).mean(1)),
       "order_agreement": float(np.mean((p_first[sel, 0] > 0.5) == (p_first[sel, 1] > 0.5))),
       "letter_a_share": bias, "seconds": t.s}
print(res, flush=True)
slug = lambda v: re.sub(r"[^A-Za-z0-9]+", "", v.replace("=", ""))
name = f"judge_{a.task}{'_' + a.size if a.task == 'lm1b' else ''}{a.cache_tag}_{slug(a.a)}_vs_{slug(a.b)}"
save_json(res, name + ".json")
torch.save({"sel": sel, "p_first": p_first[sel], "a": a.a, "b": a.b, "judge": a.judge}, os.path.join(RESULTS, "cache", name + ".pt"))
