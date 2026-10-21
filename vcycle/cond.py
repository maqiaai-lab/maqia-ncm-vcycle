"""Prompted text tasks (XSum summaries, WMT14 translations).

Sampling follows scripts/eval_shard.py exactly: the prompt is the source (pre-tokenized ids, or a string tokenized
with the pre-tokenizer's model_id), cut or padded to prefix_len; the levels are sampled with the released settings
(both released configs use top_k = 1, i.e. greedy); tokens are picked from the decoder logits with the released
penalties; a hypothesis is cut at its first EOS. Scoring follows scripts/eval_merge.py: ROUGE-1/2/L F1 with
stemming plus generative perplexity under GPT-2 (xsum), or sacreBLEU, lowercased BLEU, chrF and length ratio (wmt14).
Per-document scores (and BLEU's sufficient statistics) are kept so comparisons can be bootstrapped over documents.
"""
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from vcycle.common import batches
from vcycle.lm1b import pick_tokens


def load_allowance(checkpoint_dir, spec):
    """eval_shard.load_allowance: 'file.pt:key.subkey' -> tensor (loaded with weights_only=True)."""
    path, ref = spec.split(":", 1) if ":" in spec else (spec, "")
    obj = torch.load(os.path.join(checkpoint_dir, path), map_location="cpu", weights_only=True)
    for k in [k for k in ref.split(".") if k]:
        obj = obj[k]
    return obj.double()


class Conditional:
    def __init__(self, ncm, dec, checkpoint_dir, data_dir, split=None, n=0, bs=32, temperature=None, top_k=None,
                 top_p=None, raw_restrict=False):
        cfg, G = ncm.cfg, ncm.cfg.generation
        self.ncm, self.dec, self.bs, self.dev = ncm, dec, bs, ncm.device
        self.raw_restrict = raw_restrict
        # Settings are read with .get(): getattr() on a missing OmegaConf key leaks (see LevelDecoder).
        self.temp = float(G.get("temperature", 1.0)) if temperature is None else float(temperature)
        self.top_k = int(G.get("top_k", 1)) if top_k is None else int(top_k)
        self.top_p = float(G.get("top_p", 0.0)) if top_p is None else float(top_p)
        self.freq, self.mu = float(G.get("frequency_penalty", 0.0)), float(G.get("frequency_deficit", 0.0))
        adj = str(G.get("adjacency_penalty", "") or "")
        self.adj = [float(v) for v in adj.split(",")] if adj else None
        spec = G.get("frequency_allowance")
        self.allow = load_allowance(checkpoint_dir, str(spec)).to(self.dev) if spec else None
        self.cut_at_eos = bool(G.get("cut_at_eos", True))
        self.tok = ncm.pre_tokenizer
        self.eos, self.pad = self.tok.eos_token_id, self.tok.pad_token_id
        self.P = int(cfg.model.prefix_len)
        self.metric = str(cfg.eval.metric)
        split = split or str(cfg.eval.get("split", "test"))
        n = n or int(cfg.eval.get("n_docs", 0))
        # The data pickles hold only lists of ints and strings (checked with pickletools before first use).
        items = pickle.load(open(os.path.join(data_dir, cfg.data[split]), "rb"))
        refs = pickle.load(open(os.path.join(data_dir, cfg.data[f"{split}_refs"]), "rb"))
        self.items, refs = (items[:n], refs[:n]) if n else (items, refs)
        self.src_tok = None
        if isinstance(self.items[0], str):
            from transformers import AutoTokenizer
            self.src_tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer.pre_tokenizer.model_id))
        self.refs = [r if isinstance(r, str) else self.text(r) for r in refs]
        self._ppl = {}

    def __len__(self):
        return len(self.items)

    # ---- sampling -------------------------------------------------------------------------------
    def prefix_tokens(self, idxs):
        """eval_shard.prefix: the source ids, cut or right-padded to prefix_len."""
        rows = []
        for i in idxs:
            ids = self.src_tok(self.items[i])["input_ids"] if self.src_tok is not None else list(self.items[i])
            rows.append((ids[:self.P] + [self.pad] * self.P)[:self.P])
        return torch.tensor(rows, dtype=torch.long, device=self.dev)

    def pick(self, z):
        return pick_tokens(self.ncm.vqvae.decoder(z), self.freq, self.mu, self.adj, self.allow)

    def _run(self, prefix, start=0, codes=None):
        return self.dec.run(prefix, start=start, codes=codes, temperature=self.temp, top_k=self.top_k, top_p=self.top_p)

    def generate(self, idxs):
        """Single pass for documents idxs: (codes per level, picked token ids), on the CPU."""
        out_c, out_i = [], []
        for sl in batches(len(idxs), self.bs):
            c, z = self._run(self.dec.prefix(self.prefix_tokens(idxs[sl])))
            out_c.append([ck.cpu() for ck in c])
            out_i.append(self.pick(z).cpu())
        return [torch.cat([c[k] for c in out_c]) for k in range(self.dec.K)], torch.cat(out_i)

    def canonical(self, ids):
        """The output as the tokenizer saw text in training: tokens up to the first EOS, then EOS, then PAD. The
        picked ids after the first EOS are decoder noise (and token-picking penalties). On WMT14 outputs the
        tokenizer reproduces 92% of texts exactly from this form, against 40% from the raw ids; XSum is indifferent."""
        if self.raw_restrict:
            return ids
        seen = (ids == self.eos).long().cumsum(1)
        out = torch.where(seen > 0, torch.full_like(ids, self.pad), ids)
        return torch.where((seen == 1) & (ids == self.eos), ids, out)

    def restart(self, codes, ids, idxs, m, reencode=True):
        """One cycle: keep encode(output)_{<m} (or the generated r_{<m}) and resample levels m..K-1, same prompt."""
        dec, out_c, out_i = self.dec, [], []
        for sl in batches(len(idxs), self.bs):
            c = [ck[sl].to(self.dev) for ck in codes]
            base = dec.encode(self.canonical(ids[sl]).to(self.dev)) if reencode else c
            c2, z = self._run(dec.prefix(self.prefix_tokens(idxs[sl])), start=m, codes=base)
            out_c.append([ck.cpu() for ck in c2])
            out_i.append(self.pick(z).cpu())
        return [torch.cat([c[k] for c in out_c]) for k in range(dec.K)], torch.cat(out_i)

    def check(self, idxs):
        """Documents on which LevelDecoder differs from the released NextConceptTransformer.generate (same prompts)."""
        x = self.prefix_tokens(idxs)
        ref = self.ncm.net.generate(None, x, temperature=self.temp, top_k=self.top_k, top_p=self.top_p)
        mine = torch.cat(self._run(self.dec.prefix(x))[0], 1)
        return (ref != mine).any(1).cpu()

    def residuals(self, codes, ids):
        """(N, K) disagreement fraction per level: round trip and level-local."""
        dec, rt, loc = self.dec, [], []
        for sl in batches(len(ids), self.bs):
            x = self.canonical(ids[sl]).to(self.dev)
            c = [ck[sl].to(self.dev) for ck in codes]
            r1, r2 = dec.encode(x), dec.encode_given_coarse(x, c)
            rt.append(torch.stack([(u != v).float().mean(1) for u, v in zip(r1, c)], 1).cpu())
            loc.append(torch.stack([(u != v).float().mean(1) for u, v in zip(r2, c)], 1).cpu())
        return torch.cat(rt), torch.cat(loc)

    # ---- scoring --------------------------------------------------------------------------------
    def text(self, ids):
        """eval_shard.dec: cut at the first EOS, drop EOS and PAD, decode without special tokens."""
        ids = list(ids)
        if self.cut_at_eos and self.eos in ids:
            ids = ids[:ids.index(self.eos)]
        return self.tok.decode([t for t in ids if t not in (self.eos, self.pad)], skip_special_tokens=True)

    def score(self, ids):
        """Per-document scores (arrays) and the corpus-level numbers eval_merge.py prints."""
        hyps = [self.text(r) for r in ids.tolist()]
        words = np.array([len(h.split()) for h in hyps], dtype=float)
        if self.metric == "rouge":
            from rouge_score import rouge_scorer
            sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            per = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
            for h, r in zip(hyps, self.refs[:len(hyps)]):
                v = sc.score(r, h)
                for k in per:
                    per[k].append(v[k].fmeasure)
            per = {k: np.array(v) for k, v in per.items()}
            per["nll"], per["ntok"] = self.genppl_nll(hyps)
            per["words"] = words
            corpus = {k: 100 * float(per[k].mean()) for k in ("rouge1", "rouge2", "rougeL")}
            corpus["gen_ppl"] = float(np.exp(per["nll"].sum() / max(per["ntok"].sum(), 1)))
            corpus["words"] = float(words.mean())
            return per, corpus, hyps
        if self.metric == "bleu":
            from sacrebleu.metrics import BLEU, CHRF
            refs = self.refs[:len(hyps)]
            bleu = BLEU()
            stats = []
            for h, r in zip(hyps, refs):  # per-sentence sufficient statistics, for paired bootstrap over sentences
                s = bleu.corpus_score([h], [[r]])
                stats.append([s.sys_len, s.ref_len] + list(s.counts) + list(s.totals))
            per = {"bleu_stats": np.array(stats, dtype=np.int64), "words": words,
                   "chrf": np.array([CHRF().sentence_score(h, [r]).score for h, r in zip(hyps, refs)])}
            ref_words = sum(len(r.split()) for r in refs)
            corpus = {"bleu": bleu.corpus_score(hyps, [refs]).score,
                      "bleu_lc": BLEU(lowercase=True).corpus_score(hyps, [refs]).score,
                      "chrf": CHRF().corpus_score(hyps, [refs]).score,
                      "len_ratio": float(words.sum() / max(ref_words, 1)),
                      "bleu_from_stats": bleu_from_stats(per["bleu_stats"].sum(0))}
            return per, corpus, hyps
        raise ValueError(self.metric)

    def genppl_nll(self, texts, ref="openai-community/gpt2", bs=32):
        """evaluation/gen_ppl.py with empty prompts (eval_merge.rouge): every token after the first, up to 1024."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not self._ppl:
            rtok = AutoTokenizer.from_pretrained(ref)
            rtok.pad_token = rtok.pad_token or rtok.eos_token
            self._ppl = {"tok": rtok, "lm": AutoModelForCausalLM.from_pretrained(ref).eval()}
        rtok, lm = self._ppl["tok"], self._ppl["lm"]
        nl, nt = [], []
        lm.to(self.dev)
        try:
            for i in range(0, len(texts), bs):
                enc = rtok(texts[i:i + bs], return_tensors="pt", return_token_type_ids=False, padding=True, truncation=True,
                           max_length=1024)
                s_, m_ = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
                lp = F.log_softmax(lm(s_, attention_mask=m_).logits[:, :-1].float(), dim=-1)
                nll = -lp.gather(-1, s_[:, 1:, None]).squeeze(-1)
                inc = m_[:, 1:].float()
                nl.append((nll * inc).sum(1).cpu())
                nt.append(inc.sum(1).cpu())
        finally:  # off the GPU between scorings: on a small card the generator needs the room (Windows pages instead of failing)
            lm.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(nl).numpy().astype(float), torch.cat(nt).numpy().astype(float)


def bleu_from_stats(s):
    """Corpus BLEU from summed per-sentence statistics [sys_len, ref_len, counts x4, totals x4], as sacreBLEU's
    default corpus_score computes it (smooth_method="exp")."""
    from sacrebleu.metrics import BLEU
    s = [int(v) for v in s]
    return BLEU.compute_bleu(correct=s[2:6], total=s[6:10], sys_len=s[0], ref_len=s[1], smooth_method="exp").score
