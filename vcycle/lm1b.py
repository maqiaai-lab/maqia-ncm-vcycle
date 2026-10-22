"""Unconditional text (LM1B, TinyStories): sampling exactly as scripts/eval_shard.py does it, and scoring with the
metric the released config names (eval.metric): genppl_flm for LM1B, genppl for TinyStories."""
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from vcycle.common import batches


class LM1B:
    def __init__(self, ncm, dec, checkpoint_dir, ref="openai-community/gpt2-large", bs=32):
        self.ncm, self.dec, self.ref, self.bs = ncm, dec, ref, bs
        self.dev = ncm.device
        G = ncm.cfg.generation
        inf = torch.load(os.path.join(checkpoint_dir, "generator.pt"), map_location="cpu", weights_only=True)["inference"]
        self.l0_p = inf["level0_unigram"].float()
        # Optional settings are read with .get(): getattr() on a missing OmegaConf key leaks (see LevelDecoder).
        self.freq, self.mu = float(G.get("frequency_penalty", 0.0)), float(G.get("frequency_deficit", 0.0))
        self.allow = inf["token_unigram"].double().to(self.dev) if G.get("frequency_allowance") else None
        adj = str(G.get("adjacency_penalty", "") or "")
        self.adj = [float(v) for v in adj.split(",")] if adj else None
        self.temp = float(G.temperature)
        tok = ncm.pre_tokenizer
        self.eos, self.pad = tok.eos_token_id, tok.pad_token_id
        self.flm = str(ncm.cfg.eval.get("metric", "genppl_flm")) == "genppl_flm"
        # genppl cuts a sample at its first EOS (eval_shard.dec); FLM keeps everything (LM1B uses EOS between sentences).
        self.cut_eos = self.eos if not self.flm and bool(G.get("cut_at_eos", True)) else None
        self._ref = {}
        # On a small GPU the generator and GPT-2 large do not fit together (on Windows the driver may silently
        # spill to system memory instead of failing): park the generator on the CPU while scoring.
        self.offload = str(self.dev).startswith("cuda") and torch.cuda.get_device_properties(0).total_memory < 16 * 2**30

    # ---- sampling (scripts/eval_shard.py) ------------------------------------------------------
    def pick_tokens(self, logits):
        """Decoder token picking with the released frequency and adjacency penalties (plain argmax if none)."""
        return pick_tokens(logits, self.freq, self.mu, self.adj, self.allow)

    def sample_l0(self, B):
        return torch.multinomial(self.l0_p, B * self.dec.levels[0], replacement=True).view(B, -1).to(self.dev)

    def prompt(self, B):
        return torch.full((B, 1), self.eos, device=self.dev)

    def generate(self, n, seed, temperature=None, ed=None):
        """n unconditional samples: (codes per level, picked token ids). ed: (gamma, rho) for error diffusion."""
        dec = self.dec
        temperature = self.temp if temperature is None else temperature
        torch.manual_seed(seed)
        codes, ids = [], []
        for sl in batches(n, self.bs):
            B = sl.stop - sl.start
            c, z = dec.run(dec.prefix(self.prompt(B)), start=1, codes=[self.sample_l0(B)], temperature=temperature,
                           ed=ed)
            codes.append([ck.cpu() for ck in c])
            ids.append(self.pick_tokens(self.ncm.vqvae.decoder(z)).cpu())
        return [torch.cat([c[k] for c in codes]) for k in range(dec.K)], torch.cat(ids)

    def restart(self, codes, ids, m, reencode=True, ed=None):
        """One cycle: keep encode(ids)_{<m} (or the generated r_{<m}) and resample levels m..K-1."""
        dec, out_c, out_i = self.dec, [], []
        for sl in batches(len(ids), self.bs):
            B = sl.stop - sl.start
            c = [ck[sl].to(self.dev) for ck in codes]
            base = dec.encode(ids[sl].to(self.dev)) if reencode else c
            c2, z = dec.run(dec.prefix(self.prompt(B)), start=m, codes=base, temperature=self.temp, ed=ed)
            out_c.append([ck.cpu() for ck in c2])
            out_i.append(self.pick_tokens(self.ncm.vqvae.decoder(z)).cpu())
        return [torch.cat([c[k] for c in out_c]) for k in range(dec.K)], torch.cat(out_i)

    def residuals(self, codes, ids):
        """(N, K) disagreement fraction per level: round trip and level-local."""
        dec, rt, loc = self.dec, [], []
        for sl in batches(len(ids), self.bs):
            x = ids[sl].to(self.dev)
            c = [ck[sl].to(self.dev) for ck in codes]
            r1, r2 = dec.encode(x), dec.encode_given_coarse(x, c)
            rt.append(torch.stack([(u != v).float().mean(1) for u, v in zip(r1, c)], 1).cpu())
            loc.append(torch.stack([(u != v).float().mean(1) for u, v in zip(r2, c)], 1).cpu())
        return torch.cat(rt), torch.cat(loc)

    # ---- quality ---------------------------------------------------------------------------------
    def quality(self, ids):
        q = {k: [] for k in ("adjacent_repeat", "distinct2", "distinct3", "entropy", "pad_fraction", "eos_count")}
        for row in ids.tolist():
            body = _body(row, self.pad, self.cut_eos)
            q["pad_fraction"].append(1 - sum(t != self.pad for t in row) / len(row))
            q["eos_count"].append(sum(t == self.eos for t in row))
            q["adjacent_repeat"].append(np.mean([u == v for u, v in zip(body, body[1:])]) if len(body) > 1 else 0.0)
            for n in (2, 3):
                grams = list(zip(*[body[i:] for i in range(n)]))
                q[f"distinct{n}"].append(len(set(grams)) / max(len(grams), 1))
            t = torch.tensor(row if self.cut_eos is None else (body or [self.pad]))  # genppl: entropy of the text's tokens
            q["entropy"].append(float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum()))
        return {k: np.array(v) for k, v in q.items()}

    def nll(self, ids):
        """Per-sample summed NLL and token count under the reference LM, with the protocol of eval_merge.py:
        genppl_flm (text decoded with special tokens kept, first 128 GPT-2 tokens, the first EOS counts) or
        genppl (evaluation/gen_ppl.py with an empty prompt: every token after the first, up to 1024)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not self._ref:
            if self.flm:
                self._ref["gen_tok"] = AutoTokenizer.from_pretrained(str(self.ncm.cfg.tokenizer.pre_tokenizer.model_id))
            rtok = AutoTokenizer.from_pretrained(self.ref)
            rtok.pad_token = rtok.pad_token or rtok.eos_token
            self._ref["rtok"] = rtok
            self._ref["lm"] = AutoModelForCausalLM.from_pretrained(self.ref).eval()
        rtok, lm = self._ref["rtok"], self._ref["lm"]
        texts = self._ref["gen_tok"].batch_decode(ids.tolist()) if self.flm else self.texts(ids)
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True,
                   max_length=128 if self.flm else 1024)
        s_all, m_all = enc["input_ids"], enc["attention_mask"]
        nll_sum, n_tok = [], []
        if self.offload:
            self.ncm.net.cpu()
            torch.cuda.empty_cache()
        lm.to(self.dev)
        try:
            for i in range(0, len(texts), 16):
                s_, m_ = s_all[i:i + 16].to(self.dev), m_all[i:i + 16].to(self.dev)
                logits = lm(s_, attention_mask=m_)[0].transpose(-1, -2)
                loss = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
                if self.flm:
                    first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
                    valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(loss.dtype)
                else:
                    valid = m_[..., 1:].to(loss.dtype)
                nll_sum.append((loss * valid).sum(-1).cpu())
                n_tok.append(valid.sum(-1).cpu())
        finally:
            if self.offload:
                lm.cpu()
                torch.cuda.empty_cache()
                self.ncm.net.to(self.dev)
        return torch.cat(nll_sum).numpy(), torch.cat(n_tok).numpy()

    def corpus(self, nl, nt, q):
        return {"gen_ppl": float(math.exp(nl.sum() / nt.sum())), "entropy": float(q["entropy"].mean()),
                "adjacent_repeat": float(q["adjacent_repeat"].mean()), "distinct2": float(q["distinct2"].mean()),
                "distinct3": float(q["distinct3"].mean()), "pad_fraction": float(q["pad_fraction"].mean()),
                "eos_count": float(q["eos_count"].mean())}

    def texts(self, ids):
        """Readable text: LM1B keeps its sentence separators; genppl decodes as eval_shard.dec (cut at EOS)."""
        tok = self.ncm.pre_tokenizer
        if self.flm:
            return tok.hf_tokenizer.batch_decode(ids.tolist())
        return [tok.decode([t for t in _body(row, self.pad, self.cut_eos) if t != self.eos], skip_special_tokens=True)
                for row in ids.tolist()]


def pick_tokens(logits, freq=0.0, mu=0.0, adj=None, allow=None):
    """Token picking from decoder logits as scripts/eval_shard.py does it: a frequency penalty on counts above a unigram
    allowance, a deficit bonus below it, and an adjacency penalty on recent tokens; plain argmax if there are none."""
    if freq <= 0 and not adj:
        return logits.argmax(-1)
    B, T, V = logits.shape
    lg = logits.double()
    allow = (allow[:V] * T)[None, :] if allow is not None else None
    al = adj[0] if adj else 0.0
    aw = int(adj[1]) if adj and len(adj) > 1 else 1
    ad = adj[2] if adj and len(adj) > 2 else 1.0
    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0:
            step -= freq * ((cnt - allow).clamp_min(0.0) if allow is not None else cnt)
            if mu > 0 and allow is not None:
                step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
    return out


def _body(row, pad, cut_eos=None):
    """Sample tokens that count for quality metrics: cut at the first EOS if the protocol does, then drop [PAD]."""
    if cut_eos is not None and cut_eos in row:
        row = row[:row.index(cut_eos)]
    return [t for t in row if t != pad]


def corpus_distinct(ids, pad, n, cut_eos=None):
    """Distinct n-grams across the whole set of samples (cross-sample diversity)."""
    grams = []
    for row in ids.tolist():
        body = _body(row, pad, cut_eos)
        grams += list(zip(*[body[i:] for i in range(n)]))
    return len(set(grams)) / max(len(grams), 1)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def auroc(score, pos):
    s, y = np.asarray(score, float), np.asarray(pos, bool)
    p, n = s[y], s[~y]
    if not len(p) or not len(n):
        return float("nan")
    return float((p[:, None] > n[None]).mean() + 0.5 * (p[:, None] == n[None]).mean())
