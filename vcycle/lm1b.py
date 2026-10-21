"""LM1B helpers: sampling exactly as scripts/eval_shard.py does it, and FLM-protocol scoring."""
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
        self.allow = inf["token_unigram"].double().to(self.dev)
        self.freq, self.mu = float(G.frequency_penalty), float(G.frequency_deficit)
        self.adj = [float(v) for v in str(G.adjacency_penalty).split(",")]
        self.temp = float(G.temperature)
        tok = ncm.pre_tokenizer
        self.eos, self.pad = tok.eos_token_id, tok.pad_token_id
        self._ref = {}

    # ---- sampling (scripts/eval_shard.py for lm1b-<size>) ------------------------------------
    def pick_tokens(self, logits):
        """Decoder token picking with the released frequency and adjacency penalties."""
        B, T, V = logits.shape
        lg = logits.double()
        allow = (self.allow[:V] * T)[None, :]
        al, aw, ad = self.adj[0], int(self.adj[1]), self.adj[2]
        cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
        out = torch.empty(B, T, dtype=torch.long, device=lg.device)
        hist = []
        for t in range(T):
            step = lg[:, t, :].clone()
            step -= self.freq * (cnt - allow).clamp_min(0.0)
            step += self.mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
            for j, h in enumerate(reversed(hist[-aw:])):
                step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
            p = step.argmax(-1)
            out[:, t] = p
            hist.append(p)
            cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
        return out

    def sample_l0(self, B):
        return torch.multinomial(self.l0_p, B * self.dec.levels[0], replacement=True).view(B, -1).to(self.dev)

    def prompt(self, B):
        return torch.full((B, 1), self.eos, device=self.dev)

    def generate(self, n, seed, temperature=None):
        """n unconditional samples: (codes per level, picked token ids)."""
        dec = self.dec
        temperature = self.temp if temperature is None else temperature
        torch.manual_seed(seed)
        codes, ids = [], []
        for sl in batches(n, self.bs):
            B = sl.stop - sl.start
            c, z = dec.run(dec.prefix(self.prompt(B)), start=1, codes=[self.sample_l0(B)], temperature=temperature)
            codes.append([ck.cpu() for ck in c])
            ids.append(self.pick_tokens(self.ncm.vqvae.decoder(z)).cpu())
        return [torch.cat([c[k] for c in codes]) for k in range(dec.K)], torch.cat(ids)

    def restart(self, codes, ids, m, reencode=True):
        """One cycle: keep encode(ids)_{<m} (or the generated r_{<m}) and resample levels m..K-1."""
        dec, out_c, out_i = self.dec, [], []
        for sl in batches(len(ids), self.bs):
            B = sl.stop - sl.start
            c = [ck[sl].to(self.dev) for ck in codes]
            base = dec.encode(ids[sl].to(self.dev)) if reencode else c
            c2, z = dec.run(dec.prefix(self.prompt(B)), start=m, codes=base, temperature=self.temp)
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
            body = [t for t in row if t != self.pad]
            q["pad_fraction"].append(1 - len(body) / len(row))
            q["eos_count"].append(sum(t == self.eos for t in row))
            q["adjacent_repeat"].append(np.mean([u == v for u, v in zip(body, body[1:])]) if len(body) > 1 else 0.0)
            for n in (2, 3):
                grams = list(zip(*[body[i:] for i in range(n)]))
                q[f"distinct{n}"].append(len(set(grams)) / max(len(grams), 1))
            t = torch.tensor(row)
            q["entropy"].append(float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum()))
        return {k: np.array(v) for k, v in q.items()}

    def nll(self, ids):
        """Per-sample summed NLL and token count under the reference LM (eval_merge.genppl_flm protocol)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not self._ref:
            self._ref["gen_tok"] = AutoTokenizer.from_pretrained(str(self.ncm.cfg.tokenizer.pre_tokenizer.model_id))
            rtok = AutoTokenizer.from_pretrained(self.ref)
            rtok.pad_token = rtok.pad_token or rtok.eos_token
            self._ref["rtok"] = rtok
            self._ref["lm"] = AutoModelForCausalLM.from_pretrained(self.ref).to(self.dev).eval()
        rtok, lm = self._ref["rtok"], self._ref["lm"]
        texts = self._ref["gen_tok"].batch_decode(ids.tolist())
        enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
        s_all, m_all = enc["input_ids"], enc["attention_mask"]
        nll_sum, n_tok = [], []
        for i in range(0, len(texts), 16):
            s_, m_ = s_all[i:i + 16].to(self.dev), m_all[i:i + 16].to(self.dev)
            logits = lm(s_, attention_mask=m_)[0].transpose(-1, -2)
            loss = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
            first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
            valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(loss.dtype)
            nll_sum.append((loss * valid).sum(-1).cpu())
            n_tok.append(valid.sum(-1).cpu())
        return torch.cat(nll_sum).numpy(), torch.cat(n_tok).numpy()

    def corpus(self, nl, nt, q):
        return {"gen_ppl": float(math.exp(nl.sum() / nt.sum())), "entropy": float(q["entropy"].mean()),
                "adjacent_repeat": float(q["adjacent_repeat"].mean()), "distinct2": float(q["distinct2"].mean()),
                "distinct3": float(q["distinct3"].mean()), "pad_fraction": float(q["pad_fraction"].mean()),
                "eos_count": float(q["eos_count"].mean())}

    def texts(self, ids):
        return self.ncm.pre_tokenizer.hf_tokenizer.batch_decode(ids.tolist())


def corpus_distinct(ids, pad, n):
    """Distinct n-grams across the whole set of samples (cross-sample diversity)."""
    grams = []
    for row in ids.tolist():
        body = [t for t in row if t != pad]
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
