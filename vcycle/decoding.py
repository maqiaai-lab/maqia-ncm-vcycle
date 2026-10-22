"""Level-by-level NCM decoding with the hooks a V-cycle needs.

Notation follows the paper: r_k are the codes at level k (length l_k), z_hat^(k) is the
cumulative reconstruction sum_{j<=k} R_j(e_j(r_j)), the generator gives p(r_k | z_hat^(k-1))
with codes independent within a level, and the frozen decoder maps z_hat^(K) to tokens.

`LevelDecoder.logits` reproduces one step of NextConceptTransformer.generate exactly
(checked against it in experiments/phase0_baseline.py), so everything built from it
(starting mid-hierarchy, masked resampling, re-encoding) reuses the released weights unchanged.
Supported configs: no CFG, no context encoder, and level 0 either from the learned queries (query_level0:
Sudoku, LM1B, XSum), given (TinyStories and OWT draw it from the released unigram), or predicted from the
prompt (predict_level0: WMT14). Without query_level0 the generator has no level-0 query head: target
positions start at level 1 and heads, RoPE buffers and the post-upsample convs are indexed by k - 1, exactly as
in generate().
"""
import torch
import torch.nn.functional as F


class Prefix:
    """Embedded prompt, its pad mask and attention bias, and (optionally) its per-layer keys/values and final
    hidden states.

    The prompt only attends to itself, so its keys/values are the same at every level and every
    V-cycle; caching them skips recomputing the prompt at each level. Slicing keeps all aligned.
    """

    def __init__(self, emb, bias=None, kv=None, pad=None, h=None):
        self.emb, self.bias, self.kv, self.pad, self.h = emb, bias, kv, pad, h

    def __getitem__(self, idx):
        opt = lambda t: None if t is None else t[idx]
        return Prefix(self.emb[idx], opt(self.bias), None if self.kv is None else [(k[idx], v[idx]) for k, v in self.kv],
                      opt(self.pad), opt(self.h))

    @property
    def shape(self):
        return self.emb.shape


class LevelDecoder:
    def __init__(self, ncm, cache_prefix=True, cache_padded=False):
        """cache_prefix: cache the prompt's keys/values. A padded prompt (prefix_pad_token_id) is cached only with
        cache_padded=True, which adds the pad bias to the cached attention masks; the LM1B and TinyStories
        results were produced without it (uncached), and it is checked against generate() where it is used."""
        self.ncm = ncm
        self.cache_prefix, self.cache_padded = cache_prefix, cache_padded
        self.net, self.vqvae, self.vq = ncm.net, ncm.vqvae, ncm.vqvae.vq
        self.levels = list(self.vq.levels)
        self.K = len(self.levels)
        net = self.net
        assert not net.l0_ar and not net.use_ctx_encoder
        self.lvl_first = 0 if net.query_level0 else 1  # first level the generator predicts
        self.offsets = [sum(self.levels[self.lvl_first:k]) for k in range(self.K)]
        # Read config flags once with .get(): getattr() on a missing OmegaConf key raises and catches
        # an exception whose traceback cycle pins the caller's frame (and its cached prompt tensors).
        self.raw_token_prefix = bool(net.config.get("raw_token_prefix", False))
        self.contextual_prefix = bool(net.config.get("contextual_prefix", False))
        self.nfe = 0  # generator forward passes (one per sampled level), summed over calls

    # ---- generator -------------------------------------------------------------------------
    @torch.no_grad()
    def prefix(self, prefix_tokens):
        net, enc, P = self.net, self.vqvae.encoder, self.net.prefix_len
        if self.raw_token_prefix:
            if self.contextual_prefix and len(getattr(enc, "blocks", [])) > 0:
                z = enc.forward_nopos(prefix_tokens)[:, :P, :]
            else:
                z = enc.wte(prefix_tokens)[:, :P, :]
        else:
            z = self.vqvae.encode_z(prefix_tokens)[:, :P, :]
        e = net.transformer.wte(z)
        if not net.use_rope:
            e = e + net.wpe(net.wp_prefix)
        bias = pm = None
        if net.prefix_pad_id is not None:
            pm = prefix_tokens[:, :P] == net.prefix_pad_id
            bias = torch.where(pm[:, None, None, :], float("-inf"), 0.0).to(e.dtype)
        kv = h = None
        if self.cache_prefix and (bias is None or self.cache_padded):
            kv, h = self._prefix_kv(e, bias)
        return Prefix(e, bias, kv, pm, h)

    @torch.no_grad()
    def _prefix_kv(self, pe, bias=None):
        """The prompt's per-layer keys/values and final hidden states (the prompt attends only to its own
        non-pad positions, as in generate())."""
        net, P = self.net, self.net.prefix_len
        rope = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
        mask = torch.zeros(1, 1, P, P, device=pe.device)
        if bias is not None:
            mask = mask + bias
        h, kv = pe, []
        for block in net.transformer.blocks:
            block.attn.kv_caching(True)
            h = block(h, attn_mask=mask, rope_cos_sin=rope)
            kv.append((block.attn.cached_k, block.attn.cached_v))
            block.attn.kv_caching(False)
        return kv, h

    @torch.no_grad()
    def _cached_forward(self, x, prefix, k):
        """Run target embeddings x against the cached prompt keys/values; k is the head / RoPE index."""
        net, P, l = self.net, self.net.prefix_len, x.shape[1]
        rope = (getattr(net, f"rope_cos_gen_{k}"), getattr(net, f"rope_sin_gen_{k}")) if net.use_rope else None
        mask = torch.zeros(1, 1, l, P + l, device=x.device)
        if prefix.bias is not None:
            mask = mask + F.pad(prefix.bias, (0, l))
        for block, (pk, pv) in zip(net.transformer.blocks, prefix.kv):
            block.attn.kv_caching(True)
            block.attn.cached_k, block.attn.cached_v = pk, pv
            x = block(x, attn_mask=mask, rope_cos_sin=rope)
            block.attn.kv_caching(False)
        return net.lm_heads[k](net.transformer.ln_f(x))

    @torch.no_grad()
    def _level_pos(self, device):
        net = self.net
        if net.config.use_level_encoding:
            lpe = net.transformer.wle(net.lvl_1L)
        else:
            lpe = torch.zeros(1, net.L, net.config.n_embd, device=device)
        if not net.use_rope:
            lpe = lpe + net.wpe(net.wp_1L)
        return lpe

    @torch.no_grad()
    def logits(self, k, prefix, z_hat):
        """Logits of p(r_k | z_hat^(k-1), prompt); z_hat is the full-length reconstruction of levels < k."""
        net, P = self.net, self.net.prefix_len
        pe = prefix.emb
        B, dev = pe.shape[0], pe.device
        lpe = self._level_pos(dev)
        a, l = self.offsets[k], self.levels[k]
        hi = k - self.lvl_first  # head / RoPE / conv index
        self.nfe += 1
        if k == 0 and self.lvl_first == 1:
            assert net.predict_level0, "this model has no level-0 head: pass level 0 in codes and start at 1"
            return self._level0_logits(prefix)
        if k == 0:
            q = net.build_l0_query_input(B, pe.dtype)
            if net.use_post_upsample_conv:
                q = net._apply_post_upsample_conv_single(q, 0)
            q = net.transformer.wte(q) + lpe[:, :l]
            if prefix.kv is not None:
                return self._cached_forward(q, prefix, 0)
            h = torch.cat([pe, q], dim=1)
            m0 = torch.zeros(1, 1, P + l, P + l, device=dev)
            m0[:, :, :P, P:] = float("-inf")
            if prefix.bias is not None:
                m0 = m0 + F.pad(prefix.bias, (0, l))
            rope = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
            for block in net.transformer.blocks:
                h = block(h, attn_mask=m0, rope_cos_sin=rope)
            h = net.transformer.ln_f(h)
            return net.lm_heads[0](h[:, P:, :])
        x = F.interpolate(z_hat.transpose(-1, -2), size=(l,), mode="area").transpose(-1, -2)
        if net.use_post_upsample_conv:
            x = net._apply_post_upsample_conv_single(x, hi)
        x = net.transformer.wte(x) + lpe[:, a:a + l]
        if prefix.kv is not None:
            return self._cached_forward(x, prefix, hi)
        return net._decode_level_logits(x, pe, k - 1, prefix.bias, hidx=hi)

    @torch.no_grad()
    def _level0_logits(self, prefix):
        """predict_level0 (WMT14): level 0 is read off the prompt's last non-pad positions, as in generate()."""
        net, P = self.net, self.net.prefix_len
        h = prefix.h
        if h is None:
            rope = (net.rope_cos_gen_0, net.rope_sin_gen_0) if net.use_rope else None
            mask = torch.zeros(1, 1, P, P, device=prefix.emb.device)
            if prefix.bias is not None:
                mask = mask + prefix.bias
            h = prefix.emb
            for block in net.transformer.blocks:
                h = block(h, attn_mask=mask, rope_cos_sin=rope)
        return net.lm_head_l0(net._level0_slice(net.transformer.ln_f(h), prefix.pad))

    @torch.no_grad()
    def sample(self, logits, temperature=0.0, top_k=0, top_p=1.0):
        return self.net._sample_level_raw(logits, temperature, top_k, top_p)

    @torch.no_grad()
    def sample_ed(self, k, logits, temperature, gamma, rho):
        """Error diffusion within level k (Floyd & Steinberg 1976). Positions are scanned left to right; each one
        carries rho times the previous position's error (expected codebook vector + carried error - chosen code
        vector) and its logits are tilted by gamma * <E[c], carried error> / spread, so neighbouring codes are
        anti-correlated and their running average tracks the model's mean. Sampling is Gumbel-max with noise drawn
        once per level, so gamma = 0 is an exact sample of p(r_k | ...) and runs with the same seed share noise."""
        lg = logits.float() / temperature
        B, l, V = lg.shape
        g = -torch.empty_like(lg).exponential_().clamp_min(1e-30).log()
        if gamma == 0:
            return (lg + g).argmax(-1)
        E = self.vq.embeddings[k].weight.float()
        scale = gamma / (E - E.mean(0)).pow(2).sum(-1).mean()  # gamma = 1: one codebook spread of error ~ 1 nat
        mu = lg.softmax(-1) @ E  # (B, l, D): the model's expected code vector per position
        out = torch.empty(B, l, dtype=torch.long, device=lg.device)
        e = torch.zeros(B, E.shape[1], device=lg.device)
        for i in range(l):
            carried = rho * e
            c = (lg[:, i] + scale * (carried @ E.T) + g[:, i]).argmax(-1)
            e = mu[:, i] + carried - E[c]
            out[:, i] = c
        return out

    # ---- tokenizer -------------------------------------------------------------------------
    @torch.no_grad()
    def recon(self, k, codes_k):
        """R_k(e_k(r_k)): the level-k contribution to z_hat, at full length."""
        vq = self.vq
        q = vq.embeddings[k](codes_k)
        if k != self.K - 1:
            q = vq._up_to_full(q, k, self.levels[-1])
        return vq._apply_quant_resi(q, k)

    @torch.no_grad()
    def z_from_codes(self, codes, upto=None):
        upto = self.K if upto is None else upto
        B = codes[0].shape[0]
        z = torch.zeros(B, self.levels[-1], self.vq.embed_dim, device=codes[0].device)
        for k in range(upto):
            z = z + self.recon(k, codes[k])
        return z

    @torch.no_grad()
    def decode(self, z_hat):
        """Frozen decoder p(x | z_hat^(K)); returns argmax tokens and per-token probabilities."""
        probs = self.vqvae.decoder(z_hat).float().softmax(-1)
        return probs.argmax(-1), probs

    @torch.no_grad()
    def encode(self, tokens):
        """Restriction: tokens -> codes per level, with the frozen encoder + quantizer."""
        idx = self.vq.z_to_idx(self.vqvae.encode_z(tokens))
        return list(idx.split(self.levels, dim=1))

    @torch.no_grad()
    def encode_given_coarse(self, tokens, codes):
        """Level-local restriction: for every k, the code the quantizer picks for the residual of
        encode_z(tokens) after subtracting the *generated* reconstruction of levels < k.
        Unlike encode(), a disagreement at level k does not cascade into every finer level."""
        vq, z = self.vq, self.vqvae.encode_z(tokens).float()
        B, L, C = z.shape
        out, z_gen = [], torch.zeros_like(z)
        for k, l in enumerate(self.levels):
            rest = z - z_gen
            flat = (vq._down_for_quant(rest, k, l) if k != self.K - 1 else rest).reshape(-1, C)
            out.append(torch.argmin(vq._distances(flat, vq.embeddings[k].weight), dim=1).view(B, l))
            z_gen = z_gen + self.recon(k, codes[k])
        return out

    # ---- sampling --------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, prefix, start=0, codes=None, keep=None, temperature=0.0, top_k=0, top_p=1.0,
            conf=None, ed=None):
        """Sample levels start..K-1 coarse to fine.

        start: int, or LongTensor (B,) for a per-board start level (K = leave the board as given).
        codes: per-level codes; levels < start are taken from it as given.
        keep:  optional {k: bool (B, l_k)}; at a sampled level k, positions where keep[k] is True
               keep codes[k] and the rest are resampled (valid because codes are conditionally
               independent within a level).
        conf:  optional list; receives per level the generator's probability of the chosen code.
        ed:    optional (gamma, rho): sample each level with sample_ed (temperature > 0, no top-k/top-p); None keeps
               the released sampler unchanged.
        Returns (codes, z_hat^(K)). self.nfe counts one forward per level per batch, so with a
        per-board start use board_nfe(start) for honest per-board accounting.
        """
        B, dev = prefix.shape[0], prefix.emb.device
        start = torch.full((B,), int(start), device=dev) if not torch.is_tensor(start) else start.to(dev)
        z = torch.zeros(B, self.levels[-1], self.vq.embed_dim, device=dev)
        out = []
        for k in range(self.K):
            given = start > k
            if bool(given.all()):
                c = codes[k]
                if conf is not None:
                    conf.append(torch.ones(c.shape, device=dev))
            else:
                lg = self.logits(k, prefix, z)
                if ed is None:
                    c = self.sample(lg, temperature, top_k, top_p)
                else:
                    assert temperature > 0 and top_k == 0 and top_p >= 1.0, "error diffusion samples the full softmax"
                    c = self.sample_ed(k, lg, temperature, *ed)
                if keep is not None and k in keep:
                    c = torch.where(keep[k], codes[k], c)
                if codes is not None and bool(given.any()):
                    c = torch.where(given[:, None], codes[k], c)
                if conf is not None:
                    conf.append(lg.float().softmax(-1).gather(-1, c[..., None]).squeeze(-1))
            out.append(c)
            z = z + self.recon(k, c)
        return out, z

    def board_nfe(self, start):
        """Generator forwards each board needs when sampling levels start..K-1."""
        return (self.K - start).clamp(min=0)
