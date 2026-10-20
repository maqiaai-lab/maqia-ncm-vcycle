"""Level-by-level NCM decoding with the hooks a V-cycle needs.

Notation follows the paper: r_k are the codes at level k (length l_k), z_hat^(k) is the
cumulative reconstruction sum_{j<=k} R_j(e_j(r_j)), the generator gives p(r_k | z_hat^(k-1))
with codes independent within a level, and the frozen decoder maps z_hat^(K) to tokens.

`LevelDecoder.logits` reproduces one step of NextConceptTransformer.generate exactly
(checked against it in experiments/phase0_baseline.py), so everything built from it
(starting mid-hierarchy, masked resampling, re-encoding) reuses the released weights unchanged.
Supported configs: query_level0 (level 0 from the learned queries, or given), no CFG, no context
encoder (this covers the released Sudoku, LM1B, OWT and TinyStories models).
"""
import torch
import torch.nn.functional as F


class Prefix:
    """Embedded prompt plus its pad-attention bias; slicing keeps the two aligned."""

    def __init__(self, emb, bias=None):
        self.emb, self.bias = emb, bias

    def __getitem__(self, idx):
        return Prefix(self.emb[idx], None if self.bias is None else self.bias[idx])

    @property
    def shape(self):
        return self.emb.shape


class LevelDecoder:
    def __init__(self, ncm):
        self.ncm = ncm
        self.net, self.vqvae, self.vq = ncm.net, ncm.vqvae, ncm.vqvae.vq
        self.levels = list(self.vq.levels)
        self.K = len(self.levels)
        self.offsets = [sum(self.levels[:k]) for k in range(self.K)]
        net = self.net
        assert net.query_level0 and not net.l0_ar, "needs query_level0 generation"
        assert not net.use_ctx_encoder
        self.nfe = 0  # generator forward passes (one per sampled level), summed over calls

    # ---- generator -------------------------------------------------------------------------
    @torch.no_grad()
    def prefix(self, prefix_tokens):
        net, enc, P = self.net, self.vqvae.encoder, self.net.prefix_len
        if getattr(net.config, "raw_token_prefix", False):
            if getattr(net.config, "contextual_prefix", False) and len(getattr(enc, "blocks", [])) > 0:
                z = enc.forward_nopos(prefix_tokens)[:, :P, :]
            else:
                z = enc.wte(prefix_tokens)[:, :P, :]
        else:
            z = self.vqvae.encode_z(prefix_tokens)[:, :P, :]
        e = net.transformer.wte(z)
        if not net.use_rope:
            e = e + net.wpe(net.wp_prefix)
        bias = None
        if net.prefix_pad_id is not None:
            pm = prefix_tokens[:, :P] == net.prefix_pad_id
            bias = torch.where(pm[:, None, None, :], float("-inf"), 0.0).to(e.dtype)
        return Prefix(e, bias)

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
        self.nfe += 1
        if k == 0:
            q = net.build_l0_query_input(B, pe.dtype)
            if net.use_post_upsample_conv:
                q = net._apply_post_upsample_conv_single(q, 0)
            q = net.transformer.wte(q) + lpe[:, :l]
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
            x = net._apply_post_upsample_conv_single(x, k)
        x = net.transformer.wte(x) + lpe[:, a:a + l]
        return net._decode_level_logits(x, pe, k - 1, prefix.bias, hidx=k)

    @torch.no_grad()
    def sample(self, logits, temperature=0.0, top_k=0, top_p=1.0):
        return self.net._sample_level_raw(logits, temperature, top_k, top_p)

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
            conf=None):
        """Sample levels start..K-1 coarse to fine.

        start: int, or LongTensor (B,) for a per-board start level (K = leave the board as given).
        codes: per-level codes; levels < start are taken from it as given.
        keep:  optional {k: bool (B, l_k)}; at a sampled level k, positions where keep[k] is True
               keep codes[k] and the rest are resampled (valid because codes are conditionally
               independent within a level).
        conf:  optional list; receives per level the generator's probability of the chosen code.
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
                c = self.sample(lg, temperature, top_k, top_p)
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
