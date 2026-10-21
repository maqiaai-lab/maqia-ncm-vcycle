"""Load a released NCM checkpoint (frozen tokenizer + next-concept generator).

Mirrors scripts/eval_shard.py in HazyResearch/ncm, with two differences:
checkpoints are loaded with weights_only=True, and the device is configurable (CPU works).
"""
import contextlib
import io
import os
import sys
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCM_ROOT = os.path.abspath(os.environ.get("NCM_ROOT", os.path.join(REPO_ROOT, "third_party", "ncm")))


def import_ncm():
    if not os.path.isdir(os.path.join(NCM_ROOT, "models")):
        raise RuntimeError(f"NCM code not found at {NCM_ROOT}; run ./setup.sh or set NCM_ROOT")
    if NCM_ROOT not in sys.path:
        sys.path.insert(0, NCM_ROOT)


@dataclass
class NCM:
    cfg: object
    vqvae: torch.nn.Module
    net: torch.nn.Module
    pre_tokenizer: object
    device: str


def load(checkpoint_dir, device="cpu", quiet=True):
    import_ncm()
    import hydra
    from models.next_concept import NextConceptTransformer
    from models.vqvae import VQVAE
    from utils.registry import tokenizers

    cfg = OmegaConf.load(os.path.join(checkpoint_dir, "config.yaml"))
    tk_cfg = cfg.tokenizer
    # BPE tokenizers (TinyStories) read their vocabulary and merges from the checkpoint directory, as in
    # scripts/eval_shard.py. These are pickles: check new ones with pickletools before loading them.
    for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
        if cfg.checkpoints.get(name):
            tk_cfg.pre_tokenizer[key] = os.path.join(checkpoint_dir, cfg.checkpoints[name])
    pre_tokenizer = hydra.utils.get_class(tokenizers[tk_cfg.pre_tokenizer.name]).from_config(tk_cfg.pre_tokenizer)

    out = io.StringIO()
    with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
        vqvae = VQVAE(**tk_cfg.vqvae)
        sd = torch.load(os.path.join(checkpoint_dir, cfg.checkpoints.tokenizer), map_location="cpu", weights_only=True)
        vqvae.load_state_dict(sd["model"], strict=True)
        vqvae = vqvae.to(device).eval()
        net = NextConceptTransformer(vqvae_local=vqvae, config=cfg.model)
        sd = torch.load(os.path.join(checkpoint_dir, cfg.checkpoints.generator), map_location="cpu", weights_only=True)
        net.load_state_dict(sd["model"], strict=True)
        net = net.to(device).eval()
    return NCM(cfg=cfg, vqvae=vqvae, net=net, pre_tokenizer=pre_tokenizer, device=device)
