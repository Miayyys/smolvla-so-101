"""Advantage-conditioned SmolVLA, as in RECAP (pi*0.6, arXiv 2511.14759, Sec. IV-B and V-B).

Each training frame carries a binary improvement indicator I_t (advantage above a per-task threshold). It enters
the model as text, "Advantage: positive" or "Advantage: negative", embedded like the task instruction and placed in
the prefix after the instruction and before the state token, in an attention block of its own:

    [images + instruction] (block 0, bidirectional) | [indicator] (block 1) | [state] (block 2) | [actions]

Image and instruction tokens cannot attend to it (make_att_2d_masks: a token sees tokens whose cumulative block
number is not larger than its own), so their features are the same for either label; the state token and the
action expert see it. This is the paper's placement ("such that only the action log-likelihoods are affected").

A frame whose indicator is dropped (NONE, the paper's random omission so that the unconditioned policy is learned
too) gets the same tokens with the padding mask off: no position ids, nothing attends to them, and the model computes
exactly what plain SmolVLA computes. Inference uses DEFAULT_INDICATOR (positive), the paper's beta = 1.

The patch is on the class (VLAFlowMatching.embed_prefix) and only acts on models with ``advantage_tokens`` set by
``enable``; others are unchanged. A checkpoint trained this way has MARKER in its folder; ``load_policy`` and
so101vla.rollout enable conditioning for such checkpoints.
"""

import json
import math
from pathlib import Path

import torch

MARKER = 'advantage.json'
TEXT = ('Advantage: negative', 'Advantage: positive')  # index = indicator value
NONE = -1
DEFAULT_INDICATOR = 1


def install():
    """Patch VLAFlowMatching.embed_prefix once (no effect on models without advantage tokens)."""
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    if getattr(VLAFlowMatching, '_advantage_installed', False):
        return
    original = VLAFlowMatching.embed_prefix

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pad_masks, att_masks = original(self, images, img_masks, lang_tokens, lang_masks, state=state)
        tokens = getattr(self, 'advantage_tokens', None)
        if tokens is None:
            return embs, pad_masks, att_masks
        if state.ndim != 2 or self.prefix_length > 0:
            raise ValueError('advantage conditioning expects one state token last in an unpadded prefix')
        ids, valid = (t.to(embs.device) for t in tokens)
        batch = embs.shape[0]
        indicator = getattr(self, 'advantage', None)
        if indicator is None:
            indicator = torch.full((batch,), DEFAULT_INDICATOR, dtype=torch.long)
        indicator = indicator.to(ids.device).long()
        if indicator.shape != (batch,) or not ((indicator >= NONE) & (indicator <= 1)).all():
            raise ValueError(f'advantage indicator must be ({batch},) in {{-1, 0, 1}}, got {indicator}')
        chosen = ids[indicator.clamp(min=0)]
        mask = valid[indicator.clamp(min=0)] & (indicator >= 0)[:, None]  # NONE: all padding
        emb = self.vlm_with_expert.embed_language_tokens(chosen)
        emb = (emb * math.sqrt(emb.shape[-1])).to(embs.dtype)
        block = torch.zeros(batch, ids.shape[1], dtype=att_masks.dtype, device=att_masks.device)
        block[:, 0] = 1
        return (torch.cat([embs[:, :-1], emb, embs[:, -1:]], 1),
                torch.cat([pad_masks[:, :-1], mask.to(pad_masks.device), pad_masks[:, -1:]], 1),
                torch.cat([att_masks[:, :-1], block, att_masks[:, -1:]], 1))

    VLAFlowMatching.embed_prefix = embed_prefix
    VLAFlowMatching._advantage_installed = True


def token_table(tokenizer_dir):
    """(ids (2, L), valid (2, L)) for TEXT, right-padded to the longer one."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    encoded = [tokenizer(text, add_special_tokens=False)['input_ids'] for text in TEXT]
    length = max(map(len, encoded))
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.tensor([e + [pad] * (length - len(e)) for e in encoded])
    valid = torch.tensor([[True] * len(e) + [False] * (length - len(e)) for e in encoded])
    return ids, valid


def enable(policy, tokenizer_dir):
    """Turn on advantage conditioning for this policy (inference default: positive). No parameters are added."""
    install()
    policy.model.advantage_tokens = token_table(tokenizer_dir)
    policy.model.advantage = None
    return policy


def install_for_checkpoint(path):
    """For loaders outside this repository (lerobot-rollout): every SmolVLAPolicy built from now on is conditioned
    if the checkpoint at ``path`` was trained with indicators. Returns whether it was."""
    if not marked(path):
        return False
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    tokenizer_dir = json.loads((Path(path) / MARKER).read_text())['tokenizer_dir']
    original = SmolVLAPolicy.__init__

    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        enable(self, tokenizer_dir)

    SmolVLAPolicy.__init__ = __init__
    return True


def set_indicator(policy, indicator):
    """Per-frame indicators for the next forward pass (a (B,) tensor of -1/0/1), or None for the default."""
    policy.model.advantage = indicator


def write_marker(folder, tokenizer_dir, **info):
    (Path(folder) / MARKER).write_text(json.dumps({'text': TEXT, 'none': NONE, 'default': DEFAULT_INDICATOR,
                                                   'tokenizer_dir': str(tokenizer_dir), **info}, indent=2) + '\n')


def marked(folder):
    return (Path(folder) / MARKER).exists()
