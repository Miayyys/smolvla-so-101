"""Export a SmolVLA checkpoint as three fixed-shape ONNX graphs for RK3588, plus real-frame calibration and test data.

  uv run python -m so101vla.rknn_export --model-dir runs/so101_full_v5/best --output runs/rknn_v5

Record of results and board pitfalls: docs/experiments/edge-rk3588.md. Graphs (batch 1, two cameras at the checkpoint's
resize_imgs_with_padding; prefix P = 2 x 64 image + --lang-slots language (default 16; the tokenizer pads to 48) + 1 state):
  vision.onnx  pixel [1,3,S,S] in [-1,1] -> image tokens [1,64,960] (connector output x sqrt(960), as in embed_prefix)
  prefix.onnx / expert.onnx with --graph fast (default): the NPU-oriented rewrite in so101vla/rknn_fast.py (inputs
               include RoPE cos/sin tables; the prefix outputs the K/V the expert attends to, [1,5,P,64] per layer)
  prefix.onnx / expert.onnx with --graph lerobot: LeRobot's own modules (positions as float inputs, DynamicCache K/V)
The CPU glue that stays outside the graphs: image resize/pad/normalize, tokenizer and embedding lookup, state
normalization and state_proj, time embedding (and RoPE tables), the Euler loop and action unnormalization. Masks are
float 0/1 inputs (no BOOL/INT64 graph inputs).

Export-only rewrites, all checked against LeRobot in FP32 before export (relative error < 1e-5; prefix K/V on valid
positions): RoPE without slice assignment (ScatterND); attention mask fill -1e4 instead of float32 min (-inf in FP16);
the float mask cast to bool (--mask-op; `mask > 0.5` fails in INT8 RKNN on the board); language padding slots beyond
--lang-slots dropped (always masked, so outputs are unchanged); optional --scale-norms (norms on x / max|x|; not needed).

All references come from LeRobot's own ``predict_action_chunk`` on CPU in FP32 (hooks record every module boundary).
Frames come from the checkpoint's own real-arm rollout datasets. Graph post-processing (ReduceMin rewrite, checker)
and RKNN conversion run in .venv-rknn: scripts/rknn_compile.py.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

KV_NAMES = [f'{kind}_{i}' for i in range(16) for kind in ('key', 'value')]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def export_rope(x, positions, max_wavelength=10_000):
    """LeRobot's apply_rope without the in-place slice writes (they lower to ScatterND); same arithmetic."""
    dtype = x.dtype
    half = x.shape[-1] // 2
    v = x.to(torch.float32)
    exponents = (2.0 / x.shape[-1]) * torch.arange(half, dtype=torch.float32, device=x.device)
    radians = positions[..., None].to(torch.float32) / (max_wavelength ** exponents)[None, None, :]
    sin = torch.sin(radians)[..., None, :]
    cos = torch.cos(radians)[..., None, :]
    a, b = v.split(half, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1).to(dtype)


def export_attention(self, attention_mask, batch_size, head_dim, query_states, key_states, value_states):
    """LeRobot's eager_attention_forward with mask value -1e4 instead of float32 min.

    float32 min overflows to -inf in FP16; prefix rows of padded language tokens are fully masked, so their softmax
    becomes NaN and 0 x NaN then spreads through every later layer. In FP32 the result is unchanged: exp(-1e4) is 0
    for rows with a valid entry, and a fully masked row is uniform either way. (A multiplicative mask with a masked-max
    softmax was tried on 2026-11-07: it needs ~10 extra ops per attention, ran 2x slower and was wrong on the board
    even in FP16.)
    """
    groups = self.num_attention_heads // self.num_key_value_heads
    seq = key_states.shape[1]
    k = key_states[:, :, :, None, :].expand(batch_size, seq, self.num_key_value_heads, groups, head_dim)
    k = k.reshape(batch_size, seq, self.num_key_value_heads * groups, head_dim)
    v = value_states[:, :, :, None, :].expand(batch_size, seq, self.num_key_value_heads, groups, head_dim)
    v = v.reshape(batch_size, seq, self.num_key_value_heads * groups, head_dim)
    q = query_states.to(torch.float32).transpose(1, 2)
    k = k.to(torch.float32).transpose(1, 2)
    w = torch.matmul(q, k.transpose(2, 3)) * head_dim ** -0.5
    w = torch.where(attention_mask[:, None, :, :], w, torch.tensor(-1e4, dtype=torch.float32))
    probs = torch.nn.functional.softmax(w, dim=-1).to(v.dtype)
    out = torch.matmul(probs, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
    return out.reshape(batch_size, -1, self.num_key_value_heads * groups * head_dim)


MASK_OP = 'cast'


def to_bool(mask):
    """Float 0/1 graph input -> bool. 'greater' (mask > 0.5) fails in INT8 RKNN on the board: the input is INT8 and the
    constant float, and the CPU fallback rejects that pair (the whole attention mask is then garbage, and rknnlite
    reports no error). 'cast' (Cast to bool, default) and 'greater_self' (mask > mask * 0.5) both work: identical
    results on the board in FP16 and INT8, 2026-11-07."""
    if MASK_OP == 'cast':
        return mask.to(torch.bool)
    if MASK_OP == 'greater_self':
        return mask > mask * 0.5
    return mask > 0.5


def scaled_rms_norm(self, x):
    """LlamaRMSNorm computed on x / max|x| per token: x/m * rsqrt(mean((x/m)^2) + eps/m^2) == x * rsqrt(mean(x^2) + eps).

    The wrist-camera image tokens reach ~14,000 in a few channels (connector output x sqrt(960)), so x^2 overflows FP16
    (max 65504) and RKNN FP16 broke those tokens and everything attending to them. Dividing by the token's own max keeps
    every square <= 1; if m*m overflows, eps/m^2 becomes 0, which is the right limit.
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    m = x.abs().amax(-1, keepdim=True) + 1e-6
    y = x / m
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.variance_epsilon / (m * m))
    return (self.weight * y.to(dtype)).contiguous()


def scaled_layer_norm(self, x):
    """nn.LayerNorm with the same per-token max scaling of the centred input (vision post_layernorm input reaches ~955)."""
    c = x - x.mean(-1, keepdim=True)
    m = c.abs().amax(-1, keepdim=True) + 1e-6
    y = c / m
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps / (m * m))
    return (y * self.weight + self.bias).contiguous()  # nn.LayerNorm output is contiguous; pixel_shuffle uses view


def patch_norms(model):
    """Instance-level forward overrides for every norm; returns an undo function."""
    import types
    patched = []
    for m in model.modules():
        name = type(m).__name__
        if name == 'LlamaRMSNorm':
            m.forward = types.MethodType(scaled_rms_norm, m); patched.append(m)
        elif isinstance(m, torch.nn.LayerNorm):
            m.forward = types.MethodType(scaled_layer_norm, m); patched.append(m)

    def undo():
        for m in patched:
            del m.forward
    return len(patched), undo


class Vision(torch.nn.Module):
    """embed_image plus the sqrt(dim) scaling from embed_prefix, for a full (unpadded-patch) square image."""

    def __init__(self, policy):
        super().__init__()
        vlm = policy.model.vlm_with_expert.get_vlm_model()
        self.vision, self.connector = vlm.vision_model, vlm.connector

    def forward(self, pixel):
        emb = self.vision.embeddings
        patch = self.vision.patch_size
        h, w = pixel.shape[2] // patch, pixel.shape[3] // patch
        side = emb.num_patches_per_side
        positions = (torch.arange(h)[:, None] * side + torch.arange(w)[None, :]).reshape(1, -1)
        x = emb.patch_embedding(pixel).flatten(2).transpose(1, 2) + emb.position_embedding(positions)
        x = self.vision.post_layernorm(self.vision.encoder(inputs_embeds=x, attention_mask=None).last_hidden_state)
        x = self.connector(x)
        return x * math.sqrt(x.shape[-1])


class Prefix(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.module = policy.model.vlm_with_expert

    def forward(self, prefix, attention_mask, positions):
        _, cache = self.module.forward(attention_mask=to_bool(attention_mask), position_ids=positions,
                                       past_key_values=None, inputs_embeds=[prefix, None], use_cache=True)
        out = []
        for layer in cache.layers:
            out.extend((layer.keys, layer.values))
        return tuple(out)


class Expert(torch.nn.Module):
    """One denoise_step with the time embedding computed outside; the suffix attention mask is a constant."""

    def __init__(self, policy, suffix_mask):
        super().__init__()
        self.core = policy.model
        self.register_buffer('suffix_mask', suffix_mask)

    def forward(self, x_t, time_embedding, prefix_pad, suffix_positions, *kv):
        from transformers import DynamicCache
        core = self.core
        cache = DynamicCache()
        for i in range(16):
            cache.update(kv[2 * i], kv[2 * i + 1], i)
        action = core.action_in_proj(x_t)
        joined = torch.cat((action, time_embedding[:, None, :].expand_as(action)), dim=2)
        suffix = core.action_time_mlp_out(torch.nn.functional.silu(core.action_time_mlp_in(joined)))
        n = suffix.shape[1]
        mask = torch.cat((to_bool(prefix_pad[:, None, :].expand(1, n, prefix_pad.shape[1])), self.suffix_mask), dim=2)
        out, _ = core.vlm_with_expert.forward(attention_mask=mask, position_ids=suffix_positions, past_key_values=cache,
                                              inputs_embeds=[None, suffix], use_cache=True)
        return core.action_out_proj(out[1][:, -core.config.chunk_size:].to(torch.float32))


def pick_frames(roots, calib, test, seed):
    """Calibration frames from all but the last 4 episodes of each dataset; test frames from those last 4."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    rng = np.random.default_rng(seed)
    picks = []
    for di, root in enumerate(roots):
        ds = LeRobotDataset('local/' + Path(root).name, root=Path(root).resolve(), video_backend='pyav')
        ep = np.asarray(ds.hf_dataset['episode_index'])
        episodes = np.unique(ep)
        held = episodes[-4:]
        for split, count, pool in [('calib', calib // len(roots), episodes[:-4]), ('test', test // len(roots), held)]:
            rows = np.flatnonzero(np.isin(ep, pool))
            for r in sorted(rng.choice(rows, count, replace=False).tolist()):
                picks.append({'split': split, 'dataset': str(root), 'row': int(r), 'episode': int(ep[r]), '_ds': ds})
    return picks


def raw_observation(item):
    return {'observation.images.camera1': item['observation.images.top'][None].float(),
            'observation.images.camera2': item['observation.images.wrist'][None].float(),
            'observation.state': item['observation.state'][None].float(),
            'task': [item['task']]}


def run_reference(policy, pre, post, raw, noise):
    """LeRobot's predict_action_chunk with hooks recording every boundary the graphs will use."""
    core = policy.model
    vwe = core.vlm_with_expert
    rec = {'vision_in': [], 'vision_out': [], 'steps': []}
    old_embed, old_forward, old_step = vwe.embed_image, vwe.forward, core.denoise_step
    old_prefix = core.embed_prefix

    def embed_image(image):
        out = old_embed(image)
        rec['vision_in'].append(image.detach().clone())
        rec['vision_out'].append((out * math.sqrt(out.shape[-1])).detach().clone())
        return out

    def embed_prefix(*a, **kw):
        embs, pad, att = old_prefix(*a, **kw)
        rec['prefix_pad'] = pad.detach().clone()
        return embs, pad, att

    def forward(*a, **kw):
        out, cache = old_forward(*a, **kw)
        if kw['inputs_embeds'][1] is None:
            rec['prefix'] = kw['inputs_embeds'][0].detach().clone()
            rec['attention_mask'] = kw['attention_mask'].detach().clone()
            rec['positions'] = kw['position_ids'].detach().clone()
            rec['kv'] = [t.detach().clone() for layer in cache.layers for t in (layer.keys, layer.values)]
        return out, cache

    def denoise_step(prefix_pad_masks, past_key_values, x_t, timestep):
        v = old_step(prefix_pad_masks, past_key_values, x_t, timestep)
        rec['steps'].append({'x_t': x_t.detach().clone(), 'timestep': timestep.detach().clone(), 'v_t': v.detach().clone()})
        return v

    vwe.embed_image, vwe.forward, core.denoise_step, core.embed_prefix = embed_image, forward, denoise_step, embed_prefix
    try:
        with torch.inference_mode():
            policy.reset()
            batch = pre(raw)
            normalized = policy.predict_action_chunk(batch, noise=noise)
            rec['actions_normalized'] = normalized.detach().clone()
            rec['actions'] = post(normalized).detach().clone()
    finally:
        vwe.embed_image, vwe.forward, core.denoise_step, core.embed_prefix = old_embed, old_forward, old_step, old_prefix
    return rec


def time_embedding(policy, timestep):
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding
    core = policy.model
    return create_sinusoidal_pos_embedding(timestep, core.vlm_with_expert.expert_hidden_size, core.config.min_period,
                                           core.config.max_period, device=torch.device('cpu')).to(torch.float32)


def graph_inputs(policy, rec):
    """Float inputs for each graph, derived from one recorded reference."""
    pad = rec['prefix_pad'].to(torch.float32)
    # LeRobot's denoise_step: prefix_offsets + cumsum(ones) - 1 = valid prefix count + arange(chunk).
    suffix_positions = pad.sum(-1, keepdim=True) + torch.arange(policy.config.chunk_size, dtype=torch.float32)[None]
    vision = [x.to(torch.float32) for x in rec['vision_in']]
    # Rows of padded language tokens have no valid key; give them all valid keys (export_attention needs a non-zero
    # row sum). Only their own K/V change, which every later query masks out.
    mask = rec['attention_mask'].to(torch.float32)
    empty = mask.sum(-1, keepdim=True) == 0
    mask = torch.where(empty, pad[:, None, :].expand_as(mask), mask)
    prefix = (rec['prefix'].to(torch.float32), mask, rec['positions'].to(torch.float32))
    expert = [(s['x_t'].to(torch.float32), time_embedding(policy, s['timestep']), pad, suffix_positions, *rec['kv'])
              for s in rec['steps']]
    return vision, prefix, expert


def trim_language(rec, lang_slots, image_tokens=128):
    """Drop language padding slots beyond lang_slots (tokenizer pads to 48; both task prompts use 9).

    Padded slots are masked out for every query, positions are cumsum(pad) - 1, and the state token keeps its
    position, so outputs for valid tokens are unchanged; the expert check against LeRobot's v_t confirms it."""
    p = rec['prefix'].shape[1]
    keep = list(range(image_tokens + lang_slots)) + [p - 1]
    dropped = rec['prefix_pad'][0, image_tokens + lang_slots:p - 1]
    if dropped.any():
        raise ValueError(f'{int(dropped.sum())} language tokens beyond --lang-slots {lang_slots}')
    out = dict(rec)
    out['prefix'] = rec['prefix'][:, keep]
    out['attention_mask'] = rec['attention_mask'][:, keep][:, :, keep]
    out['positions'] = rec['positions'][:, keep]
    out['prefix_pad'] = rec['prefix_pad'][:, keep]
    out['kv'] = [t[:, :, keep] for t in rec['kv']]
    return out


def rel_err(a, b):
    """max |a - b| / max |b| (image tokens reach ~14,000, so an absolute FP32 tolerance would not mean much)."""
    a, b = a.to(torch.float32), b.to(torch.float32)
    return float((a - b).abs().max() / b.abs().max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-dir', type=Path, default=Path('runs/so101_full_v5/best'))
    p.add_argument('--vlm-assets-dir', type=Path, default=Path('artifacts/transfer/smolvlm2_assets'))
    p.add_argument('--datasets', type=Path, nargs='+',
                   default=[Path('artifacts/datasets/rollout_full_v5_doll_raised'), Path('artifacts/datasets/rollout_full_v5_key_raised')])
    p.add_argument('--output', type=Path, default=Path('runs/rknn_v5'))
    p.add_argument('--calib-frames', type=int, default=24)
    p.add_argument('--test-frames', type=int, default=8)
    p.add_argument('--expert-calib-per-step', type=int, default=4, help='expert calibration samples per denoising step')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--graph', default='fast', choices=['fast', 'lerobot'],
                   help='fast: NPU-oriented prefix/expert rewrite (so101vla/rknn_fast.py); lerobot: LeRobot modules as is')
    p.add_argument('--lang-slots', type=int, default=16,
                   help='language slots kept in the prefix (48 = no trimming); prefix length = 128 + this + 1')
    p.add_argument('--mask-op', default='cast', choices=['greater', 'cast', 'greater_self'], help=to_bool.__doc__)
    p.add_argument('--scale-norms', action='store_true',
                   help='compute norms on x / max|x| (FP16 overflow guard; not needed so far: the large board error '
                        'came from 3-core execution, and this rewrite made every graph slower)')
    a = p.parse_args()
    global MASK_OP
    MASK_OP = a.mask_op
    from types import SimpleNamespace
    import lerobot.policies.smolvla.smolvlm_with_expert as source
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from so101vla.policy import load_policy

    torch.manual_seed(a.seed)
    out = a.output
    for d in ['onnx', 'calib/vision', 'calib/prefix', 'calib/expert', 'test']:
        (out / d).mkdir(parents=True, exist_ok=True)
    policy, pre, post = load_policy(SimpleNamespace(model_dir=a.model_dir, vlm_assets_dir=a.vlm_assets_dir, device='cpu'))
    policy.float().eval()
    policy.model.vlm_with_expert.vlm.set_attn_implementation('eager')
    cfg = policy.config
    action_dim = cfg.action_feature.shape[0]

    picks = pick_frames(a.datasets, a.calib_frames, a.test_frames, a.seed)
    records = []
    for i, pk in enumerate(picks):
        item = pk.pop('_ds')[pk['row']]
        pk['task'] = item['task']
        noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(1000 + i))
        rec = run_reference(policy, pre, post, raw_observation(item), noise)
        rec['noise'] = noise
        assert len(rec['vision_in']) == 2 and len(rec['steps']) == cfg.num_steps, (len(rec['vision_in']), len(rec['steps']))
        records.append(rec)
        print(f'reference {i + 1}/{len(picks)} {pk["split"]} {Path(pk["dataset"]).name} row {pk["row"]}', flush=True)

    # Wrappers must reproduce the recorded boundaries before export (first frame: exact; others spot-checked below).
    prefix_len = 128 + a.lang_slots + 1
    suffix_mask = make_att_2d_masks(torch.ones(1, cfg.chunk_size, dtype=torch.bool),
                                    torch.ones(1, cfg.chunk_size, dtype=torch.float32))
    from so101vla import rknn_fast
    fast = a.graph == 'fast'
    vision_w = Vision(policy).eval()
    if fast:
        prefix_w = rknn_fast.FastPrefix(policy, to_bool).eval()
        expert_w = rknn_fast.FastExpert(policy, suffix_mask, to_bool).eval()
        kv_names = rknn_fast.FAST_KV_NAMES
        prefix_in = ['prefix', 'attention_mask', 'cos', 'sin']
        expert_in = ['x_t', 'time_embedding', 'prefix_pad', 'cos_self', 'sin_self', *kv_names]
    else:
        prefix_w, expert_w = Prefix(policy).eval(), Expert(policy, suffix_mask).eval()
        kv_names = KV_NAMES
        prefix_in = ['prefix', 'attention_mask', 'positions']
        expert_in = ['x_t', 'time_embedding', 'prefix_pad', 'suffix_positions', *KV_NAMES]

    def inputs(rec):
        """(vision, prefix, expert per step, prefix output references) for the selected graph form."""
        rec = trim_language(rec, a.lang_slots)
        vision, prefix, expert = graph_inputs(policy, rec)
        if not fast:
            return vision, prefix, expert, rec['kv']
        ref_kv = rknn_fast.fast_reference_kv(policy, rec)
        cos_p, sin_p = rknn_fast.rope_tables(prefix[2])
        cos_s, sin_s = rknn_fast.rope_tables(expert[0][3])
        return (vision, (prefix[0], prefix[1], cos_p, sin_p),
                [(e[0], e[1], e[2], cos_s, sin_s, *ref_kv) for e in expert], ref_kv)

    checks = {}
    import types
    vwe = policy.model.vlm_with_expert
    old_rope = source.apply_rope
    source.apply_rope = export_rope
    vwe.eager_attention_forward = types.MethodType(export_attention, vwe)  # instance attribute shadows the method
    scaled_norms, undo_norms = patch_norms(policy.model) if a.scale_norms else (0, lambda: None)
    try:
        with torch.inference_mode():
            for k, rec in enumerate(records[:3]):
                vision, prefix, expert, ref_kv = inputs(rec)
                valid = trim_language(rec, a.lang_slots)['prefix_pad'][0]
                checks[f'frame{k}'] = {
                    'vision': max(rel_err(vision_w(x), y) for x, y in zip(vision, rec['vision_out'])),
                    # K/V at padded positions differ by design (see graph_inputs) and are never used.
                    'prefix': max(rel_err(x[:, :, valid], y[:, :, valid]) for x, y in zip(prefix_w(*prefix), ref_kv)),
                    'expert': max(rel_err(expert_w(*e), s['v_t']) for e, s in zip(expert, rec['steps']))}
            print('wrapper vs LeRobot, max abs error / max abs reference:', json.dumps(checks), flush=True)
            # FP16 range check: a norm squares its input, which overflows FP16 (max 65504) once |x| > 255.9.
            norm_max = {}
            def hook(name):
                def f(module, args, output):
                    norm_max[name] = max(norm_max.get(name, 0.0), float(args[0].abs().max()))
                return f
            handles = [m.register_forward_hook(hook(n)) for n, m in policy.model.named_modules()
                       if type(m).__name__.endswith(('RMSNorm', 'LayerNorm'))]
            for rec in records[:3]:
                vision, prefix, expert, _ = inputs(rec)
                for x in vision:
                    vision_w(x)
                prefix_w(*prefix)
                for e in expert:
                    expert_w(*e)
            for h in handles:
                h.remove()
            checks['norm_input_max_abs_top'] = dict(sorted(norm_max.items(), key=lambda kv: -kv[1])[:12])
            checks['norm_inputs_over_255'] = sum(v > 255.9 for v in norm_max.values())
            print('largest norm inputs:', json.dumps(checks['norm_input_max_abs_top'], indent=1), flush=True)
            worst = max(v for k, c in checks.items() if k.startswith('frame') for v in c.values())
            if worst > 1e-5:
                raise ValueError(f'wrapper differs from LeRobot reference by {worst}')
            vision, prefix, expert, _ = inputs(records[0])
            exports = [('vision', vision_w, (vision[0],), ['pixel'], ['image_tokens']),
                       ('prefix', prefix_w, prefix, prefix_in, kv_names),
                       ('expert', expert_w, expert[0], expert_in, ['velocity'])]
            for name, module, example, in_names, out_names in exports:
                path = out / 'onnx' / f'{name}.onnx'
                torch.onnx.export(module, tuple(example), str(path), dynamo=False, opset_version=17,
                                  input_names=in_names, output_names=out_names, do_constant_folding=True)
                print('exported', path, path.stat().st_size, flush=True)
    finally:
        source.apply_rope = old_rope
        del vwe.eager_attention_forward
        undo_norms()

    # Calibration sets (RKNN dataset.txt: one sample per line, one .npy per graph input, in input order).
    def write_sample(folder, index, names, tensors):
        files = []
        for n, t in zip(names, tensors):
            f = (out / 'calib' / folder / f'{index:03d}_{n}.npy').resolve()
            np.save(f, t.detach().cpu().numpy().astype(np.float32))
            files.append(str(f))
        return ' '.join(files)

    calib = [(pk, rec) for pk, rec in zip(picks, records) if pk['split'] == 'calib']
    lines = {'vision': [], 'prefix': [], 'expert': []}
    for j, (pk, rec) in enumerate(calib):
        vision, prefix, expert, _ = inputs(rec)
        for c, x in enumerate(vision):
            lines['vision'].append(write_sample('vision', 2 * j + c, ['pixel'], [x]))
        lines['prefix'].append(write_sample('prefix', j, prefix_in, prefix))
    # Expert: every denoising step equally represented (the input distribution changes with the step).
    expert_samples = []
    for step in range(cfg.num_steps):
        for j in range(a.expert_calib_per_step):
            frame = (step * a.expert_calib_per_step + j) % len(calib)
            _, _, expert, _ = inputs(calib[frame][1])
            lines['expert'].append(write_sample('expert', len(expert_samples), expert_in, expert[step]))
            expert_samples.append({'frame': frame, 'step': step})
    for name, rows in lines.items():
        (out / 'calib' / name / 'dataset.txt').write_text('\n'.join(rows) + '\n')

    # Test frames: everything the board needs to run and check the chain from preprocessed pixels.
    tests = []
    for pk, rec in zip(picks, records):
        if pk['split'] != 'test':
            continue
        vision, prefix, expert, ref_kv = inputs(rec)
        k = len(tests)
        arrays = {'pixel_0': vision[0], 'pixel_1': vision[1], 'image_tokens_0': rec['vision_out'][0], 'image_tokens_1': rec['vision_out'][1],
                  'prefix': prefix[0], 'prefix_tail': prefix[0][:, 128:], 'attention_mask': prefix[1],
                  **({'cos': prefix[2], 'sin': prefix[3], 'cos_self': expert[0][3], 'sin_self': expert[0][4]} if fast
                     else {'positions': prefix[2], 'suffix_positions': expert[0][3]}),
                  'prefix_pad': expert[0][2], 'noise': rec['noise'],
                  'timesteps': torch.cat([s['timestep'] for s in rec['steps']]),
                  'time_embeddings': torch.cat([e[1] for e in expert]),
                  'x_t': torch.cat([s['x_t'] for s in rec['steps']]), 'v_t': torch.cat([s['v_t'] for s in rec['steps']]),
                  'actions_normalized': rec['actions_normalized'][..., :action_dim], 'actions': rec['actions'],
                  **{n: t for n, t in zip(kv_names, ref_kv)}}
        np.savez(out / 'test' / f'frame_{k}.npz', **{n: t.detach().cpu().numpy().astype(np.float32) for n, t in arrays.items()})
        tests.append({k2: v for k2, v in pk.items()})

    from safetensors.torch import load_file
    stats = load_file(str(a.model_dir / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors'))
    np.savez(out / 'test' / 'glue.npz', **{k.replace('.', '_'): v.numpy() for k, v in stats.items()},
             num_steps=np.array(cfg.num_steps), chunk_size=np.array(cfg.chunk_size), action_dim=np.array(action_dim),
             graph=np.array(a.graph))
    report = {'model_dir': str(a.model_dir), 'checkpoint_sha256': sha(a.model_dir / 'model.safetensors'),
              'image_size': list(cfg.resize_imgs_with_padding), 'prefix_tokens': prefix_len, 'num_steps': cfg.num_steps,
              'wrapper_vs_lerobot_rel_max_abs': checks, 'attention_mask_value': -1e4, 'scaled_norms': scaled_norms, 'graph': a.graph, 'mask_op': a.mask_op, 'lang_slots': a.lang_slots, 'reference': 'LeRobot predict_action_chunk, CPU FP32, eager attention',
              'calibration_frames': [pk for pk in picks if pk['split'] == 'calib'], 'test_frames': tests,
              'expert_calibration_samples': expert_samples,
              'onnx_sha256': {n: sha(out / 'onnx' / f'{n}.onnx') for n in ['vision', 'prefix', 'expert']},
              'postprocessor_stats_keys': list(stats), 'torch': torch.__version__}
    (out / 'export.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['prefix_tokens', 'scaled_norms', 'wrapper_vs_lerobot_rel_max_abs', 'onnx_sha256']}, indent=2))


if __name__ == '__main__':
    main()
