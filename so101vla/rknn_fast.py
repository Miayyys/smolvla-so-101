"""NPU-oriented rewrite of the SmolVLA prefix and expert graphs (same arithmetic as LeRobot, checked in rknn_export).

Board profile of the LeRobot-shaped graphs (2026-11-07): one INT8 expert step took 206 ms, of which matrix products
were 24%; transposes 31%, softmax 18%, other element-wise ops 24%; Sin/Cos ran on the CPU.

First attempt (2026-11-07, slower, replaced): stacking the 3 query heads that share a K/V head along the sequence
axis gave scores of shape [1, 5, 3T, S], whose softmax RKNN runs on the CPU (expert INT8 softmax 37 -> 139 ms per
step, prefix FP16 1.2 s), and RoPE as a 64x64 matmul added transposes (82 -> 154 per expert step). RKNN handles the
LeRobot attention layout [1, 15, T, S] well, so this version keeps it and changes only:

  * K/V enter and leave head-first ([1, 5, P, 64], the cache layout the attention reads), with no per-layer
    transposes of the cache. The expert's cross-attention layers (odd) project the prefix K/V with their own
    k_proj/v_proj; the result is the same for all denoising steps, so the prefix graph outputs it once.
  * RoPE with cos/sin tables computed on the CPU ([1, L, 32]) instead of Sin/Cos of positions inside the graph
    (CPU ops on the board; in INT8 the integer positions were quantized with a step of ~0.7). Rotation in the
    split/concat form that the LeRobot graph used.
  * 1/sqrt(head_dim) folded into q; the last prefix layer stops after its K/V (its output is never used).
  * Masks: float 0/1 input cast to bool (rknn_export.to_bool), -1e4 fill, built once per graph.

Prefix graph:  prefix [1,P,960], attention_mask [1,P,P], cos [1,P,32], sin [1,P,32]
               -> k_i, v_i [1,5,P,64] for i in 0..15 (even i: VLM K/V; odd i: expert-projected K/V)
Expert graph:  x_t [1,50,32], time_embedding [1,720], prefix_pad [1,P], cos_self [1,50,32], sin_self [1,50,32],
               k_i, v_i (i in 0..15) -> velocity [1,50,32]
"""

import torch

FAST_KV_NAMES = [f'{kind}_{i}' for i in range(16) for kind in ('k', 'v')]


def rope_tables(positions, head_dim=64, max_wavelength=10_000):
    """cos/sin [B, L, head_dim/2] of LeRobot's apply_rope at float positions [B, L]."""
    exponents = (2.0 / head_dim) * torch.arange(head_dim // 2, dtype=torch.float32)
    radians = positions[..., None].to(torch.float32) / (max_wavelength ** exponents)
    return torch.cos(radians), torch.sin(radians)


def rope(x, cos, sin):
    """x [1, L, H, 64]; cos/sin [1, L, 1, 32]."""
    a, b = x.split(x.shape[-1] // 2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


def rms_norm(norm, x):
    return norm.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon))


def mlp(m, x):
    return m.down_proj(torch.nn.functional.silu(m.gate_proj(x)) * m.up_proj(x))


def attention(q, k, v, mask, heads=15, kv_heads=5, head_dim=64):
    """q [1,T,15,64] (scaled, roped); k, v [1,5,S,64]; mask bool [1,1,T,S] -> [1,T,960]. LeRobot's eager layout."""
    t, s = q.shape[1], k.shape[2]
    groups = heads // kv_heads
    k = k[:, :, None].expand(1, kv_heads, groups, s, head_dim).reshape(1, heads, s, head_dim)
    v = v[:, :, None].expand(1, kv_heads, groups, s, head_dim).reshape(1, heads, s, head_dim)
    w = torch.matmul(q.transpose(1, 2), k.transpose(2, 3))                       # [1,15,T,S]
    w = torch.where(mask, w, torch.tensor(-1e4, dtype=torch.float32))
    o = torch.matmul(torch.softmax(w, dim=-1), v)                                 # [1,15,T,64]
    return o.transpose(1, 2).reshape(1, t, heads * head_dim)


class FastPrefix(torch.nn.Module):
    def __init__(self, policy, to_bool):
        super().__init__()
        vwe = policy.model.vlm_with_expert
        self.layers = vwe.get_vlm_model().text_model.layers
        self.expert_layers = vwe.lm_expert.layers
        self.to_bool = to_bool
        self.scale = 64 ** -0.5

    def forward(self, prefix, attention_mask, cos, sin):
        p = prefix.shape[1]
        mask = self.to_bool(attention_mask)[:, None]          # [1,1,P,P]
        cos4, sin4 = cos[:, :, None, :], sin[:, :, None, :]
        h = prefix
        outs = []
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            x = rms_norm(layer.input_layernorm, h)
            k = rope(a.k_proj(x).view(1, p, 5, 64), cos4, sin4)
            v = a.v_proj(x).view(1, p, 5, 64)
            kh, vh = k.transpose(1, 2), v.transpose(1, 2)    # [1,5,P,64]
            if i % 2 == 0:   # expert self-attention layer: uses the VLM K/V directly
                outs.extend((kh, vh))
            else:            # expert cross-attention layer: its own projection of the VLM K/V, once per chunk
                e = self.expert_layers[i].self_attn
                outs.extend((e.k_proj(k.reshape(1, p, 320)).view(1, p, 5, 64).transpose(1, 2),
                             e.v_proj(v.reshape(1, p, 320)).view(1, p, 5, 64).transpose(1, 2)))
            if i == len(self.layers) - 1:
                break        # the last layer's attention output and MLP are never used
            q = rope((a.q_proj(x) * self.scale).view(1, p, 15, 64), cos4, sin4)
            h = h + a.o_proj(attention(q, kh, vh, mask))
            h = h + mlp(layer.mlp, rms_norm(layer.post_attention_layernorm, h))
        return tuple(outs)


class FastExpert(torch.nn.Module):
    def __init__(self, policy, suffix_mask, to_bool):
        super().__init__()
        core = policy.model
        self.core = core
        self.layers = core.vlm_with_expert.lm_expert.layers
        self.norm = core.vlm_with_expert.lm_expert.norm
        self.to_bool = to_bool
        n = core.config.chunk_size
        self.chunk = n
        cos, sin = rope_tables(torch.arange(n, dtype=torch.float32)[None])     # cross layers: positions 0..n-1
        self.register_buffer('cos_cross', cos[:, :, None, :])
        self.register_buffer('sin_cross', sin[:, :, None, :])
        self.register_buffer('suffix_mask', suffix_mask[:, None].clone())       # [1,1,n,n] bool
        self.scale = 64 ** -0.5

    def forward(self, x_t, time_embedding, prefix_pad, cos_self, sin_self, *kv):
        core, n = self.core, self.chunk
        p = prefix_pad.shape[1]
        action = core.action_in_proj(x_t)
        joined = torch.cat((action, time_embedding[:, None, :].expand_as(action)), dim=2)
        h = core.action_time_mlp_out(torch.nn.functional.silu(core.action_time_mlp_in(joined)))
        pad = self.to_bool(prefix_pad)[:, None, None, :].expand(1, 1, n, p)
        mask_self = torch.cat((pad, self.suffix_mask), dim=3)                  # [1,1,n,P+n]
        cs, ss = cos_self[:, :, None, :], sin_self[:, :, None, :]
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            x = rms_norm(layer.input_layernorm, h)
            q = (a.q_proj(x) * self.scale).view(1, n, 15, 64)
            k, v = kv[2 * i], kv[2 * i + 1]
            if i % 2 == 0:
                q = rope(q, cs, ss)
                k = torch.cat((k, rope(a.k_proj(x).view(1, n, 5, 64), cs, ss).transpose(1, 2)), dim=2)
                v = torch.cat((v, a.v_proj(x).view(1, n, 5, 64).transpose(1, 2)), dim=2)
                att = attention(q, k, v, mask_self)
            else:
                att = attention(rope(q, self.cos_cross, self.sin_cross), k, v, pad)
            h = h + a.o_proj(att)
            h = h + mlp(layer.mlp, rms_norm(layer.post_attention_layernorm, h))
        return core.action_out_proj(rms_norm(self.norm, h))


def fast_reference_kv(policy, rec):
    """What FastPrefix should output, derived from LeRobot's recorded cache (valid positions are what matter)."""
    expert_layers = policy.model.vlm_with_expert.lm_expert.layers
    out = []
    with torch.inference_mode():
        for i in range(16):
            k, v = rec['kv'][2 * i], rec['kv'][2 * i + 1]           # [1,5,P,64] post-RoPE K, V
            p = k.shape[2]
            if i % 2 == 1:
                e = expert_layers[i].self_attn
                k = e.k_proj(k.transpose(1, 2).reshape(1, p, 320)).view(1, p, 5, 64).transpose(1, 2)
                v = e.v_proj(v.transpose(1, 2).reshape(1, p, 320)).view(1, p, 5, 64).transpose(1, 2)
            out.extend((k.contiguous(), v.contiguous()))
    return out
