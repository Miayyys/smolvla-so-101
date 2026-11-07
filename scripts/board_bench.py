"""RK3588 board benchmark of the three SmolVLA RKNN graphs: per-module latency and error, full-chain actions, memory.

Runs on the board (Python 3.10, rknnlite 2.3.2), e.g.:
  PYTHONPATH=/userdata/so101/board_env/python_site python3 board_bench.py --root /userdata/so101/rknn_v5 \
      --vision fp16 --prefix fp16 --expert fp16 --out report_fp16.json

<root>/rknn/<module>_<variant>.rknn and <root>/test/frame_*.npz, glue.npz come from so101vla/rknn_export.py and
scripts/rknn_compile.py. The chain starts from preprocessed pixels and the CPU-made language/state part of the prefix
(prefix_tail), so image resizing, tokenization and state_proj are not timed here (CPU glue, measured separately later).
Per-module errors use the reference inputs (each module isolated); chain errors feed each module the board's own
outputs: vision x2 -> prefix -> 10 expert steps (Euler, time embedding from the export) -> normalized actions. The
chain is rerun with FP32 reference image tokens and with reference K/V substituted upstream (untimed), which splits
the action error between vision, prefix and expert. --kv-resident runs the expert through the C API with the per-chunk
inputs uploaded once; --vision-parallel runs the two cameras on NPU cores 0 and 1 at the same time.
Record: docs/experiments/edge-rk3588.md.
"""

import argparse
import json
import os
import time
from pathlib import Path

import threading

import numpy as np
import psutil
from rknnlite.api import RKNNLite

import rknn_capi

KV_NAMES = ['%s_%d' % (kind, i) for i in range(16) for kind in ('key', 'value')]
FAST_KV_NAMES = ['%s_%d' % (kind, i) for i in range(16) for kind in ('k', 'v')]  # export --graph fast


def kv_names(z):
    return FAST_KV_NAMES if 'k_0' in z else KV_NAMES


def valid_kv(z, tensors):
    """Valid prefix positions of each K/V tensor [1, 5, P, 64] (padded positions are never used)."""
    v = z['prefix_pad'][0] > 0.5
    return [t[:, :, v] for t in tensors]


def prefix_feed(z, prefix):
    return [prefix, z['attention_mask']] + ([z['cos'], z['sin']] if 'k_0' in z else [z['positions']])


def expert_feed(z, x, step, kv):
    pos = [z['cos_self'], z['sin_self']] if 'k_0' in z else [z['suffix_positions']]
    return [x, z['time_embeddings'][step:step + 1], z['prefix_pad']] + pos + kv
CORES = {'0': RKNNLite.NPU_CORE_0, '1': RKNNLite.NPU_CORE_1, '2': RKNNLite.NPU_CORE_2,
         '01': RKNNLite.NPU_CORE_0_1, '012': RKNNLite.NPU_CORE_0_1_2, 'auto': RKNNLite.NPU_CORE_AUTO}


def mem_available():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('no MemAvailable')


def stats(times):
    t = np.asarray(times) * 1000
    return {'p50_ms': round(float(np.median(t)), 2), 'min_ms': round(float(t.min()), 2),
            'max_ms': round(float(t.max()), 2), 'n': len(t)}


def compare(outputs, refs):
    out = {'max_abs': 0.0, 'rel_l2': 0.0, 'min_cosine': 1.0, 'nonfinite': 0}
    for y, r in zip(outputs, refs):
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        r = np.asarray(r, dtype=np.float64).reshape(-1)
        out['nonfinite'] += int((~np.isfinite(y)).sum())
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        out['max_abs'] = max(out['max_abs'], float(np.abs(y - r).max()))
        out['rel_l2'] = max(out['rel_l2'], float(np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-12)))
        out['min_cosine'] = min(out['min_cosine'], float(y @ r / max(np.linalg.norm(y) * np.linalg.norm(r), 1e-12)))
    return out


class Graph:
    def __init__(self, path, core):
        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(str(path)) != 0:
            raise RuntimeError('load_rknn failed: %s' % path)
        if self.rknn.init_runtime(core_mask=CORES[core]) != 0:
            raise RuntimeError('init_runtime failed: %s' % path)

    def __call__(self, inputs):
        # Every input is passed in its ONNX layout; 4-D inputs would otherwise be read as NHWC.
        fmt = ['nchw' if x.ndim == 4 else 'nhwc' for x in inputs]
        outs = self.rknn.inference(inputs=[np.ascontiguousarray(x, dtype=np.float32) for x in inputs], data_format=fmt)
        if outs is None:
            raise RuntimeError('inference returned None')
        return outs


class CapiGraph:
    """Same call as Graph, through the C API (rknn_capi.py). The runtime keeps the last value of every input, so
    set_static() uploads the inputs that are fixed for a chunk once and step() re-uploads only the first `n_step`
    inputs (for the expert: x_t and the time embedding). Same outputs as uploading everything (max difference 0,
    2026-11-07); the expert's 32 K/V inputs no longer go through the CPU float->native conversion at every step."""

    def __init__(self, path, core):
        self.model = rknn_capi.Model(str(path), core)

    @staticmethod
    def _nhwc(x):
        return x.transpose(0, 2, 3, 1) if x.ndim == 4 else x   # C API float path wants NHWC for 4-D inputs

    def __call__(self, inputs):
        for i, x in enumerate(inputs):
            self.model.set_float(i, self._nhwc(x))
        return self.model.run()

    def set_static(self, first, inputs):
        for i, x in enumerate(inputs):
            self.model.set_float(first + i, self._nhwc(x))
        self._full = True

    def step(self, inputs):
        for i, x in enumerate(inputs):
            self.model.set_float(i, self._nhwc(x))
        indices = None if self._full else list(range(len(inputs)))
        self._full = False
        return self.model.run(indices)


def module_inputs(z):
    kv = [z[n] for n in kv_names(z)]
    return {'vision': ([z['pixel_0']], [z['image_tokens_0']]),
            'prefix': (prefix_feed(z, z['prefix']), kv),
            'expert': (expert_feed(z, z['x_t'][:1], 0, kv), [z['v_t'][:1]])}


def run_chain(graphs, z, steps, action_dim, timing, upstream='board'):
    """upstream: 'board' (all modules on the NPU), 'ref_vision' (FP32 reference image tokens), 'ref_vision_prefix'
    (FP32 reference K/V). Differences between the three attribute the action error to vision, prefix and expert."""
    t0 = time.perf_counter()
    if upstream == 'board' and 'vision_1' in graphs:
        # One camera per NPU core, in parallel (2026-11-07: 2.04 s vs 2.72 s for both on cores 0_1_2 in turn).
        tokens = [None, None]
        def one(c):
            tokens[c] = graphs['vision_%d' % c]([z['pixel_%d' % c]])[0]
        threads = [threading.Thread(target=one, args=(c,)) for c in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    elif upstream == 'board':
        tokens = [graphs['vision']([z['pixel_%d' % c]])[0] for c in range(2)]
    else:
        tokens = [z['image_tokens_0'], z['image_tokens_1']]
    t1 = time.perf_counter()
    prefix = np.concatenate([tokens[0].reshape(1, -1, 960), tokens[1].reshape(1, -1, 960), z['prefix_tail']], axis=1)
    names = kv_names(z)
    if upstream == 'ref_vision_prefix':
        kv = [z[n] for n in names]
    else:
        kv = [x.reshape(z[n].shape) for x, n in zip(graphs['prefix'](prefix_feed(z, prefix)), names)]
    t2 = time.perf_counter()
    x = z['noise'].copy()
    dt = -1.0 / steps
    expert = graphs['expert']
    if isinstance(expert, CapiGraph):
        feed = expert_feed(z, x, 0, kv)
        expert.set_static(2, feed[2:])                      # mask, positions/RoPE tables, K/V: once per chunk
        for s in range(steps):
            v = expert.step([x, z['time_embeddings'][s:s + 1]])[0].reshape(x.shape)
            x = x + dt * v
    else:
        for s in range(steps):
            v = expert(expert_feed(z, x, s, kv))[0].reshape(x.shape)
            x = x + dt * v
    t3 = time.perf_counter()
    timing['vision_x2'].append(t1 - t0); timing['prefix'].append(t2 - t1)
    timing['expert_x%d' % steps].append(t3 - t2); timing['total'].append(t3 - t0)
    return x[..., :action_dim], tokens, kv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    # prefix on cores 0_1_2 returns wrong K/V (FP16: 24% error by layer 15 vs 0.3% on one core, 2026-11-06); vision is
    # correct on three cores and 30% faster there.
    for m, core in [('vision', '012'), ('prefix', '0'), ('expert', '0')]:
        p.add_argument('--' + m, default='fp16', help='variant: <root>/rknn/%s_<variant>.rknn' % m)
        p.add_argument('--%s-core' % m, default=core, choices=sorted(CORES))
    p.add_argument('--kv-resident', action='store_true',
                   help='expert through the C API: K/V and other per-chunk inputs uploaded once, only x_t and the '
                        'time embedding per step')
    p.add_argument('--vision-parallel', action='store_true',
                   help='camera 1 on NPU core 0 and camera 2 on core 1 in parallel (ignores --vision-core)')
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--frames', type=int, default=0, help='test frames for the chain (0 = all)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    proc = psutil.Process()
    glue = np.load(a.root / 'test/glue.npz')
    steps, action_dim = int(glue['num_steps']), int(glue['action_dim'])
    mean = glue['action_mean'][:action_dim] if 'action_mean' in glue else None
    std = glue['action_std'][:action_dim] if 'action_std' in glue else None
    frames = sorted((a.root / 'test').glob('frame_*.npz'), key=lambda f: int(f.stem.split('_')[1]))
    frames = frames[:a.frames] if a.frames else frames
    report = {'variants': {m: getattr(a, m) for m in ['vision', 'prefix', 'expert']},
              'kv_resident': a.kv_resident, 'vision_parallel': a.vision_parallel,
              'cores': {m: getattr(a, m + '_core') for m in ['vision', 'prefix', 'expert']},
              'cpu_governor': open('/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor').read().strip(),
              'npu_freq': open('/sys/class/devfreq/fdab0000.npu/cur_freq').read().strip(), 'memory': {}, 'load_seconds': {}}

    # Memory: MemAvailable covers NPU DMA buffers too, which process RSS does not.
    avail0, rss0 = mem_available(), proc.memory_info().rss
    graphs = {}
    for m in ['vision', 'prefix', 'expert']:
        t = time.perf_counter()
        path = a.root / 'rknn' / ('%s_%s.rknn' % (m, getattr(a, m)))
        if m == 'expert' and a.kv_resident:
            graphs[m] = CapiGraph(path, getattr(a, m + '_core'))
        elif m == 'vision' and a.vision_parallel:
            graphs['vision_0'], graphs['vision_1'] = Graph(path, '0'), Graph(path, '1')
            graphs[m] = graphs['vision_0']
        else:
            graphs[m] = Graph(path, getattr(a, m + '_core'))
        report['load_seconds'][m] = round(time.perf_counter() - t, 2)
        report['memory']['after_load_' + m] = {'mem_used_delta_mb': round((avail0 - mem_available()) / 2**20, 1),
                                                'rss_delta_mb': round((proc.memory_info().rss - rss0) / 2**20, 1)}
        print(m, 'loaded', report['memory']['after_load_' + m], flush=True)

    # Isolated modules: reference inputs, latency and error vs LeRobot FP32.
    z = dict(np.load(frames[0]))
    report['modules'] = {}
    for m, (inputs, refs) in module_inputs(z).items():
        outs = graphs[m](inputs)
        times = []
        for _ in range(a.repeats):
            t = time.perf_counter(); graphs[m](inputs); times.append(time.perf_counter() - t)
        outs = [o.reshape(r.shape) for o, r in zip(outs, refs)]
        if m == 'prefix':  # K/V at padded positions are never used and differ from LeRobot by design
            outs, refs = valid_kv(z, outs), valid_kv(z, refs)
        report['modules'][m] = {'latency': stats(times), 'error_vs_reference': compare(outs, refs)}
        print(m, report['modules'][m], flush=True)

    # Full chain on every test frame (first frame also repeated for timing).
    timing = {k: [] for k in ['vision_x2', 'prefix', 'expert_x%d' % steps, 'total']}
    chain = []
    min_avail = mem_available()
    for f in frames:
        z = dict(np.load(f))
        actions, tokens, kv = run_chain(graphs, z, steps, action_dim, timing)
        min_avail = min(min_avail, mem_available())
        ref = z['actions_normalized']
        row = {'frame': f.name,
               'vision_tokens': compare(tokens, [z['image_tokens_0'], z['image_tokens_1']]),
               'kv': compare(valid_kv(z, kv), valid_kv(z, [z[n] for n in kv_names(z)])),
               'actions_normalized': compare([actions], [ref])}
        if std is not None:
            deg = np.abs(actions * std + mean - (ref * std + mean))[0]
            row['actions_abs_err_per_joint_mean'] = [round(float(v), 3) for v in deg.mean(0)]
            row['actions_abs_err_per_joint_max'] = [round(float(v), 3) for v in deg.max(0)]
            # Error attribution: same chain with FP32 reference outputs substituted upstream (untimed).
            row['attribution_mean_abs_deg'] = {'board': round(float(deg.mean()), 4)}
            for upstream in ['ref_vision', 'ref_vision_prefix']:
                sub, _, _ = run_chain(graphs, z, steps, action_dim, {k: [] for k in timing}, upstream)
                row['attribution_mean_abs_deg'][upstream] = round(float(np.abs((sub - ref) * std).mean()), 4)
        chain.append(row)
        print(f.name, json.dumps(row['actions_normalized']), row.get('actions_abs_err_per_joint_mean'),
              row.get('attribution_mean_abs_deg'), flush=True)
    z = dict(np.load(frames[0]))
    for _ in range(max(0, a.repeats - 1)):
        run_chain(graphs, z, steps, action_dim, timing)
    report['chain'] = chain
    report['chain_latency'] = {k: stats(v) for k, v in timing.items()}
    report['memory']['peak_used_delta_mb'] = round((avail0 - min_avail) / 2**20, 1)
    report['memory']['mem_available_after_mb'] = round(mem_available() / 2**20, 1)
    report['memory']['rss_mb'] = round(proc.memory_info().rss / 2**20, 1)
    print(json.dumps(report['chain_latency'], indent=1), report['memory'], flush=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
