"""Convert the ONNX graphs from so101vla/rknn_export.py into RKNN models for RK3588 (run in .venv-rknn).

  .venv-rknn/bin/python -I scripts/rknn_compile.py --root runs/rknn_v5 [--modules vision prefix expert] [--dtypes fp16 int8]

Steps per module:
  1. ONNX fix-up: the expert's single ReduceMin over the suffix positions becomes Gather(index 0) (the positions are
     valid_count + arange, so the minimum is the first entry; QVLA saw ReduceMin fail in the RK3588 CPU fallback).
  2. onnxruntime check of the fixed graph against the LeRobot FP32 reference on test frame 0.
  3. RKNN build: fp16 (no quantization) and/or int8 (w8a8, per-channel weights, calibration from calib/<module>/dataset.txt).
Outputs: <root>/rknn/<module>_<dtype>.rknn, logs, and <root>/rknn/compile_<module>.json.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

KV_NAMES = [f'{kind}_{i}' for i in range(16) for kind in ('key', 'value')]
FAST_KV_NAMES = [f'{kind}_{i}' for i in range(16) for kind in ('k', 'v')]  # --graph fast (so101vla/rknn_fast.py)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fix_onnx(src, dst):
    import onnx
    m = onnx.load(str(src))
    # Only the minimum over the suffix positions; the attention's row minimum (export_attention) stays a ReduceMin.
    reductions = [n for n in m.graph.node if n.op_type == 'ReduceMin' and list(n.input) == ['suffix_positions']]
    rewritten = []
    for node in reductions:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs != {'axes': [1], 'keepdims': 1}:
            raise ValueError(f'unexpected ReduceMin {node.name}: {list(node.input)} {attrs}')
        index = 'first_suffix_position_index'
        m.graph.initializer.append(onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), index))
        node.CopyFrom(onnx.helper.make_node('Gather', [node.input[0], index], list(node.output),
                                            name='FirstSuffixPosition', axis=1))
        rewritten.append(node.name)
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))
    return rewritten


def frame_inputs(module, z, step=0):
    if module == 'vision':
        return {'pixel': z['pixel_0']}, [z['image_tokens_0']]
    fast = 'k_0' in z
    kv = FAST_KV_NAMES if fast else KV_NAMES
    if module == 'prefix':
        pos = {'cos': z['cos'], 'sin': z['sin']} if fast else {'positions': z['positions']}
        return {'prefix': z['prefix'], 'attention_mask': z['attention_mask'], **pos}, [z[n] for n in kv]
    pos = {'cos_self': z['cos_self'], 'sin_self': z['sin_self']} if fast else {'suffix_positions': z['suffix_positions']}
    feeds = {'x_t': z['x_t'][step:step + 1], 'time_embedding': z['time_embeddings'][step:step + 1],
             'prefix_pad': z['prefix_pad'], **pos, **{n: z[n] for n in kv}}
    return feeds, [z['v_t'][step:step + 1]]


def compare(outputs, refs):
    worst = {'max_abs': 0.0, 'rel_l2': 0.0, 'min_cosine': 1.0}
    for y, r in zip(outputs, refs):
        y = np.asarray(y, dtype=np.float64).reshape(-1); r = np.asarray(r, dtype=np.float64).reshape(-1)
        worst['max_abs'] = max(worst['max_abs'], float(np.abs(y - r).max()))
        worst['rel_l2'] = max(worst['rel_l2'], float(np.linalg.norm(y - r) / max(np.linalg.norm(r), 1e-12)))
        worst['min_cosine'] = min(worst['min_cosine'], float(y @ r / max(np.linalg.norm(y) * np.linalg.norm(r), 1e-12)))
        worst['nonfinite'] = worst.get('nonfinite', 0) + int((~np.isfinite(y)).sum())
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('runs/rknn_v5'))
    p.add_argument('--modules', nargs='+', default=['vision', 'prefix', 'expert'])
    p.add_argument('--dtypes', nargs='+', default=['fp16', 'int8'])
    p.add_argument('--algorithm', default='normal', choices=['normal', 'mmse', 'kl_divergence'])
    p.add_argument('--optimization-level', type=int, default=3)
    p.add_argument('--suffix', default='', help='appended to output names, e.g. _mmse')
    a = p.parse_args()
    import onnxruntime as ort
    from rknn.api import RKNN

    root = a.root.resolve()
    (root / 'rknn').mkdir(exist_ok=True)
    z = dict(np.load(root / 'test/frame_0.npz'))
    for module in a.modules:
        src, fixed = root / 'onnx' / f'{module}.onnx', root / 'onnx' / f'{module}_fixed.onnx'
        report = {'module': module, 'onnx_sha256': sha(src), 'rewritten_nodes': fix_onnx(src, fixed),
                  'fixed_onnx_sha256': sha(fixed), 'builds': {}}
        feeds, refs = frame_inputs(module, z)
        session = ort.InferenceSession(str(fixed), providers=['CPUExecutionProvider'])
        outputs = session.run(None, feeds)
        if module == 'prefix':  # K/V at padded positions are never used and differ from LeRobot by design
            valid = z['prefix_pad'][0] > 0.5
            outputs, refs = [o[:, :, valid] for o in outputs], [r[:, :, valid] for r in refs]
        report['onnxruntime_vs_lerobot'] = compare(outputs, refs)
        del session
        print(module, 'onnxruntime vs LeRobot:', report['onnxruntime_vs_lerobot'], flush=True)
        for dtype in a.dtypes:
            name = f'{module}_{dtype}{a.suffix}'
            out = root / 'rknn' / f'{name}.rknn'
            rknn = RKNN(verbose=False)
            cfg = dict(target_platform='rk3588', optimization_level=a.optimization_level, float_dtype='float16')
            if dtype == 'int8':
                cfg.update(quantized_dtype='w8a8', quantized_method='channel', quantized_algorithm=a.algorithm)
            rknn.config(**cfg)
            started = time.time()
            if rknn.load_onnx(model=str(fixed)) != 0:
                raise RuntimeError(f'{name}: load_onnx failed')
            dataset = str(root / 'calib' / module / 'dataset.txt') if dtype == 'int8' else None
            if rknn.build(do_quantization=dtype == 'int8', dataset=dataset) != 0:
                raise RuntimeError(f'{name}: build failed')
            if rknn.export_rknn(str(out)) != 0:
                raise RuntimeError(f'{name}: export failed')
            rknn.release()
            report['builds'][dtype] = {'config': cfg, 'dataset': dataset, 'seconds': round(time.time() - started, 1),
                                       'rknn': str(out), 'bytes': out.stat().st_size, 'sha256': sha(out)}
            print(name, report['builds'][dtype], flush=True)
        (root / 'rknn' / f'compile_{module}{a.suffix}.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    sys.exit(main())
