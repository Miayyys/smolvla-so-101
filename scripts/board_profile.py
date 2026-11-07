"""Per-op RK3588 profile of one RKNN graph through the C API (librknnrt), which rknnlite does not expose.

Runs on the board, e.g.:
  python3 board_profile.py --root /userdata/so101/rknn_v5 --module expert --variant int8 --core 0 --out prof.txt

Feeds the arrays of test frame 0 as float32 NHWC (the C API's float conversion rejects NCHW for 4-D inputs, so 4-D
inputs are numerically wrong here: use this for timing only, board_bench.py for accuracy), runs twice, then saves the
RKNN_QUERY_PERF_DETAIL table and prints its per-op-type summary.
"""

import argparse
import ctypes
import re
from pathlib import Path

import numpy as np

KV_NAMES = ['%s_%d' % (kind, i) for i in range(16) for kind in ('key', 'value')]
RKNN_FLAG_COLLECT_PERF_MASK = 0x8
RKNN_QUERY_PERF_DETAIL = 3
CORES = {'0': 1, '1': 2, '2': 4, '01': 3, '012': 7}


class RknnInput(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('buf', ctypes.c_void_p), ('size', ctypes.c_uint32),
                ('pass_through', ctypes.c_uint8), ('type', ctypes.c_int), ('fmt', ctypes.c_int)]


class PerfDetail(ctypes.Structure):
    _fields_ = [('perf_data', ctypes.c_char_p), ('data_len', ctypes.c_uint64)]


def inputs_for(module, z):
    if module == 'vision':
        return [z['pixel_0']]
    fast = 'k_0' in z
    kv = [z['%s_%d' % (k, i)] for i in range(16) for k in (('k', 'v') if fast else ('key', 'value'))]
    if module == 'prefix':
        return [z['prefix'], z['attention_mask']] + ([z['cos'], z['sin']] if fast else [z['positions']])
    pos = [z['cos_self'], z['sin_self']] if fast else [z['suffix_positions']]
    return [z['x_t'][:1], z['time_embeddings'][:1], z['prefix_pad']] + pos + kv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--module', required=True, choices=['vision', 'prefix', 'expert'])
    p.add_argument('--variant', default='fp16')
    p.add_argument('--core', default='0', choices=sorted(CORES))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    lib = ctypes.CDLL('librknnrt.so')
    model = (a.root / 'rknn' / ('%s_%s.rknn' % (a.module, a.variant))).read_bytes()
    ctx = ctypes.c_uint64(0)
    buf = ctypes.create_string_buffer(model, len(model))
    assert lib.rknn_init(ctypes.byref(ctx), buf, len(model), RKNN_FLAG_COLLECT_PERF_MASK, None) == 0, 'rknn_init'
    assert lib.rknn_set_core_mask(ctx, CORES[a.core]) == 0, 'set_core_mask'
    arrays = [np.ascontiguousarray(x, dtype=np.float32) for x in inputs_for(a.module, dict(np.load(a.root / 'test/frame_0.npz')))]
    ins = (RknnInput * len(arrays))()
    for i, x in enumerate(arrays):
        ins[i].index, ins[i].buf, ins[i].size = i, x.ctypes.data, x.nbytes
        ins[i].pass_through, ins[i].type, ins[i].fmt = 0, 0, 1  # FLOAT32, NHWC
    for _ in range(2):
        assert lib.rknn_inputs_set(ctx, len(arrays), ins) == 0, 'inputs_set'
        assert lib.rknn_run(ctx, None) == 0, 'run'
    perf = PerfDetail()
    assert lib.rknn_query(ctx, RKNN_QUERY_PERF_DETAIL, ctypes.byref(perf), ctypes.sizeof(perf)) == 0, 'query'
    text = perf.perf_data.decode(errors='replace')
    a.out.write_text(text)
    lib.rknn_destroy(ctx)

    lines = text.splitlines()
    start = max(i for i, l in enumerate(lines) if re.match(r'\s*OpType\s+CallNumber', l))
    print('\n'.join(l for l in lines[start:start + 20] if l.strip() and not l.startswith('---')))
    print([l for l in lines if l.startswith('Total ') or l.startswith('Total\t')][-1:])


if __name__ == '__main__':
    main()
