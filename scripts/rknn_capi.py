"""Minimal ctypes binding of the RKNN C API (librknnrt 2.3.2) for the board: what rknnlite does not expose.

Used by board_bench.py (--kv-resident) and board_profile.py. Struct layouts follow rknn_api.h of runtime 2.3.2.
"""

import ctypes

import numpy as np

MAX_DIMS, MAX_NAME = 16, 256
QUERY_IN_OUT_NUM, QUERY_INPUT_ATTR, QUERY_OUTPUT_ATTR, QUERY_PERF_DETAIL = 0, 1, 2, 3
QUERY_NATIVE_INPUT_ATTR, QUERY_NATIVE_OUTPUT_ATTR = 8, 9
FLAG_COLLECT_PERF_MASK = 0x8
TYPES = {0: 'float32', 1: 'float16', 2: 'int8', 3: 'uint8', 4: 'int16', 6: 'int32', 8: 'int64', 9: 'bool'}
FORMATS = {0: 'NCHW', 1: 'NHWC', 2: 'NC1HWC2', 3: 'UNDEFINED'}
CORES = {'0': 1, '1': 2, '2': 4, '01': 3, '012': 7, 'auto': 0}


class TensorAttr(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('n_dims', ctypes.c_uint32), ('dims', ctypes.c_uint32 * MAX_DIMS),
                ('name', ctypes.c_char * MAX_NAME), ('n_elems', ctypes.c_uint32), ('size', ctypes.c_uint32),
                ('fmt', ctypes.c_int), ('type', ctypes.c_int), ('qnt_type', ctypes.c_int), ('fl', ctypes.c_int8),
                ('zp', ctypes.c_int32), ('scale', ctypes.c_float), ('w_stride', ctypes.c_uint32),
                ('size_with_stride', ctypes.c_uint32), ('pass_through', ctypes.c_uint8), ('h_stride', ctypes.c_uint32)]

    def describe(self):
        return {'name': self.name.decode(), 'dims': list(self.dims[:self.n_dims]), 'fmt': FORMATS.get(self.fmt, self.fmt),
                'type': TYPES.get(self.type, self.type), 'qnt': self.qnt_type, 'zp': self.zp, 'scale': self.scale,
                'size': self.size, 'size_with_stride': self.size_with_stride, 'w_stride': self.w_stride}


class Input(ctypes.Structure):
    _fields_ = [('index', ctypes.c_uint32), ('buf', ctypes.c_void_p), ('size', ctypes.c_uint32),
                ('pass_through', ctypes.c_uint8), ('type', ctypes.c_int), ('fmt', ctypes.c_int)]


class Output(ctypes.Structure):
    _fields_ = [('want_float', ctypes.c_uint8), ('is_prealloc', ctypes.c_uint8), ('index', ctypes.c_uint32),
                ('buf', ctypes.c_void_p), ('size', ctypes.c_uint32)]


class InOutNum(ctypes.Structure):
    _fields_ = [('n_input', ctypes.c_uint32), ('n_output', ctypes.c_uint32)]


class PerfDetail(ctypes.Structure):
    _fields_ = [('perf_data', ctypes.c_char_p), ('data_len', ctypes.c_uint64)]


class Model:
    def __init__(self, path, core='0', flags=0):
        self.lib = ctypes.CDLL('librknnrt.so')
        data = open(path, 'rb').read()
        self._model = ctypes.create_string_buffer(data, len(data))
        self.ctx = ctypes.c_uint64(0)
        if self.lib.rknn_init(ctypes.byref(self.ctx), self._model, len(data), flags, None) != 0:
            raise RuntimeError('rknn_init failed: %s' % path)
        if self.lib.rknn_set_core_mask(self.ctx, CORES[core]) != 0:
            raise RuntimeError('rknn_set_core_mask failed')
        num = InOutNum()
        self.query(QUERY_IN_OUT_NUM, num)
        self.n_in, self.n_out = num.n_input, num.n_output
        self.inputs = [self.attr(QUERY_INPUT_ATTR, i) for i in range(self.n_in)]
        self.native_inputs = [self.attr(QUERY_NATIVE_INPUT_ATTR, i) for i in range(self.n_in)]
        self.outputs = [self.attr(QUERY_OUTPUT_ATTR, i) for i in range(self.n_out)]
        self._in = (Input * self.n_in)()
        self._keep = [None] * self.n_in

    def query(self, cmd, struct):
        if self.lib.rknn_query(self.ctx, cmd, ctypes.byref(struct), ctypes.sizeof(struct)) != 0:
            raise RuntimeError('rknn_query %d failed' % cmd)

    def attr(self, cmd, index):
        a = TensorAttr(); a.index = index
        self.query(cmd, a)
        return a

    def set_float(self, i, x):
        """Input i as float32 for the runtime to convert (4-D arrays must be NHWC for the C API)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        self._keep[i] = x
        self._in[i].index, self._in[i].buf, self._in[i].size = i, x.ctypes.data, x.nbytes
        self._in[i].pass_through, self._in[i].type, self._in[i].fmt = 0, 0, 1

    def set_native(self, i, buf):
        """Input i already in the model's native type and layout (pass_through: no conversion by the runtime)."""
        buf = np.ascontiguousarray(buf)
        if buf.nbytes != self.native_inputs[i].size_with_stride and buf.nbytes != self.native_inputs[i].size:
            raise ValueError('input %d: %d bytes, native expects %d' % (i, buf.nbytes, self.native_inputs[i].size))
        self._keep[i] = buf
        self._in[i].index, self._in[i].buf, self._in[i].size = i, buf.ctypes.data, buf.nbytes
        self._in[i].pass_through = 1

    def run(self, indices=None):
        """Upload the inputs in `indices` (default all), run, return float32 outputs (ONNX shapes)."""
        if indices is None:
            arr, n = self._in, self.n_in
        else:
            arr = (Input * len(indices))(*[self._in[i] for i in indices]); n = len(indices)
        if self.lib.rknn_inputs_set(self.ctx, n, arr) != 0:
            raise RuntimeError('rknn_inputs_set failed')
        if self.lib.rknn_run(self.ctx, None) != 0:
            raise RuntimeError('rknn_run failed')
        outs = (Output * self.n_out)()
        for i in range(self.n_out):
            outs[i].want_float, outs[i].index = 1, i
        if self.lib.rknn_outputs_get(self.ctx, self.n_out, outs, None) != 0:
            raise RuntimeError('rknn_outputs_get failed')
        result = []
        for i in range(self.n_out):
            a = self.outputs[i]
            y = np.ctypeslib.as_array(ctypes.cast(outs[i].buf, ctypes.POINTER(ctypes.c_float)), shape=(a.n_elems,)).copy()
            result.append(y.reshape(list(a.dims[:a.n_dims])))
        self.lib.rknn_outputs_release(self.ctx, self.n_out, outs)
        return result

    def perf_detail(self):
        p = PerfDetail()
        self.query(QUERY_PERF_DETAIL, p)
        return p.perf_data.decode(errors='replace')

    def release(self):
        self.lib.rknn_destroy(self.ctx)
