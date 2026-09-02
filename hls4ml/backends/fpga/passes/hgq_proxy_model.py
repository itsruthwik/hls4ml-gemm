import numpy as np

from hls4ml.backends import Backend
from hls4ml.backends.template import FunctionCallTemplate
from hls4ml.model.layers import Layer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer, UnaryLUT
from hls4ml.model.types import Source


def to_apfixed(k, b, i, RND, SAT):
    u = 'u' if k == 0 else ''
    return f'ap_{u}fixed<{b},{i},AP_{RND},AP_{SAT}>'


def to_acfixed(k, b, i, RND, SAT):
    k = 'false' if k == 0 else 'true'
    if b == 1:
        # Currently oneAPI ac_fixed requires at least two bits for both signed and unsigned cases
        # Should be fixed in the future once oneAPI supports 1-bit unsigned ac_fixed
        b = 2
    return f'ac_fixed<{b},{i},{k},AC_{RND},AC_{SAT}>'


def _stream_type(backend: str) -> str:
    """io_stream channel type per backend dialect (Xilinx hls::stream vs Catapult ac_channel)."""
    return 'hls::stream' if backend.lower() in ('vivado', 'vitis') else 'ac_channel'


def _stream_pragmas(backend: str) -> tuple[str, str]:
    """(pipeline, unroll) pragma spellings per backend dialect."""
    if backend.lower() in ('vivado', 'vitis'):
        return '#pragma HLS PIPELINE II=1', '#pragma HLS UNROLL'
    return '#pragma hls_pipeline_init_interval 1', '#pragma hls_unroll'


def _mask_body(
    shape: tuple[int, ...], k: np.ndarray, b: np.ndarray, i: np.ndarray, RND: str, SAT: str, backend: str
) -> tuple[str, int]:
    """Build the fully-unrolled per-element quantization body (indexes inp[idx]/out[idx],
    idx in 0..N-1) shared by the io_parallel and io_stream emitters. Returns (body, N)."""
    assert k.shape[0] == b.shape[0] == i.shape[0] == 1
    assert backend.lower() in ('oneapi', 'quartus', 'vivado', 'vitis', 'catapult'), f'Backend {backend} not tested'
    Ks, Bs, Is = k[0], b[0], i[0]
    Ks, Bs, Is = np.broadcast_to(Ks, shape), np.broadcast_to(Bs, shape), np.broadcast_to(Is, shape)
    Ks, Bs, Is = Ks.ravel(), Bs.ravel(), Is.ravel()
    masks = []
    to_fixed = to_acfixed if backend.lower() in ['oneapi', 'quartus', 'catapult'] else to_apfixed
    for idx, (k, b, i) in enumerate(zip(Ks, Bs, Is)):
        if b == 0:
            fn = f'out[{idx}] = 0;'
        else:
            fn = f'out[{idx}] = {to_fixed(k, b, i, RND, SAT)}(inp[{idx}]);'
        masks.append(f'    {fn}')
    return '\n'.join(masks), len(Ks)


def generate_mask_fn(
    name: str, shape: tuple[int, ...], k: np.ndarray, b: np.ndarray, i: np.ndarray, RND: str, SAT: str, backend: str
) -> str:
    """Generate heterogenous quantization mask function, ONLY works for IOType=io_parallel"""
    body, _ = _mask_body(shape, k, b, i, RND, SAT, backend)
    arguments = 'input_t &inp, output_t &out' if backend.lower() in ['oneapi', 'quartus'] else 'input_t *inp, output_t *out'
    mask_fn = f"""
template<typename input_t, typename output_t>
void {name}({arguments}) {{
    {'#pragma HLS INLINE' if backend.lower() not in ['oneapi', 'quartus'] else ''}

{body}
}}
"""
    return mask_fn


def _mask_lanes(
    shape: tuple[int, ...], k: np.ndarray, b: np.ndarray, i: np.ndarray, RND: str, SAT: str, backend: str
) -> tuple[list[str | None], int]:
    """Per-element conversion expressions in flat tensor order, as a function of the source
    expression. Returns (exprs, N) where exprs[idx] is either None (mask is zero, emit a
    literal 0) or a format string with a single '{}' placeholder for the source. Same
    element-wise semantics as _mask_body, factored so the io_stream emitter can index by
    (beat, lane) instead of by a flat array offset."""
    assert k.shape[0] == b.shape[0] == i.shape[0] == 1
    assert backend.lower() in ('oneapi', 'quartus', 'vivado', 'vitis', 'catapult'), f'Backend {backend} not tested'
    Ks, Bs, Is = k[0], b[0], i[0]
    Ks, Bs, Is = np.broadcast_to(Ks, shape), np.broadcast_to(Bs, shape), np.broadcast_to(Is, shape)
    Ks, Bs, Is = Ks.ravel(), Bs.ravel(), Is.ravel()
    to_fixed = to_acfixed if backend.lower() in ['oneapi', 'quartus', 'catapult'] else to_apfixed
    exprs: list[str | None] = []
    for _k, _b, _i in zip(Ks, Bs, Is):
        exprs.append(None if _b == 0 else to_fixed(_k, _b, _i, RND, SAT) + '({})')
    return exprs, len(Ks)


def _beat_lane_body(beat_exprs: list[str | None], indent: str) -> str:
    """Straight-line per-lane conversion for one beat: res[p] = <cast>(beat[p]).

    Deliberately NOT a loop carrying #pragma hls_unroll: hls4ml_pipeline_stage_loops in the
    generated TCL skips any loop that has an UNROLL attribute, so an unrolled inner loop
    here risks disturbing the enclosing beat loop's pipelining directive."""
    return '\n'.join(
        f'{indent}res[{p}] = ' + ('0;' if expr is None else expr.format(f'beat[{p}]') + ';')
        for p, expr in enumerate(beat_exprs)
    )


def _generate_mask_fn_stream_beatwise(
    name: str,
    shape: tuple[int, ...],
    k: np.ndarray,
    b: np.ndarray,
    i: np.ndarray,
    RND: str,
    SAT: str,
    backend: str,
    beat_size: int,
) -> str:
    """Beat-wise io_stream quantizer. See generate_mask_fn_stream."""
    exprs, n = _mask_lanes(shape, k, b, i, RND, SAT, backend)
    n_beats = n // beat_size
    beats = [exprs[t * beat_size : (t + 1) * beat_size] for t in range(n_beats)]

    if all(beat == beats[0] for beat in beats):
        # Stream-invariant: the same conversion applies to every beat, so no beat counter
        # and no per-beat selection are needed at all.
        compute = _beat_lane_body(beats[0], ' ' * 8)
    else:
        # Heterogeneous along a streamed axis: still beat-wise, but select the lane types on
        # the beat index. Identical beats share a case, so the mux is over the DISTINCT
        # patterns, not over n_beats. Storage stays O(1) either way.
        groups: dict[tuple, list[int]] = {}
        for t, beat in enumerate(beats):
            groups.setdefault(tuple(beat), []).append(t)
        cases = []
        for beat, idxs in groups.items():
            labels = '\n'.join(f'        case {t}:' for t in idxs)
            cases.append(labels + '\n' + _beat_lane_body(list(beat), ' ' * 12) + '\n            break;')
        compute = '        switch (i) {\n' + '\n'.join(cases) + '\n            default: break;\n        }'

    stream = _stream_type(backend)
    return f"""
template<typename input_t, typename output_t>
void {name}({stream}<input_t> &inp_s, {stream}<output_t> &out_s) {{
    static const unsigned N = {n};
    static const unsigned BEAT = {beat_size};
    static_assert(N % input_t::size == 0, "{name}: input beat size must divide N");
    static_assert(input_t::size == BEAT, "{name}: input beat size disagrees with codegen");
    static_assert(output_t::size == BEAT, "{name}: output beat size disagrees with codegen");

Beat_{name}:
    for (unsigned i = 0; i < N / BEAT; i++) {{
        input_t beat = inp_s.read();
        output_t res;
{compute}
        out_s.write(res);
    }}
}}
"""


def generate_mask_fn_stream(
    name: str,
    shape: tuple[int, ...],
    k: np.ndarray,
    b: np.ndarray,
    i: np.ndarray,
    RND: str,
    SAT: str,
    backend: str,
    beat_size: int | None = None,
) -> str:
    """Generate the io_stream (ac_channel) overload of the heterogenous quantization mask
    function for the Catapult backend. Same function name as the io_parallel overload; C++
    overload resolution selects this one when the arguments are ac_channel references.

    When ``beat_size`` is known and divides N, this emits a **beat-wise** quantizer that
    conforms to hls4ml's io_stream contract: one channel read and one write per iteration,
    O(1) internal storage, one-beat latency. Bit-exactness is by construction -- every
    element receives the identical conversion as the io_parallel path, only the evaluation
    order and the storage change.

    Without a usable ``beat_size`` it falls back to the original whole-tensor buffered form,
    which is correct for any packing but stores the tensor twice."""
    if beat_size and beat_size > 0 and len(shape) and int(np.prod(shape)) % beat_size == 0:
        return _generate_mask_fn_stream_beatwise(name, shape, k, b, i, RND, SAT, backend, beat_size)
    body, n = _mask_body(shape, k, b, i, RND, SAT, backend)
    stream = _stream_type(backend)
    pipeline_pragma, unroll_pragma = _stream_pragmas(backend)
    mask_fn = f"""
template<typename input_t, typename output_t>
void {name}({stream}<input_t> &inp_s, {stream}<output_t> &out_s) {{
    static const unsigned N = {n};
    static_assert(N % input_t::size == 0, "{name}: input beat size must divide N");
    static_assert(N % output_t::size == 0, "{name}: output beat size must divide N");
    typename input_t::value_type inp[N];
    typename output_t::value_type out[N];

ReadInp_{name}:
    for (unsigned i = 0; i < N / input_t::size; i++) {{
        {pipeline_pragma}
        input_t beat = inp_s.read();
        {unroll_pragma}
        for (unsigned p = 0; p < input_t::size; p++) {{
            inp[i * input_t::size + p] = beat[p];
        }}
    }}

{body}

WriteOut_{name}:
    for (unsigned i = 0; i < N / output_t::size; i++) {{
        {pipeline_pragma}
        output_t beat;
        {unroll_pragma}
        for (unsigned p = 0; p < output_t::size; p++) {{
            beat[p] = out[i * output_t::size + p];
        }}
        out_s.write(beat);
    }}
}}
"""
    return mask_fn


def _stream_beat_size(var) -> int | None:
    """Elements per beat of a variable that will be streamed.

    This pass runs in the 'optimize' flow, which is ordered BEFORE 'catapult:transform_types'
    converts variables to PackedType -- so the packed type is not yet available to read and
    the beat width has to be predicted from the shape. StreamVariableConverter.convert()
    builds ``PackedType(..., n_elem=shape[-1], n_pack=1)`` and PackedTypeConverter emits
    ``nnet::array<T, n_elem * n_pack>``, so the beat is the last axis.

    The prediction is guarded, not trusted: the generated function static_asserts
    ``input_t::size == BEAT``, so if the packing ever differs the build fails at compile time
    with a named message rather than silently producing wrong hardware."""
    if getattr(var, 'type', None) is not None and getattr(var.type, 'n_elem', None):
        # Already converted (a later flow, or a caller that pre-converts): read it directly.
        vtype = var.type
        if getattr(vtype, 'unpack', False):
            return vtype.n_elem // vtype.n_pack if vtype.n_elem % vtype.n_pack == 0 else None
        return vtype.n_elem * vtype.n_pack
    shape = getattr(var, 'shape', None)
    if not shape:
        return None
    return int(shape[-1])


class ProcessFixedPointQuantizerLayer(OptimizerPass):
    def match(self, node: Layer):
        return isinstance(node, FixedPointQuantizer)

    def transform(self, model, node: FixedPointQuantizer):
        io_type = model.config.config['IOType']
        backend = model.config.config['Backend']
        if io_type != 'io_parallel' and not (
            io_type == 'io_stream' and backend.lower() in ('catapult', 'vivado', 'vitis')
        ):
            raise NotImplementedError(
                'Heterogenous quantization for activations is only supported with IOType=io_parallel '
                '(io_stream is supported on the Catapult, Vivado, and Vitis backends)'
            )

        name = node.name

        assert node.mask_kbi is not None
        k, b, i = node.mask_kbi
        RND = node.RND
        SAT = node.SAT
        if io_type == 'io_stream':
            # Beat width comes from the variable that is actually streamed, not from the
            # tensor shape: PackedType packs shape[-1] * n_pack elements per beat
            # (fpga_types.py, StreamVariableConverter). None -> buffered fallback.
            beat_size = _stream_beat_size(node.get_output_variable())
            mask_fn: str = generate_mask_fn_stream(
                name, node.get_input_variable().shape, k, b, i, RND, SAT, backend, beat_size
            )
        else:
            mask_fn = generate_mask_fn(name, node.get_input_variable().shape, k, b, i, RND, SAT, backend)

        node.set_attr('mask_fn_codegen', Source(mask_fn))


class ProcessFixedPointQuantizerCall(FunctionCallTemplate):
    def __init__(self):
        super().__init__(FixedPointQuantizer, include_header=[])
        self.template = '{namespace}::{name}<{input_t}, {output_t}>({input}, {output});'

    def format(self, node):
        params = self._default_function_params(node)
        namespace = node.model.config.writer_config.get('Namespace', None) or 'nnet'
        params['namespace'] = namespace

        return self.template.format(**params)


class ProcessUnaryLUTCall(FunctionCallTemplate):
    def __init__(self):
        super().__init__(UnaryLUT, include_header=[])
        self.template = 'nnet::unary_lut<{input_t}, {output_t}, {config}>({input}, {output}, {table});'
        self.include_header = [
            'nnet_utils/nnet_activation.h',
            'nnet_utils/nnet_activation_stream.h',
        ]

    def format(self, node):
        params = self._default_function_params(node)
        params['config'] = f'unary_lut_config{node.index}'
        params['table'] = node.get_weights('table').name

        return self.template.format(**params)


def register_hgq_proxy_model(backend: Backend):
    backend.register_pass('process_fixed_point_quantizer_layer', ProcessFixedPointQuantizerLayer)
    backend.register_template(ProcessFixedPointQuantizerCall)
    backend.register_template(ProcessUnaryLUTCall)
