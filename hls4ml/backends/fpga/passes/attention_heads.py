"""Split multi-head attention into H per-head GEMM lanes (transpose-free io_stream).

hls4ml decomposes ``QMultiHeadAttention`` into EinsumDense projections (Q/K/V/O),
two Einsum matmuls (QK^T, A.V) and a Softmax. The generic ``LowerEinsumToGemm``
lowers each einsum to a Gemm and materialises every operand/output permutation as a
physical ``Transpose`` — 5 per block, each a whole-tensor reorder buffer under
io_stream. Most of those permutations only exist to move the ``head`` axis so a
single ``n_inplace = H`` batched GEMM can walk it.

This pass removes that need. It rewrites the attention cluster into H independent
per-head lanes: the projections stay 2D ``[seq, d_model]``; a stateless ``HeadSplit``
fans each into H ``[seq, key_dim]`` streams (head is a within-beat lane, so this is
pure wiring — no buffer, no reorder); each head runs its own QK^T Gemm, Softmax and
A.V Gemm; a stateless ``HeadMerge`` concatenates the H contexts back to
``[seq, d_model]`` before the output projection. The head-move transposes vanish; the
per-head Gemms are plain two-operand GEMMs, identical to any other (the GEMM node
stays uniform).

Two residual within-head orientations are handled here, not as head-moves:
* **QK^T output** — a per-head orientation (``S`` vs ``S^T``). Chosen away by operand
  role assignment: emit ``A=Q, B=K`` so the canonical output is already
  ``[seq_q, seq_k]`` (softmax reduces ``seq_k`` = the last axis). No node.
* **A.V's V** — V's contract axis ``seq_k`` must be the column height, so V is fed
  as ``[key_dim, seq_k]``. Realised as one small per-head 2D ``Transpose`` on the V
  lane (the csim-correct fallback). Folding it into the GEMM IP's V-residency buffer
  write-order — so even this disappears in io_stream — is the gemm-ip-gen contract
  tracked separately; keeping the physical transpose here means nothing mis-lowers.

Runs BEFORE ``LowerEinsumToGemm``: it consumes the QK^T/A.V Einsum and the Softmax
directly (creating Gemm + per-head Softmax nodes), and leaves the now-2D EinsumDense
projections for the generic lowering. Softmax is HGQ2 bit-exact — the per-head clones
copy the identical (head-invariant) table types/scales and keep the transformed flag.

For QAT-trained models, heterogeneous (per-element) quantizers between the cluster ops
cannot fuse and survive as ``FixedPointQuantizer`` nodes; the pass navigates past them
and clones each into the per-head lanes with its head-axis precision slice. Homogeneous
/ untrained models have none and take the identical path unchanged (see the helpers at
the bottom of this module).

Lives in the shared FPGA base so every FPGABackend child inherits it (mirroring the
GEMM passes). ``match()`` requires ``Strategy: GEMM`` on the Einsum, so it stays inert
on backends with no GEMM-IP path; emission is only realised where the backend also
provides the HeadSplit / HeadMerge templates and wires the pass into its flow.
"""

from collections import OrderedDict

from hls4ml.model.layers import Einsum, EinsumDense, Softmax
from hls4ml.model.optimizer import OptimizerPass

from hls4ml.backends.fpga.passes.gemm_nodes import Gemm, _resolve_gemm_config
from hls4ml.backends.fpga.passes.split_merge_nodes import HeadSplit, HeadMerge

# Softmax attributes that fully describe the (head-invariant) bit-exact tables and
# reduction; copied verbatim onto every per-head clone. The per-head shape facts
# (n_in / n_outer / n_inner / input+output vars) are set separately.
_SOFTMAX_CARRY_KEYS = (
    'activation', 'axis', 'implementation', 'exp_scale',
    'exp_table_t', 'exp_table_size', 'inv_table_t', 'inv_table_size',
    'inv_inp_t', 'inp_norm_t', 'table_size', 'table_t', 'accum_t',
    '_bit_exact', 'bit_exact_transformed',
)


class SplitAttentionHeads(OptimizerPass):
    """Rewrite one attention cluster (QK^T -> Softmax -> A.V) into H per-head lanes."""

    def match(self, node):
        # Anchor on the QK^T Einsum: an Einsum on the GEMM path whose sole consumer
        # is a Softmax. (The A.V Einsum feeds the output projection, not a softmax.)
        if not (isinstance(node, Einsum) and node.get_attr('strategy') == 'gemm'):
            return False
        consumers = _consumers(node.model, node.outputs[0])
        return len(consumers) == 1 and isinstance(consumers[0], Softmax)

    def transform(self, model, qk):
        sm = _consumers(model, qk.outputs[0])[0]
        # A QAT-trained model keeps heterogeneous quantizers between the cluster nodes
        # (they cannot fuse into their neighbours); an untrained/homogeneous model has
        # none. Navigate past any such FixedPointQuantizer to find the real einsums /
        # projections, and remember each survivor so it can be cloned into the per-head
        # lanes below. With no survivors this reduces exactly to the homogeneous path.
        av, softmax_oq = _skip_quantizers_fwd(model, sm)
        if not (isinstance(av, Einsum) and av.get_attr('strategy') == 'gemm'):
            raise RuntimeError(f'{qk.name}: expected an Einsum (A.V) after the softmax, got {av.class_name}')
        o_proj, out_iq = _skip_quantizers_fwd(model, av)

        # Cluster geometry. QK^T canonical: n_inplace=H (head), n_free0/1 = seq, contract = key_dim.
        H = qk.get_attr('n_inplace')
        seq_k = qk.get_attr('n_free0')
        seq_q = qk.get_attr('n_free1')
        key_dim = qk.get_attr('n_contract')
        d_model = H * key_dim

        # Projections: K/Q feed QK^T (inputs [K, Q]); V feeds A.V (input1).
        k_proj, key_oq = _skip_quantizers_back(model, qk.inputs[0])
        q_proj, query_oq = _skip_quantizers_back(model, qk.inputs[1])
        v_proj, value_oq = _skip_quantizers_back(model, av.inputs[1])
        for p, tag in ((k_proj, 'K'), (q_proj, 'Q'), (v_proj, 'V')):
            if not isinstance(p, EinsumDense):
                raise RuntimeError(f'{qk.name}: {tag} projection is {p.class_name}, expected EinsumDense')
        proj_oq = {'Q': query_oq, 'K': key_oq, 'V': value_oq}

        qk_prec = qk.get_output_variable().type.precision
        sm_prec = sm.get_output_variable().type.precision
        av_prec = av.get_output_variable().type.precision

        # Resolved GEMM config (strategy/reuse_factor/... + SecondOperandRowMajor) mirrored
        # onto the per-head two-operand Gemms. row_major flips where the one per-block B
        # transpose lands: col-major feeds QK^T's B (K, already [seq_k, key_dim]) directly and
        # transposes V for A.V; row-major transposes K for QK^T and feeds V ([seq_k, key_dim])
        # directly -- one transpose either way (see the per-lane build below).
        qk_cfg = _resolve_gemm_config(model, qk)
        av_cfg = _resolve_gemm_config(model, av)
        row_major = bool(qk_cfg.get('second_operand_row_major', False))

        # --- 1. Projections stay 2D [seq, d_model]: head becomes a within-beat lane
        #        (beat = d_model), not a stream-order axis. The reshape moves no data.
        for p in (k_proj, q_proj, v_proj):
            p.get_output_variable().shape = [seq_q, d_model]

        # --- 2. HeadSplit after each Q/K/V projection: 1 x [seq, d_model] -> H x [seq, key_dim]
        splits = {}
        for p, tag in ((q_proj, 'Q'), (k_proj, 'K'), (v_proj, 'V')):
            hs = model.make_node(
                HeadSplit,
                f'{p.name}_split',
                {'n_heads': H, 'key_dim': key_dim, 'seq': seq_q, 'd_model': d_model},
                [p.outputs[0]],
                [f'{p.name}_h{h}' for h in range(H)],
            )
            for o in hs.outputs:
                hs.get_output_variable(o).type.precision = p.get_output_variable().type.precision
            splits[tag] = hs

        # --- 3. H per-head lanes: QK^T Gemm -> Softmax -> (V transpose) -> A.V Gemm
        av_head_out = []  # tensor name feeding HeadMerge for each head
        chain = []  # graph order for the per-head nodes
        for h in range(H):
            # Each Q/K/V lane starts from its HeadSplit output. If the projection's output
            # quantizer survived (heterogeneous, QAT-trained), clone it into this lane with
            # its head-axis slice; otherwise (homogeneous, fused) the raw split passes through.
            lane = {}
            for tag in ('Q', 'K', 'V'):
                t = splits[tag].outputs[h]
                if proj_oq[tag] is not None:
                    qn = _clone_quantizer_head(model, proj_oq[tag], h, t, H)
                    chain.append(qn)
                    t = qn.outputs[0]
                lane[tag] = t
            q_h, k_h, v_h = lane['Q'], lane['K'], lane['V']

            # QK^T: A=Q_h, B=K_h, contract key_dim -> [seq_q, seq_k] (softmax reduces last).
            # B beat layout: col-major feeds K_h ([seq_k, key_dim]) directly (K-inner beats,
            # gemm_k-wide); row-major transposes K_h to [key_dim, seq_k] (N-inner beats,
            # gemm_n-wide) for the mvau IP.
            if row_major:
                kt_h = model.make_node('Transpose', f'{qk.name}_kt_h{h}', {'perm': [1, 0]}, [k_h])
                chain.append(kt_h)
                qk_b = kt_h.outputs[0]
            else:
                qk_b = k_h
            qk_h = _two_op_gemm(
                model, f'gemm_{qk.name}_h{h}', q_h, qk_b,
                gemm_m=seq_q, gemm_k=key_dim, gemm_n=seq_k,
                out_shape=[seq_q, seq_k], out_prec=qk_prec, extra=qk_cfg,
            )
            chain.append(qk_h)

            # Per-head bit-exact softmax over seq_k (n_outer = seq_q rows, n_inner = 1).
            # The bit-exact tables are head-invariant, so copy them verbatim; the
            # required attrs (e.g. 'activation') must be present at construction.
            sm_attrs = {key: sm.attributes[key] for key in _SOFTMAX_CARRY_KEYS if key in sm.attributes}
            # n_slice is the reduction length (seq_k). It defaults to n_in when unset,
            # which would reduce over the whole per-head frame instead of one row and
            # break normalization (rows summing to >1) — set it explicitly.
            sm_attrs.update({'n_in': seq_q * seq_k, 'n_slice': seq_k, 'n_outer': seq_q, 'n_inner': 1})
            sm_h = model.make_node('Softmax', f'{sm.name}_h{h}', sm_attrs, [qk_h.outputs[0]])
            sm_h.get_output_variable().type.precision = sm_prec
            chain.append(sm_h)
            # The softmax output quantizer is the one genuinely in-span quantizer (on the
            # per-head attention matrix); clone it per head when it survived.
            sm_out = sm_h.outputs[0]
            if softmax_oq is not None:
                qn = _clone_quantizer_head(model, softmax_oq, h, sm_h.outputs[0], H)
                chain.append(qn)
                sm_out = qn.outputs[0]

            # A.V: contract = seq_k, output = key_dim. B beat layout: col-major needs B as
            # [key_dim, seq_k] (K-inner), so transpose the V lane ([seq_k, key_dim]); row-major
            # needs B as [seq_k, key_dim] (N-inner) = V directly, so no transpose. One transpose
            # per block either way -- row-major just moves it from A.V to QK^T.
            if row_major:
                av_b = v_h
                av_chain = []
            else:
                vt_h = model.make_node('Transpose', f'{av.name}_vt_h{h}', {'perm': [1, 0]}, [v_h])
                av_b = vt_h.outputs[0]
                av_chain = [vt_h]

            av_h = _two_op_gemm(
                model, f'gemm_{av.name}_h{h}', sm_out, av_b,
                gemm_m=seq_q, gemm_k=seq_k, gemm_n=key_dim,
                out_shape=[seq_q, key_dim], out_prec=av_prec, extra=av_cfg,
            )
            chain += av_chain + [av_h]
            # The A.V output quantizer feeds the output projection; clone per head, then merge.
            out_t = av_h.outputs[0]
            if out_iq is not None:
                qn = _clone_quantizer_head(model, out_iq, h, av_h.outputs[0], H)
                chain.append(qn)
                out_t = qn.outputs[0]
            av_head_out.append(out_t)

        # --- 4. HeadMerge: H x [seq, key_dim] -> [seq, d_model], before the output projection.
        hm = model.make_node(
            HeadMerge,
            f'{av.name}_merge',
            {'n_heads': H, 'key_dim': key_dim, 'seq': seq_q, 'd_model': d_model},
            av_head_out,
        )
        hm.get_output_variable().type.precision = (out_iq or av).get_output_variable().type.precision
        # The output projection now reads the merged 2D context (same flat data the
        # 3D [seq, head, key_dim] carried; O_proj contracts head*key_dim = d_model).
        o_in = out_iq.outputs[0] if out_iq is not None else av.outputs[0]
        _rewire_input(o_proj, o_in, hm.outputs[0])

        # --- 5. Rebuild the graph: drop qk/sm/av (plus any surviving cluster quantizers,
        #        now cloned per head), splice splits after projections and the per-head
        #        chain + merge before the output projection.
        drop = {qk.name, sm.name, av.name}
        for q in (softmax_oq, out_iq, query_oq, key_oq, value_oq):
            if q is not None:
                drop.add(q.name)
        new_nodes = list(splits.values()) + chain + [hm]
        _rebuild_graph(
            model,
            drop=drop,
            after={q_proj.name: [splits['Q']], k_proj.name: [splits['K']], v_proj.name: [splits['V']]},
            before={o_proj.name: chain + [hm]},
            new_nodes=new_nodes,
        )
        return True


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _consumers(model, tensor_name):
    return [n for n in model.graph.values() if tensor_name in n.inputs]


def _rewire_input(node, old_tensor, new_tensor):
    node.inputs = [new_tensor if i == old_tensor else i for i in node.inputs]


def _two_op_gemm(model, name, in0, in1, gemm_m, gemm_k, gemm_n, out_shape, out_prec, extra=None):
    """Build a plain two-operand Gemm (both operands activations, no constant weight).

    ``extra`` carries the resolved GEMM config (strategy/reuse_factor/... and
    second_operand_row_major) so the per-head Gemms mirror the source einsum's knobs for
    the manifest and the b_row_major config const. in1 (B) must already be oriented so its
    last axis matches the chosen beat layout (K-inner for col-major, N-inner for row-major)."""
    attrs = {
        'n_in': gemm_k,
        'n_out': gemm_n,
        'n_patches': gemm_m,
        'n_inplace': 1,
        'strategy': 'gemm',
        'gemm_m': gemm_m,
        'gemm_k': gemm_k,
        'gemm_n': gemm_n,
        'weights_in_core': False,
        '_original_type': 'Einsum',
        '_gemm_output_shape': list(out_shape),
    }
    if extra:
        attrs.update(extra)
    g = model.make_node(Gemm, name, attrs, [in0, in1])
    g.get_output_variable().type.precision = out_prec
    return g


def _rebuild_graph(model, drop, after, before, new_nodes):
    """Rebuild model.graph as an ordered dict: keep existing order minus `drop`,
    splice `after[name]` right after each named node and `before[name]` right before.
    `new_nodes` are the freshly created node objects to register in the dict.
    """
    registry = dict(model.graph)
    for n in new_nodes:
        registry[n.name] = n

    order = []
    for name in model.graph:
        if name in drop:
            continue
        for pre in before.get(name, []):
            order.append(pre.name)
        order.append(name)
        for post in after.get(name, []):
            order.append(post.name)

    model.graph = OrderedDict((name, registry[name]) for name in order)

    # Drop the removed nodes' output variables from the output registry.
    for name in drop:
        for out in registry[name].outputs:
            model.output_vars.pop(out, None)


# ---------------------------------------------------------------------------
# heterogeneous-quantizer handling (QAT-trained attention)
# ---------------------------------------------------------------------------
#
# hls4ml fuses a homogeneous FixedPointQuantizer into its neighbour, so an
# untrained / homogeneous MHA presents QK -> Softmax -> A.V directly adjacent. A
# QAT-trained model has *heterogeneous* (per-element) quantizers that cannot fuse
# and survive as nodes between the cluster ops. The pass navigates past them and
# clones each into the H per-head lanes with its head-axis precision slice, so any
# QAT model lowers through the head-split. (These helpers are inert on the
# homogeneous path -- there is simply nothing to skip or clone.)


def _producer(model, tensor_name):
    for n in model.graph.values():
        if tensor_name in n.outputs:
            return n
    return None


def _is_fixed_point_quantizer(node):
    return node is not None and node.class_name == 'FixedPointQuantizer'


def _skip_quantizers_fwd(model, node):
    """Follow the single-consumer chain past any FixedPointQuantizer nodes.

    Returns (first non-quantizer consumer, last skipped quantizer or None).
    """
    q = None
    consumers = _consumers(model, node.outputs[0])
    nxt = consumers[0] if consumers else None
    while _is_fixed_point_quantizer(nxt):
        q = nxt
        consumers = _consumers(model, nxt.outputs[0])
        nxt = consumers[0] if consumers else None
    return nxt, q


def _skip_quantizers_back(model, tensor_name):
    """Walk producers back past any FixedPointQuantizer to the real source node.

    Returns (first non-quantizer producer, last skipped quantizer or None).
    """
    node = _producer(model, tensor_name)
    q = None
    while _is_fixed_point_quantizer(node):
        q = node
        node = _producer(model, node.inputs[0])
    return node, q


def _head_axis(mask_kbi, H):
    """Axis of a quantizer's per-element (k, b, i) mask that indexes the head.

    The mask carries a leading batch axis; head is the (non-batch) axis whose length
    is H (ax1 for the attention matrix [H, seq, seq]; ax2 for a projection
    [seq, H, key_dim]).
    """
    import numpy as np

    x = np.asarray(mask_kbi[0])
    candidates = [ax for ax in range(1, x.ndim) if x.shape[ax] == H]
    if len(candidates) != 1:
        raise RuntimeError(f'SplitAttentionHeads: cannot locate a unique head axis (len H={H}) in mask shape {x.shape}')
    return candidates[0]


def _clone_quantizer_head(model, orig, h, in_tensor, H):
    """Clone a heterogeneous FixedPointQuantizer for one head, slicing its per-element
    (k, b, i) mask along the head axis so each lane keeps its own learned precision.
    """
    import numpy as np

    from hls4ml.model.optimizer.passes.hgq_proxy_model import FixedPointQuantizer

    ha = _head_axis(orig.mask_kbi, H)
    sliced = tuple(np.ascontiguousarray(np.take(np.asarray(x), h, axis=ha)) for x in orig.mask_kbi)
    name = f'{orig.name}_h{h}'
    attrs = {
        'name': name,  # the call template reads attributes['name']; make_node does not set it
        'overrides': orig.attributes['overrides'],
        'fusible': orig.attributes.get('fusible', False),
        'SAT': orig.SAT,
        'RND': orig.RND,
        'mask_kbi': sliced,
    }
    q = model.make_node(FixedPointQuantizer, name, attrs, [in_tensor])
    q.get_output_variable().type.precision = orig.get_output_variable().type.precision
    return q
