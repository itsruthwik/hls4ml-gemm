import numpy as np
from hls4ml.model.layers import Layer, Dense, Conv1D, Conv2D, Einsum, EinsumDense, Im2Col, register_layer
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.attributes import Attribute, WeightAttribute, TypeAttribute

class Gemm(Layer):
    """Math-only consumer of the GEMM IP.

    One node for every plain GEMM (Dense, pointwise Conv, and — from Phase 2 —
    the attention einsums). The interface (stream vs array) is NOT encoded in the
    class: it is read from the model's IOType in the template, exactly as every
    other hls4ml layer sources it. The two axes that DO live on the node are
    ``weights_in_core`` (is one operand a compile-time constant?) and ``n_inplace``
    (batched GEMMs, e.g. one per attention head). Together with IOType they select
    among the four gemm_* signatures.
    """
    # weight/bias are NOT required: the const_weights path (Dense/Conv/EinsumDense) passes
    # them as data and initialize() materializes the variables; the two-operand path
    # (attention QK^T / A.V) has no constant weight at all. Only accum is always needed.
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),
        Attribute('n_patches'),
        Attribute('n_inplace', default=1),
        Attribute('weights_in_core', value_type=bool, default=False),
        TypeAttribute('accum'),
    ]

    def initialize(self):
        _shape_hint = self.attributes.get('_gemm_output_shape', None)
        if _shape_hint is not None:
            shape = list(_shape_hint)
        else:
            shape = [self.attributes['n_patches'], self.attributes['n_out']]
        self.add_output_variable(shape)

        if not self.get_attr('weights_in_core', True):
            # Two-operand GEMM (attention QK^T / A.V): both operands are activations
            # (node.inputs[0]=A, node.inputs[1]=B). No constant weight; the zero bias
            # is emitted locally by the template, so no weight/bias variables here.
            return

        # Explicitly pass data to add_weights_variable to avoid NoneType errors
        weight_data = self.get_attr('weight')
        self.add_weights_variable(name='weight', data=weight_data, quantizer=self.get_attr('weight_quantizer'))

        bias_data = self.get_attr('bias')
        self.add_weights_variable(name='bias', data=bias_data, quantizer=self.get_attr('bias_quantizer'))


class Im2ColGemm(Layer):
    """Fused Im2Col + GEMM IP for strided/non-pointwise Conv.

    Kept fused (im2col is streamed straight into the GEMM) and kept out of the
    plain ``Gemm`` node — im2col is orthogonal to the stream/array and
    weights_in_core axes. Like ``Gemm``, the interface is chosen from IOType in
    the template, not by the class.
    """
    _expected_attributes = [
        Attribute('n_in'),
        Attribute('n_out'),
        Attribute('n_patches'),
        Attribute('n_inplace', default=1),
        Attribute('in_height'),
        Attribute('in_width'),
        Attribute('n_chan'),
        Attribute('filt_height'),
        Attribute('filt_width'),
        Attribute('stride_height'),
        Attribute('stride_width'),
        Attribute('pad_top'),
        Attribute('pad_bottom'),
        Attribute('pad_left'),
        Attribute('pad_right'),
        Attribute('out_height'),
        Attribute('out_width'),
        Attribute('data_format', value_type=str),
        Attribute('gemm_m'),
        Attribute('im2col_tile_rows'),
        WeightAttribute('weight'),
        WeightAttribute('bias'),
        TypeAttribute('weight'),
        TypeAttribute('bias'),
        TypeAttribute('accum'),
    ]

    def initialize(self):
        _shape_hint = self.attributes.get('_gemm_output_shape', None)
        if _shape_hint is not None:
            shape = list(_shape_hint)
        else:
            shape = [self.attributes['n_patches'], self.attributes['n_out']]
        self.add_output_variable(shape)
        
        weight_data = self.get_attr('weight')
        if hasattr(weight_data, 'data'):
            weight_data = weight_data.data
        bias_data = self.get_attr('bias')
        if hasattr(bias_data, 'data'):
            bias_data = bias_data.data

        self.add_weights_variable(name='weight', data=weight_data, quantizer=self.get_attr('weight_quantizer'))
        self.add_weights_variable(name='bias', data=bias_data, quantizer=self.get_attr('bias_quantizer'))

register_layer('Gemm', Gemm)
register_layer('Im2ColGemm', Im2ColGemm)


# ---------------------------------------------------------------------------
# weights_in_core: does this layer have a CONSTANT operand?
# ---------------------------------------------------------------------------
# This is a property of the LAYER and is independent of IOType. The two axes are
# orthogonal:
#
#   weights_in_core  -> is one operand constant?   (the layer)
#   interface        -> ac_channel or array?       (IOType)
#
# A Dense layer's kernel is constant whether it runs io_stream or io_parallel, so
# deriving this from IOType (as an earlier revision did) both mislabels io_parallel
# Dense and blocks io_parallel weight-stationary support.
#
# True  : Dense, Conv (pointwise and im2col), EinsumDense — the kernel is constant.
# False : Einsum QK^T / A.V — both operands are per-frame activations.
_CONST_OPERAND_LAYERS = ('Dense', 'Conv1D', 'Conv2D', 'PointwiseConv1D', 'PointwiseConv2D',
                         'Conv1DBatchnorm', 'Conv2DBatchnorm', 'EinsumDense')


def layer_has_const_operand(class_name):
    """True when one GEMM operand is a compile-time constant for *class_name*."""
    return class_name in _CONST_OPERAND_LAYERS


def _bias_tensor_is_nonzero(bias_data):
    """True when a bias tensor exists and is not all-zero.

    has_bias is a fact derived from the IR (the bias tensor itself), not from any
    attribute -- nothing sets a 'use_bias' attribute on the node, so reading it always
    fell back to its True default and has_bias was a dead constant.
    """
    if bias_data is None:
        return False
    try:
        return bool(np.any(np.asarray(bias_data) != 0))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Precision mirroring helper
# ---------------------------------------------------------------------------
# Precision vars that may carry per-layer overrides from the user's hls_config.
_GEMM_PRECISION_VARS = ('weight', 'bias', 'accum', 'result', 'default')


def _is_auto_precision(precision):
    return precision is None or (isinstance(precision, str) and precision.lower() == 'auto')


def _lookup_effective_precision(cfg, source_node, source_layer_name, var):
    source_name = source_layer_name.lower()
    class_name = source_node.class_name.lower() if source_node is not None else None

    candidates = [
        cfg.layer_name_precision.get(source_name + '_' + var),
        cfg.layer_name_precision.get(source_name + '_default'),
    ]

    if class_name is not None:
        candidates.extend(
            [
                cfg.layer_type_precision.get(class_name + '_' + var),
                cfg.layer_type_precision.get(class_name + '_default'),
            ]
        )

    candidates.extend(
        [
            cfg.model_precision.get(var),
            cfg.model_precision.get('default'),
        ]
    )

    for precision in candidates:
        if not _is_auto_precision(precision):
            return precision

    return None


def _mirror_precision_to_gemm_node(model, source_layer_name, dest_node_name):
    """Mirror effective source precision to a synthetic GEMM node.

    When a Dense/Conv layer (e.g. ``dense_target``) is replaced by a ``GemmStream``
    node (e.g. ``gemm_dense_target``), the new node's name is not present in
    ``HLSConfig.layer_name_precision``.  Explicitly materialize the source
    layer's effective precision under the synthetic name so its weights, bias,
    accumulators, and outputs do not fall back to backend defaults when the
    source layer precision is ``auto`` but model-level precision is specific.

    Args:
        model:             The ``ModelGraph`` (holds ``model.config``).
        source_layer_name: Original layer name (e.g. ``'dense_target'``).
        dest_node_name:    Synthetic node name  (e.g. ``'gemm_dense_target'``).
    """
    cfg = model.config
    source_node = model.graph.get(source_layer_name)

    precision_overrides = {}
    for var in _GEMM_PRECISION_VARS:
        precision = _lookup_effective_precision(cfg, source_node, source_layer_name, var)
        if precision is not None:
            precision_overrides[var] = precision

    if not precision_overrides:
        return

    synthetic_cfg = {'Precision': precision_overrides}
    cfg.parse_name_config(dest_node_name, synthetic_cfg)


# ---------------------------------------------------------------------------
# GEMM config resolution
# ---------------------------------------------------------------------------
# Resolve config values (Strategy, ReuseFactor, ...) from the SOURCE layer and store
# the concrete per-layer value on the GEMM node (resolve-then-store). Resolution MUST
# happen here: after replacement the GEMM node's name is not in HLSConfig, so a later
# read would miss the user's per-layer settings (same reason precision is mirrored
# above). hls4ml carries these into the manifest; what a gemm-ip-gen target does with
# them is the target's own concern.
def _resolve_config_key(cfg, source_node, key, default):
    """Look up *key* with a proper LayerName -> LayerType -> Model fallback by KEY presence.

    ``HLSConfig.get_layer_config_value`` short-circuits on the first scope that exists for the
    layer (so a layer with any LayerName entry never falls back to Model for a missing key).
    For a broad model-level knob like ``SecondOperandRowMajor`` we want the key to fall through
    scope by scope, so resolve it explicitly here.
    """
    hls = cfg.config.get('HLSConfig', {})
    for scope in (
        hls.get('LayerName', {}).get(source_node.name),
        hls.get('LayerType', {}).get(source_node.class_name),
        hls.get('Model'),
    ):
        if scope is not None and key in scope:
            return scope[key]
    return default


def _resolve_gemm_config(model, source_node):
    cfg = model.config

    def _safe(fn, default):
        try:
            value = fn()
        except Exception:
            return default
        return default if value is None else value

    def _strategy():
        # The source layer's init already normalized its resolved strategy onto the
        # node attribute (canonical 'gemm'/'latency'/'resource'/'distributed_arithmetic').
        # Prefer it — get_strategy returns the snake-cased config value, which mangles
        # the all-caps 'GEMM' into 'g_e_m_m'.
        strategy = source_node.get_attr('strategy')
        if strategy is None:
            strategy = cfg.get_strategy(source_node)
        return strategy.lower() if isinstance(strategy, str) else strategy

    return {
        'strategy': _safe(_strategy, 'latency'),
        'reuse_factor': _safe(lambda: cfg.get_reuse_factor(source_node), 1),
        'parallelization_factor': _safe(
            lambda: cfg.get_layer_config_value(source_node, 'ParallelizationFactor', 1), 1
        ),
        'target_cycles': _safe(lambda: cfg.get_target_cycles(source_node), None),
        'sparse': bool(_safe(lambda: cfg.get_compression(source_node), False)),
        # Two-operand only: stream the B operand row-major (N-wide beats, one contraction
        # row per beat) instead of the default col-major (K-wide beats, one output column
        # per beat). Required to route a two-operand GEMM to the mvau IP (its runtime load
        # streams for every tiling with no full-B buffer). Default False = today's behavior.
        'second_operand_row_major': bool(
            _safe(lambda: _resolve_config_key(cfg, source_node, 'SecondOperandRowMajor', False), False)
        ),
        # Im2Col-only knob (see SplitConvGemm): number of im2col rows written per
        # tile before the downstream GEMM IP is allowed to backpressure. Resolved
        # here for both branches; the Pointwise Gemm branch drops it (no im2col).
        'im2col_tile_rows': _safe(
            lambda: _resolve_config_key(cfg, source_node, 'Im2ColTileRows', None), None
        ),
    }


# ---------------------------------------------------------------------------
# Optimizer passes
# ---------------------------------------------------------------------------

class SplitConvGemm(OptimizerPass):
    def match(self, node):
        return isinstance(node, (Conv1D, Conv2D)) and node.get_attr('strategy') == 'gemm'

    def transform(self, model, node):
        is_pointwise = node.get_attr('filt_height', 1) == 1 and node.get_attr('filt_width') == 1

        # Capture the original Conv output shape so the GEMM node can restore it.
        original_output_shape = list(node.get_output_variable().shape)

        n_patches = node.get_attr('out_height', 1) * node.get_attr('out_width')
        bias_data = node.get_weights('bias').data

        # Shared GEMM attributes. The GEMM path uses row/column GEMM: gemm_m = n_patches
        # (full M, no tiling).
        gemm_attributes = {
            'n_in': node.get_attr('filt_height', 1) * node.get_attr('filt_width') * node.get_attr('n_chan'),
            'n_out': node.get_attr('n_filt'),
            'n_patches': n_patches,
            'gemm_m': n_patches,
            # Conv's kernel is constant, so this is always weight-stationary —
            # independent of IOType, and true for both the pointwise GemmStream path
            # and the fused Im2ColGemmStream path.
            'weights_in_core': layer_has_const_operand(node.class_name),
            'weight_quantizer': node.get_attr('weight_quantizer'),
            'bias_quantizer': node.get_attr('bias_quantizer'),
            'weight': node.get_weights('weight').data,
            'bias': bias_data,
            '_original_type': node.class_name,
            '_gemm_output_shape': original_output_shape,
            # Resolved config (resolve-then-store); see _resolve_gemm_config. Flows into
            # both the pointwise Gemm and the fused Im2ColGemm (via **gemm_attributes).
            'has_bias': _bias_tensor_is_nonzero(bias_data),
            **_resolve_gemm_config(model, node),
        }

        gemm_name = f'gemm_{node.name}'

        # Mirror any per-layer precision overrides from the source Conv layer BEFORE
        # make_node() so that initialize() → add_weights_variable() picks them up.
        _mirror_precision_to_gemm_node(model, node.name, gemm_name)

        # Preserve the Conv's resolved output precision (the HGQ2 activation quantizer)
        # so the GEMM intermediate's int8 code fits and rounds like Keras, and its
        # resolved weight precision so the type is not re-inferred (and widened past
        # int8) from raw float data — see ReplaceDenseGemm for the full rationale.
        original_output_precision = node.get_output_variable().type.precision
        original_weight_precision = node.get_weights('weight').type.precision

        # Interface axis (stream vs array) is NOT encoded in the node class: the
        # template reads it from IOType. The node carries only the layer facts.
        # Orthogonal to weights_in_core — see the two-axes note above.
        if is_pointwise:
            # Pointwise conv is a plain GEMM (no im2col) — drop the im2col-only knob.
            gemm_attributes.pop('im2col_tile_rows', None)
            gemm_node = model.make_node(
                Gemm, gemm_name, gemm_attributes, node.inputs.copy(), node.outputs.copy()
            )
        else:
            # Resolve/validate the im2col tile-row knob: default = out_width (one
            # output row, the natural gapless burst), else must fit 1..n_patches.
            tile_rows = gemm_attributes.get('im2col_tile_rows')
            if not tile_rows:
                tile_rows = node.get_attr('out_width')
            elif not (1 <= tile_rows <= n_patches):
                raise ValueError(
                    f"Layer '{node.name}': Im2ColTileRows={tile_rows} must satisfy "
                    f'1 <= Im2ColTileRows <= n_patches ({n_patches}).'
                )
            gemm_attributes['im2col_tile_rows'] = tile_rows
            # Non-pointwise → build the fused Im2Col+GEMM node DIRECTLY (previously this
            # was a Gemm+Im2Col pair immediately re-fused by FuseIm2ColGemm; the
            # merge removes that create-then-replace churn). _original_type must contain
            # the conv class name — gemm_transposition branches on it to pick the
            # [W,C,F]/[H,W,C,F] → [F, W*C]/[F, H*W*C] weight layout.
            fused_attributes = {
                'in_height': node.get_attr('in_height', 1),
                'in_width': node.get_attr('in_width'),
                'n_chan': node.get_attr('n_chan'),
                'filt_height': node.get_attr('filt_height', 1),
                'filt_width': node.get_attr('filt_width'),
                'stride_height': node.get_attr('stride_height', 1),
                'stride_width': node.get_attr('stride_width'),
                'pad_top': node.get_attr('pad_top', 0),
                'pad_bottom': node.get_attr('pad_bottom', 0),
                'pad_left': node.get_attr('pad_left', 0),
                'pad_right': node.get_attr('pad_right', 0),
                'out_height': node.get_attr('out_height', 1),
                'out_width': node.get_attr('out_width'),
                'data_format': node.get_attr('data_format'),
                **gemm_attributes,
                '_original_type': f'Im2Col_{node.class_name}',
            }
            gemm_node = model.make_node(
                Im2ColGemm, gemm_name, fused_attributes, node.inputs.copy(), node.outputs.copy()
            )

        gemm_node.get_output_variable().type.precision = original_output_precision
        gemm_node.get_weights('weight').type.precision = original_weight_precision

        model.replace_node(node, gemm_node)
        return True

class ReplaceDenseGemm(OptimizerPass):
    def match(self, node):
        return isinstance(node, Dense) and node.get_attr('strategy') == 'gemm'

    def transform(self, model, node):
        input_shape = node.get_input_variable().shape
        original_output_shape = list(node.get_output_variable().shape)

        n_patches = int(np.prod(input_shape[:-1])) if len(input_shape) > 1 else 1

        # Dense's kernel is constant, so this is weight-stationary regardless of
        # IOType. IOType selects only the INTERFACE (stream vs array) in the template.
        weights_in_core = layer_has_const_operand(node.class_name)
        bias_data = node.get_weights('bias').data

        # The GEMM path uses row/column streaming: gemm_m = n_patches (full M, no tiling).
        gemm_attributes = {
            'n_in': node.get_attr('n_in'),
            'n_out': node.get_attr('n_out'),
            'n_patches': n_patches,
            'gemm_m': n_patches,
            'weights_in_core': weights_in_core,
            'weight_quantizer': node.get_attr('weight_quantizer'),
            'bias_quantizer': node.get_attr('bias_quantizer'),
            'weight': node.get_weights('weight').data,
            'bias': bias_data,
            '_original_type': 'Dense',
            '_gemm_output_shape': original_output_shape,
            # Resolved config (resolve-then-store); see _resolve_gemm_config.
            'has_bias': _bias_tensor_is_nonzero(bias_data),
            **_resolve_gemm_config(model, node),
        }

        gemm_name = f'gemm_{node.name}'

        # Mirror any per-layer precision overrides from the source Dense layer BEFORE
        # make_node() so that initialize() → add_weights_variable() picks them up.
        _mirror_precision_to_gemm_node(model, node.name, gemm_name)

        # Preserve the original layer's resolved OUTPUT precision. In the standard
        # path this is the HGQ2 activation-quantizer type (e.g. fixed<9,5,RND>),
        # not the wide accumulator. The GEMM IP feeds activations as int8 codes
        # (the fixed-point mantissa), so an intermediate that keeps the wide
        # accumulator type (fixed<16,6>) overflows int8 and feeds the next GEMM a
        # corrupted code. Re-applying the activation-quantizer precision makes the
        # intermediate's code fit and round exactly like Keras (bit-exact chains).
        original_output_precision = node.get_output_variable().type.precision

        gemm_node = model.make_node(Gemm, gemm_name, gemm_attributes, node.inputs.copy(), node.outputs.copy())
        gemm_node.get_output_variable().type.precision = original_output_precision

        model.replace_node(node, gemm_node)
        return True


# ---------------------------------------------------------------------------
# Attention: lower einsum nodes (EinsumDense projections, Einsum matmuls) to Gemm.
# One pass = one transformation (the attention -> GEMM lowering). transform()
# dispatches by node type to per-type helper methods, the idiomatic hls4ml shape
# for a pass handling several layer types (cf. InferPrecisionTypes._infer_*).
# ---------------------------------------------------------------------------


def _perm_is_identity(perm):
    """True when a transpose index map is the identity permutation."""
    return list(perm) == list(range(len(perm)))


class LowerEinsumToGemm(OptimizerPass):
    def match(self, node):
        return isinstance(node, (Einsum, EinsumDense)) and node.get_attr('strategy') == 'gemm'

    def transform(self, model, node):
        if isinstance(node, EinsumDense):
            return self._lower_einsum_dense(model, node)
        return self._lower_einsum(model, node)

    def _lower_einsum_dense(self, model, node):
        """EinsumDense (constant kernel) -> const_weights Gemm.

        Mirrors ReplaceDenseGemm. The kernel is already [n_inplace, K, N] from
        init_einsum_dense; the writer packs it as [K, N] keyed on _original_type
        ('EinsumDense'), and TransposeWeightsForGemmIP leaves it untouched.
        """
        n_inplace = node.attributes['n_inplace']
        if n_inplace != 1:
            # The packed-weight writer holds one [K, N] block per header; batched
            # (multi-head) projections would need a different packing. MHA projections
            # are n_inplace == 1, so refuse anything else loudly rather than guess.
            raise NotImplementedError(
                f"EinsumDense '{node.name}': GEMM lowering supports n_inplace==1 "
                f'(got {n_inplace}).'
            )
        # Projections carry identity transposes; non-identity perms need Transpose
        # nodes, which arrive with the Einsum (two-operand) lowering.
        if not (_perm_is_identity(node.attributes['inp_tpose_idxs'])
                and _perm_is_identity(node.attributes['out_tpose_idxs'])):
            raise NotImplementedError(
                f"EinsumDense '{node.name}': non-identity transpose lowering not wired yet."
            )

        # Bias shape decides the lowering. init_einsum_dense broadcasts the bias to
        # the full output (I, L0, L1); the const_weights GEMM IP bias PORT is one value
        # per column (L1), broadcast across the L0 rows. If the bias is constant along
        # the L0 (data/row) axis it collapses to that per-column port with no cost. If
        # it varies along L0 (EinsumDense bias_axes touches the data free axis), the
        # per-column port cannot express it — feed the core a zero per-column bias and
        # add the full per-element bias in the wrapper (see the row-varying template).
        L0 = node.attributes['n_free_data']
        L1 = node.attributes['n_free_kernel']
        bias_full = np.asarray(node.get_weights('bias').data).reshape(1, L0, L1)
        row_varying_bias = not np.allclose(bias_full, bias_full[:, :1, :])
        if row_varying_bias:
            bias_data = bias_full.reshape(-1)  # full per-element [L0*L1], added in-wrapper
        else:
            bias_data = bias_full[:, 0, :].reshape(-1)  # per-column [L1], on the IP port

        original_output_shape = list(node.get_output_variable().shape)
        gemm_attributes = {
            'n_in': node.attributes['n_contract'],
            'n_out': node.attributes['n_free_kernel'],
            'n_patches': node.attributes['n_free_data'],
            'n_inplace': 1,
            'gemm_m': node.get_attr('gemm_m', node.attributes['n_free_data']),
            'gemm_k': node.get_attr('gemm_k', node.attributes['n_contract']),
            'gemm_n': node.get_attr('gemm_n', node.attributes['n_free_kernel']),
            'weights_in_core': True,  # EinsumDense kernel is a compile-time constant
            'weight_quantizer': node.get_attr('weight_quantizer'),
            'bias_quantizer': node.get_attr('bias_quantizer'),
            'weight': node.get_weights('weight').data,
            'bias': bias_data,
            '_row_varying_bias': row_varying_bias,
            '_original_type': 'EinsumDense',
            '_gemm_output_shape': original_output_shape,
            # Resolved config (resolve-then-store); see _resolve_gemm_config.
            'has_bias': _bias_tensor_is_nonzero(bias_data),
            **_resolve_gemm_config(model, node),
        }

        gemm_name = f'gemm_{node.name}'
        _mirror_precision_to_gemm_node(model, node.name, gemm_name)

        original_output_precision = node.get_output_variable().type.precision
        original_weight_precision = node.get_weights('weight').type.precision

        gemm_node = model.make_node(Gemm, gemm_name, gemm_attributes, node.inputs.copy(), node.outputs.copy())
        gemm_node.get_output_variable().type.precision = original_output_precision
        gemm_node.get_weights('weight').type.precision = original_weight_precision

        model.replace_node(node, gemm_node)
        return True

    def _lower_einsum(self, model, node):
        """Einsum (QK^T / A.V) -> two-operand Gemm, with Transpose nodes for any
        non-identity operand/output permutation.

        The Gemm is pure canonical-in/canonical-out (A rows [I, L0, C], B columns
        [I, L1, C], C rows [I, L0, L1]). Whatever reorder the einsum equation implies
        is realized as explicit Transpose IR nodes around it — identity perms fold
        away. Both operands are activations (weights_in_core=False).
        """
        a = node.attributes
        inp0_perm = list(a['inp0_tpose_idxs'])
        inp1_perm = list(a['inp1_tpose_idxs'])
        out_perm = list(a['out_tpose_idxs'])
        canonical_out_shape = list(a['out_interpert_shape'])  # [I, L0, L1] before out-transpose
        original_output_shape = list(node.get_output_variable().shape)
        original_output_precision = node.get_output_variable().type.precision

        gemm_attributes = {
            'n_in': a['n_contract'],
            'n_out': a['n_free1'],
            'n_patches': a['n_free0'],
            'n_inplace': a['n_inplace'],
            'gemm_m': node.get_attr('gemm_m', a['n_free0']),
            'gemm_k': node.get_attr('gemm_k', a['n_contract']),
            'gemm_n': node.get_attr('gemm_n', a['n_free1']),
            'weights_in_core': False,  # both operands are per-frame activations
            '_original_type': 'Einsum',
            '_gemm_output_shape': canonical_out_shape,
            # Two-operand GEMM (QK^T / A.V) has no bias operand.
            'has_bias': False,
            **_resolve_gemm_config(model, node),
        }

        # SecondOperandRowMajor: canonical B is [I, L1, C] (C=K last -> K-wide beats). For the
        # mvau IP we want [I, C, L1] (L1=N last -> N-wide beats, one contraction row per beat),
        # so swap the last two axes of B's canonicalizing perm. The Gemm's gemm_k/gemm_n and
        # its [I, L0, L1] output are unchanged; only B's stream beat layout flips. The
        # config/template/csim read B row-major off CONFIG_T::b_row_major (below).
        if gemm_attributes['second_operand_row_major']:
            inp1_perm = inp1_perm[:-2] + [inp1_perm[-1], inp1_perm[-2]]

        gemm_name = f'gemm_{node.name}'
        gemm_node = model.make_node(Gemm, gemm_name, gemm_attributes, node.inputs.copy(), node.outputs.copy())
        # The Gemm's output is the CANONICAL [I, L0, L1]; the out-transpose (below)
        # restores the einsum's declared output shape/precision.
        gemm_node.get_output_variable().type.precision = original_output_precision
        model.replace_node(node, gemm_node)

        # Input operands: Transpose only the non-identity ones into canonical order.
        for input_idx, perm in ((0, inp0_perm), (1, inp1_perm)):
            if _perm_is_identity(perm):
                continue
            src = gemm_node.inputs[input_idx]
            tpose = model.make_node(
                'Transpose', f'{gemm_name}_tpose_in{input_idx}', {'perm': perm}, [src]
            )
            model.insert_node(tpose, before=gemm_node, input_idx=input_idx)

        # Output: Transpose canonical [I, L0, L1] to the einsum's declared order.
        if not _perm_is_identity(out_perm):
            out_tpose = model.make_node(
                'Transpose', f'{gemm_name}_tpose_out', {'perm': out_perm}, [gemm_node.outputs[0]]
            )
            model.insert_node(out_tpose)
            out_tpose.get_output_variable().type.precision = original_output_precision
            assert list(out_tpose.get_output_variable().shape) == original_output_shape, (
                f'{node.name}: out-transpose shape {out_tpose.get_output_variable().shape} '
                f'!= einsum output {original_output_shape}'
            )

        return True


class ValidateGemm(OptimizerPass):
    """Structural contract checks for GEMM-IP nodes.

    Runs after the packed stream types are assigned, so it can read the real
    per-beat width the codegen will use. A home for GEMM-IP legality rules;
    add new checks as helper methods and call them from :meth:`transform`.
    Raises with an actionable message instead of letting a violation reach a
    cryptic C++ static_assert deep inside Catapult analyze.
    """

    def match(self, node):
        return isinstance(node, Gemm)

    def transform(self, model, node):
        io_type = model.config.get_config_value('IOType')
        self._check_stream_beat_carries_full_k(node, io_type)
        self._check_two_operand_b_beat(node, io_type)
        return False  # validation only; never mutates the graph

    def _check_two_operand_b_beat(self, node, io_type):
        # Two-operand io_stream GEMM: the B operand's beat width must match its declared
        # layout -- gemm_k when col-major (K-wide, one output column per beat) or gemm_n
        # when row-major (N-wide, one contraction row per beat, SecondOperandRowMajor).
        # Fail here with an actionable message rather than at the C++ static_assert.
        if io_type != 'io_stream' or node.get_attr('weights_in_core', True):
            return
        if len(node.inputs) < 2:
            return
        inp = node.get_input_variable(node.inputs[1])
        n_elem = getattr(inp.type, 'n_elem', None)
        if n_elem is None:
            return  # not a packed stream type; nothing to check
        n_pack = getattr(inp.type, 'n_pack', 1)
        beat = n_elem // n_pack if getattr(inp.type, 'unpack', False) else n_elem * n_pack
        row_major = node.get_attr('second_operand_row_major', False)
        gemm_k = node.get_attr('gemm_k', node.get_attr('n_in'))
        gemm_n = node.get_attr('gemm_n', node.get_attr('n_out'))
        want = gemm_n if row_major else gemm_k
        if beat != want:
            layout = 'row-major (N-wide)' if row_major else 'col-major (K-wide)'
            raise Exception(
                f"Gemm '{node.name}': two-operand io_stream GEMM-IP with {layout} B expects the "
                f"B beat width == {'gemm_n' if row_major else 'gemm_k'} ({want}), but the second "
                f"input beat width is {beat}. Check the B producer's layout / the "
                f"SecondOperandRowMajor setting on layer '{node.name}'."
            )

    def _check_stream_beat_carries_full_k(self, node, io_type):
        # io_stream const_weights Dense: the generated core consumes A one beat per
        # read and requires each beat to carry the whole K-row (a_beat_T::size ==
        # gemm_k). A producer whose beat is narrower than gemm_k (e.g. a Conv/Pool
        # output streamed over several beats, then Flatten -> Dense) violates this.
        if io_type != 'io_stream' or not node.get_attr('weights_in_core', True):
            return
        inp = node.get_input_variable(node.inputs[0])
        n_elem = getattr(inp.type, 'n_elem', None)
        if n_elem is None:
            return  # not a packed stream type; nothing to check
        n_pack = getattr(inp.type, 'n_pack', 1)
        beat = n_elem // n_pack if getattr(inp.type, 'unpack', False) else n_elem * n_pack
        gemm_k = node.get_attr('gemm_k', node.get_attr('n_in'))
        if beat != gemm_k:
            raise Exception(
                f"Gemm '{node.name}': io_stream GEMM-IP requires one beat to carry the full "
                f"K-row (beat width == gemm_k), but the input beat width is {beat} while "
                f"gemm_k is {gemm_k}. This happens when a multi-beat activation (e.g. a "
                f"Conv/Pool output flattened into this Dense) feeds the const_weights GEMM-IP. "
                f"Set a non-GEMM Strategy (e.g. Latency/Resource) on layer '{node.name}' so it "
                f"uses the stock io_stream Dense, which gathers the beats itself."
            )
