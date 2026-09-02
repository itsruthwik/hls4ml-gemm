"""Shared GEMM IP configuration lookups and template bases.

Used by both the Catapult and Vivado/Vitis backends so the ``Strategy: GEMM``
resolution (Model / LayerType / LayerName) and the GemmStream/GemmArray
config formatting cannot silently diverge.

Note: the template base lives here (outside the backend passes
directories) deliberately — the optimizer registry no-arg instantiates
every OptimizerPass subclass *defined* in a passes module, so an abstract
base there would break discovery.
"""

import warnings

from hls4ml.backends.template import LayerConfigTemplate


def is_gemm_strategy(layer):
    """Return True if *layer* opts into the GEMM IP path via ``Strategy: GEMM``.

    GEMM is a mutually-exclusive Strategy value — a peer of Latency / Resource /
    distributed_arithmetic, not an orthogonal flag. It is resolved by the stock
    ``get_strategy`` chain (LayerName → LayerType → Model), so a model-wide default
    is ``config['Model']['Strategy'] = 'GEMM'``. There is no separate boolean opt-in.

    ``get_strategy`` returns the snake-cased strategy, and ``convert_to_snake_case``
    mangles the all-caps acronym ``GEMM`` into ``g_e_m_m`` (unlike ``Latency`` →
    ``latency``). Strip the underscores so every spelling — ``GEMM``, ``Gemm``,
    ``gemm`` — normalizes to the canonical ``gemm``.
    """
    strategy = layer.model.config.get_strategy(layer)
    return isinstance(strategy, str) and strategy.replace('_', '').lower() == 'gemm'


class GemmIPConfigTemplateBase(LayerConfigTemplate):
    """Shared format() for the GemmStream/GemmArray config templates.

    Subclasses set ``backend_name`` ('catapult' or 'vivado') and bind their
    layer class and template string in ``__init__``.
    """

    backend_name = None

    def format(self, node):
        from hls4ml.backends import get_backend

        params = self._default_config_params(node)
        params['accum_t'] = node.types['accum_t']
        params['product_type'] = get_backend(self.backend_name).product_type(
            node.get_input_variable().type.precision, node.weights['weight'].type.precision
        )
        # Microarchitecture knobs for the generic GEMM core: the row-loop pipeline II is
        # the reuse factor; the multiplier ALLOCATION cap is ceil(K*N / reuse) so
        # reuse=1 leaves the full K*N array (Latency) and larger reuse shares it (Resource).
        rf = max(1, int(node.get_attr('reuse_factor', 1) or 1))
        gk = int(node.get_attr('gemm_k', node.get_attr('n_in')))
        gn = int(node.get_attr('gemm_n', node.get_attr('n_out')))
        params['reuse_factor'] = rf
        params['multiplier_limit'] = -(-(gk * gn) // rf)
        weight_type_name = node.weights['weight'].type.name
        bias_type_name = node.weights['bias'].type.name
        if weight_type_name == 'model_default_t' or bias_type_name == 'model_default_t':
            warnings.warn(
                f"{type(node).__name__} node '{node.name}': weight_t='{weight_type_name}' and "
                f"bias_t='{bias_type_name}' — one or both resolved to the global "
                "model_default_t. If you intended a per-layer precision override "
                "(e.g. INT8), ensure the source layer name is set in "
                "hls_config['LayerName'] before the GEMM lowering.",
                stacklevel=2,
            )
        return self.template.format(**params)
