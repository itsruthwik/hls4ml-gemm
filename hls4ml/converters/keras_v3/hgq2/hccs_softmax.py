import typing
from collections.abc import Sequence
from math import prod

from ._base import QLayerHandler
from .softmax import fixed_quantizer_to_hls4ml_t

if typing.TYPE_CHECKING:
    import hgq
    from keras import KerasTensor


class QHCCSSoftmaxHandler(QLayerHandler):
    handles = ('hgq.layers.hccs_softmax.QHCCSSoftmax',)

    def handle(
        self,
        layer: 'hgq.layers.QHCCSSoftmax',
        in_tensors: Sequence['KerasTensor'],
        out_tensors: Sequence['KerasTensor'],
    ):
        if len(in_tensors) != 1:
            raise ValueError(
                f'Too many inputs for HCCS softmax layer {layer.name}: expected 1, got {len(in_tensors)}'
            )

        ax = layer.axes[0]
        ax = ax if ax >= 0 else len(in_tensors[0].shape) + ax
        # io_stream asserts axes=-1, convert to -1 when it is
        n_outer: int = prod(in_tensors[0].shape[1:ax])  # type: ignore
        n_inner: int = prod(in_tensors[0].shape[ax + 1 :])  # type: ignore
        ax = -1 if ax == len(in_tensors[0].shape) - 1 else ax
        n_in: int = prod(in_tensors[0].shape[1:])  # type: ignore

        score_t = fixed_quantizer_to_hls4ml_t(layer.score_q.quantizer)
        inv_table_t = fixed_quantizer_to_hls4ml_t(layer.inv_oq.quantizer)
        inv_inp_t = fixed_quantizer_to_hls4ml_t(layer.inv_iq.quantizer)
        inp_norm_t = fixed_quantizer_to_hls4ml_t(layer.norm_q.quantizer)

        config = {}
        config.update(self.default_config)
        config.update(
            {
                'axis': ax,
                'n_in': n_in,
                'activation': 'softmax',
                'n_outer': n_outer,
                'n_inner': n_inner,
                'class_name': 'Softmax',
                'implementation': 'hccs',
                '_bit_exact': True,
                'recip_impl': layer.recip_impl,
                'recip_table_size': layer.recip_table_size,
                'score_t': score_t,
                'inv_table_t': inv_table_t,
                'inv_inp_t': inv_inp_t,
                'inp_norm_t': inp_norm_t,
                'hccs_b': float(layer.B),
                'hccs_s': float(layer.S),
                'hccs_dmax': float(layer.Dmax),
            }
        )

        return (config,)
