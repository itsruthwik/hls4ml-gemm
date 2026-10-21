import numpy as np

from hls4ml.model.layers import (
    GRU,
    LSTM,
    Conv1D,
    Conv2D,
    Dense,
    DepthwiseConv1D,
    DepthwiseConv2D,
    EinsumDense,
    SeparableConv1D,
    SeparableConv2D,
)
from hls4ml.backends.fpga.fpga_layers import PointwiseConv1D, PointwiseConv2D
from hls4ml.model.optimizer import OptimizerPass


def block_major_weight_keys(node):
    """Weight keys the writer stores block-major as packed words for nnet::dense_resource (see
    nnet::weight_store), the Catapult counterpart of the Vivado ARRAY_RESHAPE block factor.

    Covers layers whose weight array is read whole by dense_resource (EinsumDense: one call per
    in-place slice). Depthwise, pointwise, separable and recurrent weights keep their order. The layer config templates derive
    CONFIG_T::block_major_weights from this same function, so both sides always agree.
    """
    if str(node.get_attr('strategy', '')).lower() != 'resource':
        return ()
    # Depthwise weights go to other kernels; the Vivado pointwise conv partitions its weights
    # completely (no ROM), so pointwise keeps the flat, constant layout too.
    if isinstance(node, (DepthwiseConv1D, DepthwiseConv2D, PointwiseConv1D, PointwiseConv2D)):
        return ()
    if not isinstance(node, (Dense, EinsumDense, Conv1D, Conv2D)):
        return ()
    # Block-major positions ir*block_factor + im only tile the array when the reuse factor
    # divides the length one dense_resource call reads; otherwise keep the original order.
    weight = node.get_weights('weight')
    n_calls = weight.data.shape[0] if isinstance(node, EinsumDense) else 1
    per_call, rf = weight.data_length // n_calls, node.get_attr('reuse_factor', 1)
    # RF 1 reads the whole array in one iteration, so it stays flat (the writer never packs it).
    if weight.data_length % n_calls or not rf or rf <= 1 or per_call % rf:
        return ()
    return ('weight',)


class ApplyResourceStrategy(OptimizerPass):
    """Transposes the weights to use the dense_resource matrix multiply routine"""

    def match(self, node):
        node_matches = isinstance(node, (Dense, EinsumDense, Conv1D, SeparableConv1D, Conv2D, SeparableConv2D, LSTM, GRU))
        is_resource_strategy = node.get_attr('strategy', '').lower() == 'resource'
        already_transformed = node.get_attr('_weights_transposed', False) is True

        return node_matches and is_resource_strategy and not already_transformed

    def transform(self, model, node):
        if isinstance(node, Dense):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
        elif isinstance(node, EinsumDense):
            # init_einsum_dense stores the kernel as (n_inplace, n_contract, n_free_kernel), i.e. one
            # (n_in, n_out) latency-layout matrix per in-place slice. dense_resource reads (n_out, n_in).
            w = node.weights['weight'].data
            assert w.ndim == 3, f'EinsumDense weight expected 3-D (I, C, L1), got shape {w.shape}'
            node.weights['weight'].data = np.transpose(w, axes=[0, 2, 1])
        elif isinstance(node, Conv1D):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data, axes=[2, 0, 1])  # (W,C,F) => (F,W,C)
        elif isinstance(node, SeparableConv1D):
            node.weights['depthwise'].data = np.transpose(
                node.weights['depthwise'].data, axes=[2, 0, 1]
            )  # (W,C,F) => (F,W,C)
            node.weights['pointwise'].data = np.transpose(
                node.weights['pointwise'].data, axes=[2, 0, 1]
            )  # (W,C,F) => (F,W,C)
        elif isinstance(node, Conv2D):
            node.weights['weight'].data = np.transpose(
                node.weights['weight'].data, axes=[3, 0, 1, 2]
            )  # (H,W,C,F) => (F,H,W,C)
        elif isinstance(node, SeparableConv2D):
            node.weights['depthwise'].data = np.transpose(
                node.weights['depthwise'].data, axes=[3, 0, 1, 2]
            )  # (H,W,C,F) => (F,H,W,C)
            node.weights['pointwise'].data = np.transpose(
                node.weights['pointwise'].data, axes=[3, 0, 1, 2]
            )  # (H,W,C,F) => (F,H,W,C)
        elif isinstance(node, (LSTM, GRU)):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)
        else:
            raise Exception(f'Unexpected layer {node.class_name} with resource strategy')

        node.set_attr('_weights_transposed', True)

        return False
