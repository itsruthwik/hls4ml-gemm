import numpy as np

from hls4ml.model.layers import (
    GRU,
    LSTM,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    EinsumDense,
    SeparableConv1D,
    SeparableConv2D,
)
from hls4ml.model.optimizer import OptimizerPass


class ApplyResourceStrategy(OptimizerPass):
    """Transposes the weights to use the dense_resource matrix multiply routine"""

    def match(self, node):
        node_matches = isinstance(node, (Dense, EinsumDense, Conv1D, SeparableConv1D, Conv2D, SeparableConv2D, LSTM, GRU, Bidirectional))
        is_resource_strategy = node.get_attr('strategy', '').lower() in ['resource', 'resource_unrolled']
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
            w = np.transpose(w, axes=[0, 2, 1])  # (I, C, L1) -> (I, L1, C)

            # Single (I, L1, C) Resource layout for both io types: the io_stream kernel
            # (nnet_einsum_dense_stream.h) computes each row with the same dense_resource
            # kernel the io_parallel array core uses (nnet_einsum_dense.h), so it reads
            # weights in the identical layout -- no separate io_stream permutation.
            node.weights['weight'].data = w
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
        elif isinstance(node, (Bidirectional)):
            for d in ['forward', 'backward']:
                node.weights[f'{d}_weight'].data = np.transpose(node.weights[f'{d}_weight'].data)
                node.weights[f'{d}_recurrent_weight'].data = np.transpose(node.weights[f'{d}_recurrent_weight'].data)
        elif isinstance(node, (LSTM, GRU)):
            node.weights['weight'].data = np.transpose(node.weights['weight'].data)
            node.weights['recurrent_weight'].data = np.transpose(node.weights['recurrent_weight'].data)
        else:
            raise Exception(f'Unexpected layer {node.class_name} with resource strategy')

        node.set_attr('_weights_transposed', True)

        return False
