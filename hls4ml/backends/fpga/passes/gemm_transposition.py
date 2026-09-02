import numpy as np
from hls4ml.model.layers import Dense, Conv1D, Conv2D, SeparableConv1D, SeparableConv2D
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.backends.fpga.passes.gemm_nodes import Gemm, Im2ColGemm

class TransposeWeightsForGemmIP(OptimizerPass):
    """
    Transposes the weights for layers on the GEMM path (Strategy: GEMM).
    This ensures that the weights are stored in a format suitable for the GEMM IP
    (typically [N, K] where N is the number of filters/outputs and K is the number of inputs).
    """

    def match(self, node):
        # Match layers on the GEMM path (Strategy: GEMM) that haven't been transposed yet
        is_gemm = node.get_attr('strategy') == 'gemm'
        already_transposed = node.get_attr('_weights_transposed_for_gemm', False)

        # We only care about layers that have weights to transpose
        # After node splitting/fusion this is the unified Gemm or the fused
        # Im2ColGemm (whose _original_type 'Im2Col_Conv1D/2D' selects the conv
        # branch below — without it, raw [W,C,F]/[H,W,C,F] kernels reached the
        # writer and the packed weight columns came out scrambled).
        has_weights = isinstance(
            node, (Dense, Conv1D, Conv2D, SeparableConv1D, SeparableConv2D, Gemm, Im2ColGemm)
        )

        return is_gemm and has_weights and not already_transposed

    def transform(self, model, node):
        original_type = node.get_attr('_original_type', type(node).__name__)

        # EinsumDense-origin Gemm nodes keep their [K, N] kernel (the writer packs them
        # via the kn_source path). Do NOT transpose — and note 'Dense' is a substring of
        # 'EinsumDense', so this guard must precede the Dense branch below.
        if original_type == 'EinsumDense':
            node.set_attr('_weights_transposed_for_gemm', True)
            return False

        if 'Dense' in original_type or isinstance(node, Dense):
            # Dense weights are typically [K, N]. GEMM IP expects [N, K].
            if not node.get_attr('_weights_transposed', False):
                node.weights['weight'].data = np.transpose(node.weights['weight'].data)
                node.weights['weight'].shape = list(node.weights['weight'].data.shape)
                node.set_attr('_weights_transposed', True)
        
        elif 'Conv1D' in original_type or isinstance(node, Conv1D):
            # Conv1D weights are [W, C, F]. GEMM IP expects [F, W*C].
            data = node.weights['weight'].data
            if node.get_attr('_weights_transposed', False):
                # If already transposed by ApplyResourceStrategy, it's [F, W, C]
                node.weights['weight'].data = data.reshape(data.shape[0], -1)
            else:
                # It's [W, C, F]. Transpose to [F, W, C] then flatten.
                node.weights['weight'].data = np.transpose(data, axes=[2, 0, 1]).reshape(data.shape[2], -1)
                node.set_attr('_weights_transposed', True)
            node.weights['weight'].shape = list(node.weights['weight'].data.shape)

        elif 'Conv2D' in original_type or isinstance(node, Conv2D):
            # Conv2D weights are [H, W, C, F]. GEMM IP expects [F, H*W*C].
            data = node.weights['weight'].data
            if node.get_attr('_weights_transposed', False):
                # If already transposed by ApplyResourceStrategy, it's [F, H, W, C]
                node.weights['weight'].data = data.reshape(data.shape[0], -1)
            else:
                # It's [H, W, C, F]. Transpose to [F, H, W, C] then flatten.
                node.weights['weight'].data = np.transpose(data, axes=[3, 0, 1, 2]).reshape(data.shape[3], -1)
                node.set_attr('_weights_transposed', True)
            node.weights['weight'].shape = list(node.weights['weight'].data.shape)

        elif isinstance(node, SeparableConv1D):
            # Pointwise weights are [1, C, F].
            p_data = node.weights['pointwise'].data
            if node.get_attr('_weights_transposed', False):
                # [F, 1, C]
                node.weights['pointwise'].data = p_data.reshape(p_data.shape[0], -1)
            else:
                # [1, C, F] -> [F, 1, C] -> [F, C]
                node.weights['pointwise'].data = np.transpose(p_data, axes=[2, 0, 1]).reshape(p_data.shape[2], -1)
            
            # Depthwise weights are [W, C, 1]. 
            # Note: SeparableConv GEMM IP usually only applies to the pointwise part if we split it,
            # but if we keep it as one node, we might need to handle it.
            # For now, we focus on the pointwise part which is the GEMM part.
            pass

        elif isinstance(node, SeparableConv2D):
            # Pointwise weights are [1, 1, C, F].
            p_data = node.weights['pointwise'].data
            if node.get_attr('_weights_transposed', False):
                # [F, 1, 1, C]
                node.weights['pointwise'].data = p_data.reshape(p_data.shape[0], -1)
            else:
                # [1, 1, C, F] -> [F, 1, 1, C] -> [F, C]
                node.weights['pointwise'].data = np.transpose(p_data, axes=[3, 0, 1, 2]).reshape(p_data.shape[3], -1)
            pass

        node.set_attr('_weights_transposed_for_gemm', True)
        return False
