"""Keep GEMM IP edges packed across the graph (Vivado/Vitis, io_stream).

A GEMM IP takes packed ``ap_uint`` bit streams, so by default every GEMM node converts its
operands and result with its own pack/unpack processes (nnet_gemm_pack.h). Each of those is
a separate dataflow process whose fill, drain and per-frame start sync add to the latency.
With ``HLSConfig: Model: GemmPackedStreams: True`` this pass marks an edge ``gemm_packed``
when one end is a GEMM node and the other end is a GEMM node or a layer whose template reads
or writes the packed beat itself (nnet_stream_beat.h); that edge then carries the raw beat,
and neither end runs a separate conversion process. Every other edge is unchanged, so a
layer without packed support keeps today's pack/unpack around the GEMM.

Model inputs and outputs are never packed: Vitis will not hand a top-level argument straight
to an RTL blackbox, and the model's interface stays in hls4ml's array beats.
"""

from hls4ml.backends.fpga.gemm.gemm_nodes import Gemm, Im2Col
from hls4ml.backends.fpga.passes.clone import Clone
from hls4ml.backends.fpga.passes.split_merge_nodes import HeadMerge, HeadSplit
from hls4ml.model.layers import Activation, ParametrizedActivation, Softmax
from hls4ml.model.optimizer import ModelOptimizerPass

# Activations whose io_stream templates read and write through nnet::beat_io.
_PACKED_ACTIVATIONS = ('linear', 'relu')
_PACKED_PARAM_ACTIVATIONS = ('leakyrelu',)


def _is_stream_gemm(node):
    # The bare gemm_stream_<name> call; the n_inplace > 1 per-head loop is not a packed call.
    return isinstance(node, Gemm) and node.get_attr('n_inplace', 1) == 1


def _packed_output_ok(node):
    if _is_stream_gemm(node) or isinstance(node, (HeadSplit, HeadMerge, Clone)):
        return True
    if isinstance(node, Im2Col):
        return node.get_attr('strategy') == 'gemm'
    return _packed_activation(node)


def _packed_input_ok(node):
    return _is_stream_gemm(node) or isinstance(node, (HeadSplit, HeadMerge)) or _packed_activation(node)


def _packed_activation(node):
    if isinstance(node, Softmax):
        # Only softmax_stable (Latency strategy) reads packed beats; the template calls it directly.
        return (
            str(node.get_attr('implementation', '')).lower() == 'stable'
            and str(node.get_attr('strategy', '')).lower() != 'resource'
            and node.get_attr('n_inner', 1) == 1
        )
    activation = str(node.get_attr('activation', '')).lower()
    if isinstance(node, ParametrizedActivation):
        return activation in _PACKED_PARAM_ACTIVATIONS
    if isinstance(node, Activation):
        return activation in _PACKED_ACTIVATIONS
    return False


class MarkGemmPackedEdges(ModelOptimizerPass):
    def __init__(self):
        self.name = 'mark_gemm_packed_edges'

    def transform(self, model):
        hls_model = model.config.config.get('HLSConfig', {}).get('Model', {})
        if not hls_model.get('GemmPackedStreams', False):
            return False
        if model.config.get_config_value('IOType') != 'io_stream':
            return False

        boundary = set(model.inputs) | set(model.outputs)
        consumers = {}
        for node in model.get_layers():
            for name in node.inputs:
                consumers.setdefault(name, []).append(node)

        for producer in model.get_layers():
            for name in producer.outputs:
                users = consumers.get(name, [])
                if name in boundary or len(users) != 1:
                    continue
                consumer = users[0]
                if not (_is_stream_gemm(producer) or _is_stream_gemm(consumer)):
                    continue
                if _packed_output_ok(producer) and _packed_input_ok(consumer):
                    producer.get_output_variable(name).gemm_packed = True

        # clone_stream / split_lanes / merge_lanes give all their copies or head streams one
        # stream type, so those outputs (or a merge's inputs) are packed all or none.
        for node in model.get_layers():
            if isinstance(node, (HeadSplit, Clone)):
                group = [node.get_output_variable(o) for o in node.outputs]
            elif isinstance(node, HeadMerge):
                group = [node.get_input_variable(i) for i in node.inputs]
            else:
                continue
            if not all(getattr(v, 'gemm_packed', False) for v in group):
                for v in group:
                    v.gemm_packed = False

        return False
