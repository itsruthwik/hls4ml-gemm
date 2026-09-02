"""Head-lane Split / Merge IR nodes for multi-head attention on the GEMM path.

These keep the GEMM node uniform: a projection stays 2D ``[seq, d_model]`` and a
``HeadSplit`` fans it into H per-head ``[seq, key_dim]`` streams; the per-head GEMMs
consume clean ``[seq, key_dim]``; a ``HeadMerge`` concatenates the H head outputs
back to ``[seq, d_model]`` before the output projection. Head lives inside a beat
(``d_model = H * key_dim``), so both are stateless lane wiring (no buffer, no
reorder) — this is what lets the head-move transpose be dropped entirely.

The IR node + logical shapes are identical across io_stream and io_parallel; only
the emitted C++ differs (enumerated ac_channels vs flat-array reindex), via an
IOType branch, exactly as every other hls4ml layer does.

These node classes live in the shared FPGA base so every FPGABackend child inherits
them (mirroring the GEMM nodes). The per-backend HeadSplit / HeadMerge codegen
templates live in that backend's own ``*_templates`` module, in its own dialect.
"""

from hls4ml.model.layers import Layer, register_layer
from hls4ml.model.attributes import Attribute


class HeadSplit(Layer):
    """1 input ``[seq, d_model]`` -> H outputs ``[seq, key_dim]`` (one per head)."""

    _expected_attributes = [
        Attribute('n_heads'),
        Attribute('key_dim'),
        Attribute('seq'),
        Attribute('d_model'),
    ]

    def initialize(self):
        H = self.attributes['n_heads']
        seq = self.attributes['seq']
        key_dim = self.attributes['key_dim']
        # H distinct outputs. Each needs a DISTINCT var_name/type_name — the default
        # 'layer{index}_out' would collide across the H outputs (all share index).
        for h, out_name in enumerate(self.outputs):
            self.add_output_variable(
                [seq, key_dim],
                out_name=out_name,
                var_name=f'layer{{index}}_out_h{h}',
                type_name=f'layer{{index}}_t_h{h}',
            )

    def get_layer_precision(self):
        # add_output_variable stores every output's type under the single 'result_t'
        # key (AttributeDict.__setitem__), so the default get_layer_precision (which
        # reads self.types) would emit only the LAST head's typedef. Read each output
        # variable's CURRENT type directly instead — correct for both io_parallel
        # (scalar) and io_stream (nnet::array beat, after TransformTypes converts the
        # variables), with no stale duplicate shadowing the converted type.
        return {self.get_output_variable(o).type.name: self.get_output_variable(o).type for o in self.outputs}


class HeadMerge(Layer):
    """H inputs ``[seq, key_dim]`` -> 1 output ``[seq, d_model]`` (lane-concat)."""

    _expected_attributes = [
        Attribute('n_heads'),
        Attribute('key_dim'),
        Attribute('seq'),
        Attribute('d_model'),
    ]

    def initialize(self):
        seq = self.attributes['seq']
        d_model = self.attributes['d_model']
        self.add_output_variable([seq, d_model])


register_layer('HeadSplit', HeadSplit)
register_layer('HeadMerge', HeadMerge)
