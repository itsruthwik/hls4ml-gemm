from hls4ml.model.optimizer import OptimizerPass


class ConfigureInputFifoDepth(OptimizerPass):
    """Applies HLSConfig LayerName.<name>.InputFifoDepth to the layer's input streams.

    ``InputFifoDepth`` is a per-input-edge override: ``{1: 32}`` sets the FIFO depth of
    input index 1 (e.g. the skip-connection operand of a residual add) to 32. Only
    explicitly configured edges are touched -- everything else keeps whatever depth
    ``StreamVariableConverter``/the build_prj.tcl blanket loops give it. Must run after
    clone insertion (`catapult:clone_output`) and `catapult:transform_types`, so
    `node.inputs[idx]` already names the post-clone `*_cpyN` variable and that variable
    already carries a `('stream', depth)` pragma to overwrite. Shared across the FPGA
    backends (Catapult, Vivado, Vitis); each backend's flow places it after its own
    clone-insertion and transform_types passes at the equivalent point.
    """

    def match(self, node):
        io_type = node.model.config.get_config_value('IOType')
        if io_type != 'io_stream':
            return False
        depths = node.model.config.get_layer_config(node).get('InputFifoDepth')
        return bool(depths)

    def transform(self, model, node):
        depths = model.config.get_layer_config(node).get('InputFifoDepth')
        for idx_key, depth in depths.items():
            idx = int(idx_key)
            if idx < 0 or idx >= len(node.inputs):
                raise Exception(
                    f"Layer '{node.name}': InputFifoDepth index {idx} out of range "
                    f'(layer has {len(node.inputs)} input(s)).'
                )
            var = node.get_input_variable(node.inputs[idx])
            if var is None or var.pragma is None:
                raise Exception(
                    f"Layer '{node.name}': InputFifoDepth[{idx}] targets input "
                    f"'{node.inputs[idx]}', which has no stream pragma to override "
                    '(is IOType io_stream and has type transform already run?).'
                )
            var.pragma = ('stream', int(depth))
            var.fifo_depth_explicit = True
        return False  # config only; never mutates the graph


def register_configure_input_fifo_depth(backend):
    backend.register_pass('configure_input_fifo_depth', ConfigureInputFifoDepth)
