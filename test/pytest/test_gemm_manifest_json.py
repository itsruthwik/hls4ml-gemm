import json

import numpy as np
import pytest

import hls4ml
from hls4ml.writer.catapult_writer import CatapultWriter
from hls4ml.writer.gemm_ip_json import write_gemm_config_json
from hls4ml.writer.vivado_writer import VivadoWriter


onnx = pytest.importorskip('onnx')
qonnx = pytest.importorskip('qonnx')


def _constant_matmul_qonnx():
    from onnx import TensorProto, helper, numpy_helper
    from qonnx.core.modelwrapper import ModelWrapper

    input_value = helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3])
    output_value = helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])
    weight = numpy_helper.from_array(np.arange(6, dtype=np.float32).reshape(3, 2), 'weight')
    matmul = helper.make_node('MatMul', ['x', 'weight'], ['y'], name='matmul')
    graph = helper.make_graph([matmul], 'numpy_scalar_manifest', [input_value], [output_value], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    return ModelWrapper(model)


@pytest.mark.parametrize(
    'backend,writer_class',
    [('Vitis', VivadoWriter), ('Catapult', CatapultWriter)],
)
def test_qonnx_gemm_manifest_serializes_numpy_dimensions(backend, writer_class, tmp_path):
    model = _constant_matmul_qonnx()
    config = hls4ml.utils.config_from_onnx_model(model, granularity='name', backend=backend)
    config['LayerName']['matmul']['Strategy'] = 'GEMM'
    output_dir = tmp_path / backend.lower()
    output_dir.mkdir()

    hls_model = hls4ml.converters.convert_from_onnx_model(
        model,
        hls_config=config,
        backend=backend,
        io_type='io_stream',
        output_dir=str(output_dir),
    )
    gemm = next(layer for layer in hls_model.get_layers() if layer.class_name == 'Gemm')
    assert isinstance(gemm.get_attr('n_in'), np.integer), 'regression requires QONNX NumPy shape metadata'

    writer_class().write_gemm_config(hls_model)

    manifest = json.loads((output_dir / 'gemm_config.json').read_text())
    item = manifest[gemm.name]
    assert item['n_in'] == 3
    assert item['n_out'] == 2
    assert item['gemm_k'] == 3
    assert item['gemm_n'] == 2
    assert all(type(item[key]) is int for key in ('n_in', 'n_out', 'gemm_k', 'gemm_n'))


@pytest.mark.parametrize('value,type_name', [(object(), 'object'), (np.array([1]), 'ndarray')])
def test_gemm_manifest_serializer_rejects_unsupported_values(value, type_name, tmp_path):
    with pytest.raises(TypeError, match=f'Object of type {type_name} is not JSON serializable'):
        write_gemm_config_json(tmp_path / 'gemm_config.json', {'bad': value})
