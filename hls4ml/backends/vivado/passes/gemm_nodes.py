"""Vivado GEMM graph optimizer passes.

Reuses the shared GemmStream / Im2ColGemmStream layer classes and the
ReplaceDenseGemm / SplitConvGemm passes that were originally written for
Catapult (SplitConvGemm builds the fused Im2ColGemmStream node directly).
Those classes are already globally registered through
``register_layer``, so we only need to import them and register the passes
under the ``vivado:`` namespace.
"""

# Re-export node classes so the Vivado backend can import from one place.
from hls4ml.backends.fpga.passes.gemm_nodes import (  # noqa: F401
    Gemm,
    Im2ColGemm,
    LowerEinsumToGemm,
    ReplaceDenseGemm,
    SplitConvGemm,
    ValidateGemm,
)
from hls4ml.backends.fpga.passes.gemm_transposition import TransposeWeightsForGemmIP  # noqa: F401
