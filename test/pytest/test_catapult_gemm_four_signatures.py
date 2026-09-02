"""Global guard: the Catapult GEMM layer exposes EXACTLY four entry points.

The unified-gemm-ir-node work collapsed a ~12-name GEMM C++ layer down to four
signatures — `gemm_stream`, `gemm_stream_weightless`, `gemm_array`,
`gemm_array_weightless` — the 2x2 of (io_stream|io_parallel) x (two-operand|
weight-stationary). Every node template dispatches into this fixed set. A fifth
`gemm_*` entry point (e.g. an accum-draining core, or a resurrected wrapper) would
silently break the "four names everywhere" contract that gemm-ip-gen is built
against, so assert the set directly against the canonical headers rather than
relying on the per-test `... not in` negative checks scattered across the suite.
"""

import os
import re

import hls4ml

# The two headers that DEFINE the GEMM entry points (io_parallel + io_stream).
_NNET_UTILS = os.path.join(
    os.path.dirname(hls4ml.__file__), 'templates', 'catapult', 'nnet_utils'
)
_GEMM_HEADERS = ('nnet_gemm_ip.h', 'nnet_gemm_stream.h')

_EXPECTED = {'gemm_array', 'gemm_array_weightless', 'gemm_stream', 'gemm_stream_weightless'}

# A GEMM entry point is a free function `void gemm_<name>(...)`. Each name is
# defined twice (behavioral body + decl-only, under the GEMM_IP_HEADER `#if`), so
# compare the unique NAME set, not the count.
_DEF_RE = re.compile(r'\bvoid\s+(gemm_[A-Za-z0-9_]+)\s*\(')


def _gemm_entry_points():
    names = set()
    for header in _GEMM_HEADERS:
        text = open(os.path.join(_NNET_UTILS, header)).read()
        names.update(_DEF_RE.findall(text))
    return names


def test_catapult_exposes_exactly_four_gemm_signatures():
    found = _gemm_entry_points()
    assert found == _EXPECTED, (
        f'Catapult GEMM entry points drifted from the four-name contract. '
        f'Unexpected: {sorted(found - _EXPECTED)}; missing: {sorted(_EXPECTED - found)}.'
    )
