"""The beat-wise io_stream quantizer generator must not emit one case label per beat.

A zero-padded conv input is a frame of ~1,000 beats where the border beats are known to be
exactly zero and the interior beats carry a real conversion; the HLS front end spent minutes
per quantizer on a switch with a label for every beat index. Converting a zero gives zero, so
such beats share the interior body (no switch at all), and a genuinely beat-dependent
conversion is written as case ranges.
"""
import numpy as np

from hls4ml.backends.fpga.passes.hgq_proxy_model import _generate_mask_fn_stream_beatwise


def _gen(b):
    # one streamed axis of 8 positions x 2 lanes; k, i broadcast; b per element
    shape = (8, 2)
    k = np.ones((1, 8, 2), dtype=np.int8)
    i = np.full((1, 8, 2), 3, dtype=np.int8)
    return _generate_mask_fn_stream_beatwise('q', shape, k, b[None], i, 'RND', 'WRAP', 'vitis', beat_size=2)


def test_zero_beats_share_the_interior_body():
    b = np.full((8, 2), 7, dtype=np.int8)
    b[0] = 0   # first and last beat known zero, as on a padded border
    b[7] = 0
    src = _gen(b)
    assert 'switch (i)' not in src
    assert 'case ' not in src
    assert src.count('res[0] = ap_fixed<7,3,AP_RND,AP_WRAP>(beat[0]);') == 1


def test_heterogeneous_beats_use_case_ranges():
    b = np.full((8, 2), 7, dtype=np.int8)
    b[4:, 0] = 5   # lane 0 changes width halfway through the stream
    src = _gen(b)
    assert 'switch (i)' in src
    assert 'case 0 ... 3:' in src and 'case 4 ... 7:' in src
    assert src.count('case ') == 2
    assert 'ap_fixed<7,3,AP_RND,AP_WRAP>(beat[0])' in src and 'ap_fixed<5,3,AP_RND,AP_WRAP>(beat[0])' in src


def test_all_zero_lane_stays_literal_zero():
    b = np.full((8, 2), 7, dtype=np.int8)
    b[:, 1] = 0
    src = _gen(b)
    assert 'switch (i)' not in src
    assert 'res[1] = 0;' in src
