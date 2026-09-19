"""Shared helpers for the two-operand Einsum layer, used by the Vivado/Vitis and
Catapult backends so the row-streaming operand-selection logic (and its semantics)
is defined once. See the Vivado backend's `einsum.py` for the original derivation.
"""


def effective_perm(shape, idxs):
    """Drop size-1 axes (they carry no layout) and re-rank the remaining ones."""
    kept = [i for i in idxs if shape[i] != 1]
    rank = {ax: r for r, ax in enumerate(sorted(kept))}
    return [rank[ax] for ax in kept]


def select_row_stream_operand(
    io_type: str,
    strategy: str,
    n_inplace: int,
    n_contract: int,
    n_free0: int,
    n_free1: int,
    inp0_shape,
    inp1_shape,
    out_shape,
    inp0_tpose_idxs,
    inp1_tpose_idxs,
    out_tpose_idxs,
    in0_pack: int,
    in1_pack: int,
    out_pack: int,
):
    """Decide whether the two-operand Einsum can stream one operand a row at a time
    against the other (buffered), and if so which operand.

    Returns (row_stream: bool, row_stream_operand: int). `row_stream_operand` is 0 or
    1 even when `row_stream` is False (a harmless default in that case).

    Row streaming requires:
    - io_stream + Resource strategy, single in-place slice (n_inplace == 1).
    - The streamed operand's transpose (ignoring size-1 axes) is the identity, i.e. it
      already arrives as (L, C) with C fastest — so a stream beat is a real row prefix.
    - The output transpose (ignoring size-1 axes) is either identity ((L0, L1), rows of
      operand 0) or the swap ([1, 0], (L1, L0), rows of operand 1).
    - The streamed operand's pack size divides n_contract, and the output pack size
      divides the output row length, so beats tile whole rows.
    """
    eff_in0 = effective_perm(inp0_shape, inp0_tpose_idxs)
    eff_in1 = effective_perm(inp1_shape, inp1_tpose_idxs)
    eff_out = effective_perm(out_shape, out_tpose_idxs)
    in0_rows_ok = eff_in0 == list(range(len(eff_in0)))
    in1_rows_ok = eff_in1 == list(range(len(eff_in1)))

    row_stream_operand = None
    if eff_out == list(range(len(eff_out))) and in0_rows_ok:
        if n_contract % in0_pack == 0 and n_free1 % out_pack == 0:
            row_stream_operand = 0  # output is (L0, L1): rows of operand 0
    elif eff_out == [1, 0] and in1_rows_ok:
        if n_contract % in1_pack == 0 and n_free0 % out_pack == 0:
            row_stream_operand = 1  # output is (L1, L0): rows of operand 1

    row_stream = (
        io_type == 'io_stream' and strategy.lower() == 'resource' and n_inplace == 1 and row_stream_operand is not None
    )
    return row_stream, (row_stream_operand if row_stream else 0)
