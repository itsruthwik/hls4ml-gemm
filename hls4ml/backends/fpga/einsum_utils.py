"""Shared helpers for the two-operand Einsum layer, used by the Vivado/Vitis and
Catapult backends so the row-streaming plan (which operand supplies rows, and the row
loop's nesting order) is defined once. See the Vivado backend's original `einsum.py`
for the derivation.
"""

from hls4ml.utils.einsum_utils import _validate_einsum_expr


def equation_row_plan(equation, shape0, shape1, out_pack=1):
    """Derive, purely from the einsum equation's index letters (not by transposing/simulating
    anything numerically), which operand the io_stream kernel's rows come from, the row loop's
    nesting order, and the (I, L, C) axis layout each operand's local array needs so ReadLoop can
    scatter beats straight into it.

    Classification (the batch letter, or any size-1 letter, falls out of every check below
    because it never breaks a contiguity/ordering test):
      - inplace: in both inputs and the output
      - contracted: in both inputs, not the output
      - free0: only in input 0 and the output
      - free1: only in input 1 and the output

    The output index string's trailing indices must be exactly all of one input's free indices
    (in some order) -- that input is BUFFERED (its whole free axis is one row's contents, laid out
    in that trailing, output order); the other input is the ROW operand (one free index fixed per
    row). The remaining leading output indices classify as inplace or row-free; if, ignoring size-1
    letters, they form a single inplace block and a single row-free block (in either order), that
    order gives the row loop's nesting (outer to inner) as a flattened counter -- no lookup table.
    Anything else (no operand's free set is a suffix, or the leading indices interleave inplace
    and row-free letters) raises.

    Returns a dict: row_op (0 or 1, which operand supplies rows), row_major_i_outer (bool: True if
    inplace nests outside the row-free axis, False if the row-free axis nests outside inplace),
    perm0/perm1 (axis order, as positions in shape0/shape1, giving each operand's (inplace-axes...,
    free-axes-in-its-assigned-order..., contract-axes...) layout), and the letter lists (inplace,
    contract, free0, free1, buffered_free_order, row_free_order) for reporting/debugging.
    """
    fn, _ = _validate_einsum_expr(equation, tuple(shape0), tuple(shape1))
    inp, out = fn.split('->')
    in0, in1 = inp.split(',')
    s0, s1, so = set(in0), set(in1), set(out)
    common = s0 & s1

    contract = sorted(common - so, key=lambda x: in1.index(x))
    inplace = sorted(common & so, key=lambda x: in1.index(x))
    free0 = sorted((s0 - common) & so, key=lambda x: in0.index(x))
    free1 = sorted((s1 - common) & so, key=lambda x: in1.index(x))

    size_of = {}
    for ch, sz in zip(in0, shape0):
        size_of[ch] = sz
    for ch, sz in zip(in1, shape1):
        size_of.setdefault(ch, sz)

    def fail(msg):
        raise Exception(f'io_stream Einsum "{equation}": {msg}')

    if free1 and list(out[-len(free1):]) and set(out[-len(free1):]) == set(free1):
        row_op, buffered_free, row_free = 0, free1, free0
        leading = list(out[: -len(free1)])
    elif free0 and set(out[-len(free0):]) == set(free0):
        row_op, buffered_free, row_free = 1, free0, free1
        leading = list(out[: -len(free0)])
    else:
        fail(
            f'output "{out}" does not end with all of one input\'s free indices (free0={free0}, '
            f'free1={free1}) -- there is no operand whose whole free axis is a contiguous block of '
            'the output, so no operand can be streamed out row by row.'
        )

    cats = ['I' if ch in inplace else ('ROW' if ch in row_free else '?') for ch in leading]
    if '?' in cats:
        fail(f'leading output indices "{"".join(leading)}" contain an index that is neither inplace nor row-free.')

    eff = [(ch, cat) for ch, cat in zip(leading, cats) if size_of[ch] != 1]
    eff_cats = [cat for _, cat in eff]
    n_i = eff_cats.count('I')
    i_outer_ok = eff_cats[:n_i] == ['I'] * n_i and eff_cats[n_i:] == ['ROW'] * (len(eff_cats) - n_i)
    row_outer_ok = eff_cats[: len(eff_cats) - n_i] == ['ROW'] * (len(eff_cats) - n_i) and eff_cats[len(eff_cats) - n_i :] == [
        'I'
    ] * n_i
    if not (i_outer_ok or row_outer_ok):
        fail(
            f'leading output indices "{"".join(leading)}" interleave inplace and row-free indices '
            f'(effective order {"".join(c for _, c in eff)}); the row loop can only nest one inside '
            'the other.'
        )
    row_major_i_outer = i_outer_ok

    i_seq = [ch for ch, cat in eff if cat == 'I']
    row_seq = [ch for ch, cat in eff if cat == 'ROW']
    if i_seq != [ch for ch in inplace if size_of[ch] != 1]:
        fail(f'inplace indices appear in the output in an unsupported order: "{"".join(i_seq)}" vs canonical {inplace}.')
    if row_seq != [ch for ch in row_free if size_of[ch] != 1]:
        fail(
            f'row operand\'s free indices appear in the output in an unsupported order: '
            f'"{"".join(row_seq)}" vs canonical {row_free}.'
        )

    row_len = 1
    for ch in buffered_free:
        row_len *= size_of[ch]
    if row_len % out_pack != 0:
        fail(f'output pack size {out_pack} does not divide the row length {row_len} (buffered indices {buffered_free}).')

    return dict(
        row_op=row_op,
        row_major_i_outer=row_major_i_outer,
        inplace=inplace,
        contract=contract,
        free0=free0,
        free1=free1,
        buffered_free_order=buffered_free,
        row_free_order=row_free,
    )
