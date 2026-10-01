# The GEMM IP path in hls4ml-gemm (`Strategy: GEMM`)

## Overview

With `Strategy: GEMM`, hls4ml-gemm turns every matrix product in a model (Dense, Conv, EinsumDense and the
attention matmuls) into one uniform `Gemm` node. It implements that node as a plain C++ function call into an
external GEMM IP. The fork is 78 commits on top of upstream hls4ml (merge base `e998745`).

The boundary with gemm-ip-gen is a function signature, not a hardware contract. hls4ml emits three things:
- the `gemm_stream_<layer>` / `nnet::gemm_*` calls;
- a manifest describing each GEMM;
- the constant operand, packed in beat order.

gemm-ip-gen alone decides what the callee becomes (RTL, a blackbox or behavioral C++).

The GEMM-specific work in hls4ml is therefore graph work:
- rewrite layers into canonical GEMMs;
- put every operand into a stream layout the IP can consume;
- do any reordering either at compile time (weights) or with explicit `Transpose` nodes (activations).

## Opt-in and the Gemm node

- **Opt-in.** `Strategy: GEMM` is a peer of Latency and Resource, resolved LayerName → LayerType → Model
  (`backends/gemm_ip_config.py:is_gemm_strategy`).
- **Early rejection.** The backends reject what the path cannot express before any rewrite: Conv must be
  channels_last, with dilation 1 and valid padding.
- **The `Gemm` node** (`backends/fpga/gemm/gemm_nodes.py`) carries only the math: M = `n_patches`, K = `n_in`,
  N = `n_out`, plus `n_inplace` for batched GEMMs.

Two independent properties select the C++ entry point:

| | constant B (`weights_in_core=True`) | activation B (`weights_in_core=False`) |
|---|---|---|
| io_stream | `gemm_stream_const_weights` | `gemm_stream` |
| io_parallel | `gemm_array_const_weights` | `gemm_array` |

`weights_in_core` depends on the source layer, not on the IOType:
- True for Dense, Conv, PointwiseConv and EinsumDense.
- False for the attention Einsums (QKᵀ, A·V), where both operands are activations.

Per-layer settings are resolved once onto the node by `_resolve_gemm_config`: `strategy`, `reuse_factor`,
`SecondOperandRowMajor` and `Im2ColTileRows`. `SecondOperandRowMajor` is resolved scope by scope based on
whether the key is present. A model-wide default therefore still applies to a layer that has other LayerName
entries (the stock lookup would stop at the first scope that exists).

## GEMM-specific graph passes (in pipeline order)

Six shared passes, plus one Vivado/Vitis-only pass (7), turn a Keras graph into canonical GEMMs. Passes 1–6 live
in `backends/fpga/gemm/` and are shared by every FPGA backend; pass 7 lives in `backends/vivado/passes/`.
Catapult runs 1–4 in the `optimize` flow and 5–6 in `specific_types`; Vivado/Vitis registers 1–6 as well.

1. **`SplitConvGemm`**: Conv1D/Conv2D becomes `Im2Col` + `Gemm`.
   - Shape: M = out_h·out_w, K = kh·kw·C, N = filters.
   - A 1×1 stride-1 conv skips im2col and is a plain GEMM over the pixels.
   - A strided 1×1 conv keeps im2col, so only the strided positions are emitted.
   - The im2col node streams one K-wide row per beat. `Im2ColTileRows` limits how far it can run ahead of the IP.
2. **`ReplaceDenseGemm`**: Dense becomes a const-weight `Gemm`. M = product of the leading dims, so a Dense over
   a sequence is one M-row GEMM.
3. **`SplitAttentionHeads`** (`attention_heads.py`): rewrites each QKᵀ → Softmax → A·V cluster into H per-head
   lanes, and runs before the generic einsum lowering.
   - A stateless `HeadSplit` fans the 2D projection into H `[seq, key_dim]` streams. Head is a lane within the
     beat, so this is pure wiring with no buffer.
   - Each head runs its own two-operand GEMMs and Softmax. A `HeadMerge` then concatenates the per-head outputs.
   - This removes the head-moving transposes that generic lowering would insert: 5 whole-tensor reorder buffers
     per attention block under io_stream.
   - Quantizers left between the ops by a quantization-aware-trained model are cloned into each head lane, each
     with its head's slice of the precision.
4. **`LowerEinsumToGemm`**:
   - EinsumDense becomes a const-weight `Gemm`, with the bias handled one of two ways:
     - bias constant along rows: per-column bias on the IP port;
     - bias varying along rows: zero IP bias, plus an in-wrapper add of the full per-element bias.
   - Einsum becomes a two-operand `Gemm` in canonical form, with `Transpose` nodes inserted automatically (next section).
5. **`TransposeWeightsForGemmIP`** (`gemm_transposition.py`): reorders constant kernels into `[N, K]` at compile
   time (details in the next section).
6. **`ValidateGemm`**: runs after stream types are assigned and fails early, naming the layer:
   - an io_stream const-weight GEMM needs every A beat to carry the full K row (e.g. Conv → Flatten → Dense
     violates this, and the error suggests a non-GEMM strategy);
   - a two-operand GEMM needs B's beat width to equal K (column-major) or N (row-major).
7. **`MarkGemmPackedEdges`** (Vivado/Vitis only, `GemmPackedStreams: True`): keeps an edge in packed `ap_uint`
   form when both ends can read or write it.
   - Eligible ends: GEMM, add, pool, relu/linear/leaky-relu, quantizers.
   - This drops the per-GEMM pack/unpack dataflow processes and their fill/drain/start latency.
   - Model inputs and outputs are never packed.

## Operand layouts: repack, row-/column-major, transposes

The IP's canonical form is: A = rows `[I, L0, C]` (one K-wide row per beat); B = `[I, L1, C]`; output C =
rows `[I, L0, L1]` (one N-wide row per beat). Every reorder needed to reach that form is resolved in one of the
three ways below.

### 1. Compile-time weight transpose (constant B)

`TransposeWeightsForGemmIP` rewrites the numpy kernel once, before codegen, so no reorder hardware exists:

| Source layer | Kernel shape | After |
|---|---|---|
| Dense | `[K, N]` | `[N, K]` |
| Conv1D | `[W, C, F]` | `[F, W·C]` (reshape only if `ApplyResourceStrategy` already transposed it) |
| Conv2D | `[H, W, C, F]` | `[F, H·W·C]` |
| SeparableConv pointwise | `[1(,1), C, F]` | `[F, C]` |
| EinsumDense | `[K, N]` | untouched; the writer reads it as `[K, N]` |

The pass is idempotent (`_weights_transposed_for_gemm`). The EinsumDense guard runs before the Dense branch,
because `'Dense'` is a substring of `'EinsumDense'`. The writer chooses the source layout from the layer type,
never from the shape: square kernels are ambiguous, and an earlier shape-based guess silently packed MHA
projection weights transposed.

### 2. Weight repack into IP beat order (writer)

`writer/gemm_ip_weights.py` (shared by the Catapult and Vivado writers) writes the constant operand already in
the order the IP consumes it, as a C++ ROM header plus a raw-integer `.dat` file. The layout is chosen by
`SecondOperandRowMajor`:

| `SecondOperandRowMajor` | ROM | One beat | Manifest `weight_layout` |
|---|---|---|---|
| False (default) | `<w>_gemm_cols[N]` | `array<weight_t, K>`: one output column | `column_major` |
| True | `<w>_gemm_rows[K]` | `array<weight_t, N>`: one contraction row | `row_major` |

- The `.dat` file holds the exact fixed-point bit patterns (`round(v·2^frac)`), with no float round-trip, so
  csim and cosim stay bit-exact.
- The config exposes `weights_row_major`, `weight_beat_t` and `gemm_weight_beats()`, so csim reads either layout
  through one accessor.
- gemm-ip-gen consumes `weight_layout` as given and never reorders it; a target that can't read a layout errors out.

### 3. Second operand row-major vs. column-major (runtime B)

For two-operand GEMMs (QKᵀ, A·V), the same knob chooses how B is streamed:
- **column-major (default)**: one K-high output column per beat, K elements wide;
- **row-major**: one contraction row per beat, N elements wide.

The knob exists because targets differ: some IPs load B more cheaply one way, e.g. mvau's runtime loader
streams row-major B for every tiling without a full-B buffer. In `_lower_einsum`, row-major swaps the last two
axes of B's canonicalizing permutation. `gemm_k`, `gemm_n` and the output shape are unchanged; only B's beat
layout flips. The templates pass `b_row_major` to the IP, and `ValidateGemm` checks the beat width matches.

### 4. Automatically inserted Transpose nodes (activations)

`_lower_einsum` builds a canonical-in/canonical-out `Gemm`, then realizes whatever reordering the einsum
equation implies as explicit IR nodes:
- `gemm_<name>_tpose_in0` / `_tpose_in1`: inserted before the GEMM for each non-identity input permutation;
- `gemm_<name>_tpose_out`: inserted after the GEMM to restore the einsum's declared output order. It is
  asserted to reproduce the original output shape and precision.

Identity permutations fold away, so a well-oriented einsum gets no transposes. Inside attention, `SplitAttentionHeads`
removes most of them:
- **Head moves** disappear: heads become lanes within a beat.
- **QKᵀ output orientation** (S vs Sᵀ) is avoided by assigning operand roles `A=Q, B=K`. The output is then
  already `[seq_q, seq_k]`, and softmax reduces the last axis. No node is needed.
- **K in QKᵀ and V in A·V** each follow their own `SecondOperandRowMajor`. Column-major feeds them directly;
  row-major needs a small per-head 2D `Transpose` (the csim-correct fallback). The two matmuls are independent,
  so both, one or neither may need one.

**Not done yet:** folding that remaining transpose into the IP's B-load write order, which would remove it
under io_stream as well.

## Writer outputs

- **Manifest** (`write_gemm_config`, `gemm_config.json`), per GEMM node:
  - type and shape (`gemm_m/k/n`, `n_in/n_out`, `n_inplace`);
  - `has_bias`, `transpose_weights`;
  - input/weight(rhs)/output/accum/bias precisions;
  - interface and protocol, `weight_layout` / `second_operand_row_major`, and the resolved knobs.
  - `Im2Col` also carries `strategy='gemm'`, but it only reshapes and gets no entry.
- **Packed weights**: `firmware/weights/<w>_gemm_{cols,rows}.h` + `.dat`.
- **Call site / hook**: the node's config struct and the call.
  - Vitis: a `parameters.h` hook declares `gemm_stream_<layer>` over packed `ap_uint` streams.
  - Catapult: `nnet_gemm_ip.h` includes `gemm_ip_combined.h` when `GEMM_IP_HEADER` is defined.
- **Build scripts**: one `source gemm_ip_sources.tcl` line; gemm-ip-gen decides its contents.
- **Package-less csim fallback**: without `GEMM_IP_HEADER`, csim uses hls4ml's behavioral GEMM (`nnet_gemm_ip.h`
  on Catapult, `nnet_gemm_behavioral.h` / `nnet_gemm_pack.h` on Vitis). Synthesis gets declarations only, so a
  build without a package fails at link time on purpose.

## Supporting non-GEMM changes

These let whole CNN and transformer models run in the GEMM flow:
- **Kernels:**
  - Catapult: Einsum/EinsumDense stream, LayerNorm stream, im2col stream, split/merge, transpose, softmax
    constant tables, and reworked activations and pooling.
  - Vivado/Vitis: io_stream EinsumDense/Einsum, LayerNorm stream, im2col stream.
- **Front end:** HGQ2 converters (norm/embedding, softmax, lookup-table activations), a QKeras MHA converter,
  and bit-exactness and quantizer-proxy fixes.
- **Catapult build:** RAM-backed channel FIFOs (`ram_pipe.tcl`, `rp_ram.v`), input FIFO depth configuration,
  and a reworked `build_prj.tcl`.

## Limitations

- EinsumDense lowering supports `n_inplace == 1` and identity transposes only; the packed writer holds one
  `[K, N]` block per file.
- Conv: channels_last, no dilation, valid padding only.
- An io_stream const-weight GEMM cannot consume a multi-beat activation (e.g. a Flatten after Conv/Pool); such
  layers must use a non-GEMM strategy.
- Row-major K/V in attention still costs a physical per-head `Transpose`.
