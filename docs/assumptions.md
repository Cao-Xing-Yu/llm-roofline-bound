# What the bounds assume

A lower bound is only as good as what it leaves out. This page lists every
assumption behind the numbers `rooflinebound` reports, and which way each one
errs. Anything that could make a reported bound *wrong* (rather than merely
loose) is called out.

## Hardware

- **Peak rates.** Compute uses the dense (no 2:4 sparsity) tensor-core peak of
  the operator's dtype; memory uses peak HBM bandwidth; the interconnect uses
  half the bidirectional NVLink figure per direction. Real kernels reach less,
  so this only loosens the bound.
- **Spec values are inputs.** A bound is conditional on the spec it was
  computed with. Every spec field says whether it is vendor-published
  (OFFICIAL), from a third party, or supplied by the user. If a value is too
  low (say bandwidth understated), a bound computed from it can be wrong.
- **Memory capacity** is what the device exposes. Memory bounds such as
  `KV pool ≤ capacity − weights` take it as an upper limit.

## Memory

- **Weights** are counted tensor by tensor from the checkpoint inventory, and
  are exact for the stated layout.
- **The layout** (what TP splits and what it replicates, how EP and PP place
  layers) is spelled out in `src/rooflinebound/parallel.py`. Engines may differ.
- **KV cache** uses a uniform layout: every cached element is stored at the
  KV dtype, with no scale metadata. Formats that add per-token scales use more
  memory, so counts derived from the pool remain upper bounds.
- **Runtime memory** (activations, CUDA graphs, workspaces, fragmentation) is
  not derivable from a config. Without `--reserve-gb` it is taken as zero, and
  the results are labelled BOUND (upper). With `--reserve-gb` they are labelled
  ASSUMED.

## Time

- **Overlap.** A pass takes at least the largest of its total compute time,
  its total HBM time and its total interconnect time. This allows every
  operator to overlap perfectly with every other, so it holds for any schedule.
  The per-operator sum shown by `--ops` is tighter, but it assumes operators
  never overlap, so it is not reported as a bound.
- **FLOPs** are the standard `2·M·N·K` per matmul, plus attention over exactly
  the attended (query, key) pairs. Elementwise work (norms, activations,
  softmax, top-k selection, RoPE) is omitted. That loosens the bound.
- **Fast matrix multiplication** (Strassen-like algorithms) is assumed unused,
  as it is on tensor cores.
- **Compute dtype follows storage.** An fp8 weight runs at the fp8 peak and a
  bf16 weight at the bf16 peak; attention runs at the KV dtype's peak. An
  implementation that re-quantizes bf16 tensors to fp8 on the fly could beat
  this. *This assumption could make a bound wrong.*
- **HBM traffic** counts only what must move:
  - weights, once per pass (a GPU's shard is far larger than on-chip cache);
    the embedding table is excluded, since only the looked-up rows are read;
  - cached KV that attention or the indexer must see, once per sequence;
  - new KV, written once.

  Activations are excluded because a fused kernel could keep them on chip.
  Leaving traffic out loosens the bound.
- **Top-k sparse attention (DSA).** Each query attends `min(position + 1, topk)`
  keys. The indexer scores every earlier key, but only for queries that have
  more than `topk` of them. "Shared" layers reuse another layer's selection and
  score nothing. For HBM traffic, a pass reads at least the keys its last query
  attends, beyond the new tokens.
- **MLA.** Both the absorbed form (attention in latent space) and the expanded
  form (decompress keys and values) are valid ways to compute attention. The
  bound takes whichever is cheaper for the pass.
- **Communication.** Each tensor-parallel all-reduce of `N` bytes needs every
  rank to receive at least `(p − 1)/p · N`. Latency per message is not counted.
  The final logits gather is omitted.
- **Expert parallelism.** EP = TP with tensor-parallel attention: hidden states
  are already on every rank, each rank runs its own experts, and the MoE output
  is all-reduced. The busiest rank does at least the average work.
- **Pipeline parallelism** is not modelled for time. PP > 1 is rejected.

## Routing (EXPECTED)

The weight bytes of routed experts depend on how tokens are routed. The value
reported assumes uniform, independent routing: an expert is missed by all `T`
tokens with probability `(1 − k/E)^T`. The range printed with it runs from the
most concentrated routing (every token picks the same `k` experts) to the most
spread out (`min(E, k·T)` experts touched). Only the low end of the range
holds whatever the routing.

## MTP

Draft tokens per step and their acceptance rates are user inputs. The step
time (one verify pass with `1 + k` tokens per sequence, then `k` single-token
MTP passes) is a bound. Time per output token divides that step time by the
expected tokens per step, `1 + Σ_i Π_{j≤i} p_j`, so it is ASSUMED.

## Not modelled

The following are not modelled. Where a feature can only add time, leaving it
out keeps the bounds valid.

- queueing
- scheduling and prefill/decode interference
- kernel launch and synchronization overheads
- per-layer fixed latency
- disaggregated prefill
- KV offload
- multimodal inputs
