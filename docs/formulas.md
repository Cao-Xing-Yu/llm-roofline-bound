# Formulas

Every number `rooflinebound` reports, as it is computed, so that any figure can be
checked by hand. Each formula is tagged with the level its result carries:

| level | meaning |
|---|---|
| **EXACT** | arithmetic on the model config and the hardware spec |
| **EXPECTED** | an exact expectation under a stated statistical assumption, with a guaranteed range |
| **BOUND** | a provable inequality, either lower (≥) or upper (≤) |
| **ASSUMED** | rests on an input you supplied, not on theory |
| **ESTIMATE** | a bound scaled by an efficiency you measured |

Arithmetic propagates the weakest level of its inputs. What each formula leaves
out, and which way that errs, is in [assumptions.md](assumptions.md).

Notation. Quantities are **per GPU** unless stated otherwise.

- `H`: hidden size
- `L`: decoder layers
- `V`: vocabulary
- `p`: TP size
- `T`: tokens in a pass
- `E`, `k`: routed experts, and experts per token
- `I_e`: expert intermediate size
- `n_s`: shared experts
- `I`: dense intermediate size
- MLA:
  - `h`: heads, with `h_l = h / p` per GPU
  - `r_q`: q LoRA rank
  - `r_kv`: KV LoRA rank
  - `d_n`: no-rope q/k head dim
  - `d_r`: rope dim
  - `d_v`: value head dim
  - `d_qk = d_n + d_r`
- DSA indexer:
  - `h_i`: indexer heads
  - `d_i`: indexer head dim
  - `K`: top-k
- Hardware:
  - `P_d`: peak FLOP/s at dtype `d`
  - `BW`: peak HBM bandwidth
  - `Link`: bidirectional interconnect bandwidth
  - `C`: memory capacity

---

## 1. Weights (EXACT)

The checkpoint is described tensor by tensor (`arch/`). For GLM-5.3 the
generated inventory equals the safetensors headers exactly: 118,629 tensors,
755,617,140,416 bytes.

**MXFP4 (compressed-tensors).** `weight_packed` stores two fp4 values per byte (`out × in/2`, U8) and `weight_scale` one e8m0 byte per 32 inputs (`out × in/32`). Parameters count both values of each byte.

**Block-fp8 scales.** A quantized `out × in` weight carries an fp32 inverse
scale per `b_r × b_c` block:

```
scale bytes = ceil(out / b_r) × ceil(in / b_c) × 4
```

**Bytes on one rank** (`parallel.py`):

| shard | rank bytes |
|---|---|
| replicated | `bytes` |
| column / row split | `bytes / p` (the split dim, and its scale blocks, must divide by `p`) |
| vocab split | `ceil(rows / p) × cols × dtype bytes` (padded) |
| routed expert, EP = 1 | split like column/row |
| routed expert, EP = p | `bytes / EP` on average (each rank holds `E / EP` whole experts) |

**Pipeline stages.** Each stage gets a contiguous run of decoder layers, as even
as possible, with earlier stages taking the remainder. The embedding goes on the
first stage; the final norm, `lm_head` and MTP layers go on the last.
`weights per GPU` is the largest stage.

**Active parameters per token.** All non-expert parameters, plus `k / E` of each
MoE layer's routed experts, plus `lm_head`. The embedding (a lookup) and MTP
layers are excluded.

## 2. KV cache (EXACT for the stated layout)

Per token and layer:

| layout | MLA cache | DSA indexer key (own-indexer layers only) |
|---|---|---|
| uniform, dtype `w` | `(r_kv + d_r) × w` | `d_i × w_i` |
| GQA (any engine), dtype `w` | `2 × max(1, kv_heads / p) × head_dim × w` per rank | — |
| SGLang MLA without DSA (Kimi-K3) | `(r_kv + d_r) × w` | — |
| KDA (linear attention) | **none per token**; per sequence `(h/p)·d²·s_ssm + (K−1)·3·(h·d/p)·s_conv` | — |
| SGLang DSA, fp8, scaled kernels | `r_kv + (r_kv / 128) × 4 + d_r × 2` | `d_i + (d_i / 128) × 4` |
| SGLang DSA, fp8, raw (TRT-LLM, TileLang) | `r_kv + d_r` | `d_i + (d_i / 128) × 4` |
| SGLang DSA, bf16 | `(r_kv + d_r) × 2` | `d_i + (d_i / 128) × 4` |

For GLM-5.3 (`r_kv = 512`, `d_r = 64`, `d_i = 128`, 78 layers, 21 of them with
their own indexer), the SGLang fp8 layout gives `78 × 656 + 21 × 132 = 53,940` B per
token.

Every TP rank needs the whole latent, so the MLA cache is **replicated** across
TP, not divided. A GQA cache is split by KV head; with more ranks than KV heads,
each rank keeps one, as SGLang does. A request holding `n` tokens with page size `g` occupies
`ceil(n / g) × g` tokens of cache.

## 3. Memory bounds

```
KV pool per GPU        ≤ C − W_stage                          BOUND (upper)
KV tokens per replica  ≤ floor(pool / KV per token)           BOUND (upper), binding stage
concurrent requests    ≤ floor(tokens / ceil(n / g) × g)      BOUND (upper)
with a reserve R       pool = C − W_stage − R                 ASSUMED (upper)
GPUs for the weights   ≥ ceil(W_total / C)                    BOUND (lower), any layout
```

SGLang's `--mem-fraction-static f` is read as `R = (1 − f) × C`. That is
ASSUMED, because SGLang applies `1 − f` to free memory before model load, which
is slightly less than `C`. "GPUs for weights + 1 request (TP × PP layouts)" is
the fewest GPUs among the layouts this package models where every stage holds
its weights and one request's KV.

## 4. Operator graph (EXACT, routed experts EXPECTED)

One pass serves `S` sequences, each adding `n` tokens at KV position `s`, so
`T = S × n`. Only compulsory HBM traffic is counted (§assumptions).

### Matmuls

`2 × T × out × in` FLOPs, divided by `p` for a split matmul. Bytes: the rank's
weight bytes, read once per pass. The table lists each family; "/p" marks
TP-split ones.

| operator | FLOPs |
|---|---|
| `q_a_proj`, `kv_a_proj` | `2T·r_q·H`, `2T·(r_kv + d_r)·H` |
| `q_b_proj` /p | `2T·h·d_qk·r_q / p` |
| GQA `q_proj` /p, `k_proj`, `v_proj` | `2T·h·d·H / p`, `2T·kv_l·d·H` each (`kv_l` = KV heads on the rank) |
| GQA core | `4 · h_l · d · pairs`, reading `kv_l` heads' cached keys and values |
| KDA projections /p | q, k, v, g: `2T·h·d·H / p` each; `f_a` `2T·d·H`; `f_b` `2T·h·d·d / p`; `b` `2T·h·H / p` |
| KDA recurrence | `4 · (h/p) · d² · T` (state read-out and rank-one update) + conv `2·K·3·(h·d/p)·T`; each sequence's state read (if it exists) and written once per pass |
| latent MoE | down and up projections `2T·L·H` each (replicated); routed experts on `L`: `2·3·T·k·L·I_e / p` |
| `o_proj` /p | `2T·H·h·d_v / p` |
| dense FFN /p | `2T·2I·H / p` + `2T·H·I / p` |
| router | `2T·E·H` |
| routed experts /p | `2·3·T·k·H·I_e / p` (the busiest rank does at least the average) |
| shared experts /p | `2·3·T·n_s·I_e·H / p` |
| indexer `wq_b`, `wk`, `weights_proj` | `2T·h_i·d_i·r_q`, `2T·d_i·H`, `2T·h_i·H` |
| `lm_head` | `2·T_logits·ceil(V / p)·H` |
| MTP `eh_proj` | `2T·H·2H` |

### Attention pairs (exact sums)

A query at position `q` attends `q + 1` keys, or `min(q + 1, K)` under top-k
sparse attention. Over a pass:

```
pairs = S × Σ_{q=s}^{s+n−1} min(q + 1, K)
```

This is evaluated in closed form (`causal_pairs`), not as `S²/2`.

**MLA, absorbed** (attention in latent space):

```
absorb q (W_UK) /p   2T · h · r_kv · d_n / p
core                 2 · h_l · ((r_kv + d_r) + r_kv) · pairs
absorb o (W_UV) /p   2T · h · d_v · r_kv / p
```

**MLA, expanded** (decompress cached latents to keys and values first):

```
kv_b expand /p       2 · keys · h · (d_n + d_v) · r_kv / p,
                     keys = S × (s + n) dense, S × min(s + n, K) top-k
core                 2 · h_l · (d_qk + d_v) · pairs
```

The pass bound takes whichever form gives the smaller time.

**KV traffic.**
- MLA, per sequence, reads its cached keys: all `s` of them when dense, or
  `max(min(s + n, K) − n, 0)` under top-k. It writes `n` new tokens.
- The DSA indexer scores every earlier key, `h_i × (d_i + 1)` multiply-adds per
  pair, but only for queries with more than `K` keys to choose from:

  ```
  index pairs = S × Σ_{q=s}^{s+n−1} (q + 1)·[q + 1 > K]
  ```

  When any pair is scored it reads all `s` cached index keys, and it always
  writes `n`.

### Routed experts read (EXPECTED)

Each token picks `k` of `E` experts. Under uniform independent routing, a rank
holding `E_r = E / EP` experts reads:

```
expected  E_r × (1 − (1 − k/E)^T)
fewest    min(E_r, ceil(k / EP))      (whatever the routing)
most      min(E_r, k × T)
```

The fraction `read / E_r` scales the rank's routed-expert bytes. The range is
carried to every result built on it.

### Interconnect

A TP all-reduce of `N = T × H × 2` bytes makes each rank receive at least
`(p − 1)/p × N`. There is one after `o_proj`, one after the FFN, and one after
the embedding. Time uses `Link / 2` per direction.

## 5. Pass bound (BOUND, lower)

```
t ≥ max( Σ_i F_i / P_{d_i},   Σ_i B_i / BW,   Σ_i X_i / (Link / 2) )
```

The sum is over every operator `i`: FLOPs `F`, compulsory bytes `B`, received
bytes `X`. This assumes nothing about overlap. By default the compute dtype
`d_i` is the storage dtype. With `--strict`, every `P_d` is the device's fastest
tensor rate. `--ops` also prints `Σ_i max(F_i/P, B_i/BW)`, which holds only if
operators never overlap and so is not reported as a bound.

## 6. Decode

```
TPOT(B, c)              ≥ t(pass: B sequences, 1 token, at context c)       BOUND
tokens/s per sequence   ≤ 1 / TPOT(1, c)                                    BOUND (upper)
decode tokens/s         ≤ B_max / TPOT(B_max, c)                            BOUND (upper)
```

- `B_max` is the smaller of the KV-memory concurrency (§3) and the engine's
  `--max-running-requests`.
- FLOPs scale with `T` while weight bytes do not, so `B / t(B)` never decreases
  in `B`. That makes `B_max` the maximum.
- **Critical batch:** the smallest `B` with `Σ F/P ≥ Σ B/BW`, found by exact
  bisection (EXPECTED, with the routing range).

## 7. MTP (draft depth `k`, acceptance rates `p_j` from you)

```
step       ≥ t(verify: B sequences × (1 + k) tokens) + k × t(MTP layer: B × 1)   BOUND
tokens     E = 1 + Σ_{i=1}^{k} Π_{j≤i} p_j,   range [1, 1 + k]                   ASSUMED
TPOT       ≥ step / E                                                            ASSUMED
```

The step is bounded as one combined pass (§5 over all its operators).

## 8. Prefill

The prompt is split into chunks of size `c`. Chunk `j` is a pass of `n_j` tokens
at position `s_j`. Only the last chunk computes logits, for one token. With MTP,
the MTP layer also processes every chunk to fill its cache.

```
TTFT ≥ max over resources of the sum over all chunks     BOUND
```

## 9. Sustained capacity (any schedule)

Per request (prompt `P`, output `O`):

```
compute  = min over MLA forms of Σ_prefill F/P  +  Σ_{O decode tokens} F/P
HBM      = prefill KV traffic + decode KV traffic + O × W_step(B) / B
link     = all-reduce bytes of P + O tokens / (Link/2)

QPS per replica      ≤ 1 / max(compute, HBM, link)       BOUND (upper)
output tokens/s      ≤ QPS × O                            BOUND (upper)
```

`W_step(B)` is one decode step's weight bytes at batch `B` (routed experts
EXPECTED). `B` is the smallest of the KV-limited concurrency at `P + 1` tokens,
the engine's cap, and the SLA batch. Prefill weight reads are not charged: they
can share decode steps.

Decode KV traffic is computed in closed form. The token at context `c` reads
`min(c, K − 1)` cached latents per layer, and `c` index keys per own-indexer
layer once `c ≥ K`:

```
decode KV = Σ_layers [ main × (Σ_{c=P}^{P+O−1} min(c, K−1) + O)
                       + index × (Σ_{c=P}^{P+O−1} c·[c ≥ K] + O) ]
```

## 10. SLA, fleet, cost

```
batch meeting TPOT ≤ x      largest B with TPOT(B, c) ≤ x                       BOUND (upper)
TTFT ≤ y                    "cannot be met" if TTFT bound > y, else "not ruled out"
replicas                    ≥ ceil(Q / QPS)                                     BOUND (lower)
GPUs, this layout           ≥ replicas × GPUs per replica                       BOUND (lower)
GPUs, any layout            ≥ ceil(Q × F_request / P_fastest)                   BOUND (lower)
cost per million output     ≥ price × GPUs per replica / (3600 × tokens/s) × 1e6   ASSUMED (lower)
```

For the TPOT target, the context `c` is `--context`, or else the prompt length,
which is the most permissive. `F_request` counts total FLOPs without TP
replication.

## 11. Calibration (ESTIMATE)

```
efficiency = bound / measured        (rejected if measured < bound)
estimate   = bound / efficiency      (times);   bound × efficiency   (rates)
```

The stored point nearest in `log(batch) + log(context)` (decode) or
`log(prompt)` (prefill) is used.

## 12. Display

Bounds round in their safe direction: an upper bound up, a lower bound down, and
a guaranteed range outward. EXACT values print in full. BOUND, ASSUMED and
EXPECTED values keep 3 to 4 significant digits, and ESTIMATE values keep 2.
