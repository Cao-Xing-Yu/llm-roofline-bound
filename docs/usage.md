# Usage

This guide follows the order you would use the tool in. The running example is
GLM-5.3 on 8× H20-3e, served by SGLang. Other bundled pairs work the same way, for
example `--model qwen3-32b --hw h100-sxm`.

[中文](usage.zh-CN.md) · [Formulas](formulas.md) · [Assumptions](assumptions.md) · [Sources](sources.md)

## Contents

1. [Install](#1-install)
2. [Paste the launch command](#2-paste-the-launch-command)
3. [Memory: does it fit, how many requests](#3-memory-does-it-fit-how-many-requests)
4. [Latency and throughput bounds](#4-latency-and-throughput-bounds)
5. [Capacity planning: QPS, GPUs, cost](#5-capacity-planning-qps-gpus-cost)
6. [Roofline chart and operator detail](#6-roofline-chart-and-operator-detail)
7. [Calibration: from bounds to estimates](#7-calibration-from-bounds-to-estimates)
8. [Reading the output](#8-reading-the-output)
9. [Without a launch command](#9-without-a-launch-command)
10. [Other options](#10-other-options)
11. [Common errors](#11-common-errors)

## 1. Install

Requires Python 3.11+. There are no other runtime dependencies.

```bash
pip install git+https://github.com/Cao-Xing-Yu/llm-roofline-bound
# or, from a source checkout:
pip install -e .

rooflinebound --version
```

## 2. Paste the launch command

The simplest way to use the tool is to put the server's SGLang launch command,
unchanged, after a bare `--`. The tool reads these from it, so you do not
translate anything by hand:
- TP, EP and PP;
- the real KV layout, including its fp8 scales;
- chunk size and page size;
- the running-request cap and context length;
- MTP drafts;
- the memory fraction, which becomes a runtime reserve.

The examples below keep the command in a variable `$L`:

```bash
L='sglang serve --model-path /models --tp 8 --context-length 393216
   --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton_sparse_mla
   --dsa-decode-backend flashmla_kv --mem-fraction-static 0.92
   --chunked-prefill-size 16384 --max-running-requests 32
   --enable-hierarchical-cache --hicache-size 100'
```

Every flag lands in one of four groups, and the report's Notes list them. None
is dropped silently:

| group | examples | treatment |
|---|---|---|
| modelled | `--tp`, `--kv-cache-dtype`, `--chunked-prefill-size` | converted to the equivalent setting |
| not modelled | `--enable-hierarchical-cache`, CUDA graph flags | reported |
| no effect | `--host`, `--port`, parsers | listed and ignored |
| unrecognised | flags from newer versions | listed, so you can check they do not change memory or speed |

Configurations that cannot be modelled correctly are rejected:
- data-parallel attention;
- tree speculation (`--speculative-eagle-topk > 1`);
- quantization other than the checkpoint's own.

## 3. Memory: does it fit, how many requests

```bash
rooflinebound memory --model glm-5.3 --hw h20-3e --context 32768 -- $L
```

`--context` is the number of tokens one request holds in the KV cache, prompt
plus output.

```
Per GPU  (TP8 PP1 EP1)
  weights per GPU (largest stage)                  94.683065664 GB [EXACT]
  KV cache per token per GPU (largest stage)       53,940 B/token [EXACT]
Bounds  (no assumptions)
  KV pool per GPU                                  ≤ 56.1 GB [BOUND]
  concurrent requests at 32,768 tokens             ≤ 31 req [BOUND]
Under your assumptions
  concurrent requests at 32,768 tokens             ≤ 24 req [ASSUMED]
```

- **Bounds** take device memory minus weights, with nothing subtracted for the
  runtime, so they always hold.
- **Under your assumptions** also subtracts a runtime reserve. That reserve
  comes from `--mem-fraction-static` in the launch command, or from
  `--reserve-gb`.
- The model section lists each component's parameter count and its size in
  bytes, quantization scales included. Components can differ in storage dtype,
  so bytes do not always follow parameters. It also gives the checkpoint size
  and the active parameters per token.

## 4. Latency and throughput bounds

```bash
rooflinebound bounds --model glm-5.3 --hw h20-3e \
  --context 32768 --batch 16 --prompt 32768 -- $L
```

| option | meaning |
|---|---|
| `--context` | tokens each decoding sequence already holds |
| `--batch` | sequences decoding together. Without it, the report covers batch 1 and the largest batch that fits |
| `--prompt` | prompt length for TTFT |
| `--chunk` | prefill chunk size. Read from the launch command when there is one |

```
Decode at context 32,768
  TPOT, batch 1                                    ≥ 1.42 ms [BOUND]
  TPOT, batch 16                                   ≥ 8.96 ms [BOUND 2.03..10.9]
  decode tokens/s, max over batches                ≤ 2,030 token/s [ASSUMED 1,500..10,200]
  batch where decode turns compute-bound           601 seq [EXPECTED 43..601]
Prefill
  TTFT, 32,768-token prompt                        ≥ 1,840 ms [BOUND]
```

What each figure means:
- **TPOT and TTFT** cannot be beaten by any implementation or schedule.
- **Decode tokens/s** is bounded at the largest batch that fits in KV memory,
  and never above the launch command's `--max-running-requests`.
- **The critical batch** separates two regimes. Below it, decode is limited by
  reading weights and KV; above it, by compute.
- **A range in brackets** spans expert routing. The value assumes uniform
  routing, and the two ends are the most and least concentrated routings.

**MTP:**

```bash
rooflinebound bounds ... --mtp-draft 3 --mtp-accept 0.8,0.6,0.4 -- $L
```

`--mtp-accept` gives the acceptance rate of each draft token. Because you
supply these rates, TPOT with MTP is labelled ASSUMED.

## 5. Capacity planning: QPS, GPUs, cost

```bash
rooflinebound bounds --model glm-5.3 --hw h20-3e \
  --prompt 8192 --output 1024 \
  --sla-tpot-ms 30 --sla-ttft-ms 1500 \
  --target-qps 10 --price-gpu-hour 2.0 -- $L
```

| option | effect |
|---|---|
| `--prompt`, `--output` | input and output tokens per request (required) |
| `--sla-tpot-ms` | the largest batch that meets the TPOT target |
| `--sla-ttft-ms` | whether the TTFT target *cannot be met*, or is only *not ruled out* |
| `--target-qps` | the fewest replicas and GPUs for the target rate |
| `--price-gpu-hour` | USD per GPU-hour, giving a lower bound on cost per million output tokens |

```
  batch meeting TPOT ≤ 30 ms                       ≤ 388 seq [BOUND 388..507]
  requests/s per replica (8,192 in, 1,024 out)     ≤ 2.18 req/s [BOUND 1.52..2.18]
  output tokens/s per replica, sustained           ≤ 2,230 token/s [BOUND 1,550..2,230]
  GPUs for 10 requests/s, this layout              ≥ 40 gpu [BOUND 40..56]
  GPUs for 10 requests/s, any layout               ≥ 28 gpu [BOUND]
  cost per million output tokens                   ≥ 1.99 USD/Mtoken [ASSUMED 1.99..2.86]
```

Every request needs a minimum amount of compute, HBM traffic and interconnect
traffic. Requests per second therefore cannot exceed any resource's rate
divided by that work, whatever the schedule.

These figures are not given with MTP yet. The report says so rather than
showing a number that ignores MTP.

## 6. Roofline chart and operator detail

```bash
rooflinebound bounds ... --roofline-out chart.html -- $L
```

The output is one self-contained file. Open it in a browser; it needs no
network access and has light and dark themes.

- There is one chart per pass: decode at batch 1, at your batch, and at the
  largest batch that fits, plus prefill.
- Each point is an operator family, and its area is that family's share of the
  pass.
- A point on the slope is limited by HBM bandwidth; a point on a flat roof is
  limited by compute.
- Hover or keyboard focus shows the values, and a full table sits under each
  chart.

For per-operator FLOPs, bytes, intensity and limiter in the terminal, add
`--ops`.

## 7. Calibration: from bounds to estimates

A bound says what cannot be beaten; it does not predict. To get estimates for a
real deployment, measure once.

**1. Measure on the server.** For example:
- TPOT at batch 1 with an 8k context;
- TTFT for one 8k-token request on an idle server.

**2. Give the tool the measurement.**

```bash
rooflinebound calibrate --model glm-5.3 --hw h20-3e \
  --measured-tpot-ms <value> --batch 1 --context 8192 --save -- $L

rooflinebound calibrate --model glm-5.3 --hw h20-3e \
  --measured-ttft-ms <value> --prompt 8192 --save -- $L
```

The tool prints `efficiency = bound / measured`. The result is stored only on
your machine: in the file named by `ROOFLINEBOUND_CALIBRATIONS`, or else in the
user config directory. It never goes into the repository.

**3. Add `--estimate`, and the report gains an Estimates section.**

```bash
rooflinebound bounds ... --estimate -- $L
```

Estimates are labelled `≈ … [ESTIMATE]`. Each names the conditions its
efficiency was measured under and the conditions it is applied to.

Caveats:
- **Efficiency changes with batch.** At small batch, fixed per-step costs that
  the bound leaves out (kernel launches, synchronization) dominate. An
  efficiency measured at batch 1 is therefore low, and applying it to large
  batches gives pessimistic estimates. Calibrate near the batch you care about;
  the tool picks the stored point nearest in batch and context.
- **A measurement faster than its bound is rejected and not saved.** It means
  the hardware spec, the launch command or the measurement is off.

## 8. Reading the output

Every number carries a label that says how far to trust it:

| label | meaning | use it for |
|---|---|---|
| `[EXACT]` | arithmetic on the model config and hardware spec | as is |
| `≥ … [BOUND]`, `≤ … [BOUND]` | provable; reality is only worse | deciding feasibility. If TTFT ≥ 1.8 s, a 1.5 s target cannot be met |
| `[EXPECTED a..b]` | exact expectation under uniform expert routing, with a guaranteed range | magnitude and trends |
| `[ASSUMED]` | rests on your input (reserve, MTP acceptance, price) | as good as that input |
| `≈ … [ESTIMATE]` | bound divided by your measured efficiency | planning |

Two more points:
- Bounds only round toward weaker claims: upper bounds round up, lower bounds
  round down. A displayed number never claims more than was computed.
- **Read the Notes at the end.** They say which flags were not modelled, which
  spec values are third-party, and what `--strict` would change.

## 9. Without a launch command

You can give the deployment directly:

```bash
rooflinebound bounds --model glm-5.3 --hw h20-3e \
  --tp 8 --kv-dtype fp8 --index-kv-dtype fp8 --page-size 64 \
  --reserve-gb 12 --context 32768
```

| option | meaning | default |
|---|---|---|
| `--tp`, `--ep` (plus `--pp` for `memory`) | parallel layout. EP must be 1 or equal to TP | 1 |
| `--kv-dtype`, `--index-kv-dtype` | storage width of the KV cache and of DSA indexer keys | bf16 |
| `--page-size` | KV allocation unit, in tokens | 1 |
| `--reserve-gb` | per-GPU runtime memory besides weights and KV | none: figures that need it are not computed |

This uses a *uniform* layout: every element at one width, with no scales. For
GLM-5.3 under SGLang's fp8 layout, that undercounts KV per token by about 12%.
The results are still valid bounds, only looser, so paste the launch command
when you can.

## 10. Other options

| option | effect |
|---|---|
| `--lang zh` | Chinese report, and a Chinese roofline chart |
| `--json` | machine-readable output, always in English so field names stay stable |
| `--strict` | every operator at the fastest tensor rate: bounds that hold however an engine picks precision. A default report lists what `--strict` would change |
| `rooflinebound hw h20-3e` | the hardware spec, with each value's source and trust |
| `rooflinebound hw --list`, `rooflinebound memory --list-models` | the bundled hardware and models |
| `--hw my.toml`, `--model config.json` | your own hardware spec or Hugging Face config |

To write a hardware spec, start from `src/rooflinebound/spec/data/h20-3e.toml`.
Every field needs a unit and a trust level (`OFFICIAL`, `THIRD_PARTY` or
`USER`), and every value that is not `USER` needs an https source.

## 11. Common errors

| error | cause | fix |
|---|---|---|
| `conflicts with the launch command` | a CLI flag disagrees with the launch command | drop the duplicate flag |
| `--kv-dtype / --index-kv-dtype come from the launch command` | KV dtypes given alongside a launch command | drop them |
| `exceeds the server's --context-length` | `--context` or `--prompt` is beyond the server's limit | use a shorter length |
| `below the lower bound` | a calibration measurement beats the bound | check the hardware spec, the launch command and the measurement |
| `weights do not fit under ...` | the layout cannot hold the weights | raise TP (or PP, for `memory`) |
| `... is not modelled` | a configuration that cannot be modelled correctly | not supported yet; no approximation is given |
| `PP > 1 are not modelled yet` | latency or throughput asked for with PP > 1 | only `memory` supports PP for now |
