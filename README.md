# llm-roofline-bound

Provable lower bounds — and clearly-labelled estimates — for LLM inference capacity.

Given a model, an accelerator, a deployment (TP/PP/EP, dtypes) and a workload,
`llm-roofline-bound` computes what **cannot** be beaten: the minimum TTFT and TPOT,
the maximum throughput and concurrency, the fewest GPUs. Everything it can derive
exactly from the model config and the hardware spec, it does. Everything it cannot
is asked of you, and labelled as such.

> **Status: pre-alpha (P4).** Memory; TTFT / TPOT / throughput bounds from a per-GPU
> operator graph; sustained QPS, SLA, fleet and cost bounds; SGLang launch commands;
> roofline charts; calibrated estimates.
> Every assumption is listed in [docs/assumptions.md](docs/assumptions.md). See [docs/PLAN.md](docs/PLAN.md) (Chinese).

[中文说明](README.zh-CN.md)

## Every number says how far to trust it

```
$ rooflinebound hw h20-3e
NVIDIA H20 141GB HBM3e  (bundled:h20-3e.toml)
  mem_capacity      141 GB [EXACT]                   THIRD_PARTY
  mem_bw            4,800 GB/s [EXACT]               USER
  ...
Note: some values are not vendor-published. Every bound computed on this spec is
conditional on them.
```

| level | meaning | shown as |
|---|---|---|
| `EXACT` | arithmetic on config and spec, no assumption | `753,330,000 param [EXACT]` |
| `EXPECTED` | exact expectation under a stated assumption, with a guaranteed range | `223 expert [EXPECTED 64..256]` |
| `BOUND` | provable inequality | `≥ 1.23 s [BOUND]` |
| `ASSUMED` | a modelling input you supply | `[ASSUMED]` |
| `ESTIMATE` | a bound scaled by efficiency factors you supply | `≈ 18 ms [ESTIMATE]` |

These are enforced in code: a result is never more certain than its weakest
input, a bound keeps its direction or the computation refuses, and nothing that
needs your input gets a silent default.

## Memory

```
$ rooflinebound memory --model glm-5.3 --hw h20-3e --tp 8 --context 32768 --page-size 64
...
Per GPU  (TP8 PP1 EP1)
  weights per GPU (largest stage)                  94.683065664 GB [EXACT]
  KV cache per token per GPU (largest stage)       95,232 B/token [EXACT]

Bounds  (no assumptions)
  KV pool per GPU                                  ≤ 56.1 GB [BOUND]
  concurrent requests at 32,768 tokens             ≤ 17 req [BOUND]

Not computed
  - --reserve-gb (runtime memory per GPU besides weights and KV): needed for ...
```

Bounds round in their safe direction: an upper bound is never displayed below what was
computed.

## Latency and throughput

```
$ rooflinebound bounds --model glm-5.3 --hw h20-3e --tp 8 --context 32768 --batch 32     --prompt 32768 --chunk 16384 --mtp-draft 3 --mtp-accept 0.8,0.6,0.4
Decode at context 32,768  (bounds; their assumptions are under Notes)
  TPOT, batch 1                                    ≥ 1.46 ms [BOUND]
  TPOT, batch 32                                   ≥ 15.2 ms [BOUND 3.78..22.1]
  decode tokens/s, max over batches                ≤ 1,710 token/s [BOUND 1,400..6,390]
  batch where decode turns compute-bound           727 seq [EXPECTED 52..727]

MTP decode  (tokens per step and TPOT rest on your acceptance rates)
  tokens per step                                  2.47 token [ASSUMED 1..4]
  TPOT with MTP, batch 32                          ≥ 9.19 ms [ASSUMED 3.39..23.4]

Prefill  (bounds; their assumptions are under Notes)
  TTFT, 32,768-token prompt                        ≥ 2,380 ms [BOUND]
```

A pass takes at least the largest of total compute, total HBM traffic and total
interconnect traffic, each at peak. The range after a bound spans routing: the
value assumes uniform expert routing, and the ends are the most and least
concentrated routings. `--ops` prints the per-operator table behind each figure.

### Capacity, SLA, fleet and cost

```
$ rooflinebound bounds ... --prompt 8192 --output 1024 --sla-tpot-ms 30 --sla-ttft-ms 1500     --target-qps 10 --price-gpu-hour 2.0 -- sglang serve ...
Sustained capacity, SLA and fleet  (any schedule; assumptions under Notes)
  batch meeting TPOT ≤ 30 ms                       ≤ 388 seq [BOUND 388..507]
  requests/s per replica (8,192 in, 1,024 out)     ≤ 2.18 req/s [BOUND 1.52..2.18]
  output tokens/s per replica, sustained           ≤ 2,230 token/s [BOUND 1,550..2,230]
  GPUs for 10 requests/s, this layout              ≥ 40 gpu [BOUND 40..56]
  GPUs for 10 requests/s, any layout               ≥ 28 gpu [BOUND]
  cost per million output tokens                   ≥ 1.99 USD/Mtoken [ASSUMED 1.99..2.86]
```

Each request needs a minimum of compute (exact FLOPs), HBM (KV traffic plus its share of
decode weight reads) and interconnect, whatever the schedule, so requests per second
cannot exceed any resource's rate divided by that work.

### From a launch command

Paste the server's launch command after a bare `--`, and the layout, KV format, chunk
size, page size and running-request cap come from it. Flags that are not modelled are
listed, never silently dropped:

```
$ rooflinebound bounds --model glm-5.3 --hw h20-3e --context 32768 --prompt 32768 --     sglang serve --tp 8 --kv-cache-dtype fp8_e4m3 --dsa-decode-backend flashmla_kv     --mem-fraction-static 0.92 --chunked-prefill-size 16384 --max-running-requests 32 ...
```

By default each operator computes in its storage dtype (fp8 weights at the fp8
rate, attention at the KV dtype's rate). `--strict` puts every operator at the
fastest tensor rate, so the bounds hold however an engine picks precision. A
default report notes what `--strict` would change.

## Supported

| | status |
|---|---|
| GLM-5.3 (MLA + DSA + MoE + MTP) | memory and time bounds; inventory verified tensor by tensor against the checkpoint |
| NVIDIA H20 141GB HBM3e | spec bundled (third-party sourced) |

New models and accelerators are added as parsers and TOML spec files; see
[docs/sources.md](docs/sources.md) for every bundled value and its source.

## Roofline chart

`--roofline-out chart.html` writes every pass as a roofline: one point per operator family
at its arithmetic intensity, area proportional to its share of the pass, with the HBM slope
and a compute roof per precision. One self-contained file (no network), light and dark,
with the full table under each chart.

## Calibration and estimates

Bounds are not predictions. To turn them into estimates for *your* deployment, measure
once and let the tool derive the fraction of the roofline you reach:

```
rooflinebound calibrate ... --measured-tpot-ms 12.5 --batch 1 --context 8192 --save -- sglang serve ...
rooflinebound calibrate ... --measured-ttft-ms 1600 --prompt 8192 --save -- sglang serve ...
rooflinebound bounds ... --estimate -- sglang serve ...
```

`efficiency = bound / measured` is stored locally (`$ROOFLINEBOUND_CALIBRATIONS`, or the
user config directory), never in the repository. Estimates are labelled `≈ … [ESTIMATE]`
and name the conditions their efficiency was measured under; efficiency changes with batch
and context, so calibrate near where you will read. A measurement below its bound is
rejected: it means the spec or the inputs do not describe the deployment.

## Language

`--lang zh` prints the report in Chinese. JSON output (`--json`) stays in English, so its
field names are stable.

## Install

```bash
pip install -e ".[dev]"
pytest
```

Python 3.11+, no runtime dependencies.

## License

Apache-2.0
