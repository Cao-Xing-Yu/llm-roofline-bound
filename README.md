# llm-roofline-bound

Provable lower bounds, and clearly labelled estimates, for LLM inference capacity.

Give it a model, an accelerator, a deployment and a workload, and
`llm-roofline-bound` computes what **cannot be beaten**: the shortest TTFT and
TPOT, the most tokens per second and concurrent requests, the fewest GPUs. It
derives everything it can exactly from the published model config and the
hardware spec. Whatever it cannot derive, it asks you for, and labels as yours.

[中文说明](README.zh-CN.md) · **[Usage](docs/usage.md)** · [Formulas](docs/formulas.md) ·
[Assumptions](docs/assumptions.md) · [Sources](docs/sources.md)

> **Status: pre-alpha.** Supported today: GLM-5.3, Qwen3-32B and Kimi-K3; NVIDIA H20-3e,
> H100 SXM and B300; plus SGLang launch commands. The architecture is built to take more models and accelerators.

## Install

```bash
pip install git+https://github.com/Cao-Xing-Yu/llm-roofline-bound
```

Python 3.11+, no runtime dependencies.

## Quick start

Paste your server's launch command after a bare `--`:

```bash
# memory: weights, KV cache, what fits
rooflinebound memory --model glm-5.3 --hw h20-3e --context 32768 -- sglang serve --tp 8 ...

# latency and throughput bounds, a roofline chart, and a Chinese report
rooflinebound bounds --model glm-5.3 --hw h20-3e --context 32768 --batch 16 --prompt 32768 \
  --roofline-out chart.html --lang zh -- sglang serve --tp 8 ...

# capacity, SLA, fleet size and cost
rooflinebound bounds --model glm-5.3 --hw h20-3e --prompt 8192 --output 1024 \
  --sla-tpot-ms 30 --target-qps 10 --price-gpu-hour 2.0 -- sglang serve --tp 8 ...
```

Without a launch command, give the layout directly, for example
`--tp 8 --kv-dtype fp8 --index-kv-dtype fp8`. Run `rooflinebound <command> --help`
for every option.

## Every number says how far to trust it

| level | meaning | shown as |
|---|---|---|
| `EXACT` | arithmetic on the model config and hardware spec | `53,940 B/token [EXACT]` |
| `EXPECTED` | exact expectation under a stated assumption, with a guaranteed range | `601 seq [EXPECTED 43..601]` |
| `BOUND` | provable inequality | `≥ 1.42 ms [BOUND]`, `≤ 31 req [BOUND]` |
| `ASSUMED` | rests on an input you supplied | `≤ 24 req [ASSUMED]` |
| `ESTIMATE` | a bound scaled by an efficiency you measured | `≈ 12 ms [ESTIMATE]` |

This is enforced in code:
- A result is never more certain than its weakest input.
- A bound keeps its direction, or the computation refuses to run.
- Nothing that needs your input gets a silent default.
- A bound is never rounded into a stronger claim: upper bounds round up and
  lower bounds round down.

Every hardware value says where it came from:

```
$ rooflinebound hw h20-3e
NVIDIA H20 141GB HBM3e  (bundled:h20-3e.toml)
  mem_capacity      140.401367188 GiB [EXACT]        USER
  mem_bw            4,800 GB/s [EXACT]               USER
  tflops_bf16       148 TFLOP/s [EXACT]              THIRD_PARTY
  tflops_fp8        296 TFLOP/s [EXACT]              THIRD_PARTY
  ...
```

## What it computes

The examples below are GLM-5.3 on 8× H20-3e, read from an SGLang launch command
with an fp8 KV cache.

**Memory.** Weights are counted tensor by tensor; the generated inventory matches
the checkpoint's safetensors headers exactly. KV cache follows the engine's real
layout.

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

**Latency and throughput.** Each operator of a pass contributes FLOPs,
compulsory HBM bytes and interconnect bytes. A pass cannot finish faster than
the largest of the three totals at peak, whatever the schedule.

```
Decode at context 32,768
  TPOT, batch 1                                    ≥ 1.42 ms [BOUND]
  TPOT, batch 16                                   ≥ 8.96 ms [BOUND 2.03..10.9]
  decode tokens/s, max over batches                ≤ 2,030 token/s [ASSUMED 1,500..10,200]
  batch where decode turns compute-bound           601 seq [EXPECTED 43..601]
Prefill
  TTFT, 32,768-token prompt                        ≥ 1,840 ms [BOUND]
```

The range after a value spans expert routing: the value assumes uniform
routing, and the ends are the most and least concentrated routings. `--ops`
prints the per-operator table behind each figure. MTP is supported with
`--mtp-draft k --mtp-accept p1,p2,...`.

**Capacity, SLA, fleet and cost.** Every request needs a minimum of compute, HBM
traffic and interconnect traffic, so requests per second cannot exceed any
resource's rate divided by that work.

```
  batch meeting TPOT ≤ 30 ms                       ≤ 388 seq [BOUND 388..507]
  requests/s per replica (8,192 in, 1,024 out)     ≤ 2.18 req/s [BOUND 1.52..2.18]
  GPUs for 10 requests/s, this layout              ≥ 40 gpu [BOUND 40..56]
  GPUs for 10 requests/s, any layout               ≥ 28 gpu [BOUND]
  cost per million output tokens                   ≥ 1.99 USD/Mtoken [ASSUMED 1.99..2.86]
```

**Roofline chart.** `--roofline-out chart.html` writes one self-contained file
(no network access, light and dark themes). It has a roofline per pass, one
point per operator family, and the full table under each chart.

**Calibrated estimates.** Bounds are not predictions. Measure once and the tool
derives the fraction of the roofline your deployment reaches:

```bash
rooflinebound calibrate ... --measured-tpot-ms 12.5 --batch 1 --context 8192 --save -- sglang serve ...
rooflinebound bounds ... --estimate -- sglang serve ...
```

`efficiency = bound / measured` is stored only on your machine. Estimates are
labelled `≈ … [ESTIMATE]` and name the conditions under which their efficiency
was measured. A measurement faster than its bound is rejected, because it means
the spec or the inputs do not describe the deployment.

**Precision.** By default each operator computes in its storage dtype. With
`--strict`, every operator runs at the fastest tensor rate, so the bounds hold
however an engine picks precision. A default report notes any figure `--strict`
would change.

## Launch commands

After `--`, an SGLang command (`sglang serve` or
`python -m sglang.launch_server`) supplies:
- TP, EP and PP;
- the KV layout, including its fp8 scales;
- chunk size and page size;
- running-request cap and context length;
- MTP drafts;
- the memory fraction, which becomes an ASSUMED reserve.

Each flag is either modelled, reported as not modelled, listed as having no
effect, or listed as unrecognised; none is dropped silently. A configuration
that cannot be modelled correctly, such as tree speculation or another
quantization, is rejected. DP attention (`--attn-dp-size`, or the older
`--dp N --enable-dp-attention`) is modelled for memory; the latency bounds
refuse it until it is modelled for time too.

## Supported

| | |
|---|---|
| GLM-5.3 (MLA + DSA + MoE + MTP) | inventory verified tensor by tensor against the checkpoint |
| Qwen3-32B (dense GQA, QK-norm) | inventory verified tensor by tensor against the checkpoint |
| NVIDIA H20 141GB HBM3e | capacity and bandwidth from the maintainer's device; the rest third-party sourced |
| Kimi-K3 (hybrid KDA + MLA, latent MoE, MXFP4 experts, vision) | inventory verified tensor by tensor (497,220 tensors) against the checkpoint |
| NVIDIA H100 SXM 80GB | every value vendor-published |
| NVIDIA B300 SXM 288GB (HGX) | every value vendor-published, including the fp4 rate |
| SGLang | launch commands; DSA, MLA and MHA/GQA KV layouts, and linear-attention state, at a pinned commit |

Models are assembled from components (attention, FFN, MoE, MTP) by a parser per
Hugging Face `model_type`. Accelerators are TOML files that carry a source and a
trust level per value. Adding either is self-contained. The bundled values and
their sources are listed in [docs/sources.md](docs/sources.md).

## Development

```bash
pip install -e ".[dev]"
pytest                               # ~25 s
ruff check .
python scripts/compliance_scan.py    # private IPs, credentials; local patterns optional
```

## License

Apache-2.0. The bundled GLM-5.3 `config.json` is distributed under its own license;
see [NOTICE](NOTICE).
