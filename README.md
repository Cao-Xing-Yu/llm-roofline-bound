# llm-roofline-bound

Provable lower bounds — and clearly-labelled estimates — for LLM inference capacity.

Given a model, an accelerator, a deployment (TP/PP/EP, dtypes) and a workload,
`llm-roofline-bound` computes what **cannot** be beaten: the minimum TTFT and TPOT,
the maximum throughput and concurrency, the fewest GPUs. Everything it can derive
exactly from the model config and the hardware spec, it does. Everything it cannot
is asked of you, and labelled as such.

> **Status: pre-alpha (P1).** Memory is done: parameters, weights per GPU, KV cache and
> what fits. Latency and throughput bounds land next. See [docs/PLAN.md](docs/PLAN.md) (Chinese).

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

## Supported

| | status |
|---|---|
| GLM-5.3 (MLA + DSA + MoE + MTP) | memory; inventory verified tensor by tensor against the checkpoint |
| NVIDIA H20 141GB HBM3e | spec bundled (third-party sourced) |

New models and accelerators are added as parsers and TOML spec files; see
[docs/sources.md](docs/sources.md) for every bundled value and its source.

## Install

```bash
pip install -e ".[dev]"
pytest
```

Python 3.11+, no runtime dependencies.

## License

Apache-2.0
