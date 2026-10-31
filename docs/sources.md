# Sources

Every input value the package ships with, where it comes from, and how far that
source can be trusted. Bounds are only as good as these inputs.

## Hardware

### NVIDIA H20 141GB HBM3e (`h20-3e`)

NVIDIA publishes no public datasheet for this SKU. Values are **THIRD_PARTY** except where marked **USER**.

| field | value | source | note |
|---|---|---|---|
| mem_capacity | 143,771 MiB | **USER** (maintainer, `nvidia-smi --query-gpu=memory.total`) | 140.4 GiB; marketed as "141 GB" |
| mem_bw | 4,800 GB/s | **USER** (maintainer) | the third-party page lists 4,000 GB/s, apparently the 96 GB H20's figure |
| tflops_bf16 | 148 TFLOP/s | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | dense |
| tflops_fp8 | 296 TFLOP/s | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | dense (592 with 2:4 sparsity) |
| link_bw | 900 GB/s | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | NVLink, bidirectional |
| devices_per_node | 8 | [HGX H20 listing](https://viperatech.com/product/nvidia-hgx-h20) | |

Accessed 2026-10-30.

## Models

### GLM-5.3 (`glm-5.3`)

Shape parsed from [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)
`config.json` at revision `ba32f8c19c8738e2c57c71798c601d477a7bf844`, bundled unmodified
with the model's license (`src/rooflinebound/spec/data/models/`).

Verification: the generated tensor inventory equals the checkpoint's safetensors headers
tensor by tensor (name, shape, dtype; 118,629 tensors, 755,617,140,416 bytes). The headers
were read from all 141 shards with HTTP range requests and folded into
`tests/fixtures/glm-5.3.safetensors-summary.json`.

## Engines

### SGLang

Launch commands are read with the semantics of SGLang at commit
[`e3718813ec5f`](https://github.com/sgl-project/sglang/tree/e3718813ec5f) (2026-10-30):

| what | where |
|---|---|
| DSA fp8 KV row: `kv_lora_rank + kv_lora_rank/128 × 4 + rope × 2` bytes (scaled backends); raw `kv_lora_rank + rope` for TRT-LLM or TileLang on both sides | `python/sglang/srt/mem_cache/kv_cache_configurator.py`, `calculate_mla_kv_cache_dim` |
| indexer key: `index_head_dim + index_head_dim/128 × 4` bytes, always fp8; no rows for layers that reuse top-k | `python/sglang/srt/mem_cache/index_key_cache.py`, `memory_pool.py` (`DSATokenToKVPool`) |
| DSA on CUDA requires 64-token pages | `memory_pool.py` (`DSATokenToKVPool.__init__`) |
| KV budget keeps `(1 − mem_fraction_static) ×` free memory before model load out of the pool | `kv_cache_configurator.py`, `_profile_available_bytes` |
