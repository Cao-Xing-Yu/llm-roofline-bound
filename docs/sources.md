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

### NVIDIA H100 SXM 80GB (`h100-sxm`)

All values are **OFFICIAL**.

| field | value | source | note |
|---|---|---|---|
| mem_capacity | 80 GiB | [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/) | marketed "80GB", read as GiB so memory bounds never understate the device |
| mem_bw | 3.35 TB/s | [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/) | |
| tflops_bf16 | 989.5 TFLOP/s | [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/) | half of the listed 1,979, which the page marks "with sparsity" |
| tflops_fp8 | 1,979 TFLOP/s | [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/) | half of the listed 3,958 (with sparsity) |
| link_bw | 900 GB/s | [nvidia.com/h100](https://www.nvidia.com/en-us/data-center/h100/) | NVLink |
| devices_per_node | 8 | [HGX reference architecture](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html) | "eight H100/H200 GPUs" per baseboard |

Accessed 2026-11-01.

### NVIDIA B300 SXM 288GB, HGX (`b300`)

All values are **OFFICIAL**.

| field | value | source | note |
|---|---|---|---|
| mem_capacity | 288 GiB | [Blackwell Ultra blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) | 288 GB HBM3e, read as GiB so memory bounds never understate the device |
| mem_bw | 8 TB/s | [Blackwell Ultra blog](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) | "up to" |
| tflops_bf16 | 2,250 TFLOP/s | [nvidia.com/hgx](https://www.nvidia.com/en-us/data-center/hgx/) | 36 PFLOPS per 8 GPUs, sparse; dense is half |
| tflops_fp8 | 4,500 TFLOP/s | [nvidia.com/hgx](https://www.nvidia.com/en-us/data-center/hgx/) | 72 PFLOPS per 8 GPUs, sparse; dense is half (generic chip table: 5 PF dense; SKUs differ) |
| tflops_fp4 | 13,500 TFLOP/s | [nvidia.com/hgx](https://www.nvidia.com/en-us/data-center/hgx/) | 144 / 108 PFLOPS per 8 GPUs, sparse / dense (generic chip table: 15 PF dense) |
| link_bw | 1,800 GB/s | [nvidia.com/hgx](https://www.nvidia.com/en-us/data-center/hgx/) | NVLink 5 |
| devices_per_node | 8 | [nvidia.com/hgx](https://www.nvidia.com/en-us/data-center/hgx/) | "8x NVIDIA Blackwell Ultra SXM" |

Accessed 2026-11-01.

## Models

### GLM-5.3 (`glm-5.3`)

Shape parsed from [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)
`config.json` at revision `ba32f8c19c8738e2c57c71798c601d477a7bf844`, bundled unmodified
with the model's license (`src/rooflinebound/spec/data/models/`).

Verification: the generated tensor inventory equals the checkpoint's safetensors headers
tensor by tensor (name, shape, dtype; 118,629 tensors, 755,617,140,416 bytes). The headers
were read from all 141 shards with HTTP range requests and folded into
`tests/fixtures/glm-5.3.safetensors-summary.json`.

### Qwen3-32B (`qwen3-32b`)

Shape parsed from [`Qwen/Qwen3-32B`](https://huggingface.co/Qwen/Qwen3-32B) `config.json`
at revision `9216db5781bf21249d130ec9da846c4624c16137`, bundled unmodified with its
Apache-2.0 license. The generated inventory equals the safetensors headers tensor by tensor
(707 tensors, 65,524,246,528 bytes; `tests/fixtures/qwen3-32b.safetensors-summary.json`).

### Kimi-K3 (`kimi-k3`)

Shape parsed from [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3)
`config.json` at revision `f831ab66814297da540d832a5235f8e904f29d06`, bundled unmodified with
the Kimi K3 License. The generated inventory equals the safetensors headers tensor by tensor
(497,220 tensors in 96 shards, 1,560,860,324,864 bytes;
`tests/fixtures/kimi-k3.safetensors-summary.json`), including:
- the MXFP4 routed experts, two fp4 values per byte plus one e8m0 scale per 32;
- the fp32 KDA parameters;
- the 27-block vision tower and its projector.

The config's quantization `ignore` list does not cover the latent MoE projections, but the
checkpoint stores them in bf16; quantization is applied to routed experts only, as the
checkpoint shows.

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
| MHA/GQA pool: K and V buffers of `kv heads per rank × head_dim` per token and layer, at the KV dtype | `memory_pool.py` (`MHATokenToKVPool`) |
| KV heads per rank: `max(1, kv_heads // tp)` | `configs/model_config.py`, `get_num_kv_heads` |
| MLA pool without DSA: `kv_lora_rank + rope` elements at the KV dtype | `kv_cache_configurator.py`, `calculate_mla_kv_cache_dim` |
| KDA state per request and layer: `(heads/tp) x head_dim^2` at the SSM dtype (fp32 by default) + conv window `(K-1) x 3 x proj/tp` at bf16 | `configs/mamba_utils.py` (`KimiLinearStateShape`, `mamba2_state_dtype`) |
| Kimi-K3 vision tower runs data-parallel (plain linears), so it is replicated on every rank; `--language-only` skips it | `models/kimi_k3_vl.py`, `models/kimi_k3.py` |
| attention DP: attention TP = `tp / attn_dp_size / attn_cp_size`; `--dp-size N --enable-dp-attention` resolves to `attn_dp_size = N`, `dp_size = 1` | `runtime_context.py` (`derive_attn_tp_size`), `arg_groups/parallel_hook.py` (`handle_deprecated_dp_attention`) |
| attention DP: MLA / GQA projections on `attn_tp`; embedding vocab-split over `attn_tp` (unless `SGLANG_ENABLE_EMBED_REPLICATION`); `lm_head` over `attn_tp` only with `--enable-dp-lm-head`; dense MLP and shared experts on the full TP (unless `--moe-dense-tp-size 1`, an all-to-all MoE backend or `SGLANG_SHARED_EXPERT_TP1`) | `models/deepseek_v2.py`, `models/qwen3.py`, `models/qwen2.py`, `layers/vocab_parallel_embedding.py` (`get_embedding_tp_kwargs`) |
| latent MoE projections, KDA `f_a_proj`, MLA down-projections and attention-residual projections are replicated; `f_b` / `b` / `g` projections are column-parallel | `models/kimi_k3.py` |
