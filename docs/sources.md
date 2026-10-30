# Sources

Every input value the package ships with, where it comes from, and how far that
source can be trusted. Bounds are only as good as these inputs.

## Hardware

### NVIDIA H20 141GB HBM3e (`h20-3e`)

NVIDIA publishes no public datasheet for this SKU. All values are **THIRD_PARTY**.

| field | value | source | note |
|---|---|---|---|
| mem_capacity | 141 GB | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | marketed; physical reportedly 144 GB |
| mem_bw | 4,000 GB/s | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | **disputed**: uncited reports give 4,800 GB/s |
| tflops_bf16 | 148 TFLOP/s | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | dense |
| tflops_fp8 | 296 TFLOP/s | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | dense (592 with 2:4 sparsity) |
| link_bw | 900 GB/s | [flopper.io](https://flopper.io/gpu/nvidia-h20-141gb) | NVLink, bidirectional |
| devices_per_node | 8 | [HGX H20 listing](https://viperatech.com/product/nvidia-hgx-h20) | |

Accessed 2026-10-30.

## Models

### GLM-5.3 (`glm-5.3`)

Shape read from [`zai-org/GLM-5.3`](https://huggingface.co/zai-org/GLM-5.3)
`config.json` at revision `ba32f8c19c8738e2c57c71798c601d477a7bf844`.
