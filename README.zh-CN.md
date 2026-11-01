# llm-roofline-bound

LLM 推理容量的**可证明理论下界**，以及**明确标注的估算值**。

给定模型、加速卡、部署方式和负载，这个工具会算出**无法被超越**的极限：最短的 TTFT 和 TPOT、最高的 tokens/s 和并发数、最少需要几张卡。

- 能从公开的模型 config 和硬件规格精确推导出来的值，都会算出来。
- 推导不出来的，会要求你提供，并在结果里标明来自你的输入。

[English](README.md) · [公式](docs/formulas.md) · [假设](docs/assumptions.md) · [数据来源](docs/sources.md)

> **状态：pre-alpha。** 目前支持 GLM-5.3 × NVIDIA H20-3e，以及 SGLang 启动命令。架构上可以扩展到更多模型和加速卡。

## 安装

```bash
pip install git+https://github.com/Cao-Xing-Yu/llm-roofline-bound
```

需要 Python 3.11 及以上，没有运行时依赖。

## 快速上手

把服务的启动命令原样粘贴在 `--` 后面：

```bash
# 显存：权重、KV cache，以及能放下多少
rooflinebound memory --model glm-5.3 --hw h20-3e --context 32768 -- sglang serve --tp 8 ...

# 延迟和吞吐的界，外加 roofline 图和中文报告
rooflinebound bounds --model glm-5.3 --hw h20-3e --context 32768 --batch 16 --prompt 32768 \
  --roofline-out chart.html --lang zh -- sglang serve --tp 8 ...

# 持续容量、SLA、集群规模和成本
rooflinebound bounds --model glm-5.3 --hw h20-3e --prompt 8192 --output 1024 \
  --sla-tpot-ms 30 --target-qps 10 --price-gpu-hour 2.0 -- sglang serve --tp 8 ...
```

不粘贴启动命令时，可以直接指定布局，例如 `--tp 8 --kv-dtype fp8 --index-kv-dtype fp8`。所有选项见 `rooflinebound <命令> --help`。

## 每个数字都标明可信程度

| 级别 | 含义 | 显示方式 |
|---|---|---|
| `EXACT` | 对模型 config 和硬件规格做精确算术 | `53,940 B/token [EXACT]` |
| `EXPECTED` | 在明确的统计假设下取精确期望，并附保证范围 | `601 seq [EXPECTED 43..601]` |
| `BOUND` | 可证明成立的不等式 | `≥ 1.42 ms [BOUND]`、`≤ 31 req [BOUND]` |
| `ASSUMED` | 依赖你提供的输入 | `≤ 24 req [ASSUMED]` |
| `ESTIMATE` | 下界乘以你实测得到的效率 | `≈ 12 ms [ESTIMATE]` |

这些规则由代码强制执行：
- 结果的可信程度不会高于它最弱的输入。
- 界的方向必须保持不变，无法确定方向时直接拒绝计算。
- 需要你提供的值不会被悄悄填上默认值。
- 取整只会让界变松，不会变紧：上界向上取整，下界向下取整。

每个硬件参数都标明了来源，见 `rooflinebound hw h20-3e`。

## 能算什么

以下示例是 GLM-5.3 部署在 8 张 H20-3e 上，参数取自一条使用 fp8 KV cache 的 SGLang 启动命令。

- **显存**：权重逐个张量计算，和 checkpoint 的 safetensors header 完全一致；KV cache 按引擎的实际布局计算。示例结果：每 token 的 KV 为 53,940 B [EXACT]；32k 上下文时并发 ≤ 31 [BOUND]，扣除运行时预留后 ≤ 24 [ASSUMED]。
- **延迟和吞吐**：每个算子都会算出它的 FLOPs、必然发生的 HBM 字节数和互联字节数。不管怎么调度，一次 pass 都不可能快过三类资源按峰值速率各自所需时间中的最大值。示例结果：TPOT（batch 1）≥ 1.42 ms；TTFT（32k）≥ 1,840 ms。
- **持续容量、SLA、集群规模和成本**：每个请求对每种资源都有最小工作量，所以 QPS 不可能超过"资源速率 ÷ 每请求工作量"。示例结果：每副本 ≤ 2.18 req/s；达到 10 QPS 至少需要 40 张卡。
- **Roofline 图**：`--roofline-out chart.html` 生成单个文件，不需要联网，支持暗色模式，每张图下方附完整的数据表。
- **基于校准的估算**：`rooflinebound calibrate ... --save` 用一次实测算出效率，只保存在你本机；之后 `bounds --estimate` 会给出标为 ESTIMATE 的估算值。如果实测值比下界还快，会直接拒绝，因为这说明规格或输入和实际部署对不上。
- **计算精度**：默认每个算子按它的存储精度计算；`--strict` 让所有算子都按硬件最快的精度计算，这样不管引擎实际用什么精度，界都成立。
- **中文报告**：加 `--lang zh`。JSON 输出始终是英文，字段名保持稳定。

## 启动命令

`--` 之后粘贴 SGLang 命令（`sglang serve` 或 `python -m sglang.launch_server`），工具会从中读取：
- TP、EP、PP；
- KV 布局，包括 fp8 的 scale；
- chunk 大小和 page 大小；
- 最大运行请求数和上下文长度；
- MTP 草稿 token 数；
- 显存占比，换算成运行时预留，标为 ASSUMED。

每个参数都会被归入四类之一："已建模"、"影响结果但未建模"、"不影响结果"、"无法识别"，不会被悄悄忽略。无法正确建模的配置会直接报错，比如 DP attention、树形投机解码、其他量化方式。

## 开发

```bash
pip install -e ".[dev]"
pytest
ruff check .
python scripts/compliance_scan.py
```

## 许可证

Apache-2.0。仓库中附带的 GLM-5.3 `config.json` 遵循它自己的许可证，详见 [NOTICE](NOTICE)。
