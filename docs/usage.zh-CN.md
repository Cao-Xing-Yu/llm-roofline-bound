# 使用说明

本文以 GLM-5.3 部署在 8 张 H20-3e 上、用 SGLang 提供服务为例，按实际使用的顺序介绍各个命令。换成其他内置组合时，用法完全一样，比如 `--model qwen3-32b --hw h100-sxm`。

[English](usage.md) · [公式](formulas.md) · [假设](assumptions.md) · [数据来源](sources.md)

## 目录

1. [安装](#1-安装)
2. [粘贴启动命令](#2-粘贴启动命令)
3. [显存：放不放得下，能放多少并发](#3-显存放不放得下能放多少并发)
4. [延迟和吞吐的下界](#4-延迟和吞吐的下界)
5. [容量规划：QPS、卡数、成本](#5-容量规划qps卡数成本)
6. [Roofline 图和算子明细](#6-roofline-图和算子明细)
7. [校准：把下界变成估算](#7-校准把下界变成估算)
8. [怎么读结果](#8-怎么读结果)
9. [不粘贴启动命令时](#9-不粘贴启动命令时)
10. [其他选项](#10-其他选项)
11. [常见报错](#11-常见报错)

## 1. 安装

需要 Python 3.11 及以上，没有其他运行时依赖。

```bash
pip install git+https://github.com/Cao-Xing-Yu/llm-roofline-bound
# 或者在本地源码目录里：
pip install -e .

rooflinebound --version
```

## 2. 粘贴启动命令

最省事的用法，是把服务的 SGLang 启动命令原样放在 `--` 后面。下面这些设置都会从命令里自动读出，不用自己换算：
- TP、EP、PP；
- KV cache 的实际布局，包括 fp8 的 scale；
- chunk 大小和 page 大小；
- 最大运行请求数和上下文长度；
- MTP 草稿 token 数；
- 显存占比，会换算成运行时预留。

为了书写方便，下文把启动命令存进变量 `$L`：

```bash
L='sglang serve --model-path /models --tp 8 --context-length 393216
   --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton_sparse_mla
   --dsa-decode-backend flashmla_kv --mem-fraction-static 0.92
   --chunked-prefill-size 16384 --max-running-requests 32
   --enable-hierarchical-cache --hicache-size 100'
```

启动命令里的每个参数都会被归入下面四类之一，并在报告的"说明"部分列出来，不会被悄悄忽略：

| 类别 | 例子 | 处理方式 |
|---|---|---|
| 已建模 | `--tp`、`--kv-cache-dtype`、`--chunked-prefill-size` | 转换成工具里对应的设置 |
| 影响结果但未建模 | `--enable-hierarchical-cache`、CUDA graph 相关参数 | 在报告里写明 |
| 不影响结果 | `--host`、`--port`、各种 parser 参数 | 列出后忽略 |
| 无法识别 | 新版本才有的参数 | 列出来，请你确认它是否影响显存或速度 |

无法正确建模的配置会直接报错，比如树形投机解码（`--speculative-eagle-topk > 1`）、checkpoint 以外的量化方式。DP attention 只有 `memory` 支持（见第 3 节），`bounds` 和 `calibrate` 遇到时会报错。

## 3. 显存：放不放得下，能放多少并发

```bash
rooflinebound memory --model glm-5.3 --hw h20-3e --context 32768 -- $L
```

`--context` 填的是单个请求在 KV cache 里占用的 token 数，也就是输入加输出。

输出示例（节选）：

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

- **Bounds 部分**只考虑显存容量减去权重，不扣运行时开销，所以是一定成立的上界。
- **Under your assumptions 部分**额外扣除了运行时预留，按启动命令里的 `--mem-fraction-static`，或者你用 `--reserve-gb` 指定的值计算。
- "每卡权重"下面会按组件列出每卡分别占多少，算的是每卡权重最大的那个 stage，各项加起来正好等于每卡权重。最后一行"其中每卡都复制的部分"是不随 TP 增大而减少的那部分，包括 router、norm、MLA 的下投影，以及 TP 大于 KV head 数时的 GQA k/v 投影。
- 模型部分会列出每个组件的参数量和对应的字节数（含量化 scale；各组件的存储精度可能不同，所以字节数不一定和参数量成正比），以及 checkpoint 总大小和每 token 的激活参数量。

**DP attention。** 使用 `--attn-dp-size N`（或旧写法 `--dp N --enable-dp-attention`）时，TP 组被分成 `N` 个组，每组 `TP / N` 个 rank，各组只服务、只缓存自己的请求：
- attention 投影和词嵌入在组内切分，每组各存一份（报告里的"其中只在 attention TP 内切分的部分"）；FFN、专家和 `lm_head` 仍在完整 TP 上切分（加 `--enable-dp-lm-head` 时 `lm_head` 也在组内切分）；
- MLA 的 KV 本来就在每张卡上完整存一份，所以每卡每 token 的 KV 不变，但每副本能容纳的 token 数和并发数是单组的 `N` 倍；
- GQA 的 KV 只在组内按 KV head 切分，每卡每 token 的 KV 会变大。

以 GLM-5.3、8 × H20-3e、131,072 token 为例：attention DP 8 让每卡权重多出 11.8 GB，并发上界从 7 提高到 48。目前支持 MLA/DSA 和 GQA 模型（Kimi-K3 暂不支持），且只计算显存。

## 4. 延迟和吞吐的下界

```bash
rooflinebound bounds --model glm-5.3 --hw h20-3e \
  --context 32768 --batch 16 --prompt 32768 -- $L
```

| 参数 | 含义 |
|---|---|
| `--context` | decode 时每个序列已经持有的 token 数 |
| `--batch` | 同时 decode 的序列数。不填时只计算 batch 1，以及显存能放下的最大 batch |
| `--prompt` | 计算 TTFT 用的 prompt 长度 |
| `--chunk` | prefill 的分块大小。粘贴启动命令时会自动读取 |

输出示例：

```
Decode at context 32,768
  TPOT, batch 1                                    ≥ 1.42 ms [BOUND]
  TPOT, batch 16                                   ≥ 8.96 ms [BOUND 2.03..10.9]
  decode tokens/s, max over batches                ≤ 2,030 token/s [ASSUMED 1,500..10,200]
  batch where decode turns compute-bound           601 seq [EXPECTED 43..601]
Prefill
  TTFT, 32,768-token prompt                        ≥ 1,840 ms [BOUND]
```

各项的含义：
- **TPOT 和 TTFT 下界**：不管引擎怎么实现、怎么调度，都不可能比这个更快。
- **decode tokens/s 上界**：在显存能放下的最大 batch 处取到，同时不会超过启动命令里的 `--max-running-requests`。
- **转为算力受限的临界 batch**：batch 小于它时，decode 受限于读取权重和 KV；大于它时受限于算力。
- **括号里的范围**：数值本身假设 MoE 专家路由是均匀的，括号两端分别对应路由最集中和最分散的情况。

**开启 MTP**：

```bash
rooflinebound bounds ... --mtp-draft 3 --mtp-accept 0.8,0.6,0.4 -- $L
```

`--mtp-accept` 是每个草稿 token 的接受率，由你提供，所以带 MTP 的 TPOT 会标为 ASSUMED。

## 5. 容量规划：QPS、卡数、成本

```bash
rooflinebound bounds --model glm-5.3 --hw h20-3e \
  --prompt 8192 --output 1024 \
  --sla-tpot-ms 30 --sla-ttft-ms 1500 \
  --target-qps 10 --price-gpu-hour 2.0 -- $L
```

| 参数 | 作用 |
|---|---|
| `--prompt`、`--output` | 每个请求的输入和输出 token 数（必须给） |
| `--sla-tpot-ms` | TPOT 目标：算出满足这个目标的最大 batch |
| `--sla-ttft-ms` | TTFT 目标：判断"肯定达不到"还是"没有被下界排除" |
| `--target-qps` | 目标 QPS：算出至少需要几个副本、几张卡 |
| `--price-gpu-hour` | 每卡每小时价格（美元）：算出每百万输出 token 成本的下界 |

输出示例：

```
  batch meeting TPOT ≤ 30 ms                       ≤ 388 seq [BOUND 388..507]
  requests/s per replica (8,192 in, 1,024 out)     ≤ 2.18 req/s [BOUND 1.52..2.18]
  output tokens/s per replica, sustained           ≤ 2,230 token/s [BOUND 1,550..2,230]
  GPUs for 10 requests/s, this layout              ≥ 40 gpu [BOUND 40..56]
  GPUs for 10 requests/s, any layout               ≥ 28 gpu [BOUND]
  cost per million output tokens                   ≥ 1.99 USD/Mtoken [ASSUMED 1.99..2.86]
```

推导思路：每个请求对算力、HBM、互联三种资源都有一个最小工作量。用资源速率除以每请求工作量，取最小值就是 QPS 上界，所以它在任何调度方式下都成立。

目前带 MTP 时不计算这一部分，工具会明确提示"还没有建模"，而不是给出一个不考虑 MTP 的数字。

## 6. Roofline 图和算子明细

```bash
rooflinebound bounds ... --roofline-out chart.html -- $L
```

生成的 `chart.html` 是单个文件，用浏览器打开就行，不需要联网，支持暗色模式。

- 每次 pass 一张图：decode batch 1、指定的 batch、显存能放下的最大 batch，以及 prefill。
- 每个点是一个算子族，点的面积表示它占这次 pass 的时间比例。
- 点落在斜线上，说明受限于 HBM 带宽；落在水平线上，说明受限于算力。
- 鼠标悬停或键盘聚焦可以看具体数值，每张图下方还附了完整的数据表。

想在终端里看每个算子的 FLOPs、字节数、算术强度和瓶颈，加 `--ops`。

## 7. 校准：把下界变成估算

下界只说明"不可能更快"，不是预测值。要得到贴近实际部署的估算，需要先实测一次。

**第一步：在服务上实测。** 比如：
- TPOT：batch 1、8k 上下文时测一次；
- TTFT：服务空闲时，只发一个请求，用 8k prompt 测一次。

**第二步：把实测值交给工具，保存效率。**

```bash
rooflinebound calibrate --model glm-5.3 --hw h20-3e \
  --measured-tpot-ms <实测值> --batch 1 --context 8192 --save -- $L

rooflinebound calibrate --model glm-5.3 --hw h20-3e \
  --measured-ttft-ms <实测值> --prompt 8192 --save -- $L
```

工具会输出 `效率 = 下界 / 实测值`。数据只保存在本机：路径是环境变量 `ROOFLINEBOUND_CALIBRATIONS` 指定的文件，没有设置时放在用户配置目录下，不会写进仓库。

**第三步：加 `--estimate`，报告会多出一个"估算"部分。**

```bash
rooflinebound bounds ... --estimate -- $L
```

估算值标为 `≈ … [ESTIMATE]`，并写明效率是在什么条件下测的、现在用在什么条件下。

注意事项：
- **效率会随 batch 明显变化。** batch 很小时，每步的固定开销（kernel launch、同步）占大头，下界没有包含这部分，所以 batch 1 测出的效率偏低，拿去估大 batch 会偏悲观。最好在关心的 batch 附近也测一次。工具会自动选 batch 和上下文最接近的那个校准点。
- **实测值比下界还小时会直接报错，而且不会保存。** 这说明硬件规格、启动参数或测量方法有问题，值得查一下原因。

## 8. 怎么读结果

每个数字后面都有一个标签，表示它的可信程度：

| 标签 | 含义 | 怎么用 |
|---|---|---|
| `[EXACT]` | 对模型 config 和硬件规格做精确算术 | 直接用 |
| `≥ … [BOUND]`、`≤ … [BOUND]` | 可证明成立的界，实际值只会更差 | 判断"能不能做到"。比如 TTFT ≥ 1.8 s，那 1.5 s 的目标就一定做不到 |
| `[EXPECTED a..b]` | 假设专家路由均匀时的精确期望，括号里是保证范围 | 看量级和趋势 |
| `[ASSUMED]` | 依赖你提供的输入，比如预留显存、MTP 接受率、单价 | 输入准，结果才准 |
| `≈ … [ESTIMATE]` | 下界除以你实测得到的效率 | 做规划时参考 |

另外有两点：
- 界在显示时只会往更松的方向取整：上界向上取整，下界向下取整。显示出来的数字不会比算出来的结论更强。
- **一定要看报告末尾的 Notes（说明）。** 哪些参数没有建模、哪些硬件规格来自第三方、`--strict` 下结果会怎么变，都写在这里。

## 9. 不粘贴启动命令时

也可以直接用参数指定部署方式：

```bash
rooflinebound bounds --model glm-5.3 --hw h20-3e \
  --tp 8 --kv-dtype fp8 --index-kv-dtype fp8 --page-size 64 \
  --reserve-gb 12 --context 32768
```

| 参数 | 含义 | 默认值 |
|---|---|---|
| `--tp`、`--ep`（`memory` 命令还有 `--pp`） | 并行方式。EP 只能是 1，或者等于 TP | 1 |
| `--kv-dtype`、`--index-kv-dtype` | KV 和 DSA indexer key 的存储精度 | bf16 |
| `--page-size` | KV 的分配粒度，单位是 token | 1 |
| `--reserve-gb` | 每卡除权重和 KV 之外的运行时显存 | 不填就不计算依赖它的结果 |

这种方式用的是"统一精度"布局：每个元素都按同一种宽度存储，不含 scale。以 GLM-5.3 的 SGLang fp8 布局为例，这样每个 token 会少算约 12% 的 KV。结果仍然成立，只是界更松。所以能粘贴启动命令时，尽量粘贴。

## 10. 其他选项

| 选项 | 作用 |
|---|---|
| `--lang zh` | 输出中文报告，roofline 图也会是中文 |
| `--json` | 输出给程序读的 JSON（始终是英文，字段名保持稳定） |
| `--attn-dp N`、`--dp-lm-head` | 不粘贴启动命令时指定 DP attention（只用于显存计算） |
| `--strict` | 所有算子都按硬件最快的精度计算，得到不依赖引擎实现方式的界。默认模式下，报告会自动列出 `--strict` 会改变的数值 |
| `rooflinebound hw h20-3e` | 查看硬件规格，以及每个值的来源和可信度 |
| `rooflinebound hw --list`、`rooflinebound memory --list-models` | 列出内置的硬件和模型 |
| `--hw my.toml`、`--model config.json` | 使用自己的硬件规格文件或 Hugging Face config |

自定义硬件规格可以参考内置的 `src/rooflinebound/spec/data/h20-3e.toml`：每个字段都要写明单位、可信度（`OFFICIAL`、`THIRD_PARTY` 或 `USER`），非 `USER` 的值还必须附上 https 来源。

## 11. 常见报错

| 报错 | 原因 | 处理 |
|---|---|---|
| `conflicts with the launch command` | 命令行参数和启动命令里的不一致 | 删掉命令行里重复的参数 |
| `--kv-dtype / --index-kv-dtype come from the launch command` | 粘贴了启动命令，又另外指定了 KV 精度 | 删掉 `--kv-dtype` / `--index-kv-dtype` |
| `exceeds the server's --context-length` | `--context` 或 `--prompt` 超过了服务端允许的最大长度 | 改小输入长度 |
| `below the lower bound` | 校准时实测值比理论下界还小 | 检查硬件规格、启动参数和测量方法 |
| `weights do not fit under ...` | 这个布局下每卡放不下权重 | 增大 TP（或者 PP，只适用于 `memory` 命令） |
| `... is not modelled` | 遇到了无法正确建模的配置 | 暂不支持，工具不会给出近似结果 |
| `PP > 1 are not modelled yet` | 对 PP > 1 的部署计算延迟和吞吐 | 目前只有 `memory` 命令支持 PP |
