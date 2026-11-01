"""Report text in other languages.

Reports are built in English, and JSON output stays English so that field
names are stable. The CLI translates text at render time. A string is split on
"; " and each piece is matched against exact entries, then against patterns.
Pattern groups named ``t_*`` are translated recursively, and a trailing
quantity such as ``≥ 1.84 s [BOUND]`` is kept verbatim. Anything without a
translation is left in English and recorded in ``UNTRANSLATED``, which the
tests check stays empty.
"""

from __future__ import annotations

import re

__all__ = ["LANGS", "t", "UNTRANSLATED"]

LANGS = ("en", "zh")
UNTRANSLATED: set[str] = set()

_ZH_EXACT = {
    # section titles and headings (before .format)
    "Model  (EXACT: arithmetic on the published config)": "模型  （EXACT：对公开 config 的精确算术）",
    "Per GPU  ({layout})": "每卡  （{layout}）",
    "Bounds  (no assumptions)": "界  （不含假设）",
    "Under your assumptions": "基于你提供的假设",
    "Decode at context {context}  (bounds; their assumptions are under Notes)":
        "Decode，上下文 {context}  （界；所依赖的假设见“说明”）",
    "MTP decode  (tokens per step and TPOT rest on your acceptance rates)":
        "MTP decode  （每步 token 数和 TPOT 依赖你给的接受率）",
    "Prefill  (bounds; their assumptions are under Notes)": "Prefill  （界；所依赖的假设见“说明”）",
    "Estimates  (ESTIMATE: bound divided by your measured efficiency)":
        "估算  （ESTIMATE：下界除以你实测得到的效率）",
    "Sustained capacity, SLA and fleet  (any schedule; assumptions under Notes)":
        "持续容量、SLA 与集群规模  （任意调度下成立；假设见“说明”）",
    "Not computed": "未计算",
    "Notes": "说明",
    "Inputs": "输入",
    "on": "运行于",
    # memory
    "parameters (main model)": "参数量（主模型）",
    "routed_expert": "路由专家", "attn": "注意力", "shared_expert": "共享专家",
    "embed": "词嵌入", "lm_head": "lm_head", "dense_ffn": "稠密 FFN", "indexer": "DSA indexer",
    "router": "路由器", "norm": "归一化", "MTP layers": "MTP 层",
    "moe_latent": "MoE 隐空间投影", "vision": "视觉编码器",
    "linear-attention state per request per GPU (largest stage)": "每卡每请求的线性注意力状态（最大的 stage）",
    "fixed per sequence, whatever its length": "每个序列固定大小，与长度无关",
    "per-sequence state not subtracted": "未扣除每序列状态",
    "KV and per-sequence state": "KV 与每序列状态",
    "of which replicated on every rank": "其中每卡都复制的部分",
    "does not shrink as TP grows": "不随 TP 增大而减少",
    "parameters (MTP layers)": "参数量（MTP 层）",
    "checkpoint size": "checkpoint 大小",
    "active parameters per token": "每 token 激活参数量",
    "weights per GPU (largest stage)": "每卡权重（最大的 stage）",
    "KV cache per token per GPU (largest stage)": "每卡每 token 的 KV cache（最大的 stage）",
    "GPUs to hold the weights (any layout)": "放下权重所需卡数（任意布局）",
    "ceil(weights / capacity per GPU)": "ceil(权重 / 每卡容量)",
    "GPUs for weights + 1 request (TP x PP layouts)": "放下权重和 1 个请求所需卡数（TP×PP 布局）",
    "KV pool per GPU": "每卡 KV 池",
    "capacity - weights": "容量 - 权重",
    "runtime memory not subtracted": "未扣除运行时显存",
    "tokens of KV per replica": "每副本可容纳的 KV token 数",
    "pool / KV per token": "KV 池 / 每 token KV",
    "KV memory only": "只考虑 KV 显存",
    "says nothing about latency": "不代表延迟可接受",
    "capacity - weights - reserve": "容量 - 权重 - 预留",
    "--reserve-gb (runtime memory per GPU besides weights and KV): needed for the KV pool, "
    "token and concurrency figures under that assumption":
        "--reserve-gb（每卡除权重和 KV 之外的运行时显存）：计算该假设下的 KV 池、token 数和并发数需要它",
    # bounds
    "tokens/s per sequence, batch 1": "单序列 tokens/s，batch 1",
    "decode tokens/s, max over batches": "decode tokens/s，所有 batch 中的最大值",
    "batch where decode turns compute-bound": "decode 转为算力受限的 batch",
    "uniform routing": "均匀路由",
    "range spans the most and least concentrated routing": "区间覆盖路由最集中到最分散的情况",
    "prefill tokens/s, this request": "prefill tokens/s，本请求",
    "tokens per step": "每步 token 数",
    "1 + Σ Π acceptance rates": "1 + Σ Π 接受率",
    "step bound / tokens per step": "每步下界 / 每步 token 数",
    "memory-bound": "受访存限制", "compute-bound": "受算力限制",
    "interconnect-bound": "受互联限制",
    "HBM-bound": "受 HBM 限制",
    "MLA absorbed": "MLA 吸收形式", "MLA expanded": "MLA 展开形式",
    "--mtp-accept (acceptance rate per draft token): needed for TPOT with MTP":
        "--mtp-accept（每个草稿 token 的接受率）：计算带 MTP 的 TPOT 需要它",
    "bounds assume datasheet peak rates, perfect overlap of compute, HBM and interconnect, and "
    "count only compulsory HBM traffic (weights, KV)":
        "界的前提：使用标称峰值；算力、HBM、互联完全重叠；只计必然发生的 HBM 流量（权重、KV）",
    "each operator computes in its storage dtype (docs/assumptions.md)":
        "每个算子按存储精度计算（见 docs/assumptions.md）",
    "every operator at the fastest tensor rate (--strict) (docs/assumptions.md)":
        "所有算子按最快的 tensor 算力计算（--strict）（见 docs/assumptions.md）",
    "maximum decode throughput: the weights leave no KV room under this layout":
        "decode 最大吞吐：此布局下权重放下后没有 KV 空间",
    "maximum decode throughput: not one sequence fits in KV memory":
        "decode 最大吞吐：KV 显存连一个序列都放不下",
    # capacity
    "output tokens/s per replica, sustained": "每副本持续输出 tokens/s",
    "prefill and decode together, any schedule": "prefill 与 decode 合计，任意调度",
    "compute alone at the fastest tensor rate": "只考虑算力，按最快的 tensor 算力",
    "cost per million output tokens": "每百万输出 token 成本",
    "every larger batch misses it": "更大的 batch 都达不到",
    "--prompt and --output (tokens in and out per request): needed for sustained capacity, "
    "SLA batch, fleet size and cost":
        "--prompt 和 --output（每个请求的输入、输出 token 数）：计算持续容量、SLA batch、集群规模和成本需要它们",
    "sustained capacity, SLA and fleet figures with MTP are not modelled yet":
        "带 MTP 的持续容量、SLA 和集群规模还没有建模",
    "run without --mtp-draft": "请去掉 --mtp-draft 运行",
    "sustained capacity: the weights leave no KV room": "持续容量：权重放下后没有 KV 空间",
    "sustained capacity: one request does not fit in KV memory": "持续容量：KV 显存放不下一个请求",
    "queueing can still break it": "排队仍可能使其无法满足",
    # launch
    "CUDA graphs: launch overhead is not modelled": "CUDA graph：launch 开销未建模",
    "hierarchical (host-memory) prefix cache: prefix hits shorten TTFT": "分层（主机内存）prefix 缓存：命中时 TTFT 会变短",
    "bounds assume no hit": "界按未命中计算",
    "prefix (radix) caching is not modelled either way": "prefix（radix）缓存未建模",
    "scheduling is not modelled": "调度未建模",
    "kernel choices are not modelled": "kernel 选择未建模",
    "bounds hold for any kernel": "界对任意 kernel 都成立",
    "figures are per replica": "结果均为每副本数值",
    "no --mem-fraction-static: SGLang picks one at startup, so the runtime reserve is unknown":
        "未指定 --mem-fraction-static：SGLang 会在启动时自动选择，因此运行时预留未知",
}

_N = r"[\d,.]+(?:e[+-]?\d+)?"
_ZH_PATTERNS: list[tuple[str, str]] = [
    (rf"KV cache per request at (?P<n>{_N}) tokens", "单请求 KV cache（{n} token）"),
    (rf"concurrent requests at (?P<n>{_N}) tokens", "{n} token 上下文下的并发请求数"),
    (rf"KV cache and state per request at (?P<n>{_N}) tokens", "单请求 KV cache 与状态（{n} token）"),
    (r"linear-attention state: recurrent state at (?P<a>\S+), conv window at (?P<b>\S+) \(SGLang defaults unless --mamba-ssm-dtype is given\)",
     "线性注意力状态：递推状态为 {a}，卷积窗口为 {b}（SGLang 默认值，可用 --mamba-ssm-dtype 修改）"),
    (r"tensor bytes per rank under (?P<l>.+)", "{l} 下每个 rank 的张量字节数"),
    (r"layout (?P<l>.+)", "KV 布局：{l}"),
    (rf"ceil\((?P<a>{_N})/(?P<b>{_N})\) x (?P<c>{_N}) tokens x KV per token",
     "ceil({a}/{b}) × {c} token × 每 token KV"),
    (r"smallest feasible: (?P<l>.+)", "最小可行布局：{l}"),
    (r"pool / KV per token \(stage (?P<s>\d+) binds\)", "KV 池 / 每 token KV（stage {s} 是瓶颈）"),
    (r"vocab padded by (?P<n>\d+) rows to split over TP=(?P<tp>\d+)",
     "词表补齐了 {n} 行，以便按 TP={tp} 切分"),
    (rf"weights do not fit under (?P<l>.+?): (?P<w>{_N}) GB per GPU vs (?P<c>{_N}) GB capacity",
     "{l} 下权重放不下：每卡 {w} GB，容量 {c} GB"),
    (r"KV pool and concurrency not computed", "未计算 KV 池和并发数"),
    (r"no TP x PP layout up to 512 GPUs fits the weights", "512 卡以内没有能放下权重的 TP×PP 布局"),
    (r"weights plus reserve exceed GPU memory on stage (?P<s>\d+): nothing left for KV",
     "stage {s} 上权重加预留超过了显存：没有 KV 空间"),
    (r"(?P<hw>.+): some spec values are not vendor-published",
     "{hw}：部分规格不是厂商公开的数值"),
    (r"bounds are conditional on them \(rooflinebound hw (?P<n>.+)\)",
     "界以这些数值为前提（rooflinebound hw {n}）"),
    (r"TPOT, batch (?P<b>\d+)", "TPOT，batch {b}"),
    (r"decode tokens/s, batch (?P<b>\d+)", "decode tokens/s，batch {b}"),
    (rf"compute (?P<c>{_N}) / HBM (?P<m>{_N}) / interconnect (?P<x>{_N}) ms",
     "算力 {c} / HBM {m} / 互联 {x} ms"),
    (r"(?P<n>\d+) passes", "共 {n} 次前向"),
    (r"at batch (?P<b>\d+), the most KV memory admits at this context",
     "在 batch {b}，即此上下文下 KV 显存能容纳的最大值"),
    (r"at batch (?P<b>\d+), the engine's max running requests \(KV memory admits (?P<k>\d+)\)",
     "在 batch {b}，即引擎的 max running requests（KV 显存可容纳 {k}）"),
    (r"verify \+ (?P<k>\d+) draft steps, batch (?P<b>\d+)", "验证 + {k} 步草稿，batch {b}"),
    (r"TPOT with MTP, batch (?P<b>\d+)", "带 MTP 的 TPOT，batch {b}"),
    (rf"TTFT, (?P<n>{_N})-token prompt", "TTFT，{n} token 的 prompt"),
    (r"batch (?P<b>\d+) exceeds the (?P<k>\d+) sequences KV memory admits at context (?P<c>[\d,]+)",
     "batch {b} 超过了上下文 {c} 时 KV 显存能容纳的 {k} 个序列"),
    (r"batch (?P<b>\d+) exceeds the engine's max running requests \((?P<k>\d+)\)",
     "batch {b} 超过了引擎的 max running requests（{k}）"),
    (r"decode stays HBM-bound at every batch at context (?P<c>[\d,]+): KV reads per sequence "
     r"outpace its compute", "上下文 {c} 时 decode 在任何 batch 下都受 HBM 限制：每个序列读 KV 的时间超过其计算时间"),
    (r"if an implementation computed every operator at the fastest tensor rate \(--strict\): "
     r"(?P<t_x>.+)", "若实现按最快的 tensor 算力计算所有算子（--strict）：{t_x}"),
    (r"batch meeting TPOT ≤ (?P<v>\S+) ms", "满足 TPOT ≤ {v} ms 的 batch"),
    (r"at context (?P<c>[\d,]+)", "上下文 {c}"),
    (r"even batch 1 misses it at context (?P<c>[\d,]+): infeasible",
     "上下文 {c} 时 batch 1 也达不到：不可行"),
    (r"requests/s per replica \((?P<i>[\d,]+) in, (?P<o>[\d,]+) out\)",
     "每副本 requests/s（输入 {i}，输出 {o}）"),
    (r"decode weight reads shared by ≤ (?P<b>\d+) sequences \((?P<t_why>.+)\)",
     "decode 权重读取最多由 {b} 个序列分摊（{t_why}）"),
    (r"KV memory at the prompt length", "prompt 长度下的 KV 显存"),
    (r"the engine's max running requests", "引擎的 max running requests"),
    (r"the TPOT target of (?P<v>\S+) ms", "TPOT 目标 {v} ms"),
    (r"replicas for (?P<q>\S+) requests/s", "达到 {q} requests/s 所需副本数"),
    (r"(?P<l>TP\d+ PP\d+ EP\d+) each", "每副本 {l}"),
    (r"GPUs for (?P<q>\S+) requests/s, this layout", "达到 {q} requests/s 所需卡数（此布局）"),
    (r"GPUs for (?P<q>\S+) requests/s, any layout", "达到 {q} requests/s 所需卡数（任意布局）"),
    (r"(?P<n>\d+) GPUs per replica at the sustained bound", "每副本 {n} 卡，按持续吞吐上界计算"),
    (r"SLA: TTFT ≤ (?P<v>\S+) ms cannot be met for a (?P<n>[\d,]+)-token prompt: TTFT "
     r"(?P<q>.+)", "SLA：{n} token 的 prompt 无法满足 TTFT ≤ {v} ms：TTFT {q}"),
    (r"SLA: TTFT ≤ (?P<v>\S+) ms is not ruled out by the bound \((?P<q>.+)\)",
     "SLA：TTFT ≤ {v} ms 未被下界排除（{q}）"),
    (r"SLA: TPOT ≤ (?P<v>\S+) ms does not limit the batch \(met up to (?P<n>[\d,]+) sequences at "
     r"context (?P<c>[\d,]+)\)", "SLA：TPOT ≤ {v} ms 不限制 batch（上下文 {c} 时直到 {n} 个序列都满足）"),
    (r"bound ÷ efficiency (?P<e>[\d.]+), measured at (?P<t_c>.+)", "下界 ÷ 效率 {e}，测于 {t_c}"),
    (r"batch (?P<b>\d+), context (?P<c>[\d,]+)", "batch {b}，上下文 {c}"),
    (r"(?P<p>[\d,]+)-token prompt, chunk (?P<c>[\d,]+)", "{p} token 的 prompt，chunk {c}"),
    (r"(?P<p>[\d,]+)-token prompt", "{p} token 的 prompt"),
    (r"applied at (?P<t_c>.+), where efficiency differs", "用于 {t_c}，效率在不同条件下会变化"),
    (r"(?P<k>decode|prefill) estimates: no calibration for this deployment",
     "{k} 估算：没有这个部署的校准数据"),
    (r"run rooflinebound calibrate \.\.\. (?P<f>\S+) \.\.\. --save",
     "请运行 rooflinebound calibrate ... {f} ... --save"),
    (r"launch: (?P<t_x>.+)", "启动命令：{t_x}"),
    (r"launch flags with no effect on these figures: (?P<f>.+)", "对这些结果没有影响的启动参数：{f}"),
    (r"launch flags not recognised \(check they do not change memory or speed\): (?P<f>.+)",
     "无法识别的启动参数（请确认它们不影响显存或速度）：{f}"),
    (r"--mem-fraction-static (?P<f>\S+): runtime reserve taken as \(1 - (?P=f)\) x device memory "
     r"= (?P<g>\S+) GiB \(ASSUMED: SGLang uses free memory before model load, slightly less\)",
     "--mem-fraction-static {f}：运行时预留按 (1 - {f}) × 设备显存 = {g} GiB 计算（ASSUMED：SGLang "
     "实际按加载模型前的空闲显存计算，会略少）"),
    (r"--dp (?P<n>\d+): (?P=n) independent replicas", "--dp {n}：{n} 个独立副本"),
    (r"(?P<f>--[\w-]+): (?P<t_why>.+)", "{f}：{t_why}"),
    # a label followed by a formatted quantity, e.g. inside the --strict note
    (r"(?P<t_label>.+?) (?P<q>(?:≈ )?[≥≤] \S+(?: \S+)? \[[A-Z]+[^\]]*\])", "{t_label} {q}"),
]

_COMPILED = [(re.compile(p), z) for p, z in _ZH_PATTERNS]


def _piece(s: str) -> str:
    lead = s[: len(s) - len(s.lstrip())]
    body = s.strip()
    if not body:
        return s
    if body in _ZH_EXACT:
        return lead + _ZH_EXACT[body]
    for rx, tpl in _COMPILED:
        m = rx.fullmatch(body)
        if m:
            vals = {k: (t(v, "zh") if k.startswith("t_") else v)
                    for k, v in m.groupdict().items()}
            return lead + tpl.format(**vals)
    UNTRANSLATED.add(body)
    return s


def t(s: str | None, lang: str = "en") -> str | None:
    """Translate one report string. English (and None) pass through unchanged."""
    if s is None or lang == "en":
        return s
    if lang != "zh":
        raise ValueError(f"unsupported language {lang!r}; choose from {LANGS}")
    if s.strip() in _ZH_EXACT:  # whole strings first: some contain "; " themselves
        return _piece(s)
    return "；".join(_piece(p) for p in s.split("; "))
