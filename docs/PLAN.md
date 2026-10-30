# llm-roofline-bound 实施计划

## 定位

给定 **模型 × 硬件 × 部署参数 × 负载**，输出可证明的**理论下界**，以及（可选的）**估算值**：估算值等于下界除以用户提供的效率系数。

首个支持的组合是 GLM-5.3 × NVIDIA H20-3e。要支持其他组合，只需新增"模型结构组件"或"硬件规格文件"，不改核心代码。

## 两条设计原则

1. **能通过理论计算精确确定的值，都尽量算进来。**
2. **估算值和需要用户提供的值都明确标注，不给出容易误解的结果。**

### 确定性级别

每个输出都是一个 `Quantity`，同时携带数值、单位、级别、界的方向、区间和来源。

| 级别 | 含义 | 例子 |
|---|---|---|
| EXACT | 只做算术，输入是 config 和规格，不含任何假设 | 参数量、KV/token、FLOPs、通信字节数 |
| EXPECTED | 在明确的统计假设下取精确期望，同时给出上下界 | MoE 被触及的专家数、MTP 期望接受长度 |
| BOUND | 在已声明的假设下可证明成立的不等式，带方向 | `TTFT ≥ x`、`吞吐 ≤ y` |
| ASSUMED | 理论推不出、必须由用户提供的建模输入 | 引擎预留显存、激活值 workspace |
| ESTIMATE | 下界乘以用户提供的效率系数 | `TPOT ≈ 18 ms` |

以下规则由代码强制执行，不靠使用者自觉：

- 运算结果的级别等于输入中最弱的那个，无法人为调高。
- 界的方向按代数规则传递。方向无法确定的组合会直接报错，例如 `lower − lower`、`lower + upper`。
- ESTIMATE 不能带界的方向，因为它是一个猜测，不是界。
- JSON 输出里不允许出现裸数字。
- 显示的有效数字随级别递减：EXACT 保留完整精度，BOUND 保留 3 位，ESTIMATE 保留 2 位。
- 每个输入都记录来源和可信度（Trust）：
  - DEFINITION（定义常数）
  - PUBLIC_CONFIG（公开的模型配置）
  - OFFICIAL（厂商官方规格）
  - THIRD_PARTY（第三方来源）
  - USER（用户提供）
- ASSUMED 和 ESTIMATE 类输入**默认为空**。用户没有提供时，依赖它们的输出不计算，只提示需要哪个参数。
- 不支持的结构或参数一律报错，不做近似；传入但没有生效的参数会给出警告。

## 架构

```
ModelSpec + 部署参数 + 阶段(prefill chunk / decode batch / ctx / MTP)
        │  arch 组件(attention: MLA/DSA/GQA; ffn: dense/MoE; MTP)生成算子
        ▼
OpGraph = [Op, ...]      ← 核心中间表示(IR)，每个算子有 FLOPs、读写字节数、计算精度、形状
        ├─ memory  显存：权重、KV、KV 池、最大并发
        ├─ bounds  Σ max(flops/peak, bytes/BW) → TTFT_min / TPOT_min / 吞吐上限 / 最少卡数
        ├─ estimate 按算子类别分别设效率系数（由用户提供）
        ├─ report  文本 / JSON，按级别分区
        └─ plot    roofline 图，每个算子一个点
```

- **模型侧**：HF config 由 parser 按 `architectures` 字段分发，再拼装结构组件。新增一个模型 = 写一个 parser，必要时补结构组件。
- **硬件侧**：纯数据驱动（TOML），每个字段都要附来源；精度回退顺序为 fp4 → fp8 → bf16。
- **引擎侧**：`EngineProfile` 描述显存语义和权重吸收方式。第一版只提供通用默认值。

## 计算范围

**EXACT**
- 逐张量的参数量，和 safetensors header 逐张量对齐；fp8 block scale 的精确字节数
- 按 TP、PP、EP 切分到每张卡的分片，被复制到每张卡的张量单独列出；vocab 按 TP 对齐后的 padding
- KV/token，分项列出：主 KV、DSA indexer key、MTP 层 KV、KV scale；按分页 block 取整后的实际占用
- 拓扑合法性检查：`heads % TP`、`experts % EP`、PP 分层
- 逐算子的 FLOPs 和字节数；attention pairs 做离散精确求和（`S(S+1)/2`），分块 prefill 逐块累加
- 算术强度、各精度的 ridge point、临界 batch B*（decode 从访存受限转为计算受限的点）
- 通信字节数：TP all-reduce、EP all-to-all、PP 激活传递；PP 气泡比例

**EXPECTED**
- MoE 被触及的专家数：期望 `E(1-(1-k/E)^B)`，上下界 `[ceil(kB/E), min(E, kB)]`
- MTP 期望接受长度：`1 + Σ Π p_j`

**BOUND**
- TTFT_min、TPOT_min（含 MTP 和不含 MTP 两组），单副本吞吐上限，QPS 上限
- 给定 SLA 时可行的 batch、上下文、并发区间；所需最少卡数；成本下界（需要用户提供单价）

**不算 EXACT 的，必须由用户提供（ASSUMED）**：引擎预留显存、CUDA graph 占用、激活值 workspace、碎片、通信延迟项、kernel launch 开销。

**下界依赖的假设**，写入 `assumptions.md`：
- 使用峰值算力和峰值带宽
- 计算和访存完全重叠
- 算子融合、launch 开销、tile 量化损失都计为 0
- 不考虑排队

## 里程碑

| 阶段 | 内容 | 状态 |
|---|---|---|
| P0 | `Quantity`（级别、方向、区间、来源）；硬件规格及其校验；CLI 骨架；CI；合规扫描 | ✅ |
| P1 | GLM-5.3 config 解析；结构组件 MLA、DSA、MoE、MTP；显存 EXACT 项和界；safetensors 逐张量对齐 | ✅ |
| P2 | OpGraph（从 P1 移到这里）；decode、prefill、MTP 下界；临界 batch；精确 pairs；MoE 和 MTP 期望；`docs/assumptions.md` | ✅ |
| P3 | 通信字节数；容量和吞吐上限；SLA 可行区间；报告（中英文）和 JSON | |
| P4 | roofline HTML 图；`calibrate`（校准数据只存在用户本地） | |
| P5 | 文档（formulas、assumptions、sources）；README；发布前合规扫描 → 仓库改为公开 | |
| P6 | 第二个组合（如 GQA 稠密模型 × H100），用来验证扩展性 | |

## 测试

- EXACT 和 EXPECTED 的值做精确断言，不留容差。
- BOUND 测公式本身和性质（单调性、守恒性）。
- 级别和方向的传递规则做穷举测试。
- 报告和 JSON 做快照测试：每个数字都必须带级别标签。

## 合规

- 只使用公开数据，每个数值都附公开来源。不包含任何组织内部的测量值、地址、凭据或名称。
- `scripts/compliance_scan.py`：通用规则（私网 IP、凭据、密钥）写在仓库里；组织相关的规则放在本地被 git 忽略的 `.compliance-patterns.local` 中，不进仓库。
- 校准数据只保存在用户本地，不进仓库。
