# MiniMind 代码学习指南

> 使用方法：配合 `llm-from-scratch/notes/` 中的笔记一起阅读。
> 先读笔记理解原理，再读本目录下的代码消化实现细节。

---

## 一、学习方法论

### 1.1 核心原则：增量对比学习

以 llm-from-scratch 项目（Llama 风格、纯 PyTorch、从零手写）为锚点，
对 MiniMind 的每个模块问三个问题：

1. **MiniMind 做了什么不同的？**（对比差异）
2. **为什么这样设计？**（设计决策）
3. **这在工程中意味着什么？**（实用考量）

### 1.2 费曼学习法循环

每阶段严格按以下步骤进行，确保"真懂"而非"以为懂"：

```
1. 读代码    → 先看 MiniMind 源码，理解整体结构
2. 关掉复述  → 不看代码，用自己的话解释这段代码做了什么 + 为什么这么做
3. 对比      → 和 llm-from-scratch 的对应实现对比，讨论孰优孰劣
4. 产出笔记  → 写注释 + 补充笔记，把理解固化为文档
5. 审查      → 用独立视角审查注释和笔记的质量（代码本身不审查）：
   - 注释是否讲清楚了"为什么"
   - 前置知识是否完整
   - 公式推导是否正确
   - 概念解释是否准确
```

### 1.3 学习策略：自下而上 + 管道串行

- **先搞懂底层**（模型架构）→ 再按数据流走完整条管道
- **先看差异** → MiniMind 训练脚本共享约 90% 模板代码，只需关注每个脚本的差异部分
- **对比优先** → 与 llm-from-scratch 的对应实现逐项对比，理解两种风格的取舍

---

## 二、学习路径（4 阶段）

### 阶段 1：模型架构深入

**目标**：彻底理解 MiniMind 的 Transformer 实现，特别是与标准 Llama 的差异。

**核心文件**：

| 文件 | 说明 |
|------|------|
| `model/model_minimind.py` | 完整模型架构（~290 行），已补充中文注释 |
| `model/model_lora.py` | LoRA 低秩适配器（~65 行），已补充中文注释 |

**配套笔记**：

| 笔记 | 内容 |
|------|------|
| `llm-from-scratch/notes/03-model-architecture.md` §13 | MiniMind vs llm-from-scratch 架构全面对比 |
| `llm-from-scratch/notes/08b-moe.md` | MoE 原理、路由机制、负载均衡、显存估算 |
| `llm-from-scratch/notes/08-gqa.md` | GQA 原理（已有，作为前置知识） |
| `llm-from-scratch/notes/11-lora.md` | LoRA 原理、初始化策略、闭包陷阱、merge 机制 |

**重点关注的差异**：

| 特性 | llm-from-scratch | MiniMind |
|------|-----------------|----------|
| 框架集成 | 纯 `nn.Module` | 继承 `PreTrainedModel`, `GenerationMixin` |
| QK 归一化 | 无 | `RMSNorm` 作用于 query/key |
| 中间层维度 | `8/3 * d_model` | `π * d_model` |
| MoE | 不支持 | Top-K 路由 + 负载均衡辅助 loss |
| RoPE 扩展 | 仅基础 RoPE | 支持 YaRN 动态缩放 |

---

### 阶段 2：训练管道串联

**关键发现**：所有训练脚本共享约 90% 的样板代码。唯一不同的是 **数据格式 + loss 函数**。

统一模板：

```
1. 初始化环境和随机种子
2. 配置目录、模型参数、检查 ckp
3. 设置混合精度
4. 配置 swanlab (国产 wandb 替代)
5. 定义模型、数据、优化器
6. 从 ckp 恢复状态（断点续训）
7. 编译和分布式包装
8. 开始训练（for epoch → for step）
9. 清理分布式进程
```

**学习顺序（按依赖关系）**：

#### 2a. 分词器训练

| 文件 | 说明 |
|------|------|
| `trainer/train_tokenizer.py` | BPE 分词器训练脚本，已补充中文注释 |

**配套笔记**：`llm-from-scratch/notes/01-data-pipeline.md` §12

**重点**：vocab=6400 的设计考量、BPE vs SentencePiece vs Tiktoken、ByteLevel 预分词器、特殊 token 设计、chat_template 作用

#### 2b. 预训练

| 文件 | 说明 |
|------|------|
| `trainer/train_pretrain.py` | 预训练脚本，已补充中文注释 |
| `dataset/lm_dataset.py` | `PretrainDataset` 类 |

**配套笔记**：`llm-from-scratch/notes/04-trainer.md` §16

**与 llm-from-scratch trainer.py 的核心差异**：DDP 支持、`torch.cuda.amp` API、断点续训（完整恢复 optimizer + scaler + step）、swanlab 替代 wandb、梯度累积

#### 2c. 全参数 SFT

| 文件 | 说明 |
|------|------|
| `trainer/train_full_sft.py` | SFT 训练脚本，已补充中文注释 |
| `dataset/lm_dataset.py` | `SFTDataset` 类 |

**配套笔记**：`llm-from-scratch/notes/10-alignment.md` §11

**重点**：chat_template 如何工作、为什么只训练 assistant 回复、label mask 策略、数据增强（随机 system prompt、空 think 移除）

#### 2d. LoRA 微调

| 文件 | 说明 |
|------|------|
| `model/model_lora.py` | LoRA 实现，已补充中文注释 |
| `trainer/train_lora.py` | LoRA 训练脚本 |

**配套笔记**：`llm-from-scratch/notes/11-lora.md`

**重点**：低秩分解原理、A 高斯/B 全零初始化、monkey-patch 闭包陷阱、只对 Q/K/V/O 投影加 LoRA、merge 到原权重、与 PEFT 库对比

#### 2e. DPO

| 文件 | 说明 |
|------|------|
| `trainer/train_dpo.py` | DPO 训练脚本，已补充中文注释 |
| `dataset/lm_dataset.py` | `DPODataset` 类 |

**配套笔记**：`llm-from-scratch/notes/10-alignment.md` §12

**重点**：Bradley-Terry 偏好模型、隐式奖励公式（含 Z(x) 配分函数）、参考模型锚定作用、beta 参数调优、DPO 学习率为什么极小（4e-8）

#### 2f. PPO / GRPO（RLAIF）

| 文件 | 说明 |
|------|------|
| `trainer/train_grpo.py` | GRPO 训练脚本，已补充中文注释 |
| `trainer/rollout_engine.py` | Rollout 引擎（Torch + SGLang 两种后端） |

**配套笔记**：`llm-from-scratch/notes/12-rlhf-rlaif.md`

**重点**：组内对比替代 Critic、PPO clipped loss、k3 KL 估计器、Reward Model + 规则奖励混合、Rollout Engine 可插拔设计、CISPO vs GRPO、显存瓶颈分析

---

### 阶段 3：辅助基础设施

| 文件 | 说明 |
|------|------|
| `dataset/lm_dataset.py` | 5 种 Dataset 实现（Pretrain/SFT/DPO/RLAIF/AgentRL），已补充中文注释 |
| `trainer/trainer_utils.py` | 共享工具：DDP 初始化、checkpoint、学习率调度、SkipBatchSampler |
| `scripts/convert_model.py` | 模型格式转换（torch ↔ transformers ↔ llama.cpp/Ollama） |

**配套笔记**：`llm-from-scratch/notes/09-distributed-training.md`

**重点**：DDP 工作原理、all-reduce 梯度同步、DistributedSampler、断点续训的 GPU 数量适配、FSDP/ZeRO 概念、3D 并行概念

---

### 阶段 4：横向对比总结（最重要！）

**配套笔记**：`llm-from-scratch/notes/14-pipeline-comparison.md`

四大管道全景图：数据流、loss 函数、学习率规律、显存演进、管道选择决策树、检验清单。

完成所有模块后，理解四条管道的全局关系：

| 阶段 | 数据格式 | 计算 loss 的位置 | 学习率 | 额外模型 | 目的 |
|------|---------|-----------------|--------|---------|------|
| Pretrain | 纯文本 | 全部 token | 5e-4 | 无 | 从零学习语言 |
| SFT | 多轮对话 | 仅 assistant token | 1e-5 | 无 | 学习对话格式 |
| DPO | chosen/rejected 对 | 全部 assistant token (DPO loss) | 4e-8 | ref_model | 偏好对齐 |
| GRPO | 单轮 prompt | PPO clipped loss | 3e-7 | ref_model + RM + rollout | 在线探索优化 |

**后训练学习率规律**：每深入一个阶段，lr 约降一个数量级。5e-4 → 1e-5 → 4e-8。GRPO 虽为 3e-7，仍在微小 lr 范围内，因为 KL 惩罚和 PPO clip 提供了额外约束。

**灾难性遗忘**是所有后训练阶段的核心风险——详细解释见 `llm-from-scratch/notes/10-alignment.md` §11。

---

## 三、笔记文件索引

### llm-from-scratch/notes/ 中的笔记

| 编号 | 文件 | 主题 | MiniMind 相关内容 |
|:---:|------|------|------|
| 01 | `01-data-pipeline.md` | 数据管道、分词器 | §12 MiniMind Tokenizer 对比 |
| 02 | `02-config-system.md` | 配置系统 | — |
| 03 | `03-model-architecture.md` | 模型架构 | §13 MiniMind 架构对比（QK Norm, π维度, YaRN, 框架集成） |
| 04 | `04-trainer.md` | 训练引擎 | §16 MiniMind 训练实践（DDP, amp API, 断点续训, 梯度累积） |
| 05 | `05-inference.md` | 推理/KV-Cache | — |
| 06 | `06-main-entry.md` | CLI 入口 | — |
| 07 | `07-experiments.md` | 实验管理 | — |
| 08 | `08-gqa.md` | GQA 原理 | — |
| 08b | `08b-moe.md` | **←新建** | MoE 原理与实践 |
| 09 | `09-distributed-training.md` | **←新建** | DDP, FSDP, 分布式训练概念 |
| 10 | `10-alignment.md` | 对齐技术 | §11 SFT 训练实践, §12 DPO 训练实践, 灾难性遗忘解释 |
| 11 | `11-lora.md` | **←新建** | LoRA 原理与实现 |
| 12 | `12-rlhf-rlaif.md` | **←新建** | PPO/GRPO 原理与实践 |
| 14 | `14-pipeline-comparison.md` | **←新建** | 四条管道横向对比总结（收官） |

### minimind/ 中已注释的代码文件

| 文件 | 行数 | 注释重点 |
|------|:---:|------|
| `model/model_minimind.py` | ~290 | 完整模型架构、QK Norm、YaRN、MoE 路由、Qwen3 约定 |
| `model/model_lora.py` | ~65 | LoRA 低秩分解、初始化策略、闭包陷阱、merge |
| `trainer/train_tokenizer.py` | ~170 | BPE 训练、vocab=6400 设计、chat_template |
| `trainer/train_pretrain.py` | ~250 | 训练循环模板、DDP、混合精度、梯度累积、断点续训 |
| `trainer/train_full_sft.py` | ~210 | SFT 特有：label mask、学习率、数据增强 |
| `trainer/train_dpo.py` | ~350 | DPO 数学推导、隐式奖励、ref_model 锚定 |
| `trainer/train_grpo.py` | ~340 | GRPO 组内对比、PPO clip、k3 KL、Rollout Engine |
| `dataset/lm_dataset.py` | ~310 | 5 种 Dataset（Pretrain/SFT/DPO/RLAIF/AgentRL） |

---

## 四、面试重点速查

以下主题在本项目的笔记和注释中均有覆盖，按面试出现频率排序：

| 主题 | 对应笔记 | 对应代码 |
|------|---------|---------|
| 手写 Attention（含 causal mask） | 03 | model_minimind.py → `Attention.forward()` |
| KV-Cache 原理与显存计算 | 05, 08 | model_minimind.py → `past_key_value` |
| RoPE 旋转位置编码 | 03 | model_minimind.py → `apply_rotary_pos_emb()` |
| GQA vs MHA vs MQA | 08 | model_minimind.py → `repeat_kv()` |
| RMSNorm vs LayerNorm | 03 | model_minimind.py → `RMSNorm` |
| SwiGLU 激活函数 | 03 | model_minimind.py → `FeedForward` |
| Pre-Norm vs Post-Norm | 03 | model_minimind.py → `MiniMindBlock` |
| 混合精度训练 (AMP) | 04 §16 | train_pretrain.py → autocast + scaler |
| DDP 分布式训练 | 09 | trainer_utils.py → `init_distributed_mode()` |
| 梯度累积 | 04 §16 | train_pretrain.py → `accumulation_steps` |
| 断点续训 | 04 §16 | trainer_utils.py → `lm_checkpoint()` |
| MoE 路由与负载均衡 | 08b | model_minimind.py → `MOEFeedForward` |
| YaRN 位置编码扩展 | 03 §13 | model_minimind.py → `precompute_freqs_cis()` |
| LoRA 原理与实现 | 11 | model_lora.py → `LoRA`, `apply_lora()` |
| SFT label mask 策略 | 10 §11 | lm_dataset.py → `SFTDataset.generate_labels()` |
| DPO vs RLHF vs GRPO | 10 §12, 12 | train_dpo.py, train_grpo.py |
| PPO clipped loss | 12 | train_grpo.py → `per_token_loss` |
| 灾难性遗忘 | 10 §11 | — |

---

## 五、关键概念速查

### 5.1 参数量估算 (MiniMind Dense 64M)

```
Embedding (6400×768):                        4.92M（与 lm_head 权重绑定）
Attention ×8 层 (GQA, kv_heads=4):          14.16M（QKV 有 GQA 节省）
FFN ×8 层 (SwiGLU, intermediate=2432):      44.83M（gate/up/down 三个投影）
RMSNorm 等微小参数:                          ~14K（可忽略）
─────────────────────────────────────────────────
总计:                                        ~64M
```

### 5.2 SwiGLU 参数量纠正

常见误区："SwiGLU 比标准 FFN 参数多"。正解：

- 标准 FFN(4x)：`d×4d + 4d×d = 8d²`（2 个矩阵）
- SwiGLU(8/3x)：`3 × d×(8/3)d = 8d²`（3 个矩阵）
- **参数量相等**。SwiGLU 的优势在门控机制，而非更多参数。

### 5.3 AdamW 训练显存公式

```
训练显存 ≈ 参数量 × 12 bytes
= 权重(FP16, 2B) + 梯度(FP16, 2B) + momentum(FP32, 4B) + variance(FP32, 4B)
```

不含激活值（通常额外占 30-50%）。64M 模型训练约需 770MB + 激活值。

### 5.4 后训练学习率规律

```
Pretrain: 5e-4 —— 从零学语言，需要大步更新
SFT:      1e-5 —— 只调整对话格式，~50x 差异
DPO:      4e-8 —— 只做偏好微调，~250x 差异
GRPO:     3e-7 —— 在线探索，有 KL 惩罚兜底
```

---

## 六、如何开始

1. **先通读本指南**，建立全局心智模型
2. **按阶段 1 → 2 → 3 → 4 顺序学习**，不要跳步
3. **每阶段先读笔记**（llm-from-scratch/notes/），**再读代码**（minimind/）
4. **关掉代码，自己复述**——这是最关键的一步
5. **阶段 4 的横向对比表**是最终检验：能不看资料完整画出四条管道的差异图，说明真正掌握了
