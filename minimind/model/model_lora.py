"""
MiniMind LoRA (Low-Rank Adaptation) 实现
========================================
LoRA 的核心思想：冻结预训练权重 W，只训练一个低秩增量 ΔW = BA，
其中 B ∈ R^{d×r}, A ∈ R^{r×k}，rank r << min(d, k)。

前向传播: h = Wx + BAx = Wx + (低秩修正)
         ^^^^^^        ^^^^
         冻结不动      只训练这部分

为什么有效？（来自论文《LoRA: Low-Rank Adaptation of Large Language Models》）
  - 预训练权重 W 已经学到了通用的语言知识
  - 微调只需要在低秩子空间内调整
  - 假设: 微调时的权重更新 ΔW 是低秩的（有大量实验证据支持）

优势:
  - 参数效率：rank=16 时，LoRA 参数 ≈ 原模型的 0.1-1%
  - 多任务：每个下游任务只需保存一组 LoRA 权重（几 MB），共享同一个基座模型
  - 推理无额外延迟：训练后将 LoRA 合并到原权重 W' = W + BA，推理速度不变
  - 显存节省：不需要存储冻结参数对应的 optimizer states

MiniMind 实现特点:
  - 只对 in_features == out_features 的 Linear 层添加 LoRA（即 Attention 的 Q/K/V/O 投影）
  - 用 monkey-patch 替换 module.forward 而非包装 module
  - A 用高斯初始化（std=0.02），B 用全零初始化（保证 ΔW=0 起步）
  - 不支持 torch.compile（monkey-patch forward 与 JIT 编译冲突）
"""
import torch
from torch import optim, nn


class LoRA(nn.Module):
    """
    LoRA 低秩适配器。

    公式: ΔW = BA
      A ∈ R^{rank × in_features}：从输入压缩到低秩空间
      B ∈ R^{out_features × rank}：从低秩空间恢复到输出空间

    初始化:
      A ~ N(0, 0.02)：高斯随机初始化，打破对称性
      B = 0：确保初始时 ΔW = 0，模型行为与原始预训练模型完全相同
            如果 B 也随机初始化，ΔW ≠ 0 → 初始模型行为被随机扰动

    rank 的选择:
      8  → 激进压缩，适合简单任务
      16 → 常用默认值，大部分任务够用
      64 → 复杂任务（代码、数学等）
      MiniMind 默认 rank=16
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)   # 降维: d → r
        self.B = nn.Linear(rank, out_features, bias=False)   # 升维: r → d
        self.A.weight.data.normal_(mean=0.0, std=0.02)      # 高斯初始化 A
        self.B.weight.data.zero_()                           # 全零初始化 B

    def forward(self, x):
        return self.B(self.A(x))  # ΔWx = B(Ax)


def apply_lora(model, rank=16):
    """
    ★ 核心函数：给模型的所有 Attention 投影层添加 LoRA ★

    筛选条件: module.in_features == module.out_features
      → 只有 Q/K/V/O 投影矩阵满足（它们是方阵）
      → FFN 的 gate/up/down 投影不满足（intermediate_size ≠ hidden_size）
      → Embedding 层不加 LoRA（不是 Linear）
      → lm_head 也不加（与 embedding 权重绑定）

    Monkey-patch 方式:
      直接替换 module.forward 函数，在原始输出上加 LoRA 输出。
      优点：简单直接，不需要修改模型定义
      缺点：破坏了 torch.compile 兼容性，DDP 下需要额外处理

    ★ 闭包陷阱修复 ★
      `layer1=original_forward, layer2=lora` 通过默认参数绑定了当前迭代的值。
      如果不这样做，Python 闭包会捕获变量引用(不是值)，
      所有模块的 LoRA forward 都会指向最后一个模块的 original_forward 和 lora。
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 显式绑定：通过默认参数捕获当前循环的值
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    """加载 LoRA 权重到已应用 LoRA 的模型"""
    state_dict = torch.load(path, map_location=model.device)
    # 去掉 DDP 的 'module.' 前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v
                  for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {k.replace(f'{name}.lora.', ''): v
                         for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    只保存 LoRA 权重（不保存冻结的基座权重）。

    保存的 key 格式: {module_name}.lora.A.weight, {module_name}.lora.B.weight
    一个 rank=16 的 64M 模型，LoRA 文件约 2-5MB（vs 完整模型的 128MB）。
    """
    raw_model = getattr(model, '_orig_mod', model)  # 解开 torch.compile
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 去掉 DDP 的 'module.' 前缀
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half()
                         for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """
    ★ 将 LoRA 权重合并回原始模型 ★

    操作: W' = W + BA（对每个加了 LoRA 的 Linear 层）

    合并后的模型不需要 LoRA 推理开销，与完整训练的模型一样快。
    这是 LoRA 的关键优势：训练时参数高效，推理时零额外开销。

    典型流程:
      1. 训练 LoRA (只更新 A, B)
      2. merge_lora(model, 'lora.pth', 'merged.pth')
      3. 部署 merged.pth — 不需要 LoRA 代码
    """
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)

    # 先收集所有非 LoRA 权重
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items()
                  if '.lora.' not in k}

    # 对每个 Linear 层，将 LoRA 权重加到原始权重上
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # W' = W + BA
                state_dict[f'{name}.weight'] += (
                    module.lora.B.weight.data @ module.lora.A.weight.data
                ).cpu().half()

    torch.save(state_dict, save_path)
