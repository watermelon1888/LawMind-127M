"""
MiniMind 核心模型架构
=====================
架构对齐 Qwen3，支持 Dense 和 MoE 两种模式。

与 llm-from-scratch 项目 (Llama 风格) 的关键差异:
  1. 框架集成: 继承 PreTrainedModel/GenerationMixin，可直接用 transformers 生态
  2. QK 归一化: Attention 中对 query 和 key 做 RMSNorm（Qwen3 特性，提升训练稳定性）
  3. 中间层维度: 用 π * d_model 而非 8/3 * d_model（Qwen3 约定）
  4. MoE: 支持 Top-k 路由 + 负载均衡辅助 loss
  5. YaRN: 支持动态 RoPE 缩放，扩展上下文窗口
  6. 位置编码: 用 register_buffer 预计算 cos/sin 表，而非类内缓存

参数量估算 (Dense, hidden_size=768, num_hidden_layers=16, GQA kv_heads=4):
  - 每层 Self-Attention: Q768² + KV各768×384 + O768² ≈ 1.77M (Q/K/V/O 无 bias, GQA 节省 KV 参数量)
  - 每层 FFN (SwiGLU): 3 * 768 * 2432 ≈ 5.60M (gate/up/down, intermediate_size 向上取整至 64 倍数)
  - Embedding + lm_head: 12000 * 768 ≈ 9.22M (权重绑定，embed 与 lm_head 共享，只计一次)
  - 总计约 127M 参数 (16层 attention 28.3M + 16层 FFN 89.7M + embedding 9.2M；
    RMSNorm 等微小参数未计入，在大局上可忽略)

更多架构细节见笔记:
  - notes/03-model-architecture.md §13 架构对比
  - notes/08b-moe.md MoE 原理与实现详解
  - notes/08-gqa.md GQA 原理
"""
import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


def _has_repeated_tail(token_ids, *, repeat_count=3, min_ngram_size=2, max_ngram_size=64):
    """判断序列尾部是否连续重复同一片段。"""
    values = token_ids.tolist() if torch.is_tensor(token_ids) else list(token_ids)
    if repeat_count < 2 or len(values) < min_ngram_size * repeat_count:
        return False
    upper = min(max_ngram_size, len(values) // repeat_count)
    for ngram_size in range(upper, min_ngram_size - 1, -1):
        suffix = values[-ngram_size:]
        repeats = 1
        cursor = len(values) - ngram_size
        while cursor >= ngram_size and values[cursor - ngram_size:cursor] == suffix:
            repeats += 1
            cursor -= ngram_size
        if repeats >= repeat_count:
            return True
    return False


def _apply_generated_repetition_penalties(
    logits,
    generated_token_ids,
    *,
    frequency_penalty=0.0,
    repetition_path_penalty=0.0,
    repetition_path_min_ngram_size=4,
    repetition_path_max_ngram_size=12,
):
    """只根据已生成 token 对高频 token 和重复路径进行软降权。"""
    if frequency_penalty < 0 or repetition_path_penalty < 0:
        raise ValueError("重复惩罚必须是非负数")
    if repetition_path_min_ngram_size < 2:
        raise ValueError("重复路径最小 n-gram 必须至少为 2")
    if repetition_path_max_ngram_size < repetition_path_min_ngram_size:
        raise ValueError("重复路径最大 n-gram 不能小于最小 n-gram")
    if generated_token_ids.ndim != 2 or logits.ndim != 2:
        raise ValueError("logits 和 generated_token_ids 必须是二维张量")
    if logits.shape[0] != generated_token_ids.shape[0]:
        raise ValueError("logits 与 generated_token_ids 的 batch 大小必须一致")

    for batch_index in range(logits.shape[0]):
        sequence = generated_token_ids[batch_index]
        if frequency_penalty and sequence.numel():
            counts = torch.bincount(sequence, minlength=logits.shape[-1])
            repeated = torch.nonzero(counts > 1, as_tuple=False).flatten()
            if repeated.numel():
                logits[batch_index, repeated] -= (
                    counts[repeated].to(logits.dtype) - 1
                ) * frequency_penalty

        if not repetition_path_penalty or sequence.numel() < 2:
            continue
        values = sequence.tolist()
        upper = min(repetition_path_max_ngram_size, len(values) + 1)
        penalized = set()
        for ngram_size in range(repetition_path_min_ngram_size, upper + 1):
            prefix = tuple(values[-(ngram_size - 1):])
            for start in range(len(values) - ngram_size + 1):
                if tuple(values[start:start + ngram_size - 1]) == prefix:
                    penalized.add(values[start + ngram_size - 1])
        if penalized:
            token_ids = torch.tensor(
                tuple(penalized), device=logits.device, dtype=torch.long
            )
            logits[batch_index, token_ids] -= repetition_path_penalty
    return logits


def _contrastive_search_scores(
    candidate_probabilities,
    candidate_hidden_states,
    generated_hidden_states,
    *,
    penalty_alpha,
):
    """按标准 Contrastive Search 公式计算候选分数。"""
    if not 0 < penalty_alpha < 1:
        raise ValueError("penalty_alpha 必须位于 0 和 1 之间")
    if candidate_probabilities.ndim != 2 or candidate_hidden_states.ndim != 3:
        raise ValueError("候选概率和候选隐藏状态维度不合法")
    if candidate_probabilities.shape[:2] != candidate_hidden_states.shape[:2]:
        raise ValueError("候选概率与候选隐藏状态形状不一致")
    if generated_hidden_states is None or generated_hidden_states.shape[1] == 0:
        return (1.0 - penalty_alpha) * candidate_probabilities
    if generated_hidden_states.ndim != 3:
        raise ValueError("generated_hidden_states 必须是三维张量")
    if (
        generated_hidden_states.shape[0] != candidate_hidden_states.shape[0]
        or generated_hidden_states.shape[2] != candidate_hidden_states.shape[2]
    ):
        raise ValueError("候选隐藏状态与生成历史形状不一致")

    normalized_candidates = F.normalize(candidate_hidden_states.float(), dim=-1)
    normalized_history = F.normalize(generated_hidden_states.float(), dim=-1)
    degeneration_penalty = torch.einsum(
        "bkh,bgh->bkg", normalized_candidates, normalized_history
    ).amax(dim=-1)
    return (
        (1.0 - penalty_alpha) * candidate_probabilities.float()
        - penalty_alpha * degeneration_penalty
    )


def _repeat_past_key_values(past_key_values, repeats):
    """沿 batch 维复制 KV cache，供同一步的多个候选共享上下文。"""
    return [
        tuple(value.repeat_interleave(repeats, dim=0) for value in layer)
        for layer in past_key_values
    ]


# ============================================================================
# MiniMindConfig — 模型配置，兼容 HuggingFace PretrainedConfig
# ============================================================================
class MiniMindConfig(PretrainedConfig):
    """
    继承 PretrainedConfig 的好处：
      - 可以用 .from_pretrained() 保存/加载配置 
      - 自动注册到 transformers 的 model_type 系统
      - GenerationMixin 需要 config 提供 bos/eos token id 等
    """
    model_type = "minimind"
    def __init__(self, hidden_size=768, num_hidden_layers=16, use_moe=False, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size          # 隐藏层维度，默认 768
        self.num_hidden_layers = num_hidden_layers  # Transformer 层数，默认 16
        self.use_moe = use_moe                  # 是否使用 MoE 架构

        # === 基础配置 ===
        self.dropout = kwargs.get("dropout", 0.0)           # 默认无 dropout（小模型不需要正则化）
        self.vocab_size = kwargs.get("vocab_size", 12000)   # 阶段 A 固定的 BPE 词表大小
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        self.eos_token_id = kwargs.get("eos_token_id", 2)

        # === 注意力配置 ===
        self.flash_attn = kwargs.get("flash_attn", True)    # 优先使用 Flash Attention
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # GQA: KV 头数 < Q 头数
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)  # 96 = 768/8

        # === FFN 配置 ===
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # SwiGLU 用 SiLU 激活
        # Qwen3 约定: intermediate_size = π * hidden_size，向上取整到 64 的倍数
        # 而非 Llama 的 8/3 * hidden_size。Qwen 技术报告将此描述为刻意选择
        # （π 作为"自然"乘数，与 8/3 在实验中无显著差异，但保持架构风格一致）
        self.intermediate_size = kwargs.get("intermediate_size",
                                             math.ceil(hidden_size * math.pi / 64) * 64)

        # === 位置编码配置 ===
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 32K 上下文
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.rope_theta = kwargs.get("rope_theta", 1e6)      # RoPE 基频，Qwen3 默认 1e6

        # === 权重绑定 ===
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)  # lm_head 与 embed 共享权重

        # === YaRN 位置编码扩展（推理时扩展上下文窗口） ===
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
        self.rope_scaling = {
            "beta_fast": 32,       # 高频阈值——波长<beta_fast倍训练长度的维度保持原频率不变
            "beta_slow": 1,         # 低频阈值——波长>beta_slow倍训练长度的维度按1/factor缩放
            "factor": 16,           # 扩展倍数（32K / 2K = 16）
            "original_max_position_embeddings": 2048,  # 训练时的原始上下文长度
            "attention_factor": 1.0,  # 注意力分数缩放因子
            "type": "yarn"
        } if self.inference_rope_scaling else None

        # === MoE 配置（use_moe=False 时忽略） ===
        self.num_experts = kwargs.get("num_experts", 4)              # 专家总数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 激活的专家数 (Top-K)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)     # 归一化 Top-K 权重
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 辅助 loss 系数


# ============================================================================
# RMSNorm — 均方根归一化（同 llm-from-scratch）
# ============================================================================
class RMSNorm(torch.nn.Module):
    """
    为什么 LLM 都用 RMSNorm？
      - 计算量减半（省去均值计算）
      - 实验证明去掉均值不影响效果——Transformer 的残差连接本身
        已经提供了某种"均值规范化"（梯度通过残差直接回传），
        单独的中心化操作是冗余的（B. Zhang et al., 2019）
      - 配合 Pre-Norm 使用，数值更稳定
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 γ

    def norm(self, x):
        # x / sqrt(mean(x²) + ε)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 转为 float32 计算，再转回原始 dtype（混合精度训练时很重要）
        return (self.weight * self.norm(x.float())).type_as(x)


# ============================================================================
# RoPE 位置编码 — 预计算 cos/sin 频率表
# ============================================================================
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: dict = None):
    """
    预计算所有位置的 RoPE cos/sin 值。

    RoPE 的核心思想：
      用旋转矩阵给 attention 注入位置信息，使得：
        <f(q_m, m), f(k_n, n)> = g(q_m, k_n, m - n)
      即：query 和 key 的内积只依赖于它们的内容和相对位置 (m-n)，而非绝对位置。这让模型天然具有外推能力。

    频率计算：
      θ_i = 1 / (rope_base^(2i/d))  →  i 越大，频率越低（波长越长）
      低频捕捉长距离依赖，高频捕捉短距离依赖。

    YaRN 扩展（可选）：
      当推理长度超过训练长度时，用 ramp 函数对不同频率做不同缩放：
      高频几乎不变（本来就捕捉局部），低频按 1/factor 压缩（扩展"视野"）。
      这样避免直接外推时的"频率混叠"问题。
    """
    # 基础频率: shape (dim//2, )
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    attn_factor = 1.0

    if rope_scaling is not None:
        # YaRN 公式: f'(i) = f(i) * ((1-γ) + γ/s)
        # γ = ramp，高频→0(不变)，低频→1(完全缩放)，中间线性过渡
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 16)
        beta_fast = rope_scaling.get("beta_fast", 32.0)
        beta_slow = rope_scaling.get("beta_slow", 1.0)
        attn_factor = rope_scaling.get("attention_factor", 1.0)

        if end / orig_max > 1.0:  # 只有推理长度 > 训练长度时才启用
            # 分界点: low 对应高频(小i, 短波长, 不变), high 对应低频(大i, 长波长, 缩放)
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # ramp: 0=高频不变 → 1=低频完全缩放
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001),
                0, 1
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    # 所有位置的频率: outer product → shape (end, dim//2)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    # 构造 cos 和 sin：拼接自身得到 (end, dim)，消除奇偶维度的相位差
    # 因为 RoPE 公式中相邻维度共享同一个频率但相位差 90°
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    对 query 和 key 应用旋转位置编码。

    RoPE 公式（对每一对相邻维度 [x_{2i}, x_{2i+1}]）:
      [x_{2i}']   = [cos θ_i,  -sin θ_i] [x_{2i}]
      [x_{2i+1}']   [sin θ_i,   cos θ_i] [x_{2i+1}]

    即：x' = x * cos + rotate_half(x) * sin
    """
    def rotate_half(x):
        """将向量后半部分取负，拼到前面 → 实现 90° 旋转"""
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) +
               (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) +
               (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


# ============================================================================
# GQA 辅助函数 — 将 KV 头复制到 Q 头数量
# ============================================================================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    分组查询注意力 (GQA) 的核心操作：
      当 num_kv_heads < num_attention_heads 时，
      每个 KV 头需要被 n_rep 个 Q 头共享。

    例：num_attention_heads=8, num_key_value_heads=4 → n_rep=2
      KV 头 0 → Q 头 0,1
      KV 头 1 → Q 头 2,3
      ...

    用 expand (view 操作，不拷贝内存) 而非 repeat (会拷贝)，
    但 reshape 回来时会触发 contiguous copy（所以注释中的"内存浪费"是必然的）。
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]                                                    # (bs, slen, n_kv_heads, 1, head_dim)
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)                # 广播到 n_rep 份
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)              # 合并维度
    )


# ============================================================================
# Attention — 因果自注意力（支持 Flash Attention + GQA + QK Norm）
# ============================================================================
class Attention(nn.Module):
    """
    与 llm-from-scratch MultiHeadAttention 的关键差异:
      1. QK 归一化: 在 RoPE 前对 query 和 key 做 RMSNorm
         → 防止 attention logits 过大导致 softmax 梯度消失
         → Qwen3/Gemma 等新架构的标配
      2. 使用 nn.Linear 而非自定义 Linear 层
         → 可以利用 cublas 融合算子
      3. 通过 hasattr(F, 'scaled_dot_product_attention') 检测 Flash Attention
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # GQA 配置
        self.num_key_value_heads = (config.num_attention_heads
                                    if config.num_key_value_heads is None
                                    else config.num_key_value_heads)
        self.n_local_heads = config.num_attention_heads       # Q 头数
        self.n_local_kv_heads = self.num_key_value_heads      # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个 KV 头对应几个 Q 头
        self.head_dim = config.head_dim

        self.is_causal = True  # 因果注意力（只看当前位置及之前）

        # QKV 投影：注意 KV 的输出维度按 KV 头数计算（GQA 的参数量节省在此），多个q共用一个kv
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # ★ QK 归一化 — MiniMind/Qwen3 的关键特性 ★
        # 对每个 head 的 query 和 key 分别做 RMSNorm
        # 为什么有用？softmax(QK^T / √d) 中，如果 Q 或 K 的模过大，
        # softmax 会趋于 one-hot → 梯度消失。QK Norm 确保每个 head 的Q 和 K 分布稳定，让 attention 更均匀。
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 检测 Flash Attention 是否可用（PyTorch 2.0+）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape

        # === 1. 线性投影 ===
        xq = self.q_proj(x)  # (bsz, seq_len, n_heads * head_dim)
        xk = self.k_proj(x)  # (bsz, seq_len, n_kv_heads * head_dim)
        xv = self.v_proj(x)

        # === 2. 重塑为多头形状 ===
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # === 3. QK 归一化（★ MiniMind 独有，llm-from-scratch 无此步骤） ===
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # === 4. RoPE 位置编码 ===
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # === 5. KV-Cache 拼接（推理加速） ===
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 沿序列维度拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # === 6. GQA: KV 头广播到 Q 头数量，然后转置为 (bsz, heads, seq, dim) ===
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # === 7. 注意力计算：优先 Flash Attention，回退手动实现 ===
        if (self.flash and (seq_len > 1)  # Flash 对短序列优势不大
                and (not self.is_causal or past_key_value is None)  # KV-Cache 增量推理时不走 flash
                and (attention_mask is None or torch.all(attention_mask == 1))):  # 标准因果 mask
            # PyTorch 2.0+ sdpa: 自动选择 Flash Attention / Memory Efficient Attention
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal
            )
        else:
            # 手动实现: softmax(QK^T / sqrt(d)) * V
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 因果 mask: 上三角填充 -inf（看不到未来 token）
            if self.is_causal:
                scores[:, :, :, -seq_len:] += torch.full(
                    (seq_len, seq_len), float("-inf"), device=scores.device
                ).triu(1)
            # Padding mask: 将 padding 位置的 attention 置零
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # softmax → dropout → matmul
            output = self.attn_dropout(
                F.softmax(scores.float(), dim=-1).type_as(xq)  # float32 做 softmax 防溢出
            ) @ xv

        # === 8. 合并多头 → 输出投影 ===
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


# ============================================================================
# FeedForward — SwiGLU 前馈网络（同 llm-from-scratch）
# ============================================================================
class FeedForward(nn.Module):
    """
    SwiGLU 公式: output = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

    与标准 FFN (output = down(ReLU(up(x)))) 的差异:
      - SwiGLU 有门控机制 (gate * up)，使用 3 个投影矩阵（vs 标准 FFN 的 2 个）
      - 同等参数量下效果更好: 标准 FFN (4x扩展) = 2dm = 8d², SwiGLU (8/3扩展) = 3d×(8/3)d = 8d²，参数量持平
      - LLaMA/Qwen 均采用 SwiGLU

    中间层维度:
      - Llama: 8/3 * d_model → SwiGLU 总参数 = 3d × (8/3)d = 8d² = 标准FFN(4x)的参数量
      - Qwen3: π * d_model → ≈ 3.14d，约 9.42d²，略大于 8d²，理由更"自然"
    """
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # 从 transformers 获取激活函数（默认 SiLU）

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# MOEFeedForward — 混合专家前馈网络
# ============================================================================
class MOEFeedForward(nn.Module):
    """
    MoE 核心思想：用多个"专家"(FeedForward)替代单一 FFN，
    每个 token 只激活其中 Top-K 个，从而在不增加计算量的前提下扩大模型容量。

    MiniMind MoE 规格: 4 个专家，每个 token 激活 1 个 (Top-1 路由)。
      - 16 层默认配置的总参数量约 396M（vs Dense 127M）
      - 激活参数量约 127M（因为每个 token 只激活 1/4 专家）
      - 所以主要计算量和 Dense 127M 接近，但模型"知识容量"更大

    路由机制:
      1. tokens → gate (Linear) → 4 个得分 → softmax → Top-K 选择
      2. 每个被选中的 token 由对应专家处理，结果乘以路由权重
      3. 辅助 loss: 鼓励专家负载均衡（防止所有 token 都选同一个专家）
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由门控：hidden_size → num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 多个独立的 FeedForward 专家
        self.experts = nn.ModuleList([
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        ])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # (total_tokens, hidden_dim) — 所有 token 平等对待

        # === 1. 路由得分 ===
        scores = F.softmax(self.gate(x_flat), dim=-1)  # (total_tokens, num_experts)

        # === 2. Top-K 选择 ===
        topk_weight, topk_idx = torch.topk(
            scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False
        )
        # 归一化被选中的权重（如果选了多个专家）
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # === 3. 每个专家处理分配给它的 token ===
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # 找出所有被分配给专家 i 的 token
            mask = (topk_idx == i)  # (total_tokens, top_k)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()  # 哪些 token
                weight = topk_weight[mask].view(-1, 1)            # 对应的路由权重
                # index_add_: 将专家输出累加到 y 的对应位置
                y.index_add_(0, token_idx,
                            (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # ★ 关键 trick: 即使某个专家没被任何 token 选中，
                # 仍然创建一个零梯度依赖（值为0的张量操作），否则 DDP 会报 unused parameters 错误
                # 用 0.0 而非 0 确保 PyTorch 正确处理为浮点零张量
                y[0, 0] += 0.0 * sum(p.sum() for p in expert.parameters())

        # === 4. 负载均衡辅助 loss ===
        # 目标: 让 4 个专家的使用率尽量均匀（各 25%）
        # 公式: aux_loss = num_experts * sum(load_i * score_i) * coef
        #   load_i = 平均每个 token 选专家 i 的比例
        #   score_i = 路由器给专家 i 的平均得分
        # 实际使用时加到主 loss 上（系数很小，5e-4）
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)  # 各专家实际负载
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


# ============================================================================
# MiniMindBlock — 一个 Transformer 层（Pre-Norm 结构）
# ============================================================================
class MiniMindBlock(nn.Module):
    """
    结构（与 llm-from-scratch TransformerBlock 相同，均为 Pre-Norm）:

        x → input_layernorm → Self-Attention → + residual
          → post_attention_layernorm → FFN/MoE → + residual

    与 llm-from-scratch 的差异:
      - 使用 self.self_attn() 而非 self.attn()（命名不同）
      - MoE 模式下 self.mlp 是 MOEFeedForward（而非 FeedForward）
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)       # Pre-Attention Norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # Pre-FFN Norm
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None,
                use_cache=False, attention_mask=None):
        # Attention 子层（Pre-Norm）
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )
        hidden_states += residual

        # FFN/MoE 子层（Pre-Norm）
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


# ============================================================================
# MiniMindModel — 完整的 Transformer 骨干网络
# ============================================================================
class MiniMindModel(nn.Module):
    """
    组装完整的 Transformer Decoder：
      Embedding → [MiniMindBlock × L] → Final RMSNorm

    负责管理位置编码的预计算和分发。
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers

        # Token 嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer 层堆叠
        self.layers = nn.ModuleList([
            MiniMindBlock(l, config) for l in range(self.num_hidden_layers)
        ])

        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ★ 预计算 RoPE cos/sin 表 ★
        # 用 register_buffer 而非类属性，好处：
        #   - 自动跟随 model.to(device) 移动
        #   - persistent=False: 不保存到 state_dict（推理时可重新计算）
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape

        # 兼容 transformers>=5.x 的 Cache 对象 — 转为普通 tuple
        if hasattr(past_key_values, 'layers'):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)

        # 计算当前处理的起始位置（KV-Cache 已缓存的长度）
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # Embedding + Dropout
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ★ 修复 meta device 初始化导致的 buffer 丢失 (transformers>=5.x) ★
        # 当模型通过 PreTrainedModel.post_init() 在 meta device 上初始化后，
        # register_buffer 的张量可能变成全零。这里检测并重新计算。
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_position_embeddings,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)

        # 取当前序列所需的位置编码切片
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        # 逐层前向传播
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 最终 LayerNorm
        hidden_states = self.norm(hidden_states)

        # 收集各层 MoE 的辅助 loss
        aux_loss = sum(
            [l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
            hidden_states.new_zeros(1).squeeze()
        )
        return hidden_states, presents, aux_loss


# ============================================================================
# MiniMindForCausalLM — 完整的因果语言模型（训练 + 推理）
# ============================================================================
class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    完整的因果语言模型，可直接用于训练和推理。

    三层结构：
      input_ids → MiniMindModel (Transformer骨干) → hidden_states
               → lm_head (hidden_size → vocab_size)  → logits
               → cross_entropy(labels)                → loss

    通过继承获得的能力：
      - PreTrainedModel → save_pretrained / from_pretrained、与 Trainer 兼容
      - GenerationMixin → 框架层 generate() 接口（本类覆盖了它，见下方 generate 方法）
    """
    config_class = MiniMindConfig
    # 告诉 transformers：lm_head.weight 和 embed_tokens.weight 是同一块内存
    # save_pretrained 时只存一份，避免权重文件里出现两个相同的张量
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # lm_head: 把最后一层 hidden_state (768维) 映射到词表 (12000维)，得到每个 token 的"得分"
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        # 权重绑定：让 lm_head 和 embed_tokens 共用同一个矩阵
        # 原理：embed 把 token 映射到向量，lm_head 把向量映射回 token
        #       两者在数学上是"互逆"操作，共享权重相当于让它们互为转置
        # 收益：省 vocab_size × hidden_size ≈ 922 万参数
        if self.config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight

        # post_init() 做两件事：
        # 1. 递归初始化所有子模块的权重（调用 _init_weights）
        # 2. transformers>=5.x: 先在 meta device 上创建模型结构，再加载真实权重
        #    （meta device 不分配内存，只是"占位"，大幅加速大模型初始化）
        self.post_init()

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False,
                logits_to_keep=0, labels=None, **kwargs):
        """
        参数说明：
          input_ids:      (batch, seq_len) token ID 序列
          labels:         训练时的目标序列，与 input_ids 等长。不需要预测的位置填 -100
          logits_to_keep: 只保留最后 N 个位置的 logits（默认 0 = 全部保留）
                          增量推理时设为 1，因为只需要最新 token 的 logits 来采样下一个 token
          use_cache:      是否返回 KV Cache（训练时为 False，推理时为 True）
        """
        # === 第一步：通过 Transformer 骨干得到每层的 hidden_states ===
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache, **kwargs
        )

        # 增量推理时，hidden_states 包含所有已缓存位置的输出
        # 但我们只需要最后几个位置的 logits → 用 logits_to_keep 截断，省内存和计算
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        # === 第二步：计算交叉熵 loss（仅训练时） ===
        loss = None
        if labels is not None:
            # 语言模型的核心 trick：用 t 位置的输出预测 t+1 位置的 token
            # 例如 input = [A, B, C, D]，模型看到 [A, B, C] 应预测 [B, C, D]
            # logits[:, :-1, :] 取位置 0~N-2 的预测 → 对应预测 1~N-1 位置的 token
            # labels[:, 1:]    取位置 1~N-1 的真实 token → 与上面的预测对齐
            x = logits[..., :-1, :].contiguous()   # (batch, seq-1, vocab)
            y = labels[..., 1:].contiguous()        # (batch, seq-1)
            # 展平后一次性算交叉熵，ignore_index=-100 跳过不需要预测的位置（如 padding）
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,       # MoE 负载均衡 loss，非 MoE 模型时为 0
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )

    @torch.inference_mode()  # 禁用梯度计算和 autograd，推理更快更省内存
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192,
                 temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2,
                 streamer=None, use_cache=True, num_return_sequences=1,
                 do_sample=True, repetition_penalty=1.0, no_repeat_ngram_size=0,
                 repetition_control_start_tokens=0, repetition_tail_repeat_count=0,
                 frequency_penalty=0.0, repetition_path_penalty=0.0,
                 repetition_path_min_ngram_size=4,
                 repetition_path_max_ngram_size=12, penalty_alpha=0.0,
                 contrastive_search_top_k=0, **kwargs):
        """
        自回归生成：逐 token 预测，每次把新 token 拼回输入，直到达到最大长度或全部结束。

        采样参数：
          temperature:         缩放 logits。0.85 → 概率分布更尖锐（更确定），>1.0 → 更平坦（更随机）
          top_k:               只保留概率最高的 K 个 token，其余置为 -inf（硬截断）
          top_p (nucleus):     从高到低累加概率，超过 p 后的 token 全部置为 -inf（动态截断）
          repetition_penalty:  >1 降低当前回答中已生成 token 的概率，避免重复。=1 不生效
          no_repeat_ngram_size: 禁止当前回答重复生成指定长度的 n-gram；0 表示关闭
          repetition_control_start_tokens: 生成达到该长度后再启用上述两项控制
          repetition_tail_repeat_count: 尾部同一片段连续重复达到该次数时提前结束；0 表示关闭
          frequency_penalty: 生成 token 第三次出现前开始按频次软降权；0 表示关闭
          repetition_path_penalty: 对继续既有重复路径的候选 token 软降权；0 表示关闭
          penalty_alpha: Contrastive Search 的退化惩罚权重；0 表示关闭
          contrastive_search_top_k: Contrastive Search 每步参与比较的候选数
          do_sample:           True=从分布采样，False=贪心取最大（temperature 设为 1.0 即等价 greedy）
          streamer:            是一个流式输出工具，让生成的 token 逐字往外吐，而不是等全部生成完才一次性返回
          num_return_sequences:控制一次返回几条不同的生成结果。默认是 1。        


        为什么要自己写 generate 而不是直接用 GenerationMixin 的？
          HuggingFace 的 generate 在 KV-Cache 更新时机和 repetition_penalty 实现上
          与本项目的训练细节有细微不一致，这里自己写反而更可控、更透明。
        """
        # 初始化
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        prompt_length = input_ids.shape[1]
        contrastive_enabled = penalty_alpha != 0 or contrastive_search_top_k != 0
        if contrastive_enabled:
            if not 0 < penalty_alpha < 1:
                raise ValueError("启用 Contrastive Search 时 penalty_alpha 必须位于 0 和 1 之间")
            if not isinstance(contrastive_search_top_k, int) or isinstance(
                contrastive_search_top_k, bool
            ) or contrastive_search_top_k < 2:
                raise ValueError("contrastive_search_top_k 必须是至少为 2 的整数")
            if do_sample:
                raise ValueError("Contrastive Search 不支持 do_sample=True")
            if not use_cache:
                raise ValueError("Contrastive Search 必须启用 KV cache")
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        generated_hidden_states = None
        # finished[i]=True 表示第 i 条序列已经生成了 eos，后续只需填充
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer:
            streamer.put(input_ids.cpu())

        # === 自回归循环：每次生成一个 token ===
        for _ in range(max_new_tokens):
            # KV Cache 已缓存了前 past_len 个位置的 K、V
            # 所以只需把最新一个 token（input_ids[:, past_len:]）送入模型
            # 这也是 KV Cache 加速推理的核心：不用每次都重算整个序列
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(
                input_ids[:, past_len:], attention_mask, past_key_values,
                use_cache=use_cache, **kwargs
            )

            # attention_mask 随每个新 token 延长 1 位
            attention_mask = (
                torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1)
                if attention_mask is not None else None
            )

            # 只取最后一个位置（最新 token）的 logits，除以 temperature 控制随机性
            logits = outputs.logits[:, -1, :] / temperature
            generated_length = input_ids.shape[1] - prompt_length
            repetition_controls_enabled = (
                generated_length >= int(repetition_control_start_tokens)
            )

            # === repetition_penalty: 降低回答中已生成 token 的概率 ===
            # prompt 包含协议字段和示例，不能参与惩罚，否则会破坏结构化输出。
            # score > 0 (模型喜欢) → 除以 penalty → 值变小 → 更不容易被选中
            # score < 0 (模型讨厌) → 乘以 penalty → 更负 → 更讨厌
            if repetition_controls_enabled and repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i, prompt_length:])
                    if seen.numel() == 0:
                        continue
                    score = logits[i, seen]                     # 这些 token 的当前得分
                    logits[i, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            if frequency_penalty or repetition_path_penalty:
                logits = _apply_generated_repetition_penalties(
                    logits,
                    input_ids[:, prompt_length:],
                    frequency_penalty=frequency_penalty,
                    repetition_path_penalty=repetition_path_penalty,
                    repetition_path_min_ngram_size=repetition_path_min_ngram_size,
                    repetition_path_max_ngram_size=repetition_path_max_ngram_size,
                )

            # === top-k: 只保留得分最高的 K 个候选 ===
            # 找到第 K 大的值作为阈值，比它小的全部置为 -inf（exp(-inf)=0，不可能被采样）
            # === no_repeat_ngram：禁止复现当前回答已经生成过的 n-gram ===
            if repetition_controls_enabled and no_repeat_ngram_size and no_repeat_ngram_size > 0:
                ngram_size = int(no_repeat_ngram_size)
                if generated_length >= ngram_size:
                    for i in range(input_ids.shape[0]):
                        sequence = input_ids[i, prompt_length:].tolist()
                        prefix = tuple(sequence[-(ngram_size - 1):]) if ngram_size > 1 else ()
                        banned = {
                            sequence[start + ngram_size - 1]
                            for start in range(len(sequence) - ngram_size + 1)
                            if tuple(sequence[start:start + ngram_size - 1]) == prefix
                        }
                        if eos_token_id is not None:
                            banned.discard(int(eos_token_id))
                        if banned:
                            banned_ids = torch.tensor(tuple(banned), device=logits.device, dtype=torch.long)
                            logits[i, banned_ids] = -float("inf")
                            if not torch.isfinite(logits[i]).any() and eos_token_id is not None:
                                logits[i, int(eos_token_id)] = 0.0

            selected_hidden_state = None
            if contrastive_enabled:
                candidate_count = min(contrastive_search_top_k, logits.shape[-1])
                probabilities = torch.softmax(logits.float(), dim=-1)
                candidate_probabilities, candidate_ids = torch.topk(
                    probabilities, candidate_count, dim=-1
                )
                candidate_outputs = self.forward(
                    candidate_ids.reshape(-1, 1),
                    (
                        attention_mask.repeat_interleave(candidate_count, dim=0)
                        if attention_mask is not None
                        else None
                    ),
                    _repeat_past_key_values(
                        outputs.past_key_values, candidate_count
                    ),
                    use_cache=True,
                    **kwargs,
                )
                candidate_hidden_states = candidate_outputs.hidden_states[:, -1, :].reshape(
                    input_ids.shape[0], candidate_count, -1
                )
                contrastive_scores = _contrastive_search_scores(
                    candidate_probabilities,
                    candidate_hidden_states,
                    generated_hidden_states,
                    penalty_alpha=penalty_alpha,
                )
                selected_positions = contrastive_scores.argmax(dim=-1, keepdim=True)
                next_token = candidate_ids.gather(1, selected_positions)
                selected_hidden_state = candidate_hidden_states.gather(
                    1,
                    selected_positions.unsqueeze(-1).expand(
                        -1, -1, candidate_hidden_states.shape[-1]
                    ),
                )
            else:
                # === top-k: 只保留得分最高的 K 个候选 ===
                if 0 < top_k < logits.shape[-1]:
                    logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

                # === top-p (nucleus sampling): 动态截断 ===
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = 0
                    logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

                next_token = (
                    torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                    if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
                )

            # === 重复尾部早停：避免重复片段继续写入输出 ===
            tail_finished = torch.zeros_like(finished)
            if repetition_tail_repeat_count and repetition_tail_repeat_count > 1:
                for i in range(input_ids.shape[0]):
                    if finished[i]:
                        continue
                    candidate = torch.cat((input_ids[i, prompt_length:], next_token[i]))
                    if _has_repeated_tail(candidate, repeat_count=int(repetition_tail_repeat_count)):
                        tail_finished[i] = True
                        if eos_token_id is not None:
                            next_token[i] = next_token.new_tensor([eos_token_id])
            # 已结束的序列，统一填充 eos（避免模型继续产生无意义的输出）
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.shape[0], 1), eos_token_id),
                    next_token
                )

            # 拼接新 token 到序列末尾
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if selected_hidden_state is not None:
                generated_hidden_states = (
                    selected_hidden_state
                    if generated_hidden_states is None
                    else torch.cat(
                        (generated_hidden_states, selected_hidden_state), dim=1
                    )
                )

            if streamer:
                streamer.put(next_token.cpu())

            # 检查哪些序列刚刚生成了 eos → 标记为 finished
            # 全部 finished 时提前退出，不等 max_new_tokens 耗尽
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
            finished |= tail_finished
            if finished.all():
                break

        if streamer:
            streamer.end()

        if kwargs.get("return_kv"):
            return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
