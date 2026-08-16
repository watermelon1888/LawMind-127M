"""
GRPO (Group Relative Policy Optimization) 训练脚本
=================================================
在 SFT 模型基础上，用 RLAIF (RL from AI Feedback) 做对齐。

为什么需要 GRPO？DPO 不够吗？
  DPO: 从静态偏好数据中学（chosen vs rejected），简单但依赖数据质量
  GRPO: 模型自己生成回答 → Reward Model 打分 → 用组内对比学习
     → 模型可以探索 DPO 数据中没有覆盖的"更好回答"
     → 不需要人工标注的对比对，只需要 prompt

GRPO vs PPO:
  PPO: 用一个学到的 value function (Critic) 估计 baseline
  GRPO: 直接在同一个 prompt 的多个回答之间比较（组内相对优势）
     → 省掉了 Critic 模型，节省一半显存
     → 组内比较天然抵消了 prompt 难度差异

核心流程（每个 step）:
  1. Rollout (生成): 对每个 prompt 生成 num_generations 个回答
  2. Reward (打分): Reward Model + 规则奖励（长度、格式）
  3. Advantage (优势): 组内标准化 (reward - mean) / std
  4. Policy Update: PPO clipped loss + KL penalty
  5. Sync: 更新 rollout engine 中的模型权重

与 DPO 的关键区别:
  - DPO 有"标准答案"（chosen/rejected），GRPO 只有"好坏打分"
  - DPO 需要参考模型做基线，GRPO 用组内对比做基线
  - GRPO 需要 Reward Model（一个外部的评分模型），DPO 不需要
"""
import os, sys, math, re, gc, warnings, argparse
import torch, torch.nn.functional as F, torch.distributed as dist
from transformers import AutoTokenizer, AutoModel
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, init_distributed_mode,
    setup_seed, SkipBatchSampler, init_model, LMForRewardModel
)
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


# ============================================================================
# 奖励计算
# ============================================================================

def rep_penalty(text, n=3, cap=0.5):
    """
    重复惩罚（基于 n-gram 级别的重复度）。

    n=3 表示检测 3-gram 重复。重复率越高 → 惩罚越大。
    cap=0.5: 惩罚上限，避免对正确答案也过度惩罚。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """
    计算每个回答的总奖励。

    奖励 = 规则奖励 + Reward Model 评分

    规则奖励（启发式）:
      +0.5: 回答长度 20-800 字符（太短或太长扣分）
      +1.0: thinking 长度 20-300 字符（合理的思考长度）
      +0.25: 只有一个 </think> 标记（格式正确）
      -rep_penalty: n-gram 重复惩罚

    这些规则奖励的作用：在 Reward Model 之前，先给模型一个"格式正确"的信号。
    如果不加规则奖励，模型可能学会只输出一个字符来 cheat（Reward Model 对此不够敏感）。

    6 个专家合在一起：奖励 = Reward Model + 规则
    这不是标准方法，是小模型的实用 trick。
    """
    rewards = torch.zeros(len(responses), device=args.device)
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 prompt 中解析消息列表（给 Reward Model 用）
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response

                # 规则奖励: 长度
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # 规则奖励: thinking 格式
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    # thinking 长度合理
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # 只有一个 </think>
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()

                # 重复惩罚
                rewards[response_idx] -= rep_penalty(answer)

                # Reward Model 评分（来自外部的 1.8B 奖励模型）
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


# ============================================================================
# GRPO 训练循环
# ============================================================================
def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model,
                     start_step=0, wandb=None, use_sglang=False):
    for step, batch in enumerate(loader, start=start_step + 1):
        # === 1. 准备 prompt ===
        prompts = batch['prompt']  # list[str]
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # === 2. Rollout: 对每个 prompt 生成 num_generations 个回答 ===
        # rollout_engine 抽象了"谁来做推理"：可以是本地模型（TorchRolloutEngine）
        # 也可以是 SGLang HTTP 服务器（SGLangRolloutEngine，用于加速）
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,  # 每个 prompt 生成 6 个回答
            max_new_tokens=args.max_gen_len,
            temperature=0.8,  # 较高温度 → 生成多样性 → 组内有区分度
        )
        outputs = rollout_result.output_ids            # [B*G, P+R]
        completion_ids = rollout_result.completion_ids # [B*G, R] — 只答部分
        completions = rollout_result.completions       # list[str]
        # old_per_token_logps: 生成时策略模型给每个 token 的 log prob
        # 用于计算 importance sampling ratio
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)

        full_mask = (outputs != tokenizer.pad_token_id).long()

        # 只计算生成部分（completion）的 log prob 的起始位置
        logp_pos = (prompt_lens.unsqueeze(1) - 1 +
                    torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0))

        # === 3. 计算奖励 ===
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)

        # === 4. 当前策略模型的 per-token log prob ===
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(outputs, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 从 logits 计算每个 completion token 的 log prob
            per_token_logps = (
                F.log_softmax(res.logits[:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        # === 5. 参考模型的 per-token log prob（计算 KL 散度用）===
        with torch.no_grad():
            ref_per_token_logps = (
                F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1)
                .gather(2, outputs[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .gather(1, logp_pos)
            )

        # === 6. ★ GRPO 核心: 组内标准化计算 advantage ★ ===
        # grouped_rewards: (B, G) — 把每个 prompt 的 G 个回答的奖励放在一起
        # mean_r: 组内平均 → 作为 baseline
        # std_r: 组内标准差 → 衡量"这个回答相对于同组其他回答有多好"
        # advantages = (reward - 组内平均) / 组内标准差
        #
        # 直觉: 如果 prompt_A 很简单（所有回答都很高分），需要区分"特别好的回答"vs"只是不错的回答"
        #        如果 prompt_B 很难（所有回答都很低分），也需要找出相对最好的
        #   组内标准化自动适应这种差异
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # === 7. completion mask: 只对有效 token 计算 loss ===
        # 截断到 EOS token 之后的部分不参与计算
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
            (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1))
            & completion_pad_mask
        ).int()

        # === 8. ★ Policy loss: PPO clipped + KL penalty ★ ===
        # KL 散度: k3 估计器 (http://joschu.net/blog/kl-approx.html)
        #   k3(π_θ, π_ref) = exp(kl_div) - kl_div - 1
        #   比平凡的 MSE 更稳定，且永远 >= 0
        kl_div = ref_per_token_logps - per_token_logps  # log(π_ref/π_θ)
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        # Importance sampling ratio: π_θ(token) / π_old(token)
        # 因为生成时用的是旧策略 π_old，需要这个 ratio 做 off-policy 修正
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            # CISPO: Clipped Importance Sampling Policy Optimization
            # 只在 ratio > epsilon_high 时 clip（防止 outlier 的 ratio 过大）
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(
                clamped_ratio * advantages.unsqueeze(1) * per_token_logps
                - args.beta * per_token_kl  # KL 惩罚：防止模型偏离参考模型太远
            )
        else:
            # 标准 GRPO (PPO-style clipped loss)
            # L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A) - β * KL
            # clipped 部分：如果 ratio 超出 [1-ε, 1+ε]，用 clipped ratio
            # 这防止单步更新过大（trust region 的思想）
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(
                torch.min(per_token_loss1, per_token_loss2)  # ★ PPO 的 pessimistic bound
                - args.beta * per_token_kl
            )

        # 只在有效 token 上平均
        policy_loss = (
            (per_token_loss * completion_mask).sum(dim=1)
            / completion_mask.sum(dim=1).clamp(min=1)
        ).mean()

        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # === 9. 参数更新 ===
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()  # ★ GRPO 用 CosineAnnealingLR 而非每个 step 手动设 lr
            optimizer.zero_grad()

        # === 10. 日志 ===
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            # KL 散度（监控指标 — 太大说明模型"学飞了"）
            kl_ref_val = (
                (ref_per_token_logps - per_token_logps) * completion_mask
            ).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_std_val = advantages.std().item()  # 组内区分度
            current_lr = optimizer.param_groups[0]['lr']
            Logger(
                f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), '
                f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                f'Adv Std: {advantages_std_val:.4f}, Actor Loss: {policy_loss_val:.4f}, '
                f'Avg Resp Len: {avg_len_val:.2f}, LR: {current_lr:.8f}'
            )
            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val, "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val, "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val, "learning_rate": current_lr
                })

        # === 11. 保存 checkpoint ===
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            torch.save({k: v.half().cpu() for k, v in raw_model.state_dict().items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                          scheduler=scheduler)
            model.train()

        # ★ 同步模型权重到 rollout engine ★
        # 这是 RL 训练的关键：每次更新后，生成器也要用最新模型
        if step % args.save_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask

    # epoch 结束：处理未完成的梯度累积
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO")
    # 基础配置
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='grpo', type=str)
    # ★ GRPO 超参 ★
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)   # 每 batch 只有 2 个 prompt
    parser.add_argument("--learning_rate", type=float, default=3e-7)  # 极低 lr
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument('--max_seq_len', default=768, type=int)
    parser.add_argument("--max_gen_len", type=int, default=1024,
                        help="生成的最大长度 — DPO/GRPO 需要长上下文")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl")
    # ★ GRPO 特有超参 ★
    parser.add_argument("--num_generations", type=int, default=6,
                        help="每个 prompt 生成几个回答 — G 越大组内比较越可靠但越慢")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL 惩罚系数 — 防止模型偏离参考模型太远")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"],
                        help="cispo 对 ratio outlier 更鲁棒")
    parser.add_argument("--epsilon", type=float, default=0.2,
                        help="PPO clip 范围")
    parser.add_argument("--epsilon_high", type=float, default=5.0,
                        help="CISPO: ratio 上界（允许 5 倍以内的 ratio 通过）")
    parser.add_argument('--from_weight', default='full_sft',
                        help="★ GRPO 必须在 SFT 模型基础上训练 ★")
    parser.add_argument("--reward_model_path", type=str,
                        default="../../internlm2-1_8b-reward",
                        help="外部 Reward Model 路径（约 1.8B 参数）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    parser.add_argument("--debug_mode", action="store_true",
                        help="打印每个 step 的生成样本")
    parser.add_argument("--debug_interval", type=int, default=20)
    parser.add_argument("--thinking_ratio", type=float, default=0.9,
                        help="90% 的回答开启 thinking")
    # ★ Rollout Engine ★
    parser.add_argument("--rollout_engine", type=str, default="torch",
                        choices=["torch", "sglang"])
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998")
    parser.add_argument("--sglang_model_path", type=str, default="../model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo")
    args = parser.parse_args()

    # ========== 1-4 标准初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe)
    )
    ckp_data = (lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
                if args.from_resume == 1 else None)
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"MiniMind-GRPO-...", id=wandb_id, resume=resume)

    # ========== 5. 三模型 + Rollout Engine ==========
    # Policy 模型 — 要训练的模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # Reference 模型 — 冻结的 SFT 模型，计算 KL 惩罚用
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward 模型 — 外部的 1.8B 评分模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout Engine — 生成回答的引擎（可插拔：Torch 或 SGLang）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine, policy_model=model, tokenizer=tokenizer,
        device=args.device, autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url, sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len,
                            thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # ★ 用 CosineAnnealingLR（GRPO 不需要 warmup，直接用余弦退火）★
    total_optimizer_steps = math.ceil(
        len(DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler))
        / args.accumulation_steps
    ) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps,
                                  eta_min=args.learning_rate / 10)

    # ========== 6-9 标准流程 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch, start_step = ckp_data['epoch'], ckp_data.get('step', 0)
    if args.use_compile == 1:
        model = torch.compile(model); rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)  # 确保 engine 用的是最新权重

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                           num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: 跳过前{start_step}step，从{start_step+1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine,
                           ref_model, reward_model, start_step, wandb,
                           use_sglang=(args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine,
                           ref_model, reward_model, 0, wandb,
                           use_sglang=(args.rollout_engine == "sglang"))

    if dist.is_initialized():
        dist.barrier(); dist.destroy_process_group()
