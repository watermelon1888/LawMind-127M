"""
MiniMind DPO (Direct Preference Optimization) 训练脚本
=====================================================
在 SFT 模型基础上，用偏好数据 (chosen vs rejected) 直接优化模型。

为什么需要 DPO？
  预训练: 学会"语言是什么"
  SFT: 学会"对话怎么进行"
  但 SFT 有个问题——它只教模型"这是正确答案"，不教"那个是错误答案"。
  模型需要知道：同样的问题，什么样的回答更好、什么样更差。
  DPO 就是干这个的——给模型展示"好回答 vs 差回答"的对比，让模型倾向好回答。

DPO vs RLHF PPO 的区别:
  RLHF: 需要训练一个独立的 Reward Model → 用它给 PPO 提供奖励信号
        训练流程: SFT → RM → PPO（三步，复杂）
  DPO: 直接从偏好数据中推导出隐式奖励，不需要显式的 Reward Model
        训练流程: SFT → DPO（两步，更简单）
  两者在数学上等价（在特定假设下），DPO 是更"优雅"的解法。

与 pretrain/SFT 训练脚本的差异:
  1. 数据格式: DPODataset（chosen/rejected 对比对）
  2. Loss: DPO loss（不是 cross_entropy）
  3. 参考模型: 需要一份冻结的 SFT 模型（ref_model），不参与训练
  4. 学习率: 4e-8 — 极小！DPO 只做"偏好微调"，大 lr 会破坏 SFT 学会的对话能力

模板代码（与 train_pretrain.py 相同部分）只注释 DPO 特有逻辑。
"""
import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import DPODataset  # ★ DPO 专用数据集
from trainer.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint,
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

warnings.filterwarnings('ignore')


# ============================================================================
# DPO 核心数学
# ============================================================================

def logits_to_log_probs(logits, labels):
    """
    将 logits 转为每个 token 的对数概率。

    输入:
      logits: (batch_size, seq_len, vocab_size) — 模型原始输出
      labels: (batch_size, seq_len) — 目标 token IDs

    输出:
      log_probs_per_token: (batch_size, seq_len) — 每个位置上正确 token 的 log 概率

    过程:
      logits → log_softmax (得到每个 token 的 log 概率分布)
            → gather (取出 labels 对应的那个 token 的 log 概率)
    """
    log_probs = F.log_softmax(logits, dim=2)  # (B, S, V)
    # gather: 在 dim=2 (vocab 维度) 上按 labels 索引取值
    log_probs_per_token = torch.gather(
        log_probs, dim=2, index=labels.unsqueeze(2)
    ).squeeze(-1)  # (B, S)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    DPO 损失函数（核心公式）。

    数学推导（完整版）:

    1. 隐式奖励（从 RLHF 最优策略的闭式解反解）:
       r(x, y) = β * log[π_θ(y|x) / π_ref(y|x)] + β * log Z(x)
       Z(x) 是配分函数。log Z(x) 在做 chosen/rejected 差时会被消去:
         r(x, y_chosen) - r(x, y_rejected)
       = β * (log π_θ(c)/π_ref(c) - log π_θ(r)/π_ref(r)) + β * (log Z(x) - log Z(x))
       = β * (log π_θ(c)/π_ref(c) - log π_θ(r)/π_ref(r))
       所以实际代码中可以省略 Z(x) 项。

    2. Bradley-Terry 偏好模型:
       P(y_chosen > y_rejected) = σ(r(x, y_chosen) - r(x, y_rejected))
       即：chosen 更好的概率 = sigmoid(隐式奖励差)

    3. DPO loss = -log σ(β * ((log π_θ(chosen)/π_ref(chosen)) - (log π_θ(rejected)/π_ref(rejected))))
       = -log σ(β * (logratios_θ - logratios_ref))

    化简后（代码中的实现）:
      logratios_θ = log π_θ(chosen) - log π_θ(rejected)   (policy 模型对两者的偏好差)
      logratios_ref = log π_ref(chosen) - log π_ref(rejected) (参考模型对两者的偏好差)
      loss = -log sigmoid(β * (logratios_θ - logratios_ref))

    直觉理解:
      - 如果模型给 chosen 的打分 > rejected → logratios_θ 为正 → loss 小（好）
      - β 控制偏好强度: β 大 → 模型更激进地区分好坏（可能 overfit）；β 小 → 更保守
      - 参考模型基线: 因为 DPO 奖励 = 相对参考模型的提升，
        防止模型通过"对所有回答都给高分"来作弊

    ┌─────────────────────────────────────────────────┐
    │  参数说明                                          │
    │  ref_log_probs:   (2*B, S) — 参考模型(SFT)的log prob│
    │  policy_log_probs:(2*B, S) — 当前模型的log prob    │
    │    前 B 个是 chosen，后 B 个是 rejected            │
    │  mask: (2*B, S) — 只计算 assistant token 的 loss   │
    │  beta: 温度系数 — MiniMind 默认 0.15                │
    └─────────────────────────────────────────────────┘
    """
    # 只计算 assistant 回复区域的 log prob 总和（mask 过滤非 assistant token）
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)       # (2*B, )
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)  # (2*B, )

    # 将 batch 拆成 chosen 和 rejected 两半
    # 因为 DataLoader 返回的数据是 chosen 和 rejected 交替的
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]          # (B, ) — 参考模型对 chosen 的打分
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]          # (B, ) — 参考模型对 rejected 的打分
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]    # (B, ) — 当前模型对 chosen 的打分
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]    # (B, ) — 当前模型对 rejected 的打分

    # DPO 公式: loss = -log σ(β * ((log_πθ(c) - log_πθ(r)) - (log_πref(c) - log_πref(r))))
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs     # log_πθ(c) - log_πθ(r)
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs          # log_πref(c) - log_πref(r)
    logits = pi_logratios - ref_logratios  # 新模型相对参考模型的偏好提升

    # ★ 为什么是 -logsigmoid？因为我们要最大化 sigmoid(logits)，等价于最小化 -log sigmoid(logits)
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


# ============================================================================
# 训练循环（DPO 版本）
# ============================================================================
def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step

        # === DPO 特有的数据 ===
        # 一个 batch 包含 chosen 和 rejected 各一份
        x_chosen = batch['x_chosen'].to(args.device)       # chosen 的 input_ids (B, S-1)
        x_rejected = batch['x_rejected'].to(args.device)   # rejected 的 input_ids (B, S-1)
        y_chosen = batch['y_chosen'].to(args.device)       # chosen 的 labels (B, S-1)
        y_rejected = batch['y_rejected'].to(args.device)   # rejected 的 labels (B, S-1)
        mask_chosen = batch['mask_chosen'].to(args.device) # chosen 的 loss mask (B, S-1)
        mask_rejected = batch['mask_rejected'].to(args.device) # rejected 的 loss mask (B, S-1)

        # ★ 拼接成一个大 batch：前一半是 chosen，后一半是 rejected
        x = torch.cat([x_chosen, x_rejected], dim=0)       # (2*B, S-1)
        y = torch.cat([y_chosen, y_rejected], dim=0)       # (2*B, S-1)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)  # (2*B, S-1)

        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # === 参考模型前向（冻结，不计算梯度）===
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_log_probs = logits_to_log_probs(ref_logits, y)  # (2*B, S-1)

            # === 策略模型前向（计算梯度）===
            outputs = model(x)
            logits = outputs.logits
            policy_log_probs = logits_to_log_probs(logits, y)   # (2*B, S-1)

            # === DPO loss ===
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss  # 加上 MoE 辅助 loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                f'loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, '
                f'aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, '
                f'epoch_time: {eta_min:.3f}min'
            )
            if wandb:
                wandb.log({
                    "loss": current_loss, "dpo_loss": current_dpo_loss,
                    "aux_loss": current_aux_loss, "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                          scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 显式释放 DPO 大量中间变量
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    # ★ DPO 超参 ★
    parser.add_argument("--epochs", type=int, default=1)      # DPO 只需 1 个 epoch
    parser.add_argument("--batch_size", type=int, default=4)   # DPO 显存占用大（需要 ref_model + policy_model）
    # ★ DPO 学习率极小 — 只做偏好微调，大 lr 会遗忘 SFT 学到的对话能力
    parser.add_argument("--learning_rate", type=float, default=4e-8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)  # DPO 的 chosen/rejected 可能较长
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1])
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")
    parser.add_argument('--from_weight', default='full_sft', type=str,
                        help="★ DPO 必须在 SFT 模型基础上训练 ★")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1])
    # ★ beta 参数 — DPO 的核心超参，控制偏好强度
    parser.add_argument('--beta', default=0.15, type=float,
                        help="0.1推荐避免遗忘；0.25~0.5偏好更强但风险稍大；>1不推荐")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1])
    args = parser.parse_args()

    # ========== 1-4 与通用模板相同 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义策略模型和参考模型 ==========
    # 策略模型 (policy model): 需要训练 — 就是要优化的模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')

    # ★ 参考模型 (reference model): 冻结的 SFT 模型，提供"锚定基线"
    # 为什么需要参考模型？
    #   DPO 优化的是"相对于参考模型的偏好变化":
    #     objective = β * (log(π_θ/π_ref)_chosen - log(π_θ/π_ref)_rejected)
    #   如果没有参考模型，模型只需让 log π_θ(chosen) - log π_θ(rejected) > 0，
    #   → 模型可以简单地将 chosen 的概率推至 1、rejected 推至 0，导致灾难性遗忘。
    #   有了参考模型，模型只能放大 chosen 的概率"相对于参考模型更多"，
    #   而不是绝对地推高 chosen 的概率——这保护了 SFT 学到的泛化能力。
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)  # 冻结所有参数
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')

    # ★ DPO 数据集 ★
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
