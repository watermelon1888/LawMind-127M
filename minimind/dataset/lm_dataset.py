"""
数据加载模块 — 四种 Dataset 实现
===============================
覆盖 LLM 训练全管道的四种数据格式。

设计模式:
  PretrainDataset: 预打包 token shards → 固定长度 input_ids + labels
  SFTDataset: 多轮对话 → input_ids + labels (只 assistant 回复参与 loss)
  DPODataset: 偏好对 → x_chosen/y_chosen/mask + x_rejected/y_rejected/mask
  RLAIFDataset: 单轮 prompt → prompt (不含回答，模型在线生成)
  AgentRLDataset: 多轮工具调用 → messages + tools + ground truth

辅助函数:
  pre_processing_chat(): 随机添加 system prompt（20% 概率）
  post_processing_chat(): 随机移除空 <think> 标记（80% 概率）
"""
from torch.utils.data import Dataset
import torch, json, os, random
from datasets import load_dataset, Features, Sequence, Value
from .pretrain_dataset import PretrainDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# 对话预处理工具函数
# ============================================================================

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    随机添加 system prompt（20% 概率）。

    为什么需要这个？
      - 不是所有用户对话都会带 system prompt
      - 模型需要学会"有 system"和"没有 system"两种情况
      - 随机化让模型对不同输入格式更鲁棒

    tool use 数据不参与此处理（保留原始的 tools 定义）。
    """
    if any(conv.get('tools') for conv in conversations):
        return conversations  # tool use 数据原样保留

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]

    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """
    随机移除空的 <think> 标记（80% 概率）。

    为什么需要混合空/非空 thinking？
      - 有些问题不需要思考（"你好"），有些需要（"证明费马大定理"）
      - 随机混合让模型学会灵活切换"快回答"和"慢思考"
      - 80% 空 think：多数情况走快速通道
      - 20% 有 think：保留深度推理能力
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


# ============================================================================
# SFTDataset — 监督微调：多轮对话 → 只训练 assistant 回复
# ============================================================================
class SFTDataset(Dataset):
    """
    最复杂的 Dataset — 核心难点在于 chat_template 和 label mask。

    数据格式: JSONL，每行 {"conversations": [{role, content, ...}, ...]}

    处理流程:
      1. pre_processing_chat: 随机加 system prompt
      2. create_chat_prompt: apply_chat_template → 格式化字符串
      3. post_processing_chat: 随机去空 think
      4. tokenize + padding
      5. generate_labels: 找到 assistant 位置 → 设 labels = input_ids，其余 -100

    ★ generate_labels 是 SFT 最核心的逻辑 ★
    原理: 在 tokenized 序列中搜索 <|im_start|>assistant\n 和 <|im_end|>\n
    将两者之间的所有 token 标记为"参与 loss 计算"
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 定义 features 确保 conversations 字段正确解析
        features = Features({
            'conversations': [{
                'role': Value('string'), 'content': Value('string'),
                'reasoning_content': Value('string'), 'tools': Value('string'),
                'tool_calls': Value('string')
            }]
        })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 预计算 special token 的 id 序列（用于在 input_ids 中搜索）
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        将 conversation list 转为格式化的 prompt 字符串。

        调用 tokenizer.apply_chat_template() — 使用 tokenizer_config.json 中
        的 Jinja2 模板自动处理 tool calling、thinking 等格式。
        """
        messages, tools = [], None
        for message in conversations:
            message = dict(message)
            # 处理 system 消息中的 tools 定义
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # 处理 tool_calls (可能是 JSON 字符串)
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, tools=tools)

    def generate_labels(self, input_ids):
        """
        ★ SFT 核心: 生成 loss mask ★

        只在 <|im_start|>assistant\n ... <|im_end|>\n 之间的 token 参与 loss。

        算法:
          1. 初始化 labels = [-100] * len (全部不参与 loss)
          2. 搜索 bos_id (<|im_start|>assistant\n) 的位置
          3. 搜索 eos_id (<|im_end|>\n) 的位置
          4. 两者之间的 token: labels[i] = input_ids[i] (参与 loss)
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到 assistant 块的开始
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 找到 assistant 块的结束
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # assistant 回复范围内的 token → 参与 loss
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ============================================================================
# DPODataset — DPO: chosen/rejected 偏好对
# ============================================================================
class DPODataset(Dataset):
    """
    DPO 数据格式: {"chosen": [messages], "rejected": [messages]}

    输出:
      x_chosen/y_chosen/mask_chosen: chosen 回答的 input/label/mask
      x_rejected/y_rejected/mask_rejected: rejected 回答的 input/label/mask

    ★ generate_loss_mask 与 SFT 的 generate_labels 逻辑相同:
      只在 assistant 回复部分 mask=1
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self): return len(self.samples)

    def generate_loss_mask(self, input_ids):
        """与 SFTDataset.generate_labels 逻辑相同，但返回 0/1 mask 而非 labels"""
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start, end = i + len(self.bos_id), i + len(self.bos_id)
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id: break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 对 chosen 和 rejected 分别编码
        chosen_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(sample['chosen'], tokenize=False, add_generation_prompt=False))
        rejected_prompt = post_processing_chat(
            self.tokenizer.apply_chat_template(sample['rejected'], tokenize=False, add_generation_prompt=False))

        chosen_enc = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_enc = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        chosen_ids, chosen_mask = chosen_enc['input_ids'], self.generate_loss_mask(chosen_enc['input_ids'])
        rejected_ids, rejected_mask = rejected_enc['input_ids'], self.generate_loss_mask(rejected_enc['input_ids'])

        # 返回 (x=input[:-1], y=input[1:]) — 标准的 next-token prediction 格式
        return {
            'x_chosen': torch.tensor(chosen_ids[:-1], dtype=torch.long),
            'y_chosen': torch.tensor(chosen_ids[1:], dtype=torch.long),
            'mask_chosen': torch.tensor(chosen_mask[1:], dtype=torch.long),
            'x_rejected': torch.tensor(rejected_ids[:-1], dtype=torch.long),
            'y_rejected': torch.tensor(rejected_ids[1:], dtype=torch.long),
            'mask_rejected': torch.tensor(rejected_mask[1:], dtype=torch.long),
        }


# ============================================================================
# RLAIFDataset — GRPO/PPO: 只有 prompt，模型在线生成回答
# ============================================================================
class RLAIFDataset(Dataset):
    """
    RLAIF 数据只包含 prompt（不含回答）— 回答由模型在线生成。

    数据格式: {"conversations": [messages]}
    输出: {"prompt": str, "answer": ""}

    ★ 与 SFT/DPO 的关键区别：只返回 prompt，不含任何回答 ★
    回答由 Rollout Engine 在训练过程中在线生成（自回归采样），
    这是 RL 训练的核心——模型必须自己探索回答空间。

    thinking_ratio: 控制多少比例的 prompt 在生成时开启 thinking。
      → 90% (GRPO 默认): 多数回答带思考链，让模型学会"先想后说"
      → open_thinking=True 时 chat_template 会在末尾追加 \n<think>\n
        让模型先生成推理过程，再生成最终回答

    ★ add_generation_prompt=True 的效果 ★
      chat_template 会在 prompt 末尾追加 <|im_start|>assistant\n（以及可选的 <think>\n），
      使输入以"半成品 assistant"结尾。模型拿到这个半成品后直接续写 token，
      等效于"请你以 assistant 的身份回复"。
      这个效果和推理时的 "add_generation_prompt=True" 是同一个机制——
      确保训练和推理的输入格式完全一致。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self): return len(self.samples)

    def create_chat_prompt(self, conversations):
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        # ★ add_generation_prompt=True: 追加 <|im_start|>assistant\n<think>\n
        # 让模型在这个"半成品"之后继续生成
        return self.tokenizer.apply_chat_template(
            conversations[:-1], tokenize=False, open_thinking=use_thinking, add_generation_prompt=True)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        return {'prompt': prompt, 'answer': ""}


# ============================================================================
# AgentRLDataset — Agent RL: 多轮工具调用
# ============================================================================
class AgentRLDataset(Dataset):
    """用于训练模型的多轮工具调用能力。数据格式较复杂，不在本次学习范围内展开。"""
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self): return len(self.samples)

    def parse_conversations(self, conversations):
        messages, tools = [], None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools  # 去掉最后一条（ground truth answer）

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
