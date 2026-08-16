"""使用 MiniMind 实际生成模板计算法律回答 prompt 长度。"""

from rag.answering.evidence import EvidencePackage
from rag.answering.protocol import build_answer_prompt


class PromptTokenCountError(RuntimeError):
    """运行时 prompt 无法按固定模板可靠计数。"""


class AnswerPromptTokenCounter:
    """镜像 API 服务的无思考生成模板，并返回完整 prompt token 数。"""

    def __init__(self, tokenizer):
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer 必须提供 apply_chat_template")
        if not callable(tokenizer):
            raise TypeError("tokenizer 必须可调用")
        bos_token = getattr(tokenizer, "bos_token", None)
        if not isinstance(bos_token, str) or not bos_token:
            raise ValueError("tokenizer 缺少 bos_token")
        self._tokenizer = tokenizer
        self._expected_tail = f"{bos_token}assistant\n<think>\n\n</think>\n\n"

    def __call__(self, package):
        if not isinstance(package, EvidencePackage):
            raise TypeError("package 必须是 EvidencePackage")
        try:
            prompt = self._tokenizer.apply_chat_template(
                build_answer_prompt(package),
                tokenize=False,
                add_generation_prompt=True,
                tools=None,
                open_thinking=False,
            )
        except Exception as error:
            raise PromptTokenCountError("无法应用运行时 chat template") from error
        if not isinstance(prompt, str) or not prompt.endswith(self._expected_tail):
            raise PromptTokenCountError("运行时 chat template 的空 think 前缀发生变化")

        try:
            encoded = self._tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            input_ids = (
                encoded.get("input_ids")
                if isinstance(encoded, dict)
                else getattr(encoded, "input_ids", None)
            )
        except Exception as error:
            raise PromptTokenCountError("无法编码运行时 prompt") from error
        if (
            not isinstance(input_ids, list)
            or not input_ids
            or any(type(token_id) is not int for token_id in input_ids)
        ):
            raise PromptTokenCountError("tokenizer 没有返回一维非空 input_ids")
        return len(input_ids)


__all__ = ["AnswerPromptTokenCounter", "PromptTokenCountError"]
