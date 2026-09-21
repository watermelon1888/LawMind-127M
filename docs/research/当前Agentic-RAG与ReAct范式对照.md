# 当前 Agentic RAG 与 ReAct 范式对照

## 结论

严格来说，当前整体法律 RAG 不是经典 ReAct；更准确的表述是：**整体为受控 Agentic RAG，其中证据检索阶段是一个有界的 ReAct-style 工具调用循环**。

它符合 ReAct 的一项关键机制：外部模型可根据当前证据选择 `search_law`，系统执行后将 Observation 追加到同一消息历史，模型再据此决定继续检索或 `finalize_evidence`。但它不生成、保留或展示 ReAct 定义中的显式 reasoning trace，也不由同一 Agent 在循环中产生并反思最终答案。

## 判定标准

ReAct 原论文将其方法定义为：让语言模型以交错方式生成 reasoning traces 与 task-specific actions；reasoning 用于归纳、跟踪和更新行动计划，action 则与知识库或环境交互以获得新信息。[ReAct 原论文](https://arxiv.org/abs/2210.03629) 和[作者项目页](https://react-lm.github.io/) 都将“reasoning 与 acting 交错”作为核心，而不是把“调用过工具”本身等同于 ReAct。

用户指定的 [Datawhale Hello-Agents 4.2](https://hello-agents.datawhale.cc/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=_42-react) 给出了更明确的工程化口径：不断重复 `Thought -> Action -> Observation`，把新的行动与观察追加到历史，直到 Agent 认为任务完成。按这个口径，完整经典 ReAct 至少包含：

1. 模型根据问题和既有轨迹做出当前推理与行动决策。
2. 环境执行行动并返回观察。
3. 观察进入同一轨迹，以影响后续推理与行动。
4. 模型在该循环内判断停止并完成任务。

## 与当前实现对照

| ReAct 要素 | 当前实现 | 判定 |
| --- | --- | --- |
| 根据状态选择行动 | 外部模型检查 Hybrid top-20 与 Cross-Encoder top-5，每轮必须选择 `search_law` 或 `finalize_evidence` | 符合 |
| Action | `search_law(query)` 执行新的检索；`finalize_evidence(...)` 显式结束证据检查 | 符合 |
| Observation 回灌 | `_append_tool_exchange` 把 assistant tool call 和 tool observation 追加到原 `messages` | 符合 |
| 多轮交错 | `while True` 使模型在补查后重新看到观察，最多允许两次补查 | 符合，但仅限需补查的少数问题 |
| 显式 Thought / reasoning trace | 工具接口设置 `thinking: disabled`，提示词只要求结构化工具决策，运行轨迹不保留显式推理 | 不符合经典形式 |
| 同一 Agent 完成最终答案 | 外部证据助理只选证；程序确定性融合后，另由 MiniMind 单次生成最终答案 | 不符合端到端 ReAct |
| 基于最终答案的后续观察与修正 | MiniMind 输出只做本地协议与引用校验；成功则直接交付，失败则报错 | 不具备 |

对应的本地代码证据：

- [`rag/query/agentic_retrieval.py`](../../rag/query/agentic_retrieval.py) 第 283–306 行追加工具调用与 Observation，第 333–396 行实现“决策 -> 工具 -> 观察 -> 再决策”循环。
- [`rag/external/client.py`](../../rag/external/client.py) 第 329–337 行强制调用工具并关闭 thinking，因此当前轨迹是结构化的 Action / Observation，而非显式 Thought / Action / Observation。
- [`rag/core/legal_rag.py`](../../rag/core/legal_rag.py) 第 275–399 行在进入证据 Agent 前完成确定性路由与检索前澄清；第 512–615 行先完成 Hybrid 检索和精排，再调用证据 Agent。
- [`rag/core/legal_rag.py`](../../rag/core/legal_rag.py) 第 685–791 行将 Agent 选出的证据打包后交给 MiniMind 生成，且不把生成结果送回证据 Agent。

## 命名建议

对外介绍时可以说：

> 项目实现了一个受控 Agentic RAG。其外部证据助理采用有界的 ReAct-style 工具循环，在同一上下文中根据检索 Observation 决定是否补查，再将审计友好的证据交给轻量法律模型作答。

不建议直接说“整个系统是标准 ReAct Agent”。这会忽略两个主要差异：当前没有显式 reasoning trace，且证据决策与最终作答由两个模型、两个独立阶段完成。这种受控分工不是缺陷；对“轻量法律模型 + 可审计 RAG”的项目目标，它比放开端到端 ReAct 的自由度更符合现有边界。

## 来源

- Shunyu Yao 等，[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)，arXiv:2210.03629，2022（v3，2023）。
- ReAct 作者项目页，[ReAct: Synergizing Reasoning and Acting in Language Models](https://react-lm.github.io/)。
- Datawhale Hello-Agents，[第四章 4.2 ReAct](https://hello-agents.datawhale.cc/#/./chapter4/%E7%AC%AC%E5%9B%9B%E7%AB%A0%20%E6%99%BA%E8%83%BD%E4%BD%93%E7%BB%8F%E5%85%B8%E8%8C%83%E5%BC%8F%E6%9E%84%E5%BB%BA?id=_42-react)。
