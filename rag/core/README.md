# 法律 RAG 总编排

`rag/core` 定义统一结果契约，并由 `CurrentLawRAG.answer(query)` 编排一次从问题到最终结果的完整处理。

## 统一流程

```text
route_query
├─ GENERAL_CHAT -> 外部模型通用回答
├─ CLARIFY      -> 策略拒答或固定澄清
└─ ANSWER
    ├─ exact_lookup -> 程序精确查条 -> 直接交付
    └─ retrieval
         -> 外部模型检索前判断 answer / clarify
              ├─ clarify -> 返回一个简短澄清问题，不进入检索
              └─ answer  -> 继续检索
         -> 原始 query Hybrid top-20
         -> Cross-Encoder top-5 稳定基线
         -> 外部 Agent 检查全部 top-20
              ├─ top-5 已充分 -> finalize_evidence([])
              ├─ top-20 有精排遗漏 -> 提升一至五条候选
              └─ top-20 确实不足 -> 最多两次受控补充检索
         -> 程序保留 CE 前两条、最多纳入三条 Agent 提升建议并补齐 top-5
         -> 证据构包
         -> 本地 RAG 模型生成答案并选择 citations
         -> 本地 JSON、引用和证据边界校验
         -> 直接交付通过校验的 MiniMind 回答
```

## 安全边界

- 精确查条不调用回答模型和外部答案审查。
- 检索前判断为 `clarify` 时不调用检索器、证据 Agent 或 MiniMind。
- 检索为空时不生成无证据答案，直接返回固定澄清。
- 生成失败返回 `PROCESSING_FAILED + diagnostic_code`；本地协议校验失败还会在 `candidate_answer` 中保留 127M 原始输出。
- 检索前判断或证据 Agent 调用失败、超时、服务不可用或输出非法时，返回 `PROCESSING_FAILED`，不静默回退。
- MiniMind 输出只有通过本地严格 JSON、引用和证据边界校验后才会交付；生产链路不再调用外部答案审查或调整。
- 通用对话与法律回答共用 `LegalRAGResult`，但通用对话不携带法律证据。

## 主要契约

- `BusinessRoute`：`ANSWER`、`CLARIFY`、`GENERAL_CHAT`。
- `AnswerMode`：`EXACT_LOOKUP`、`RETRIEVAL`。
- `AnswerStatus`：对外可见的具体处理状态。
- `LegalRAGResult`：统一承载展示结果、127M 候选答案、证据、诊断码和 `AuditTrace`。

`LegalRAGResult.query_enhancement` 记录 Agentic 检索实际使用的 query 序列；无补查时为 `NOT_ATTEMPTED`，成功补查时为 `APPLIED`。

## 审计事件

核心事件按真实执行顺序记录，主要包括：

- `request_received`、`route_decided`；
- `exact_lookup`、`evidence_selected`；
- `retrieval`、`reranking`、`agentic_retrieval`、`evidence_packaging`；
- `answer_generation`、`protocol_validation`；
- `query_assessment`、`clarification`；
- `general_chat_generation`、`final_render`、`completed`。

已退出运行时审计的阶段包括 `external_analysis`、`route_revised`、`task_classification`、`answerability`、`clarification_planning`、`answer_review` 和 `answer_adjustment`。

## 验证

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-pytest.ps1 rag/core rag/app -q
```
