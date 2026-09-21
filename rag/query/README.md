# Query 路由与证据助理

`rag/query` 负责三个运行时边界：确定性前置路由、检索前澄清判断，以及受控 Agentic 证据检查与补充检索。

## 当前运行流程

```text
用户问题
├─ 明确法名和条号 -> ANSWER(exact_lookup)
├─ 明确非法律模式 -> GENERAL_CHAT
├─ 历史法、不支持来源或不支持任务 -> CLARIFY 路由下的 REFUSED
└─ 其他请求 -> ANSWER(retrieval)
                  -> 外部模型判断 answer / clarify
                       ├─ clarify：返回一个简短澄清问题，不进入检索
                       └─ answer：继续检索
                  -> 原始 query Hybrid top-20
                  -> Cross-Encoder top-5 稳定基线
                  -> Agent 检查全部 top-20
                       ├─ top-5 已充分：不提升候选
                       ├─ top-20 有精排遗漏：提升一至五条候选
                       └─ top-20 确实不足：最多两次受控补充检索
                  -> 程序融合 CE 基线与 Agent 提升建议，固定得到 top-5
                  -> 证据构包，轻量模型通过 citations 选择实际依据
                  -> 本地生成答案并执行严格协议校验
                  -> 直接交付通过校验的 MiniMind 回答
```

确定性路由只处理稳定模式。普通法律问题在检索前由外部模型判断能否给出一般性或条件式回答；只有缺少决定适用制度的核心对象或范围时才澄清。澄清问题不会进入检索、证据 Agent 或 MiniMind 生成。

## 运行时模块

- `router.py`：生成 `RouteDecision`，只使用 `BusinessRoute` 与 `AnswerMode`。
- `exact_reference.py`：解析法名和条号引用。
- `query_assessment.py`：执行检索前 `answer / clarify` 判断并严格校验两字段 JSON。
- `agentic_retrieval.py`：执行单工具、最多两轮的补充检索循环和 query 边界校验。
- `answer_review.py`：保留外部答案审查协议，仅供离线评估，不进入生产回答链路。

证据助理同时看到原问题 Hybrid top-20、每条 Hybrid 排名和 Cross-Encoder top-5 排名。top-5 已充分时调用 `finalize_evidence` 并提交空数组；必需证据已在 top-20 时直接提交一至五个提升建议，不调用 `search_law`；只有全部 top-20 都缺少必需证据时才允许在原问题边界内补查，最多两次。补查结束后仍必须调用 `finalize_evidence`。最终 top-5 由程序确定性融合：保留 Cross-Encoder 前两条，最多纳入三条新的 Agent 建议，再按 Cross-Encoder 顺序补齐。Agent 不直接生成答案，也不能独占最终排序。

`EvidencePackager` 按融合顺序尝试装入完整法条；某条超出上下文预算时跳过它并继续尝试后续候选，不截断正文。轻量模型再通过 `citations` 选择实际支持答案的证据。

检索前判断协议固定为：

```json
{
  "decision": "answer | clarify",
  "clarification": "one short question | null"
}
```

约束如下：

- `answer` 的 `clarification` 必须为 `null`；
- `clarify` 只在问题缺少决定性适用范围、无法给出一般性或条件式回答时使用，并且只问一个关键事实；
- 输出必须是无 Markdown、无额外字段、无重复键的严格 JSON；
- 调用失败或协议非法时，core 返回 `PROCESSING_FAILED`，不回退为默认检索。

## 已退出运行链路的能力

- 法律任务分类；
- 检索后的程序回答条件判断；
- 动态澄清规划；
- 生产链路中的外部答案 `accept / clarify / adjust` 审查；
- 旧版 Query 增强协议仅保留为历史实验资产；当前运行时使用 `agentic_retrieval.py`。

`enhancement.py` 及相关训练代码仅保留为历史实验资产，不由当前应用装配和 core 调用。`answer_review.py` 只保留给离线评估使用。

## 验证

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-pytest.ps1 rag/query -q
```
