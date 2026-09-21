# 法律 RAG 流程分支场景矩阵

本文描述当前生产代码实际执行的法律 RAG 链路，并给出可重复的业务分支与异常分支验收场景。历史上的法律任务分类、Query 增强、生成后 `accept / clarify / adjust` 审查均不再是运行节点。本轮不覆盖 API 和前端。

## 1. 总体链路

```text
用户问题
├─ 空输入、纯标点、超长问题或超过三条精确引用
│  └─ 程序固定澄清 -> CLARIFICATION_REQUIRED
├─ 策略拒答
│  └─ 历史时点 / 不支持的法律来源 / 不支持的任务 -> REFUSED
├─ 明确非法律问题
│  └─ 外部模型通用回答 -> GENERAL_CHAT
├─ 一至三条明确法名和条号
│  ├─ 仓库命中 -> 程序直接返回完整法条 -> VERIFIED_LOOKUP
│  └─ 仓库未命中 -> 程序固定澄清 -> CLARIFICATION_REQUIRED
└─ 其他问题 -> RETRIEVAL
   ├─ 外部模型检索前判断 clarify
   │  └─ 返回一个关键澄清问题，不进入检索
   └─ 外部模型检索前判断 answer
      -> Hybrid top-20
      -> Cross-Encoder top-5
      -> 外部证据助理检查全部 top-20
         ├─ top-5 已充分 -> finalize_evidence([])
         ├─ top-20 中有精排遗漏 -> finalize_evidence([1～5 个 chunk_id])
         └─ top-20 确实缺少必需证据
            -> search_law(query)，最多两次
            -> finalize_evidence([0～5 个 chunk_id])
      -> 程序确定性融合为有序 top-5
      -> 在上下文预算内装入完整法条
      -> MiniMind 生成严格 JSON 回答
      -> 本地协议与引用校验
         ├─ 通过 -> 直接交付 RETRIEVED_EVIDENCE
         └─ 失败 -> PROCESSING_FAILED，不回退
```

外部模型不生成最终法律答案。`search_law` 只补充候选，`finalize_evidence` 只提交已观察到的证据提升建议；最终 top-5 仍由程序融合，最终引用由 MiniMind 从实际证据包中选择。

## 2. 验收口径

- `branch_pass`：实际 route、关键审计阶段和工具模式符合预期。它用于判断问题是否走到正确分支。
- `end_to_end_pass`：在 `branch_pass` 基础上，最终 status 也符合预期。检索工具走对但 MiniMind 输出协议失败时，前者通过、后者失败。
- 工具模式 `none`：不出现 `search_law` 或 `finalize_evidence`。
- 工具模式 `finalize_empty`：不调用 `search_law`，恰好调用一次 `finalize_evidence([])`。
- 工具模式 `finalize_promote`：不调用 `search_law`，恰好调用一次携带非空 chunk ID 的 `finalize_evidence`。
- 工具模式 `search_law`：调用 `search_law` 一至两次，最后恰好调用一次 `finalize_evidence`。
- 业务场景使用真实生产入口和真实模型；异常场景通过公开依赖 seam 注入，不通过断网或破坏本地文件制造故障。

场景事实来源为 [legal_rag_scenarios_v1.jsonl](../rag/eval/legal_rag_scenarios_v1.jsonl)，运行器为 [legal_rag_scenario_evaluation.py](../rag/eval/legal_rag_scenario_evaluation.py)。

## 3. 精确查条：5 题

共同预期：`ANSWER / VERIFIED_LOOKUP`；存在 `exact_lookup`、`evidence_selected`；不存在 `query_assessment`、`retrieval`、生成和 Agent 工具调用。

| 编号 | Query |
| --- | --- |
| EX-01 | `《中华人民共和国刑法》第75条规定了什么？` |
| EX-02 | `《中华人民共和国民法典》第143条规定了什么？` |
| EX-03 | `《中华人民共和国劳动合同法》第19条和第20条怎么规定？` |
| EX-04 | `《中华人民共和国消防法》第60条规定了什么？` |
| EX-05 | `《农业综合行政执法管理办法》第5条规定了什么？` |

## 4. 明确非法律通用回答：5 题

共同预期：`GENERAL_CHAT / GENERAL_CHAT`；存在 `general_chat_generation`；不存在 `query_assessment`、法律检索和证据。

| 编号 | Query |
| --- | --- |
| GC-01 | `今天天气怎么样？` |
| GC-02 | `推荐一道简单的家常菜。` |
| GC-03 | `Python 初学者应该怎么学习？` |
| GC-04 | `解一道数学题：2x+3=7。` |
| GC-05 | `请写一首关于秋天的诗。` |

## 5. 策略拒答：5 题

共同预期：`REFUSE / REFUSED`；不调用外部模型、检索器、Agent 或 MiniMind。

| 编号 | Query | 原因 |
| --- | --- | --- |
| RF-01 | `2020年修法前劳动法如何规定？` | `time_sensitive` |
| RF-02 | `司法解释对民间借贷利率怎么规定？` | `unsupported_legal_source` |
| RF-03 | `请帮我代写一份离婚起诉状。` | `unsupported_legal_task` |
| RF-04 | `请预测这个案件的胜诉率有多少。` | `unsupported_legal_task` |
| RF-05 | `怎样绕过市场监管部门的检查？` | `unsupported_legal_task` |

## 6. 检索前澄清：5 题

共同预期：外部问题判断返回 `clarify`，最终为 `CLARIFY / CLARIFICATION_REQUIRED`；存在 `query_assessment` 和 `clarification`；不存在 `retrieval`、Agent 与 MiniMind 生成。

| 编号 | Query |
| --- | --- |
| CL-01 | `资质证书丢了怎么补办？` |
| CL-02 | `备案应该去哪个部门办理？` |
| CL-03 | `申请材料需要提交几份？` |
| CL-04 | `审批流程一般要多长时间？` |
| CL-05 | `从业资格证要什么条件才能考？` |

## 7. top-5 已充分：5 题

共同预期：`ANSWER / RETRIEVED_EVIDENCE`；工具模式为 `finalize_empty`。证据助理检查全部 Hybrid top-20 后不提升证据、不补查。

| 编号 | Query |
| --- | --- |
| FE-01 | `外商投资企业分立后各家注册资本怎么定？` |
| FE-02 | `对拟被强制注销的公司登记有异议，怎么提出异议？要交什么材料？` |
| FE-03 | `什么是农作物种子质量监督抽查？由谁组织实施？` |
| FE-04 | `消防临时查封由谁决定？紧急情况下能当场查封吗？` |
| FE-05 | `医疗器械通用名称由几部分组成？核心词和特征词是什么？` |

## 8. top-20 内纠偏：5 题

共同预期：`ANSWER / RETRIEVED_EVIDENCE`；工具模式为 `finalize_promote`。至少提升一个 Hybrid top-20 内候选，不调用 `search_law`。

| 编号 | Query |
| --- | --- |
| FP-01 | `食用农产品批发市场开办者签订质量安全协议有什么要求，不签会怎么处罚？` |
| FP-02 | `项目审批部门超越权限审批资本金注入项目要承担什么责任？` |
| FP-03 | `机电产品出口招标委员会由哪些单位组成？` |
| FP-04 | `北交所股票注册申请文件受理之后还能改吗？出现重大事项怎么处理？` |
| FP-05 | `办法里说的餐饮服务经营活动包括哪些？适用范围是什么？` |

## 9. 补充检索：5 题

共同预期：`ANSWER / RETRIEVED_EVIDENCE`；工具模式为 `search_law`。补充 query 不得越过原问题边界，补查一至两次后必须调用一次 `finalize_evidence`。

| 编号 | Query |
| --- | --- |
| SL-01 | `高层民用建筑单位的消防设施不符合标准或未保持完好有效时，依据什么规定处罚，罚多少？` |
| SL-02 | `公安机关安全检查发现违禁品和涉案物品时怎么处理，实施扣押还要遵守哪些行政强制程序？` |
| SL-03 | `哪些情况属于国境口岸突发公共卫生事件，直属海关接报后多久要上报？` |
| SL-04 | `进出口商品复验方案要包括哪些内容？复验中发现原检验有错怎么办？` |
| SL-05 | `建设工程消防设计怎么审查？什么工程要报消防设计文件审查？` |

这些 query 是分支探针，不是永久保证模型一定调用 `search_law` 的固定规则。若真实运行改为 `finalize_empty` 或 `finalize_promote`，应记录为分支失败并分析候选变化，不得为了凑通过数修改审计结果。

## 10. 外部模型与工具协议异常：3 个

| 编号 | 注入条件 | 预期 |
| --- | --- | --- |
| EF-01 | 检索前问题判断返回非法 JSON | `PROCESSING_FAILED / query_assessment_failed`，不进入检索 |
| EF-02 | 未装配外部模型 | `PROCESSING_FAILED / query_assessment_failed`，不静默回退 |
| EF-03 | 证据助理调用未知工具或不按协议结束 | `PROCESSING_FAILED / agentic_retrieval_failed`，不进入生成 |

## 11. 检索与证据处理异常：3 个

| 编号 | 注入条件 | 预期 |
| --- | --- | --- |
| RE-01 | Hybrid 检索器抛异常 | `PROCESSING_FAILED / semantic_retrieval_failed` |
| RE-02 | Cross-Encoder 精排抛异常或返回空结果 | `PROCESSING_FAILED / reranking_failed` |
| RE-03 | 所有候选完整法条均无法装入上下文预算 | `PROCESSING_FAILED / evidence_packaging_failed` |

检索成功但返回零候选不是异常，而是固定澄清：`CLARIFY / CLARIFICATION_REQUIRED`，且不进入 Agent 和生成。

## 12. 本地生成与回答协议异常：3 个

| 编号 | 注入条件 | 预期 |
| --- | --- | --- |
| GF-01 | MiniMind 生成抛异常 | `PROCESSING_FAILED / generation_failed` |
| GF-02 | MiniMind 返回非严格 JSON | `PROCESSING_FAILED / output_validation_failed` |
| GF-03 | MiniMind 引用证据包外的 `E<n>` | `PROCESSING_FAILED / output_validation_failed` |

上述失败均默认报错，不交付未经协议校验的候选答案，也不调用外部模型代答。

## 13. 执行方式

异常注入与场景契约测试：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-pytest.ps1 `
  rag/core/test_legal_rag.py `
  rag/query/test_agentic_retrieval.py `
  rag/eval/legal_rag_scenario_evaluation_test.py
```

35 条真实 query 验收：

```powershell
conda run --no-capture-output -n minimind python -m rag.eval.legal_rag_scenario_evaluation `
  --device cuda:0 `
  --details rag/eval/legal_rag_scenario_results_v2.json
```

真实验收只输出 route、status、审计阶段、工具计数、诊断码和耗时，不使用外部模型评价最终答案质量。运行失败时保留结果，不覆盖为预期值。

## 14. 本轮验收结果

运行时间：2026-09-20。有效结果见 [legal_rag_scenario_results_v1.json](../rag/eval/legal_rag_scenario_results_v1.json)。首次沙箱内运行因外部 API 网络连接失败，不纳入业务结论；联网重跑使用同一代码、场景与 `deepseek-v4-flash`。后续重跑必须使用新的版本路径，不覆盖本次结果。

确定性异常与场景契约测试：`28 passed`。第 10～12 节的 9 个故障注入场景均得到预期诊断码，没有发生静默回退。

真实 query 总结果：

| 类别 | 分支通过 | 端到端通过 | `search_law` | `finalize_evidence` |
| --- | ---: | ---: | ---: | ---: |
| 精确查条 | 5 / 5 | 5 / 5 | 0 | 0 |
| 非法律通用回答 | 5 / 5 | 5 / 5 | 0 | 0 |
| 策略拒答 | 5 / 5 | 5 / 5 | 0 | 0 |
| 检索前澄清 | 5 / 5 | 5 / 5 | 0 | 0 |
| top-5 已充分 | 2 / 5 | 2 / 5 | 0 | 5 |
| top-20 内纠偏 | 5 / 5 | 5 / 5 | 0 | 5 |
| 补充检索 | 4 / 5 | 4 / 5 | 4 | 5 |
| **合计** | **31 / 35** | **31 / 35** | **4** | **15** |

4 条未通过分支预期的样本均成功返回 `ANSWER / RETRIEVED_EVIDENCE`，没有 `diagnostic_code`：

| 样本 | 预期工具模式 | 实际工具模式 | 实际行为 |
| --- | --- | --- | --- |
| FE-01 | `finalize_empty` | `finalize_promote` | 未补查，提升 2 条 top-20 候选 |
| FE-02 | `finalize_empty` | `finalize_promote` | 未补查，提升 2 条 top-20 候选 |
| FE-05 | `finalize_empty` | `finalize_promote` | 未补查，提升 1 条 top-20 候选 |
| SL-05 | `search_law` | `finalize_promote` | 未补查，提升 1 条 top-20 候选 |

结论：确定性路由、检索前澄清、失败即报错和最终交付链路均通过；`finalize_promote` 与 `search_law` 能按协议工作。自然语言 query 对 Agent 工具模式不是稳定的一一映射：同一 query 在候选、模型服务或证据判断变化后可以合法改走另一条证据分支。因此这 15 条 Agent 样本适合作为行为探针，不应作为要求某个 query 永久调用某个工具的硬编码回归测试。需要稳定验证工具协议时，应继续使用公开 seam 的确定性测试；真实 query 验收负责监测分支分布和异常漂移。
