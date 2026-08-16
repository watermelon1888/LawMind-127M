# RAG-SFT v2 回答语义评审规则 v1

## 评审目标

本规则只判断模型在给定问题和可见 EvidencePackage 下生成的回答质量。评审者不得使用包外法律知识补全模型答案，也不得根据模型身份、自动指标、required GT 或 hard-negative 标签调整结论。

## 评审输入

评审者只能使用以下内容：

- 原始问题；
- 模型实际看到的有序 EvidencePackage；
- 模型输出的 `summary` 与 `citations`。

Evidence 编号只在当前题目内有效。不得推测未展示法条，也不得把法名、条号或检索位置本身视为语义支持。

## 回答可能性

`answerability` 使用三个值：

- `sufficient`：可见证据足以完整回答问题中的必要事项；
- `insufficient`：可见证据缺少完成回答所必需的法律依据；
- `uncertain`：仅靠当前材料无法可靠裁决，必须升级。

证据不足时仍应审核模型已经写出的每个主张是否受其引用证据支持，但不得把缺失证据造成的回答不完整静默归因给模型。

## 原子主张

一个原子主张是可以单独判断真假的最小法律结论。复合句应按其中可分离的主体、条件、例外、期限、数额、规范强度或法律后果拆分。

每个原子主张的 `support` 只能是：

- `fully_supported`：引用证据直接、完整支持该主张；
- `partially_supported`：引用证据只支持主张的一部分，或缺少决定边界的条件；
- `unsupported`：引用证据不支持该主张。

`supporting_evidence_ids` 只能列出模型实际引用且确实支持该主张的 Evidence 编号。

## 必要事项与完整性

`necessary_matters` 列出完整回答问题必须覆盖的最小事项，并逐项标记 `covered`。只有 `answerability=sufficient` 的记录进入回答完整率分母。证据不足时仍可列出能够确认的必要事项，但不得据此计算模型完整率。

## 引用审核

对模型给出的每个 citation 分别判断：

- `supports_any_claim`：是否至少直接支持一个 summary 原子主张；
- `necessary_for_summary`：删除该证据后，summary 是否失去必要支持。

引用未支持任何主张属于无关引用；支持主张但并非必要属于非最小引用；summary 中存在没有任何引用证据完整支持的主张属于引用不充分。

## 边界与包外事实

- `query_responsive`：回答是否正面回应问题，而不是复述无关证据；
- `legal_boundaries_preserved`：决定结论的条件、例外、期限、主体、数额、规范强度和法律后果是否得到保留；
- `unsupported_fact_absent`：回答是否没有添加问题和引用证据之外的事实或法律结论。

## 决策与归因

`review_decision` 使用 `pass`、`fail`、`escalate`。

`pass` 必须同时满足：证据足以回答、所有原子主张完整受支持、全部必要事项已覆盖、引用充分且最小、回答响应问题、法律边界保留、无包外事实、错误归因为 `none`。

`fail` 表示存在可以可靠指出的问题。`escalate` 表示当前材料不足以稳定裁决。错误归因使用：

- `none`：没有实质错误；
- `retrieval_blocked`：主要问题来自可见证据不足；
- `model_semantic_failure`：证据足够，但模型回答或引用错误；
- `both`：检索不足和模型错误同时存在；
- `uncertain`：无法可靠归因。

不得用单一主观总分替代结构化字段。理由必须简短指出决定性证据或缺失事项。
