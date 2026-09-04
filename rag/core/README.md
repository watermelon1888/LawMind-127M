# core

## 模块定位

`core` 是现行法律 RAG 的公共结果契约和总编排层。它接收用户问题，连接 query、knowledge、retrieval、answering 与外部生成函数，执行四条路由之一，并始终返回结构统一、可渲染、可审计的 `LegalRAGResult`。

## 文件结构

```text
rag/core/
├── contracts.py       # 定义回答状态、证据、模型决策、渲染结果和统一接口
├── exact_lookup.py    # 连接 query 精确引用与 knowledge 确定性查条
├── semantic_search.py # 守卫并转发 semantic_search 路由
├── legal_rag.py       # 实现 CurrentLawRAG 四路总编排与失败映射
└── __init__.py        # 导出公共契约并延迟加载 CurrentLawRAG
```

## 当前实现概览

### 已实现

- `LegalRAG.answer(query) -> LegalRAGResult` 统一接口；
- `verified_lookup`、`retrieved_evidence`、`clarification_required`、`refused`、`processing_failed` 五种结果状态；
- 未作答原因与处理失败诊断码的分层契约；
- exact_lookup、semantic_search、clarify、refuse 四条执行路径；
- 精确查条一至三条原文展示，完全绕过生成模型；
- 语义检索、证据构包、模型生成、严格协议校验和引用渲染闭环；
- 仅在 semantic_search 中调用可选 Query 增强器，并记录结构化 `QueryEnhancementTrace`；
- 增强调用、超时、协议失败和增强检索失败时整份回退原始单 query 基线；
- 检索空结果、策略拒答和各处理阶段异常的安全映射；
- 所有路径统一使用 `RenderedAnswer` 和 `LegalRAGResult`。

### 尚未实现

- 项目级应用入口尚未负责创建 `ArticleRepository`、`SemanticRetriever`、`EvidencePackager`、token 计数器、真实 `generate` 函数和可选 `query_enhancer`；
- 尚未形成连接 MiniMind API 的正式端到端启动命令或服务入口；
- 当前没有独立的程序化 citation 语义蕴含验证器，模型归纳是否被引用法条完整支持仍需后续评估。

### 当前限制

- 只回答当前静态知识库中的现行法，不处理历史版本；
- 不提供多轮对话状态、Agent 工具循环或自动追问后继续执行；
- 处理失败只保留稳定阶段诊断码，不向外部结果暴露底层异常文本；
- 不执行通用自动重试，也不在 dense、BM25、reranker 或回答模型失败时静默降级；唯一例外是可选 Query 增强层按结构化 trace 整份回退已验证的原始单 query 基线；
- 语义回答能力受 retrieval 候选、上下文预算和回答模型格式遵循能力共同限制；
- 非空证据包不经过线上完整性判定，必要证据缺失时模型可能形成不完整回答；
- 生成端一次只返回一个原子法律结论，多事项问题暂不保证一次完整回答。

## 输入、输出与所在链路

```text
用户问题 -> CurrentLawRAG -> query / knowledge / retrieval / answering -> LegalRAGResult
```

| 上游依赖 | 需要的产出 | 用途 |
|---|---|---|
| 应用调用方 | 用户原始问题字符串 | 作为统一 `answer()` 输入 |
| query | `QueryDecision`、Query 增强协议与检索腿编译器 | 决定四条路径，并在语义路径验收增强输出 |
| knowledge | `ArticleRepository`、`LegalArticle` | 精确查条和 canonical 法条来源 |
| retrieval | `search()`、`search_many()` 返回的 `tuple[RankedArticle, ...]` | 提供单 query 或多 query 融合后的候选法条 |
| answering | 证据构包、协议校验和 renderer | 形成模型输入及公开回答 |
| 应用组装入口 | `generate(messages, temperature, max_tokens)` | 执行真实模型生成 |
| 应用组装入口 | 可选 `query_enhancer(query) -> raw_text` | 执行 Query 增强模型调用；未注入时保持单 query 基线 |

| 下游消费 | 接口 | 用途 |
|---|---|---|
| API、CLI 或其他应用入口 | `LegalRAG.answer(query) -> LegalRAGResult` | 获取统一结构结果和中文展示 |
| 评估与审计工具 | `LegalRAGResult` | 区分执行状态、证据、模型决策和失败阶段 |

## 核心对象与公开接口

### AnswerStatus

| 状态 | 含义 | 是否形成法律回答 |
|---|---|---|
| `VERIFIED_LOOKUP` | 程序已精确定位并展示用户指定法条 | 是，原文展示 |
| `RETRIEVED_EVIDENCE` | 检索证据经过模型归纳并通过协议校验 | 是，证据加简短归纳 |
| `CLARIFICATION_REQUIRED` | 输入不足以安全确定问题或引用 | 否 |
| `REFUSED` | 请求超出当前系统支持范围 | 否 |
| `PROCESSING_FAILED` | 检索、构包、生成或协议处理未完整完成 | 否 |

状态保持粗粒度，只回答“本次请求采用了哪种结果路径”。具体未作答原因和内部停止阶段由其他字段表达。

### UnansweredReason

| 原因 | 对应场景 |
|---|---|
| `NON_LEGAL` | 明确非法律问题 |
| `TIME_SENSITIVE` | 历史版本或过去适用问题 |
| `UNSUPPORTED_LEGAL_SOURCE` | 当前知识库不支持的规范或案例来源 |
| `UNSUPPORTED_LEGAL_TASK` | 文书代写、结果预测或规避监管等任务 |
| `NO_VERIFIABLE_EVIDENCE` | 历史兼容原因；当前空检索统一进入 `CLARIFY` |
| `CLARIFICATION_REQUIRED` | 当前输入需要补充或收窄 |
| `PROCESSING_FAILED` | 处理链路未完整完成 |

成功状态不能携带 `unanswered_reason`。澄清、拒答和处理失败必须携带与状态兼容的闭合原因。

### diagnostic_code

`diagnostic_code` 只允许出现在 `PROCESSING_FAILED` 中，用于指出请求停止在哪个阶段：

| 诊断码 | 停止阶段 |
|---|---|
| `exact_lookup_failed` | 精确引用解析或仓库访问异常 |
| `semantic_retrieval_failed` | dense、BM25、RRF、canonical 补全或 reranker 异常 |
| `evidence_packaging_failed` | 完整证据无法按预算构包 |
| `generation_failed` | 外部生成函数调用异常 |
| `output_validation_failed` | 模型输出不符合严格协议 |

诊断码服务归因和排查，不直接作为用户话术，也不替代 `AnswerStatus` 或 `UnansweredReason`。

### QueryEnhancementTrace

`QueryEnhancementTrace` 与最终回答状态正交，只记录可选增强层如何处理本次请求，不进入 `RenderedAnswer`：

| 字段 | 含义 |
|---|---|
| `status` | `NOT_ATTEMPTED`、`APPLIED` 或 `FALLBACK` |
| `failure_reason` | 仅 fallback 使用的稳定原因码 |
| `retrieval_queries` | 最终实际用于形成候选的唯一 query 列表 |

失败原因包括 `CALL_FAILED`、`TIMEOUT`、`INVALID_OUTPUT` 和 `ENHANCED_RETRIEVAL_FAILED`。`FALLBACK` 必须只携带原始 query；详细异常写日志，不复用只服务终止请求的 `diagnostic_code`，也不显示在中文回答中。合法 no-op 仍记为 `APPLIED`，其 `retrieval_queries` 只有原始 query。

### Evidence

| 字段 | 含义 |
|---|---|
| `law_name` | canonical 正式法名 |
| `article_no` | canonical 条号 |
| `content` | 完整法条正文 |

`Evidence` 是不含请求内编号的内部证据。exact_lookup 从 `LegalArticle` 直接转换；semantic_search 由 `EvidencePackager` 选择实际入包法条后创建。

### ModelAnswer

| 字段 | 含义 |
|---|---|
| `summary` | 模型输出的单行原子化法律结论 |
| `citations` | 支持该结论的当前证据包临时编号 |

`ModelAnswer` 只表示已经通过 answering 严格协议的模型决策。原始模型文本不会进入 `LegalRAGResult`。

### RenderedEvidence 与 RenderedAnswer

`RenderedEvidence` 是允许向用户展示的法条投影，字段与 `Evidence` 相同，但类型单独存在以区分内部证据和公开展示。

`RenderedAnswer` 使用统一三字段结构：

| 字段 | 法律回答 | 消息型结果 |
|---|---|---|
| `evidence` | 一至多条公开法条 | 空 |
| `summary` | 语义回答可包含；精确查条为空 | 空 |
| `message` | 空 | 澄清、拒答或失败消息 |

`to_text()` 是唯一固定中文序列化入口。统一的是数据结构和排版规则，不是所有路径都输出相同文案。

### LegalRAGResult

| 字段 | 含义 |
|---|---|
| `query` | 用户原始问题 |
| `status` | 本次请求的粗粒度结果状态 |
| `rendered_answer` | 可直接展示的程序化回答 |
| `unanswered_reason` | 未形成法律回答时的稳定原因 |
| `diagnostic_code` | 仅处理失败时的阶段标识 |
| `evidence` | 本次实际使用或已完成构包的 canonical 证据 |
| `model_answer` | 通过协议的模型决策；未调用模型时为空 |
| `query_enhancement` | 不参与中文渲染的 Query 增强状态与有效检索 query |

对象在创建时校验状态、原因、诊断码和字段组合，阻止“成功回答携带失败原因”“普通拒答携带诊断码”等矛盾结果。

### LegalRAG 与 CurrentLawRAG

```text
LegalRAG.answer(query: str) -> LegalRAGResult
```

`LegalRAG` 是调用方依赖的抽象接口；`CurrentLawRAG` 是当前现行法实现。应用显式注入 `ArticleRepository`、带 `search()` 的 retriever、`EvidencePackager` 和 `generate` 函数；启用增强时额外注入 `query_enhancer`，且 retriever 必须提供 `search_many()`。生成输出预算必须与构包预留预算一致。

## 总体执行流程

```text
CurrentLawRAG.answer(原始 query)
    -> [1] route_query()
        |- [2] CLARIFY -> 程序渲染澄清
        |- [3] REFUSE -> 映射原因并程序渲染拒答
        |- [4] EXACT_LOOKUP -> knowledge 核验 -> 程序展示法条原文
        `- SEMANTIC_SEARCH
            -> [5] 调用可选 query_enhancer 并生成 QueryEnhancementTrace
                |- 未配置 -> NOT_ATTEMPTED + 原始单 query
                |- 调用/超时/协议失败 -> FALLBACK + 原始单 query
                `- 成功 -> APPLIED + 唯一 retrieval_queries
            -> [6] retrieval
                |- APPLIED -> search_many()；异常时整份丢弃并重跑 search(原始 query)
                `- 其他状态 -> search(原始 query)
            -> [7] EvidencePackager 构造完整法条证据包
            -> [8] 回答模型生成并严格校验两字段 JSON
            -> [9] 程序映射 citation 并形成 LegalRAGResult
```

1. **接收问题并路由**
   - 位置：`legal_rag.py::CurrentLawRAG.answer()`、`query::route_query()`
   - 对象：原始 query -> `QueryDecision`
   - 行为：根据路由立即进入澄清、策略拒答、精确查条或语义检索。

2. **处理 clarify**
   - 位置：`legal_rag.py::CurrentLawRAG.answer()`、`answering::render_clarification()`
   - 对象：`CLARIFY` -> `LegalRAGResult`
   - 行为：优先调用外部澄清规划生成一个针对性问题；外部不可用或失败时返回固定澄清消息，不调用 knowledge 或 retrieval。

3. **处理 refuse**
   - 位置：`legal_rag.py::CurrentLawRAG.answer()`、`answering::render_refusal()`
   - 对象：`QueryReason` -> `UnansweredReason` -> `LegalRAGResult`
   - 行为：映射闭合原因并返回策略拒答，不调用检索或模型。

4. **处理 exact_lookup**
   - 位置：`legal_rag.py::_answer_exact_lookup()`、`exact_lookup.py::resolve_exact_lookup()`
   - 对象：`EXACT_LOOKUP` -> 一至三条 `LegalArticle` -> `Evidence`
   - 行为：解析引用并逐条调用仓库；全部命中时返回 `VERIFIED_LOOKUP`，任何正常未命中统一请求澄清。

5. **准备 Query 增强计划与 trace**
   - 位置：`legal_rag.py::_answer_semantic_search()`、`query::parse_and_validate_query_enhancement()`、`compile_retrieval_queries()`
   - 对象：`SEMANTIC_SEARCH` 原始 query -> `QueryEnhancementTrace`
   - 行为：只有注入增强器时才调用；合法输出编译为最多六条唯一 query，调用、超时或协议失败显式回退原始 query。其他三条路由保持 `NOT_ATTEMPTED`。

6. **执行 semantic_search**
   - 位置：`legal_rag.py::_answer_semantic_search()`、`semantic_search.py::resolve_semantic_search()`、`retrieval::SemanticRetriever.search_many()`
   - 对象：原始 query、trace 中的检索 query -> `tuple[RankedArticle, ...]`
   - 行为：增强成功时执行多 query；增强检索异常会丢弃全部增强候选并重跑原始单 query。基线也失败才进入 `semantic_retrieval_failed`；空候选进入回答条件判断并返回 `CLARIFY`。

7. **构造模型证据包**
   - 位置：`legal_rag.py::_answer_semantic_search()`、`answering::EvidencePackager.build()`
   - 对象：有序 `LegalArticle` -> `EvidencePackage`
   - 行为：选择预算内完整证据；构包异常返回 `evidence_packaging_failed`。

8. **调用生成模型并校验**
   - 位置：`legal_rag.py::_answer_semantic_search()`、`answering::build_answer_prompt()`、`parse_and_validate_answer()`
   - 对象：`EvidencePackage` -> 原始文本 -> `ModelAnswer`
   - 行为：以 `temperature=0` 和固定输出预算生成；调用异常或协议错误分别映射为对应诊断码。

9. **形成最终语义结果**
   - 位置：`legal_rag.py::_answer_semantic_search()`、`answering::render_semantic_answer()`
   - 对象：`ModelAnswer` -> `LegalRAGResult`
   - 行为：先通过回答条件判断，再调用两字段协议回答模型；校验通过后返回 `RETRIEVED_EVIDENCE`，并只展示被引用法条；不使用未经校准的检索分数阈值。

## 具体实现细节

### contracts.py

#### 状态、原因与诊断分层

- **要解决的问题**：用户行为、未作答业务原因和内部故障阶段如果使用同一字段，会形成大量含义重叠的状态。
- **当前选择**：`AnswerStatus` 表示结果路径，`UnansweredReason` 表示未作答原因，`diagnostic_code` 只表示处理失败阶段；`QueryEnhancementTrace` 独立描述可选增强层是否应用或回退。
- **选择理由**：最终结果、未作答原因、终止故障和非终止增强回退各自闭合，评估、界面和故障排查不会混用字段。
- **影响与限制**：所有对象组合都在 `LegalRAGResult.__post_init__()` 中约束；新增原因必须同时明确兼容状态和 renderer 行为。

#### 内部对象与公开对象分离

- **要解决的问题**：模型输入证据、模型决策和用户展示结构具有不同可信边界。
- **当前选择**：分别使用 `Evidence`、`ModelAnswer`、`RenderedEvidence`、`RenderedAnswer`，最终由 `LegalRAGResult` 聚合。
- **选择理由**：模型不能直接控制公开法条文本，renderer 只能从可信程序对象投影。
- **影响与限制**：对象数量略多，但每个对象只承担一种边界职责；调用方通常只需消费 `LegalRAGResult`。

#### 统一 RenderedAnswer

- **要解决的问题**：四条路径字段部分重叠，分别定义结果对象会让外部调用方产生重复分支。
- **当前选择**：法律回答使用 `evidence + optional summary`，未作答使用 `message`，两种形态互斥。
- **选择理由**：统一结构保留自然中文差异，同时阻止消息与法律结论同时出现。
- **影响与限制**：`to_text()` 提供固定样式；需要 JSON API 时可以直接读取字段而不解析中文文本。

### exact_lookup.py

#### 一至三条全有或全无

- **要解决的问题**：用户一次指定多条法条时，部分命中不能代表完整满足请求。
- **当前选择**：解析和仓库查找必须全部成功才返回 `FOUND`；任一正常未命中都返回不携带部分法条的澄清结果。
- **选择理由**：避免展示部分依据却让用户误认为请求已经完整处理。
- **影响与限制**：精确查条不自动降级为语义检索，也不猜测相近法律或条号。

#### 精确查条绕过模型

- **要解决的问题**：用户明确请求法条原文时，生成模型归纳会引入不必要的改写风险。
- **当前选择**：命中法条直接转换为 `Evidence` 并调用 `render_exact_lookup()`。
- **选择理由**：法名、条号和正文全部由程序确定，结果最可审计。
- **影响与限制**：该路径只展示原文，不回答开放式法律归纳。

### semantic_search.py

该文件只验证输入确实是 `SEMANTIC_SEARCH` 决策，再调用注入对象的单 query `search(query)`，供未启用增强和显式回退路径复用。增强成功时由 `legal_rag.py` 调用 retrieval 的 `search_many()`；core 仍不解释 dense、BM25、RRF、reranker 参数或候选分数。

### legal_rag.py

#### 显式依赖注入

- **要解决的问题**：总编排需要共享仓库、重量级检索器、token 预算和外部模型服务，但不能在导入时隐藏创建它们。
- **当前选择**：`CurrentLawRAG` 构造器显式接收四项必需依赖和可选 `query_enhancer`，并核对生成输出预算与 EvidencePackager 预算一致；启用增强时提前检查 retriever 支持 `search_many()`。
- **选择理由**：资源生命周期由应用入口控制，同一进程可以复用同一仓库和模型实例。
- **影响与限制**：当前没有默认 factory；调用方必须自行完成组装。

#### Query 增强允许显式回退

- **要解决的问题**：Query 增强是提高召回的可选层，其模型服务或协议失败不应让已经可用的原始单 query 基线失效。
- **当前选择**：调用失败、超时或输出非法时直接使用原始 query；增强检索任一腿异常时丢弃整个增强结果，再从头执行原始单 query。每次回退都写入 `QueryEnhancementTrace`，合法 no-op 则记为 `APPLIED`。
- **选择理由**：原始 query 基线本身是完整受支持路径；整份回退不会把部分检索候选伪装成完整增强结果。
- **影响与限制**：回退后可能失去增强带来的必要证据；程序不据此判断证据完整性，调用方可通过 trace 单独统计回退率。

#### 必需阶段不允许静默降级

- **要解决的问题**：某一路召回、reranker、构包或模型失败后继续使用残缺链路，可能产生表面正常但证据不完整的回答。
- **当前选择**：原始单 query 检索、构包、回答生成和协议校验等必需阶段异常都转换为 `PROCESSING_FAILED` 和稳定诊断码；不使用部分 dense/BM25 结果、不跳过协议校验。
- **选择理由**：法律回答优先保证证据链完整，处理失败比无提示降级更诚实。
- **影响与限制**：局部依赖暂时不可用时可用性会下降；后续若引入重试，必须保证不改变证据和协议语义。

#### 非空证据包直接回答

- **要解决的问题**：当前回答模型只学习充分证据下的原子法律结论，不承担线上证据充分性分类。
- **当前选择**：retrieval 返回非空候选且构包成功后先执行回答条件判断；通过后才调用模型，合法两字段输出形成回答，JSON 或字段不合法映射为 `PROCESSING_FAILED + output_validation_failed`。
- **选择理由**：程序不输出 `complete | partial | none`，模型也不决定是否应因证据不足拒答；请求范围拒答、空召回和服务故障继续由既有程序路径处理。
- **影响与限制**：必要证据未完整召回时，合法模型输出仍可能是不完整回答，这是当前明确接受的剩余风险。

#### 证据保留

- **要解决的问题**：生成或输出校验失败时，排查需要知道模型实际获得了哪些法条。
- **当前选择**：构包成功后的生成失败和输出校验失败仍在 `LegalRAGResult.evidence` 中保留本次证据包。
- **选择理由**：便于审计检索问题与生成问题，不把两者混为一体。
- **影响与限制**：公开失败话术不展示这些内部证据；是否记录到日志由应用层决定。

### __init__.py

该文件直接导出轻量公共契约，并通过 `__getattr__` 延迟导入 `CurrentLawRAG`。这样 answering 可以导入 core 契约而不触发总编排加载，也避免形成 core 与 answering 的循环导入。
