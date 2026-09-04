# query

## 模块定位

`query` 负责法律 RAG 的确定性前置路由，以及语义检索路径使用的 Query 增强协议。路由先选择 `exact_lookup`、`semantic_search`、`clarify`、`refuse` 四条路径之一；只有 `semantic_search` 才由 core 调用可选增强器，并使用本模块严格解析增强 JSON、编译多条检索 query。`REFUSE` 只处理请求范围和任务边界，不表示程序判断了检索证据是否完整。

## 文件结构

```text
rag/query/
├── router.py           # 定义路由契约并按固定优先级判断执行路径
├── exact_reference.py  # 从原始问题中提取一至三条完整法条引用
├── enhancement.py      # 定义增强协议、严格校验和检索腿编译
├── clarification.py    # 定义外部澄清规划协议与严格校验
└── __init__.py         # 导出 query 的公开接口
```

## 当前实现概览

### 已实现

- 四条互斥执行路径及不可变 `QueryDecision`；
- 四种闭合策略拒答原因；
- 非法律、历史版本、未收录法律来源和不支持任务的规则识别；
- 信息不足、范围过宽、混合意图和超长查询的澄清路由；
- 一至三条法条引用的确定性提取，支持并列条号和法名继承；
- 固定 `rewrite / expansion_terms / subqueries` 三字段增强协议；
- 固定 `missing_information / question` 两字段澄清规划协议；
- 增强输出的全有或全无校验、空白规范化和稳定去重；
- 原始 query、rewrite、单条术语扩展 query 和最多三条 subquery 的固定编译；
- 对路由顺序和对象组合执行运行时约束。



### 尚未实现

当前没有已确认但尚未实现的 query 内部能力。部门预测、HyDE 和多轮指代消解不进入当前增强协议或正式运行链路。

### 当前限制

- 路由基于闭合规则和关键词，不具备开放域语义分类能力；
- 只处理单轮文本，不读取历史对话；
- 澄清判断只能覆盖已编码的典型信息缺失形式，不能保证识别所有宽泛问题；
- 精确引用解析只支持一至三条完整引用，超过三条或法名归属不清时要求澄清；
- query 只解析引用语法，不证明法律或条号真实存在；
- 增强协议只能机械校验 JSON、字段和数量，不能证明改写保持了法律语义；
- query 不接收检索候选，也不输出 `complete | partial | none` 等证据完整性状态。



## 输入、输出与所在链路

```text
用户原始问题 -> 确定性路由 -> semantic_search 时可选 Query 增强 -> core
```


| 上游依赖  | 需要的产出     | 用途            |
| ----- | --------- | ------------- |
| 应用调用方 | 用户原始问题字符串 | 保留用户表达并执行前置路由 |



| 下游消费      | 接口                                                                                | 用途                     |
| --------- | --------------------------------------------------------------------------------- | ---------------------- |
| core 总编排  | `route_query(query) -> QueryDecision`                                             | 选择精确查条、语义检索、澄清或拒答      |
| core 精确查条 | `extract_exact_references(query) -> ExactReferenceResult`                         | 提取待 knowledge 核验的法名与条号 |
| core 语义检索 | `parse_and_validate_query_enhancement(raw_text)`、`compile_retrieval_queries(...)` | 验收增强输出并编译检索腿           |




## 核心对象与公开接口



### QueryRoute


| 枚举值               | 含义                   | 后续路径                    |
| ----------------- | -------------------- | ----------------------- |
| `EXACT_LOOKUP`    | 问题中含有可识别法条引用         | 解析法名与条号后进入 knowledge 查找 |
| `SEMANTIC_SEARCH` | 法律问题信息足以检索，但没有明确法条引用 | 进入 retrieval 召回与排序      |
| `CLARIFY`         | 当前输入不足以安全确定任务或范围     | 返回统一澄清提示                |
| `REFUSE`          | 问题明确落在当前能力范围之外       | 按闭合原因返回拒答               |


`scenario_advice`、`semantic_search` 等旧查询类型不再作为运行时分类。当前只保留会改变执行路径的 `QueryRoute`。

### QueryReason

`QueryReason` 只服务 `REFUSE` 路由，不能附着在其他路径上。它解释用户请求为何不进入回答链路，不表达 semantic search 返回的法条能否完整支持回答。


| 枚举值                        | 触发范围                |
| -------------------------- | ------------------- |
| `NON_LEGAL`                | 明确的非法律问题            |
| `TIME_SENSITIVE`           | 历史版本内容或过去事实应适用哪版法律  |
| `UNSUPPORTED_LEGAL_SOURCE` | 当前知识库不支持的规范来源或案例来源  |
| `UNSUPPORTED_LEGAL_TASK`   | 法律文书代写、结果预测或规避监管等任务 |




### QueryDecision

字段与枚举的类型对应关系如下：

```text
QueryDecision.route  : QueryRoute
QueryDecision.reason : QueryReason | None
```


| 字段       | 含义         | 关键约束              |
| -------- | ---------- | ----------------- |
| `query`  | 未改写的用户原始问题 | 必须是字符串            |
| `route`  | 唯一执行路径     | 类型为 `QueryRoute`  |
| `reason` | 策略拒答原因     | 类型为 `QueryReason` |


公开入口为：

```text
route_query(query) -> QueryDecision
```



### QueryEnhancement

`QueryEnhancement` 是通过严格协议校验并完成最小规范化的增强结果，不包含原始 query、路由、回答、置信度或检索权重：


| 字段                | 含义                | 关键约束    |
| ----------------- | ----------------- | ------- |
| `rewrite`         | 覆盖原问题全部事项的规范化等价改写 | 必填、非空   |
| `expansion_terms` | 弥补口语与法律术语差异的短语    | 去重后最多四项 |
| `subqueries`      | 真实多事项问题的完整原子问句    | 去重后最多三项 |


```text
build_query_enhancement_prompt(query) -> system/user messages
parse_and_validate_query_enhancement(raw_text) -> QueryEnhancement
compile_retrieval_queries(original_query, enhancement) -> tuple[str, ...]
```

模型输出必须是字段及顺序精确匹配的单个 JSON 对象。任一字段非法都会抛出 `QueryEnhancementProtocolError`，由 core 记录 `INVALID_OUTPUT` 并回退原始 query；本模块不会尝试修复 Markdown、附加解释或半份合法输出。

### ClarificationPlan

`ClarificationPlan` 用于在已有信息不足时，由外部模型规划一个最关键的补充问题。它不生成法律结论，也不改变业务路由。

```text
build_clarification_prompt(query, task_type, evidence)
parse_and_validate_clarification(raw_text)
plan_clarification(query, task_type, evidence, external_llm)
```

外部输出必须严格包含 `missing_information` 和 `question` 两个字段；缺失信息最多三项，问题只能有一个。调用失败和协议失败分别抛出受控异常，由 core 后续步骤决定是否使用程序固定模板降级。

### ExactReference 与 ExactReferenceResult

`ExactReference` 保存尚未经过知识库真实性核验的 `law_name` 和原始 `article_no`。`ExactReferenceResult` 采用原子结果：成功时包含一至三条完整引用；任何法名归属不清或数量超限都会返回 `CLARIFICATION_REQUIRED`，不携带部分引用。

支持接口为：

```text
extract_exact_references(query) -> ExactReferenceResult
count_article_references(query) -> int
contains_article_reference(query) -> bool
```

这些解析接口由 core 的精确查条连接层使用；法律与条号是否存在，仍由 [knowledge](../knowledge/README.md) 的 `ArticleRepository` 核验。

## 总体执行流程



### 核心对象如何配合

`QueryDecision` 是路由阶段交给 core 的唯一决策对象。它不保存增强结果或检索结果，只保留原始问题并指定执行路径。`QueryEnhancement` 是另一条独立契约，只会在 `SEMANTIC_SEARCH` 已经确定后创建，不能反过来改变路由。

```text
CurrentLawRAG.answer(query)
    -> route_query(query)
        -> [1] 接收并保留原始问题
        -> [2] 提取法律、非法律与任务边界信号
        -> [3] 按固定优先级判断是否需要澄清或策略拒答
            |- 是：创建 QueryDecision(route=CLARIFY 或 REFUSE)
            |      -> [5] 交给 core 直接渲染澄清或拒答
            `- 否：进入 [4] 区分精确查条与语义检索
                   -> 创建 QueryDecision(route=EXACT_LOOKUP 或 SEMANTIC_SEARCH)
                   -> [5] 交给 core 执行对应路径
                       |- EXACT_LOOKUP： [6] 调用 exact_reference，再由 knowledge 核验
                       `- SEMANTIC_SEARCH：进入 [7] 可选 Query 增强

    [7] core 调用注入的 query_enhancer（未配置时跳过）
        -> [8] enhancement.py 严格解析三字段 JSON
            |- 调用、超时或协议失败：显式记录 fallback，只保留原始 query
            `- 成功：形成 QueryEnhancement
                -> [9] 编译并稳定去重检索 query
                    -> retrieval.search_many()、EvidencePackager 和回答模型
```

四条分支的衔接如下：

```text
QueryDecision
    |- route=CLARIFY
    |   -> core 直接返回 clarification_required
    |
    |- route=REFUSE, reason=NON_LEGAL / TIME_SENSITIVE / ...
    |   -> core 不调用 knowledge、retrieval 或模型
    |   -> 映射为 refused + 对应 UnansweredReason
    |
    |- route=EXACT_LOOKUP
    |   -> exact_reference 解析法名和条号
    |   -> knowledge 核验并程序化展示完整法条
    |
    `- route=SEMANTIC_SEARCH
        -> 可选 Query 增强 -> 多 query retrieval -> EvidencePackager -> 模型按证据回答
```

1. **接收并保留原始问题**
  - 位置：`router.py::route_query()`
  - 对象：用户输入 -> 原始 `query`
  - 行为：要求输入为字符串；空白、无有效字符或超过 256 个紧凑字符的输入进入澄清。
2. **提取法律与非法律信号**
  - 位置：`router.py::_has_legal_signal()`、`_has_non_legal_signal()`、`_is_unsupported_task()`
  - 对象：紧凑查询 -> 规则信号
  - 行为：识别法条、法律语汇、明确非法律主题和不支持的法律任务。
3. **按固定优先级决定是否提前结束 query 路由**
  - 位置：`router.py::route_query()`
  - 对象：规则信号 -> `CLARIFY | REFUSE | 继续判断`
  - 行为：混合意图或信息不足返回 `CLARIFY`；任务边界、法律来源、历史时效和明确非法律问题返回 `REFUSE + QueryReason`；其他问题继续判断。
4. **区分精确查条与语义检索**
  - 位置：`router.py::route_query()`
  - 对象：剩余法律问题 -> `QueryDecision(route=EXACT_LOOKUP | SEMANTIC_SEARCH)`
  - 行为：存在一至三条法条引用时创建 `EXACT_LOOKUP` 决策；引用超过三条时创建 `CLARIFY` 决策；其余问题创建 `SEMANTIC_SEARCH` 决策。
5. **交给 core 执行**
  - 位置：`core/legal_rag.py::CurrentLawRAG.answer()`
  - 对象：`QueryDecision` -> 对应执行路径
  - 行为：core 消费 `route`；`REFUSE` 分支同时消费 `reason`，并映射为最终 `UnansweredReason`。
6. **仅在精确查条分支提取完整引用**
  - 位置：`core/exact_lookup.py::resolve_exact_lookup()` -> `exact_reference.py::extract_exact_references()`
  - 对象：`EXACT_LOOKUP` 决策的原始 `query` -> `ExactReferenceResult`
  - 行为：core 调用解析器展开并列条号、确定法名归属，再由 knowledge 核验；无法形成一至三条完整引用时返回澄清结果。
7. **仅在语义检索分支调用可选增强器**
  - 位置：`core/legal_rag.py::CurrentLawRAG._answer_semantic_search()`
  - 对象：原始 query -> 增强模型原始文本
  - 行为：core 调用显式注入的 `query_enhancer`；其他三条路由不会调用。未配置增强器时保留原始单 query 基线。
8. **严格验收增强输出**
  - 位置：`enhancement.py::parse_and_validate_query_enhancement()`
  - 对象：模型原始文本 -> `QueryEnhancement`
  - 行为：字段、顺序、类型、非空性和数量任一不合法时整份作废，不使用部分字段；core 显式记录回退原因。
9. **编译固定检索腿**
  - 位置：`enhancement.py::compile_retrieval_queries()`
  - 对象：原始 query、`QueryEnhancement` -> 最多六条唯一 query
  - 行为：依次加入原始问题、rewrite、一次性拼接全部术语的扩展 query 和最多三条 subquery，规范化后保留首次出现顺序。



## 具体实现细节



### router.py



#### 确定性四路路由

- **要解决的问题**：检索、模型生成、澄清和策略拒答具有完全不同的安全边界，不能让生成模型临时决定是否执行。
- **当前选择**：使用固定规则产生唯一 `QueryDecision`，core 只按路由执行。
- **选择理由**：结果可审计、低成本，并能在模型服务不可用时保持边界行为稳定。
- **影响与限制**：新增表达方式必须显式补充规则；未覆盖的法律表达默认进入 `semantic_search`，不代表规则已经理解其全部含义。



#### 判断优先级

- **要解决的问题**：同一句话可能同时包含法律信号、非法律任务、历史时间和法条引用。
- **当前选择**：混合意图澄清优先于拒答；不支持任务、来源和历史问题优先于精确查条；信息不足优先于普通语义检索。
- **选择理由**：先收敛输入和能力边界，避免一个法条编号掩盖不支持的真实任务。
- **影响与限制**：优先级是策略的一部分，改变顺序会直接改变用户行为，不能把规则简单视为无序关键词集合。



#### clarify 与 semantic_search

- **要解决的问题**：宽泛但可检索的问题和缺少关键事实的问题都可能没有法条编号。
- **当前选择**：只对已识别的范围过宽、指代缺失、金额性质不清、主题残缺、案件事实不足、缺少待审文档和条号不清等模式澄清，其余法律问题进入语义检索。
- **选择理由**：完全依赖规则识别“是否需要收窄”不可行；首版只拦截高置信度缺失形式，避免过度澄清降低覆盖率。
- **影响与限制**：类似“老板不给加班费怎么办”这类具有明确争议主题的问题会直接检索；模型只能依据实际证据回答，不能补造案件事实。



#### 策略拒答原因

- **要解决的问题**：外部话术、运行状态和拒答原因如果混在自由文本中，无法稳定审计。
- **当前选择**：query 只产生四个闭合 `QueryReason`，core 再映射为统一的 `UnansweredReason`。
- **选择理由**：query 负责请求边界，core 负责全链路结果契约，两层职责清晰。
- **影响与限制**：检索空候选和处理失败不是 query 阶段原因，不进入 `QueryReason`；非空候选是否完整也不由 query 判断。



#### 历史法律问题

- **要解决的问题**：当前知识库只有现行有效文本，直接回答“2020 年当时如何规定”会把现行法错误替代为历史法。
- **当前选择**：识别明确历史版本短语、截止年份、带年份的法条引用和过去事实适用提示，路由为 `TIME_SENSITIVE`。
- **选择理由**：项目已弱化历史版本功能，明确拒答比猜测版本更可靠。
- **影响与限制**：年份仅用于描述法律名称或公布背景时可能不是历史适用问题，因此规则排除了部分制定年份上下文。



### exact_reference.py



#### 一至三条原子引用

- **要解决的问题**：精确查条需要明确每个条号属于哪部法律，部分解析成功不能安全展示为完整结果。
- **当前选择**：最多提取三条；任一引用缺少法名、出现多个候选法名或数量超限时，整个结果要求澄清。
- **选择理由**：避免把部分法条误当作用户全部请求，也防止精确查条演变为超长法条拼接。
- **影响与限制**：确实需要四条以上法条的请求应先收窄或进入更适合的语义问题，不在精确展示路径中硬扩容。



### enhancement.py



#### 全有或全无的严格协议

- **要解决的问题**：增强模型可能输出 Markdown、附加解释、缺字段、错误类型或只有部分字段可用的对象。
- **当前选择**：严格拒绝重复键、非标准 JSON 常量、字段或顺序不匹配、空字符串及数组超限；整份输出只能全部接受或全部回退。
- **选择理由**：部分拼接会产生无法审计的半份检索计划，且容易掩盖模型格式能力问题。
- **影响与限制**：格式合法但语义漂移无法靠最小运行时校验识别，仍需通过 Query-SFT 审核和离线检索评估降低。



#### 固定检索腿与稳定去重

- **要解决的问题**：逐术语组合或给每条 subquery 再附加全部术语会造成检索腿指数式膨胀，并让重复改写在 RRF 中重复投票。
- **当前选择**：固定编译原始 query、rewrite、一次性术语扩展 query 和最多三条 subquery；只折叠空白并按规范化后的精确字符串稳定去重。
- **选择理由**：最多六腿、顺序可复现，不引入额外 embedding 相似度阈值；原始 query 永远位于第一条。
- **影响与限制**：近义但文本不同的 query 仍会分别投票；首版接受这一点，不做不透明的语义去重。

