# answering

## 模块定位

`answering` 负责把有序完整法条转换为受上下文预算约束的模型证据包，构造法律回答 prompt，严格校验模型的两字段 JSON，并把可信内部对象渲染为稳定中文回答。程序控制证据身份、格式和引用映射；模型收到非空证据包后生成一至三个短句，用于完整回答紧密相关事项，并选择实际支持这些结论的最小充分证据集合。

Query 增强完全位于 answering 上游。无论候选来自原始单 query 还是多 query 融合，answering 只能看到用户原始问题和最终有序 canonical 法条；rewrite、expansion terms、subqueries、RRF/rerank 分数和 `QueryEnhancementTrace` 均不进入 `EvidencePackage`、prompt 或中文渲染。

## 文件结构

```text
rag/answering/
├── evidence.py    # 构造完整法条证据包并执行 prompt 预算选择
├── protocol.py    # 定义固定 prompt 与严格两字段模型输出协议
├── token_count.py # 使用实际 MiniMind chat template 计算完整 prompt 长度
├── render.py      # 渲染精确查条、语义回答、澄清、拒答和处理失败
└── __init__.py    # 导出 answering 的公开接口
```

## 当前实现概览

### 已实现

- 将有序 `LegalArticle` 投影为不可变 `EvidencePackage`；
- 在真实 prompt token 预算内选择 retrieval 前五条完整法条的最大有序前缀；
- 为单次模型请求分配 `E1`、`E2` 等临时证据编号；
- 构造固定 system prompt 和紧凑 user JSON；
- 严格解析并校验 `summary / citations` 两字段 JSON；
- 只展示模型实际引用的法条，并提供统一中文结构和范围提示；
- 为精确查条、澄清、策略拒答和处理失败提供程序化渲染。



### 尚未实现

当前没有已确认但尚未实现的 answering 内部能力。具体生成模型和 API 调用通过 core 注入的 `generate` 函数完成，不由 answering 创建。

### 当前限制

- 证据包考虑 retrieval 返回的全部前五条候选，并按原顺序选择完整前缀；
- 不启用 excerpt selector，每条模型证据只有一个完整法条正文；
- 首条完整法条也无法装入上下文时直接构包失败，不裁剪正文；
- 默认预留 150 个输出 token；旧 RAG-SFT 的 833 条目标投影为两字段后，目标正文 token 的 p95、p99、最大值分别为 108、125、143。150 比历史最大目标多 7 tokens，但当前仍缺少两字段回答模型的真实生成长度和触顶率，因此该值需要在模型评估中重点复核；实际上下文上限由应用创建 `EvidencePackager` 时传入；
- prompt 协议要求模型稳定输出严格 JSON，格式不合法会进入处理失败而不是尝试修复；
- 模型一次只输出一个原子化法律结论，多事项问题暂不保证在一次响应中完整回答；
- 程序不判断非空证据包是完整、部分还是无支持；必要证据缺失时，模型可能形成不完整回答；
- 当前中文渲染是固定程序结构，不提供自由格式回答样式。



## 输入、输出与所在链路

```text
用户原始问题 -> core 可选 Query 增强 -> retrieval 最终排序 -> 有序 LegalArticle
    -> EvidencePackage（仍保存原始问题） -> prompt -> ModelAnswer -> RenderedAnswer
```


| 上游依赖                  | 需要的产出                            | 用途                     |
| --------------------- | -------------------------------- | ---------------------- |
| retrieval 或 core 精确查条 | 有序、完整的 `LegalArticle`            | 构造 canonical 证据并保持候选顺序 |
| MiniMind tokenizer    | 与运行时一致的 chat template 和 token 编码 | 计算完整 prompt 长度         |
| core 注入的生成函数          | 模型原始文本输出                         | 进入严格协议解析               |



| 下游消费 | 接口                                                    | 用途                        |
| ---- | ----------------------------------------------------- | ------------------------- |
| core | `EvidencePackager.build()`                            | 得到可安全送入模型的证据包             |
| core | `build_answer_prompt()`、`parse_and_validate_answer()` | 构造模型输入并取得可信 `ModelAnswer` |
| core | `render_*()`                                          | 生成所有执行路径共用的公开中文结构         |




## 核心对象与公开接口



### EvidencePackage


| 字段         | 含义                      | 关键约束           |
| ---------- | ----------------------- | -------------- |
| `query`    | 未改写的用户问题                | 非空字符串          |
| `evidence` | 有序 canonical `Evidence` | 至少一条，法名与条号不能重复 |


`Evidence` 由 [core](../core/README.md) 定义，包含 `law_name`、`article_no`、`content`。`EvidencePackage.to_model_json()` 为每条证据临时分配 `E1...En`，并把完整正文放入单元素 `excerpts` 数组；临时编号不会写回 knowledge 或检索索引。

### EvidencePackager

```text
EvidencePackager.build(query, articles) -> (EvidencePackage, prompt_tokens)
```

构造器接收 `context_limit`、`max_output_tokens` 和 `count_prompt_tokens`。`build()` 考虑 retrieval 返回的前五条候选，返回能完整装入预算的最大有序前缀及实际 prompt token 数。重复候选、无效计数或首候选超预算会抛出 `EvidencePackagingError`。

### ModelAnswer

`ModelAnswer` 由 core 定义，是通过协议校验后的原子法律结论：


| 字段          | 含义             | 关键约束                         |
| ----------- | -------------- | ---------------------------- |
| `summary`   | 单个原子化法律结论     | 非空、单行，不包含法名、条号或 Markdown     |
| `citations` | 支持该结论的证据编号集合 | 非空、非重复，只能引用当前 EvidencePackage |


answering 只摘要该对象的使用方式；完整结果契约以 [core README](../core/README.md) 为准。

### 协议与渲染接口

```text
build_answer_prompt(package) -> list[message]
parse_and_validate_answer(package, raw_text) -> ModelAnswer
render_exact_lookup(evidence) -> RenderedAnswer
render_semantic_answer(package, answer) -> RenderedAnswer
render_clarification(question=None) -> RenderedAnswer
render_refusal(reason) -> RenderedAnswer
render_failure(diagnostic_code) -> RenderedAnswer
```

`AnswerProtocolError` 表示模型输出不满足固定 JSON 协议；`PromptTokenCountError` 表示当前 tokenizer 无法可靠复现运行时 prompt；这两类错误都应由 core 转换为安全处理失败。

## 总体执行流程

```text
语义问答路径（由 core 调度）
    原始 query + retrieval 已完成单/多 query 排序的完整 LegalArticle
        -> [1] 接收有序完整法条
            -> retrieval 候选前五条
        -> [2] 选择预算内最大完整前缀
            -> EvidencePackage + 实际 prompt token 数
        -> [3] 生成唯一模型视图
            -> 带 E1...En 临时编号的 user JSON
        -> [4] 构造模型消息
            -> 固定 system message + user message
        -> [5] core 调用注入的 generate 函数（answering 边界之外）
            -> 模型原始文本
        -> [6] 校验模型原始输出
            -> 合法两字段回答 -> ModelAnswer
                -> [7] 构造公开中文回答
                    -> RenderedAnswer

其他执行路径（不调用回答模型）
    可信的精确查条结果、澄清状态、策略拒答原因或处理失败状态
        -> [8] 处理其他执行路径
            -> RenderedAnswer
```

1. **接收有序完整法条**
  - 位置：`evidence.py::EvidencePackager.build()`
  - 对象：query、`Sequence[LegalArticle]` -> retrieval 候选前五条
  - 行为：保持 retrieval 最终顺序，拒绝重复 `chunk_id`，不重新排序或读取检索分数；只使用原始 query，不接收任何 Query 增强中间字段或 trace。
2. **选择预算内最大完整前缀**
  - 位置：`evidence.py::EvidencePackager.build()`
  - 对象：候选法条 -> `EvidencePackage`
  - 行为：逐条加入完整正文并计算完整 prompt token；首次超预算时停止，已入选法条保持不变。
3. **生成唯一模型视图**
  - 位置：`evidence.py::EvidencePackage.to_model_json()`
  - 对象：`EvidencePackage` -> 紧凑 user JSON
  - 行为：按顺序分配临时证据编号，模型只能看到问题、法名、条号和完整法条正文。
4. **构造模型消息**
  - 位置：`protocol.py::build_answer_prompt()`
  - 对象：`EvidencePackage` -> system 与 user messages
  - 行为：加入固定证据约束、原子结论要求和两字段输出要求，不拼接自由格式指令。
5. **调用注入的生成函数**
  - 位置：`core/legal_rag.py::CurrentLawRAG._answer_semantic_search()`
  - 对象：system 与 user messages -> 原始模型文本
  - 行为：core 使用应用注入的 `generate` 函数调用实际模型；answering 不创建 API 客户端、不管理模型生命周期，也不解释模型服务异常。
6. **校验模型原始输出**
  - 位置：`protocol.py::parse_and_validate_answer()`
  - 对象：原始文本 -> `ModelAnswer`
  - 行为：严格解析 JSON，核对字段集合、类型和 citation 范围，并按证据包顺序规范化引用。
7. **构造公开中文回答**
  - 位置：`render.py::render_semantic_answer()`
  - 对象：`EvidencePackage`、`ModelAnswer` -> `RenderedAnswer`
  - 行为：只投影模型实际引用的完整法条，附加简短归纳和固定回答范围。
8. **处理其他执行路径**
  - 位置：`render.py::render_exact_lookup()`、`render_clarification()`、`render_refusal()`、`render_failure(diagnostic_code)`
  - 对象：可信程序状态 -> `RenderedAnswer`
  - 行为：不调用法律回答模型；澄清路径可展示外部规划生成的问题，失败时使用固定消息。



## 具体实现细节



### evidence.py



#### 完整法条而非 selector 摘录

- **要解决的问题**：自动摘录可能截断条件、例外或上下句关系，模型也可能引用了程序无法准确映射的局部文本。
- **当前选择**：每条 `Evidence` 保存一整条 canonical 法条；模型 JSON 中保持单元素 `excerpts` 结构，但内容就是完整正文。
- **选择理由**：证据边界与法条边界一致，程序可以精确映射模型引用，不需要推断模型使用了哪个摘录。
- **影响与限制**：长法条占用更多上下文，覆盖率通过减少入选法条数量而不是裁剪正文来控制。



#### 最大有序前缀

- **要解决的问题**：retrieval 已经给出相关性顺序，但上下文无法保证容纳全部候选。
- **当前选择**：检查 retrieval 返回的前五条，逐条测量完整 prompt，保留能装入预算的最大有序前缀。
- **选择理由**：算法确定、可复现，不重新解释检索分数，也不会从中间跳过一条再拼接后续法条。
- **影响与限制**：高排名长法条可能阻止后续短法条进入；当前优先保持排序语义和证据完整性。



#### 临时证据编号

- **要解决的问题**：模型需要简短稳定的引用键，但 `chunk_id` 和完整法名不适合作为生成协议字段。
- **当前选择**：每次构包按顺序分配 `E1...En`，只在当前 `EvidencePackage` 内有效。
- **选择理由**：编号紧凑，程序可以确定性反向映射到法名、条号和正文。
- **影响与限制**：编号不能跨请求保存，也不能被当作知识库永久身份。



### token_count.py



#### 使用实际生成模板计数

- **要解决的问题**：只统计 user JSON 会漏掉 system prompt、角色标记、特殊 token 和生成前缀，导致训练或推理时意外截断。
- **当前选择**：`AnswerPromptTokenCounter` 调用 MiniMind tokenizer 的 `apply_chat_template()`，使用与无思考生成一致的空 think 前缀，再对完整 prompt 编码。
- **选择理由**：预算针对真正送入模型的序列，而不是字符数或局部 token 估计。
- **影响与限制**：chat template 的尾部格式改变时计数器会明确失败，需要同步更新协议；未来启用更大上下文或 YaRN 时只调整传入预算，完整法条不裁剪规则不变。



### protocol.py



#### 内部严格 JSON 协议

- **要解决的问题**：127M 模型能力有限，自由文本同时承担格式、证据映射和回答内容会增加不可控错误。
- **当前选择**：模型只输出精确两个字段：`summary`、`citations`；程序负责其他结构。
- **选择理由**：SFT 可以专门增强格式遵循和证据选择，运行时也能做确定性校验。
- **影响与限制**：缺字段、多字段、重复键、非标准常量、Markdown 包裹或附加解释都视为协议失败，不尝试启发式修复。



#### 原子结论与非空引用

- **要解决的问题**：一次响应包含多个独立事项会使扁平 citations 无法表达每个结论的证据归属。
- **当前选择**：每次只允许一个非空单行原子法律结论，并要求至少一个合法 citation；一个结论可以由一条或多条证据共同支持。
- **选择理由**：保持 `summary + citations` 协议简单，同时让 citations 表示支持当前结论所必需的证据集合。
- **影响与限制**：程序只能校验结构和引用范围，不能自动证明结论原子性或语义支持关系；多事项问题暂不保证一次完整回答。



#### Citation 规范化

- **要解决的问题**：模型可能改变引用顺序、重复引用或引用证据包之外的编号。
- **当前选择**：拒绝未知和重复编号；合法编号按 EvidencePackage 原顺序规范化。
- **选择理由**：公开证据顺序由程序决定，模型只表达实际采用哪些证据。
- **影响与限制**：协议只能验证编号存在，不能单独证明 summary 被引用法条语义支持，也不能识别未被引用的 hard negative；这些能力依赖模型训练和后续评估。



### render.py



#### 统一 RenderedAnswer 结构

- **要解决的问题**：四条执行路径存在重合字段，如果每条路径返回不同对象，调用方需要大量分支。
- **当前选择**：所有路径都返回 core 定义的 `RenderedAnswer`，使用 `evidence + summary` 表示法律回答，使用 `message` 表示澄清、拒答或失败。
- **选择理由**：外部结构统一，同时允许中文内容根据执行路径自然变化。
- **影响与限制**：这不意味着所有回答文本完全相同；相同的是数据结构和排版规则。



#### 程序控制证据展示

- **要解决的问题**：模型不应直接生成法名、条号和法条原文，否则可能编造或改写引用。
- **当前选择**：模型只返回 citation ID；程序从 EvidencePackage 取回对应完整法条并构造 `RenderedEvidence`。
- **选择理由**：公开引用与内部证据一一对应，模型无法凭文本制造新法条。
- **影响与限制**：语义回答只显示被 citation 选中的法条；未引用的包内候选不会展示。



#### 固定中文消息与回答范围

- **要解决的问题**：策略拒答、空召回和运行故障需要稳定且不泄露内部异常的用户表达。
- **当前选择**：按 `UnansweredReason` 选择固定消息；所有处理失败共用安全话术；正常语义回答附加固定范围说明。
- **选择理由**：内部诊断与外部话术分离，避免把异常详情或自由模型文本暴露给用户。
- **影响与限制**：当前不提供可配置文案主题；需要修改对外表达时应在 renderer 集中调整。



### **[init**.py](http://init.py)

该文件导出证据构包、prompt 协议、token 计数和五类 renderer。公开对象集中在 `rag.answering`，core 不需要依赖各实现文件的私有辅助函数。
