# knowledge

## 模块定位

`knowledge` 是法律 RAG 的只读 canonical 法条仓库。它把静态 `rag/chunk/article_index.jsonl` 加载为可确定性访问的 `LegalArticle`，为精确查条、混合检索、RAG-SFT 数据构造和评估工具提供同一份法名、条号与完整正文。仓库保证法条身份与正文的一致来源，不判断若干法条能否完整回答某个问题。

## 文件结构

```text
rag/knowledge/
├── repository.py  # 定义 canonical 法条、静态仓库、加载校验与确定性查找
└── __init__.py    # 导出 knowledge 的公开接口
```

## 当前实现概览

### 已实现

- 从 JSONL 一次性加载静态法条，建立只读内存仓库；
- 将每条记录投影为四字段不可变 `LegalArticle`；
- 支持 `(law_name, article_no)` 与 `chunk_id` 两种确定性访问；
- 支持书名号、空白、确定性法名简称和常见条号写法的规范化；
- 对运行时真正依赖的索引契约执行最小完整性校验；
- 当前索引包含 22,199 条法条记录和 302 个不同法名。

### 尚未实现

当前没有已确认但尚未实现的 knowledge 内部能力。应用如何创建并注入仓库由应用组装入口决定，不属于仓库自身的缺失功能。

### 当前限制

- 只访问项目选定的静态现行法集合，不维护历史版本时间线；
- 311 份源 DOCX 中有 302 个法名形成法条级索引，另外 9 份无明确条号结构的文档不进入 `LegalArticle` 仓库；
- 不维护“刑诉法、民诉法”等人工简称词典，也不执行模糊法名搜索；
- 不在运行时判断法律动态有效性或证明来源权威性；
- 正文始终以完整法条返回，不在仓库层生成摘录或执行预算裁剪；
- 索引契约完整性只表示记录和映射结构有效，不等于回答所需证据在语义上完整。



## 输入、输出与所在链路

```text
article_index.jsonl -> ArticleRepository -> exact_lookup / retrieval / RAG-SFT
```


| 上游依赖                            | 需要的产出                                                 | 用途                   |
| ------------------------------- | ----------------------------------------------------- | -------------------- |
| `rag/chunk/article_index.jsonl` | 包含 `chunk_id`、`law_name`、`article_no`、`content` 的法条记录 | 建立 canonical 法条仓库    |
| 应用组装入口                          | 显式索引路径                                                | 在进程启动时创建仓库，不依赖隐藏默认路径 |



| 下游消费          | 接口                                            | 用途                                |
| ------------- | --------------------------------------------- | --------------------------------- |
| core 精确查条     | `lookup(law_name, article_no)`                | 把用户指定的法名与条号解析为唯一法条                |
| retrieval     | `get_by_chunk_id(chunk_id)`                   | 把检索候选补全为 canonical `LegalArticle` |
| RAG-SFT 与评估工具 | `from_jsonl()`、`lookup()`、`get_by_chunk_id()` | 使用与运行时相同的法条身份和正文                  |




## 核心对象与公开接口



### LegalArticle

`LegalArticle` 是 knowledge 对外提供的 canonical 法条对象，由 `ArticleRepository` 创建并交给 retrieval、core 和数据构造工具消费。


| 字段           | 含义                         | 关键约束                           |
| ------------ | -------------------------- | ------------------------------ |
| `chunk_id`   | 检索产物与 canonical 仓库之间的稳定关联键 | 必须等于 `{law_name}#{article_no}` |
| `law_name`   | 索引保存的正式法律名称                | 非空，引用展示以此为准                    |
| `article_no` | canonical 条号               | 使用 `264`、`133之一` 等格式           |
| `content`    | 不含条号前缀的完整法条正文              | 非空，仓库不改写、不截断                   |


对象使用 `frozen=True` 保持不可变。同一实例可以被多个访问路径和请求共享，下游需要构造证据时复制所需字段，而不是修改仓库中的法条。

逐法条 `content_hash` 不属于该对象。当前法条集合不做运行时更新，也没有外部哈希基准；哈希不参与查找、完整性校验或证据映射，因此不沿运行链路复制。

### ArticleRepository


| 接口                                   | 返回或行为                                     |
| ------------------------------------ | ----------------------------------------- |
| `ArticleRepository.from_jsonl(path)` | 加载静态索引、校验核心契约并建立内存映射                      |
| `lookup(law_name, article_no)`       | 唯一命中返回 `LegalArticle`；正常未命中返回 `None`      |
| `get_by_chunk_id(chunk_id)`          | 命中返回 `LegalArticle`；未知内部 ID 抛出 `KeyError` |


仓库路径由调用方显式传入。推荐每个应用进程创建一次仓库，再把同一对象注入 retrieval 和 `CurrentLawRAG`；仓库不提供新增、修改、删除、reload 或热更新接口。

### IndexIntegrityError

`IndexIntegrityError` 表示 JSONL 结构、canonical 条号、法条身份或唯一性契约被破坏。文件不存在和权限不足继续使用 Python 原生的 `FileNotFoundError` 与 `PermissionError`，便于区分存储问题和数据结构问题。

## 总体执行流程

```text
rag/chunk/article_index.jsonl
    -> ArticleRepository.from_jsonl(path)
        -> 逐行解析 JSONL
        -> 校验 canonical 契约
            |- 不合法、重复或空索引 -> IndexIntegrityError
            `- 合法记录
                -> 创建不可变 LegalArticle
                -> 建立两张指向同一 LegalArticle 的映射
                    |- (law_name, article_no) -> LegalArticle
                    `- chunk_id -> LegalArticle

调用方
    |- core 精确查条
    |   -> lookup(law_name, article_no)
    |   -> 规范化法名和条号
    |   -> LegalArticle | None
    |       `- None 是正常业务未命中，由 core 转为澄清
    |
    `- retrieval / RAG-SFT / 评估工具
        -> get_by_chunk_id(chunk_id)
        -> LegalArticle | KeyError
            `- KeyError 是内部索引不完整，必须阻断调用链
```

1. **加载静态索引**
  - 位置：`repository.py::ArticleRepository.from_jsonl()`
  - 对象：JSONL 行 -> 原始法条记录
  - 行为：逐行解析 JSON，忽略空行，并读取四个核心字段。
2. **校验 canonical 契约**
  - 位置：`repository.py::ArticleRepository.from_jsonl()`
  - 对象：原始法条记录 -> 合法法条记录
  - 行为：校验非空字段、canonical 条号、`chunk_id` 一致性、法条键唯一性和非空索引。
3. **建立只读法条与双键映射**
  - 位置：`repository.py::LegalArticle`、`ArticleRepository.__init__()`
  - 对象：合法法条记录 -> `LegalArticle`
  - 行为：建立 `(law_name, article_no)` 和 `chunk_id` 到同一法条对象的映射，并派生确定性法名简称。
4. **执行面向用户的精确查条**
  - 位置：`repository.py::ArticleRepository.lookup()`
  - 对象：法名、条号 -> `LegalArticle | None`
  - 行为：规范化输入，仅在法名与条号能唯一确定时返回法条。
5. **执行面向内部索引的法条补全**
  - 位置：`repository.py::ArticleRepository.get_by_chunk_id()`
  - 对象：`chunk_id` -> `LegalArticle`
  - 行为：返回 canonical 法条；未知 ID 直接报错，阻止不配套索引继续进入回答链路。



## 具体实现细节



### repository.py



#### 静态 canonical 仓库

- **要解决的问题**：精确查条、dense、BM25、reranker 和训练数据不能各自保存并传播不同版本的法条正文。
- **当前选择**：所有运行路径通过一个只读 `ArticleRepository` 获取完整 `LegalArticle`。
- **选择理由**：法条正文只有一个事实来源，检索产物只需保存 `chunk_id`，避免旧字典或元数据覆盖 canonical 内容。
- **影响与限制**：整体替换静态数据时必须重建仓库及配套检索索引；当前不支持在线更新。



#### 最小加载契约

- **要解决的问题**：上游 JSONL 包含部门、层级、日期和 token 统计等附加字段，但 knowledge 的查找行为并不依赖它们。
- **当前选择**：只依赖并校验 `chunk_id`、`law_name`、`article_no`、`content`，允许并忽略其他字段。
- **选择理由**：把运行时仓库与上游统计元数据解耦，同时保留保护证据身份所需的约束。
- **影响与限制**：附加字段是否正确由其所属模块负责；knowledge 不用它们推导时效、部门或展示内容。



#### 双键访问

- **要解决的问题**：用户精确查条使用法名和条号，检索索引则只应传播内部 `chunk_id`。
- **当前选择**：同时建立 `(law_name, article_no) -> LegalArticle` 和 `chunk_id -> LegalArticle` 两张内存映射。
- **选择理由**：两种访问均为 O(1)，并确保不同路径最终回到同一对象。
- **影响与限制**：`chunk_id` 必须与法名、条号严格一致；不一致会在加载阶段失败。



#### 确定性法名规范化

- **要解决的问题**：用户可能输入书名号、空白、去掉“中华人民共和国”的简称或去掉版本括号的法名。
- **当前选择**：先匹配正式法名，再匹配从正式法名自动派生的唯一简称；多个正式法名共享简称时返回未命中。
- **选择理由**：规则可以审计，不会因相似名称猜错法律。
- **影响与限制**：不支持残缺前缀、编辑距离、向量近似或 LLM 猜测；人工简称只有在出现明确高频需求后才考虑加入。



#### Canonical 条号规范化

- **要解决的问题**：`第264条`、`第二百六十四条` 和 `264` 应定位同一法条。
- **当前选择**：将规范中文数字与阿拉伯数字转换为 canonical 主条号，并支持“之几”后缀。
- **选择理由**：仓库键保持单一格式，调用方不必自行拼接条号。
- **影响与限制**：中阿混写、非条号单位和不规范中文数字不做猜测，统一表现为 `lookup()` 未命中。



#### 两种失败语义

- **要解决的问题**：用户写错法名或条号是正常业务分支，而检索索引引用不存在的 ID 是系统完整性问题。
- **当前选择**：`lookup()` 未命中返回 `None`；`get_by_chunk_id()` 未命中抛出 `KeyError`。
- **选择理由**：前者交给 core 请求澄清，后者必须阻断链路，不能静默丢弃候选。
- **影响与限制**：knowledge 不进一步细分法名不存在、简称歧义、条号不存在和格式无效，因为这些情况当前具有相同下游行为。



### **[init**.py](http://init.py)

该文件稳定导出 `ArticleRepository`、`LegalArticle` 和 `IndexIntegrityError`。调用方通过 `rag.knowledge` 使用公开契约，不依赖 `repository.py` 中的私有规范化函数。
