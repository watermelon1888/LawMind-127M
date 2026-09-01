# retrieval

## 模块定位

`retrieval` 负责从静态现行法知识库中召回并排序候选法条。`SemanticRetriever.search()` 保留原始单 query 基线，`search_many()` 对 core 已编译的多条 query 分别执行 dense + BM25、两层等权 RRF 和一次统一 Cross-Encoder 重排。两条接口都只返回 canonical `LegalArticle` 候选，不把候选分类为 `complete | partial | none`。

## 文件结构

```text
rag/retrieval/
├── text.py          # 构造统一法条标题和 tokenizer 感知的检索窗口
├── dense.py         # 构建及运行 BGE + FAISS 稠密检索
├── sparse.py        # 构建及运行 jieba + BM25 稀疏检索
├── fusion.py        # 对多路法条排名执行等权 RRF
├── rerank.py        # 对候选法条窗口执行 Cross-Encoder 精排
├── semantic.py      # 编排召回、融合、canonical 补全和最终排序
├── loader.py        # 显式加载本地索引与重量级模型并校验产物
├── build_indexes.py # 从 article_index.jsonl 显式构建新索引
├── artifacts/       # 新检索实现的本地生成物目录
└── __init__.py      # 导出轻量运行时接口并延迟加载重量级依赖
```



## 当前实现概览



### 已实现

- BGE 查询指令、归一化向量与 FAISS `IndexFlatIP` 稠密召回；
- jieba 分词、法条引用锁词与 BM25 稀疏召回；
- dense 与 sparse 的法条级等权 RRF 融合；
- 多 query 第一层逐 query 混合召回、第二层跨 query 等权 RRF；
- 跨 query 融合后使用用户原始 query 执行一次统一 rerank；
- 长法条 tokenizer 滑窗、窗口命中折叠和 Cross-Encoder 最佳窗口聚合；
- 通过 `chunk_id` 从 `ArticleRepository` 补全 canonical 法条；
- 显式索引构建、显式模型加载和加载期索引完整性校验；
- `rag/retrieval/artifacts/` 已生成与新窗口协议匹配的 FAISS、dense metadata 和 BM25 产物；除旧 64 条校准 query 外，2026-08-09 已对统一 canonical 的 574 条长度合格 query 完成真实物化，无空检索、检索失败或构包失败；
- 任一路异常时严格失败，不静默退回单路召回或 RRF 顺序。



### 尚未实现

- 窗口重叠和 reranker batch size 尚未进行独立参数评估；
- 项目级应用入口尚未创建并注入可运行的 `SemanticRetriever`。



### 当前限制

- 默认使用 `BAAI/bge-small-zh-v1.5` 和 `BAAI/bge-reranker-base`，受各自输入长度与本地算力约束；
- dense 和 BM25 使用 core 编译的各条检索 query；reranker 始终使用用户原始 query；
- retrieval 不调用 Query 增强模型，也不解释 rewrite、术语或 subquery 的业务含义；HyDE 和部门过滤仍未接入；
- RRF 分数和 reranker 分数只用于排序，不作为可跨查询比较的回答置信度；
- 当前没有分数阈值拒答，只有空候选会被 core 映射为无可核验证据；
- 历史单 query、top-4 基线为 `complete_hit@4 = 74.29%`，不代表当前多 query、top-5 链路结果，也不转换为线上证据完整性状态或拒答条件；
- 当前原始单 query、top-5 基线为 `complete_hit@5 = 80.71%`、required GT Macro Recall@5 `= 84.64%`；固定参数、构包结果和逐题归因见 `rag/eval/results/retrieval-baseline-original-top5-v1/report.md`；
- 非空候选缺少必要证据时，下游模型可能形成不完整回答，当前接受这一剩余风险；
- 索引面向静态单版本法条，替换知识集合后必须重建配套产物。



## 输入、输出与所在链路

```text
原始问题 + 唯一检索 query 列表 -> 每 query dense / BM25 / RRF -> 跨 query RRF -> 原始问题 rerank -> RankedArticle
```


| 上游依赖      | 需要的产出                                          | 用途                         |
| --------- | ---------------------------------------------- | -------------------------- |
| core      | 原始 query，以及一至六条已规范化且去重的检索 query                | 执行单 query 基线或多 query 融合与重排 |
| knowledge | `ArticleRepository` 和 canonical `LegalArticle` | 构建索引文本并补全检索候选正文            |
| chunk     | `article_index.jsonl`                          | 离线构建 dense 与 BM25 索引       |



| 下游消费   | 接口                                                                 | 用途                   |
| ------ | ------------------------------------------------------------------ | -------------------- |
| core   | `SemanticRetriever.search(query) -> tuple[RankedArticle, ...]`     | 取得已完成重排的法条候选         |
| core   | `SemanticRetriever.search_many(original_query, retrieval_queries)` | 跨 query 融合后按原始意图统一重排 |
| 应用组装入口 | `load_semantic_retriever(...) -> SemanticRetriever`                | 显式加载本地索引和模型后注入 core  |




## 核心对象与公开接口



### SemanticRetrievalConfig


| 字段                      | 默认值 | 含义                 |
| ----------------------- | --- | ------------------ |
| `dense_top_k`           | 10  | dense 返回的唯一法条数量    |
| `sparse_top_k`          | 30  | BM25 返回的法条数量       |
| `rrf_k`                 | 60  | RRF 排名平滑常数         |
| `candidate_pool`        | 20  | 进入 reranker 的法条数量  |
| `top_k`                 | 5   | 最终返回给 core 的法条数量   |
| `batch_size`            | 32  | 索引编码或 reranker 批大小 |
| `window_overlap_tokens` | 64  | 相邻长法条窗口的正文重叠量      |


这些参数构成当前代码默认值。其中 `dense_top_k=10`、`sparse_top_k=30`、`rrf_k=60`、`candidate_pool=20` 和等权 RRF `1:1` 来自统一开发集的参数评估；详细结果见 `rag/eval/results/retrieval-parameter-selection.md`。配置要求所有数量为正整数、重叠量为非负整数，且 `top_k <= candidate_pool`。

### ScoredChunk 与 FusedChunk

`ScoredChunk` 是 dense 和 BM25 的共同法条级结果，只包含 `chunk_id` 与单路内部 `score`。`FusedChunk` 额外保存 RRF `rank` 和 `score`。两者都不携带法条正文，canonical 内容由 knowledge 提供。

### RetrievalWindow


| 字段                        | 含义                  |
| ------------------------- | ------------------- |
| `chunk_id`                | 窗口所属法条              |
| `window_index`            | 同一法条内的窗口序号          |
| `start_token`、`end_token` | 正文 token 区间         |
| `content`                 | 当前窗口正文              |
| `text`                    | 法名、规范条号和窗口正文组成的模型输入 |


`RetrievalWindow` 只存在于检索模型内部，不改变 knowledge 中完整 `LegalArticle` 的粒度。

### RankedArticle


| 字段             | 含义                       | 关键约束                     |
| -------------- | ------------------------ | ------------------------ |
| `article`      | canonical `LegalArticle` | 必须来自 `ArticleRepository` |
| `rrf_rank`     | 融合阶段名次                   | 正整数                      |
| `rrf_score`    | 融合阶段分数                   | 有限正数，仅用于诊断和排序            |
| `rerank_score` | Cross-Encoder 最佳窗口分数     | 有限数值                     |


`RankedArticle` 由 `SemanticRetriever` 创建并交给 core。core 使用候选顺序和完整法条，不根据中间分数自行设置拒答阈值。

### 公开接口

```text
SemanticRetriever.search(query) -> tuple[RankedArticle, ...]
SemanticRetriever.search_many(original_query, retrieval_queries) -> tuple[RankedArticle, ...]
SemanticRetriever.retrieve_candidates_many(retrieval_queries) -> tuple[FusedChunk, ...]
load_semantic_retriever(...) -> SemanticRetriever
python -m rag.retrieval.build_indexes --target sparse|dense|all
```

`retrieve_candidates_many()` 复用生产两层 RRF 并暴露统一 candidate pool，供 `search_many()` 和离线分层评估共同使用；它不执行 reranker，也不判断 GT 或证据完整性。`RetrievalIntegrityError` 表示检索候选、reranker 分数和 canonical 仓库之间的契约被破坏。这类错误必须进入 core 的处理失败路径。

## 总体执行流程

```text
请求前，每个应用进程一次
    [0] load_semantic_retriever(...)
        -> 读取 FAISS、窗口 metadata、BM25 和模型
        -> 校验索引产物与 ArticleRepository 的完整性
        -> SemanticRetriever

每个语义检索请求
    core 的原始 query + 已去重 retrieval_queries
        -> [1] SemanticRetriever.search() 或 search_many()
            -> 对每条检索 query 执行：
                |- [2] DenseSearcher.search()
                |      -> BGE 编码 -> FAISS 窗口召回 -> 按 chunk_id 折叠最佳窗口
                `- [3] SparseSearcher.search()
                       -> jieba 分词 -> BM25 法条级召回

            [2] dense 排名 + [3] sparse 排名
                -> [4] 每条 query 独立执行第一层等权 RRF
                    -> 每条 query 一份 FusedChunk 排名
                -> [5] 多 query 时执行第二层等权 RRF
                    -> 按 chunk_id 合并并截取统一 candidate_pool
                -> [6] ArticleRepository.get_by_chunk_id()
                    -> 完整 canonical LegalArticle
                -> [7] WindowReranker.score(original_query, articles)
                    -> 所有候选只按原始 query 统一打分
                -> [8] 稳定排序并截取 top-k
                    -> tuple[RankedArticle, ...] -> core

任一阶段异常或完整性校验失败
    -> retrieval 不保留部分成功的 query 腿
    -> 向 core 抛出失败；core 可整份丢弃增强尝试并显式回退单 query 基线
```

1. **显式加载并校验检索运行时**
  - 位置：`loader.py::load_semantic_retriever()`
  - 对象：本地索引产物、模型与 `ArticleRepository` -> `SemanticRetriever`
  - 行为：进程启动时显式加载 FAISS、BM25、embedding 和 reranker，并校验产物数量、维度、窗口身份和 `chunk_id` 可解析性。
2. **接收语义检索请求**
  - 位置：`semantic.py::SemanticRetriever.search()`、`search_many()`
  - 对象：非空原始 query、唯一检索 query -> 两路召回请求
  - 行为：`search()` 执行单 query 基线；`search_many()` 对每条已编译 query 调用相同候选召回逻辑。
3. **执行窗口级 dense 召回**
  - 位置：`dense.py::DenseSearcher.search()`
  - 对象：query -> `tuple[DenseHit, ...]`
  - 行为：编码查询、搜索 FAISS 窗口，并把同一法条的多个窗口折叠为最佳窗口。
4. **执行法条级 BM25 召回**
  - 位置：`sparse.py::SparseSearcher.search()`
  - 对象：query -> `tuple[ScoredChunk, ...]`
  - 行为：jieba 分词后计算 BM25，只保留有限且大于零的法条分数。
5. **逐 query 融合 dense 与 BM25**
  - 位置：`semantic.py::retrieve_candidates()`、`fusion.py::rrf_fusion()`
  - 对象：每条 query 的 dense 与 sparse 排名 -> 一份 `tuple[FusedChunk, ...]`
  - 行为：忽略不可直接比较的原始分数，以排名执行第一层等权 RRF，并截取单 query 候选池。
6. **跨 query 融合候选**
  - 位置：`semantic.py::SemanticRetriever.search_many()`、`fusion.py::rrf_fusion()`
  - 对象：每条唯一 query 的第一层排名 -> 统一 `FusedChunk` 候选池
  - 行为：每条 query 在第二层提供一份等权排名；只有一条 query 时直接沿用第一层结果，不重复改变 RRF 分数。
7. **补全 canonical 法条**
  - 位置：`semantic.py::rerank_candidates()`、`knowledge::ArticleRepository.get_by_chunk_id()`
  - 对象：`FusedChunk` -> `LegalArticle`
  - 行为：逐个使用 `chunk_id` 获取完整法条；未知 ID 转换为 `RetrievalIntegrityError`。
8. **执行窗口级 Cross-Encoder 重排**
  - 位置：`rerank.py::WindowReranker.score()`
  - 对象：用户原始 query 与候选法条 -> 每条法条的最佳窗口分数
  - 行为：增强 query 只影响召回；所有候选统一使用原始 query 构造窗口 pair、批量打分，并以最大窗口分数代表整条法条。
9. **形成最终法条排名**
  - 位置：`semantic.py::SemanticRetriever.rerank_candidates()`
  - 对象：法条、RRF 诊断、rerank 分数 -> `tuple[RankedArticle, ...]`
  - 行为：按 rerank 分数降序、RRF 名次和 `chunk_id` 稳定排序，返回最终 top-k。



## 具体实现细节



### text.py



#### 统一法条检索文本

- **要解决的问题**：dense、BM25 和 reranker 如果使用不同法名或条号格式，会引入不必要的表示差异。
- **当前选择**：模型文本统一以 `《正式法名》第X条` 开头，后接法条正文；法名和条号来自 canonical `LegalArticle`。
- **选择理由**：法律名称和条号本身具有强检索信号，并能稳定连接不同检索阶段。
- **影响与限制**：标题不代替正文；BM25 可以额外加入 hierarchy，dense 和 reranker 仍使用统一标题加正文窗口。



#### Tokenizer 感知滑窗

- **要解决的问题**：长法条可能超过 embedding 或 reranker 的模型输入上限，直接截断会永久丢失后半段。
- **当前选择**：先为法名、条号、query 和特殊 token 预留空间，再按模型 tokenizer 切正文重叠窗口；每个窗口都重新核对实际模型输入长度。
- **选择理由**：token 数而不是字符数决定模型是否截断，复核编码长度可以处理 decode 后边界变化。
- **影响与限制**：窗口增加索引数量和 reranker 计算量；固定输入已经占满上下文时直接失败，不生成空正文窗口。



#### 窗口预算与默认参数

- **当前参数**：当前 dense artifacts 记录 `tokenizer_max_length=512`、`window_overlap_tokens=64`；构建命令的默认 `batch_size=32`。22,199 条法条被展开为 22,301 个 dense 窗口，说明只有少量长法条需要多窗口覆盖。
- **预算公式**：`content_capacity = max_length - heading_tokens - query_tokens - special_tokens`。`heading` 是 `《正式法名》第X条\n`；dense 构建时 `query=None`，reranker 时还会扣除用户 query 和 pair special tokens，因此两者的正文容量可能不同。
- **边界复核**：先按 `content_capacity` 切 tokenizer token，再 decode 为正文；若重新编码后的完整模型输入仍超过 `max_length`，逐 token 缩短窗口直到满足限制。相邻窗口的下一起点为 `end - 64`，所以 64 个正文 token 会重复出现，以避免关键句恰好落在窗口边界时被切断。
- **影响与限制**：`batch_size=32` 只控制一次编码或打分的批量，不改变窗口内容。改变 dense 的 `max_length`、overlap 或检索文本格式会改变窗口身份和向量集合，必须重建 dense artifacts；只改运行时 reranker 参数则不改变已有向量索引，但会改变精排输入分布。



### dense.py



#### BGE + FAISS 稠密召回

- **要解决的问题**：用户问题与法条正文可能没有相同关键词，仅靠 BM25 难以召回语义相关法条。
- **当前选择**：默认使用 `BAAI/bge-small-zh-v1.5`，查询添加中文检索指令，向量 L2 归一化后写入 FAISS `IndexFlatIP`。
- **选择理由**：归一化内积等价于余弦相似度；小型中文 embedding 模型适合当前本地资源和法条规模。
- **影响与限制**：模型变更会改变向量维度和表示空间，必须重建索引；loader 会核对模型维度与 FAISS 维度。



#### 向量维度与取舍

- **当前维度**：`BAAI/bge-small-zh-v1.5` 的当前输出为 512 维；代码不把 512 写死，而是在构建时写入 metadata，并在加载时比较 `encoder.get_sentence_embedding_dimension()` 与 `IndexFlatIP.d`。当前 22,301 个窗口的原始 `float32` 向量约占 43.6 MiB，未含 FAISS 对象和 metadata 开销。
- **如何取舍**：向量维度是 embedding 模型表示能力的一部分，不是可以只改 FAISS 参数的独立旋钮。更高维模型通常带来更高的向量内存、编码成本和 CPU 内积搜索成本，但可能改善语义区分；更低维模型或额外压缩会降低资源占用，却可能损失法律术语、条件和例外的区分能力。
- **当前决策**：首版沿用 bge-small 的原生 512 维，不做 PCA、截断或量化压缩，以避免在尚未建立新索引正式基线前额外引入不可解释的表示损失。若更换 embedding 模型、降维或量化，必须重新生成全部窗口向量、索引和检索评估，不能复用当前 artifacts。



#### 是否使用 ANN 索引

- **当前选择**：没有使用 ANN。`faiss.IndexFlatIP` 对全部窗口执行精确内积搜索；归一化后等价于精确余弦相似度排序。`IndexFlatIP` 的含义：`Flat`：不做近似索引，查询时遍历全部向量，得到精确 top-k。`IP`：按 Inner Product（内积）排序。
- **选择理由**：当前仅 22,301 个窗口，CPU 全扫描的成本可控，且精确检索避免 HNSW、IVF 或 PQ 的近似误差、训练参数和额外召回损失。
- **演进条件**：若法条规模、窗口数或并发使 CPU 全扫描成为可测瓶颈，再以独立召回与延迟评估比较 ANN。届时 ANN 的候选数、索引构建参数和近似召回损失都属于新检索版本，不能与当前 `IndexFlatIP` 基线混用。



#### Dense 构建中的窗口向量

- **构建方式**：`build_dense_index()` 对每个完整 `LegalArticle` 调用共享的 `build_retrieval_windows()`，以 encoder 的实际 `max_seq_length` 和默认 64-token overlap 生成窗口；每个窗口的 `《法名》第X条 + 正文` 独立编码为一个 512 维向量，并记录 `chunk_id`、窗口序号和 token 边界。
- **查询方式**：查询只编码一次，前置 BGE 中文检索指令并做 L2 归一化；FAISS 先取不少于目标法条数两倍的窗口命中，再按 `chunk_id` 保留最高分窗口。唯一法条数量不足时，将窗口请求量逐步扩大至索引总量。



#### 窗口命中折叠

- **要解决的问题**：同一长法条的多个窗口可能占满 top-k，导致法条级候选数量不足。
- **当前选择**：FAISS 先取更多窗口，同一 `chunk_id` 只保留最高分窗口；唯一法条不足时逐步扩大窗口请求量。
- **选择理由**：后续 RRF 和 reranker 的单位是完整法条，不是窗口。
- **影响与限制**：dense 的诊断分数代表最佳窗口，不代表整条法条所有内容都与 query 相关。



### sparse.py



#### jieba + BM25

- **要解决的问题**：法律名称、条号和固定法律术语需要精确字面召回，dense 可能漏掉这些稀有信号。
- **当前选择**：使用 jieba 分词与 BM25，并把规范“第X条”引用锁定为单个 token。
- **选择理由**：BM25 对精确术语和编号敏感，与 dense 形成互补。
- **影响与限制**：未进入锁词规则的引用形式仍受 jieba 切分影响；运行时只返回大于零的 BM25 候选。



#### BM25 索引文本

- **要解决的问题**：编、章、节标题可能包含正文未重复出现的主题词。
- **当前选择**：BM25 文档由统一法条标题、非空 hierarchy 和完整正文组成，法名不重复添加。
- **选择理由**：层级信息增强字面召回，但不污染 canonical `LegalArticle` 或回答证据。
- **影响与限制**：hierarchy 只在离线构建时从 JSONL 读取；运行时 BM25 产物只保存位置到 `chunk_id` 的映射。



### fusion.py



#### 等权 RRF

- **要解决的问题**：dense 相似度和 BM25 分数不在同一数值空间，直接加权相加需要额外校准。
- **当前选择**：按各路名次计算 `1 / (k + rank)` 并等权累加，默认 `k=60`。
- **选择理由**：RRF 不依赖原始分数量纲，能稳定奖励两路共同命中，同时保留单路独有候选。
- **影响与限制**：当前没有学习路线权重；同分使用 `chunk_id` 保证确定性排序，完整精度分数不会提前舍入。



### rerank.py



#### Cross-Encoder 窗口重排

- **要解决的问题**：召回模型独立编码 query 和文档，候选前几名仍可能不精确支持问题。
- **当前选择**：默认使用 `BAAI/bge-reranker-base` 联合编码 query 与候选窗口，整条法条取最高窗口分数。
- **选择理由**：Cross-Encoder 能直接建模 query 与证据文本的细粒度匹配；最佳窗口聚合避免长法条后半段被截断。
- **影响与限制**：需要对候选的所有窗口打分，延迟高于单次截断；任一返回数量、类型或有限性异常都会失败，不使用 RRF 顺序继续回答。



#### 512-token pair 预算与最佳窗口聚合

- **当前参数**：loader 以 `max_length=512` 创建 `BAAI/bge-reranker-base`，`WindowReranker` 也使用 512、64-token overlap 和 `batch_size=32`；默认最多对 RRF `candidate_pool=20` 条法条精排。
- **窗口构造**：对每条候选法条重新调用 `build_retrieval_windows(article, query=query)`。因此 512 token 预算同时包含 query、`《法名》第X条` 标题、正文和 pair special tokens；query 越长，单个窗口可容纳的正文越少。固定部分已占满预算时抛出错误，不让模型静默截断正文。
- **打分与归属**：所有 `(query, window.text)` pair 按法条原始顺序汇总，由 `CrossEncoder.predict()` 以 32 条为一批打分；代码用 owner 记录窗口属于哪条候选法条，再取每条法条全部窗口中的最大有限分数。返回分数数量、类型或有限性异常都会直接失败。
- **最终排序**：最大窗口分数是整条 `LegalArticle` 的 `rerank_score`，不是只把该窗口下游传递；下游仍接收完整法条。`rerank_score` 只在同一 query 内排序，若分数相同再用 RRF 名次和 `chunk_id` 稳定打破平局，不把它解释为跨 query 的置信度或拒答阈值。



### semantic.py



#### 单 query 基线与多 query 编排

- **要解决的问题**：core 不应理解 dense、BM25、RRF 和 reranker 的内部参数或中间对象。
- **当前选择**：`search()` 保留单 query 基线；`retrieve_candidates()` 执行第一层融合，`retrieve_candidates_many()` 执行可复用的第二层融合，`rerank_candidates()` 统一使用原始 query 重排；`search_many()` 组合这些接口。最终搜索结果仍只包含有序 `RankedArticle`。
- **选择理由**：多 query 复用与基线完全相同的 dense、BM25、canonical 补全和 reranker，不复制检索算法，也不让 retrieval 调用增强模型。
- **影响与限制**：任一增强检索腿异常都会向 core 抛出，retrieval 自身不保留部分结果；是否整份回退原始基线由 core 显式决定并记录。



#### 分数不承担拒答

- **要解决的问题**：不同 query 的 RRF 与 reranker 分数没有经过概率校准，固定阈值可能错误拒绝或错误放行。
- **当前选择**：分数只用于单次 query 内排序；空候选返回空元组，非空候选直接交给 core 构包，不判断回答所需证据是否完整。
- **选择理由**：把候选排序与线上回答控制分开，避免用未校准分数伪装证据充分性。历史 `complete_hit@4` 只保留用于旧报告复现；当前链路按 top-5 和独立构包阶段重新评估。
- **影响与限制**：模型收到非空证据包后直接回答；必要证据未完整召回时可能形成不完整回答，当前不设置阈值、完整性分类器或模型拒答分支来消除该风险。



### loader.py



#### 显式重量级加载

- **要解决的问题**：仅导入模块或构造轻量对象时不应隐藏加载 FAISS、模型权重和 pickle 的高成本操作。
- **当前选择**：只有调用 `load_semantic_retriever()` 才读取本地产物并创建 embedding、BM25 和 reranker 实例。
- **选择理由**：资源开销和失败时点对应用入口可见，也便于在每个进程中只加载一次。
- **影响与限制**：调用方必须明确提供产物目录、设备和可用模型；当前没有自动寻找旧索引或兼容旧 pickle 的回退路径。



#### 加载期完整性校验

- **要解决的问题**：FAISS、窗口 metadata、BM25 映射和 canonical 仓库可能来自不同构建批次。
- **当前选择**：接收请求前校验向量数量、向量维度、窗口身份、BM25 文档数量及所有 `chunk_id` 的仓库可解析性。
- **选择理由**：索引不配套是系统错误，不能等到某次用户请求命中坏记录后才暴露。
- **影响与限制**：校验增加启动时间，但请求阶段无需重复扫描全部索引。



### build_indexes.py

- **要解决的问题**：dense 构建需要完整窗口 metadata，BM25 构建需要 hierarchy，但运行时证据只能来自 canonical 仓库。
- **当前选择**：从同一 `article_index.jsonl` 创建 `ArticleRepository`；法条身份与正文取自仓库，hierarchy 只用于 BM25 构建文本。
- **选择理由**：离线索引与运行时 canonical 内容保持一致，同时避免把完整 article 字典复制进检索产物。
- **影响与限制**：构建是显式长任务，可以选择 `dense`、`sparse` 或 `all`；本 README 改写不会触发索引构建。



### **[init**.py](http://init.py)

包级接口直接导出 `SemanticRetriever`、`SemanticRetrievalConfig`、`RankedArticle` 和 `RetrievalIntegrityError`。`load_semantic_retriever()` 采用函数内导入，使 `import rag.retrieval` 不会隐式加载重量级第三方库、模型或索引。