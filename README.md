# LawMind-127M

基于 [MiniMind](https://github.com/jingyaogong/minimind) 构建的轻量法律语言模型与可审计 RAG 系统。项目从原始 64M 模型出发，完成 Tokenizer、预训练、法律持续训练（CPT）、法律 SFT 和 RAG-SFT 的完整实验闭环，并将模型接入覆盖 311 份法律法规的检索增强生成链路。

## 项目成果

| 方向 | 结果 |
| --- | --- |
| 模型规模 | 将词表从 6,400 扩展至 12,000、网络层数从 8 扩展至 16，得到 127M 参数模型。 |
| 通用预训练 | 清洗 443 万条通用语料，完成 `4.8B tokens × 2 epochs` 训练；最终 Loss 为 `2.51`，PPL 为 `12.42`。 |
| 法律持续训练 | 基于 154.6 万份法律数据完成 `2.4B tokens` CPT；法律 Loss 下降 `47.12%`，通用 Loss 上升 `38.06%`。 |
| RAG-SFT 寻优 | 选择 Clean:HN=`2:1`、上下文长度 `768`、`2 epochs`、父权重 `cpt_250m`；协议率 `94.94%`、Required Recall `82.91%`、Exact-set `55.70%`。 |
| 检索效果 | 在 100 条留出集上，Top-20 候选池完整法条覆盖率 `96.67%`、Top-5 精排覆盖率 `91.67%`、MRR@5 `0.9139`。 |

## 训练闭环

```text
12,000 ByteLevel BPE Tokenizer
  -> 通用预训练
  -> 法律 CPT
  -> 法律 SFT
  -> RAG-SFT 对照实验与模型选择
  -> 可审计法律 RAG
```

原始 64M MiniMind 在 RAG 场景中存在法条遗漏，难以作为可靠的回答模型。为此，项目在原有架构上扩展词表与层数，并以高占比法律数据训练 12,000 ByteLevel BPE Tokenizer。

法律 CPT 后保留 5 个父权重，并分别进行一轮法律 SFT。此阶段模型已具备基础法律问答能力，但仍存在输出协议不稳定与重复生成问题；后续通过 RAG-SFT 的数据配比、上下文长度、训练轮数和父权重对照实验确定最终模型。

## 可审计法律 RAG

系统将 311 份法律法规标准化为 22,199 条法条 chunk，支持精确查条、语义检索、澄清与拒答四种互斥 Query 路径。

```text
用户问题
  -> Query 理解与路由
  -> BGE-small + FAISS 稠密召回、BM25 稀疏召回、RRF 融合
  -> Cross-Encoder 精排
  -> 完整法条证据构包
  -> 模型生成与 JSON 协议校验
  -> 摘要、引用与异常回退
```

### 模型与程序职责划分

| 组件 | 职责 |
| --- | --- |
| 模型 | 仅输出 `summary` 与 `citations` 两字段 JSON，负责基于证据生成简短摘要和引用选择。 |
| 程序 | 负责法条解析与标准化、Query 理解与路由、混合检索与精排、完整证据构包、JSON 协议校验、引用映射与异常回退。 |

这种划分将法条身份、证据边界、格式与失败处理固定在程序侧，使回答过程可以审计，而不把可靠性完全交给模型自由生成。

### 长法条窗口策略

针对 BGE-small 与 BGE-reranker 的输入预算限制，检索模块实现 `512 tokens` 滑动窗口与 `64 tokens` 重叠策略：以最高窗口得分排序，再由完整法条参与回答。该策略使证据包完整命中率从 `72.86%` 提升至 `78.33%`。

### Query 增强实验

项目尝试过 Query 改写、问题分解、关键词提取和部门路由等增强方式，候选池覆盖率仅由 `87.14%` 提升至 `89.29%`。当前判断是 127M 模型承担自由生成增强任务的负担较重；后续计划改为由模型输出受限的 `scope` / `goal` 选择，再由程序补充检索逻辑。

## 项目结构

```text
LawMind-127M/
├── minimind/                 # 模型、训练、数据处理与推理脚本
│   ├── model/                # 模型定义、LoRA 与 Tokenizer 配置
│   ├── trainer/              # 预训练、CPT、SFT、RAG-SFT、评估与推理
│   ├── dataset/              # 数据准备、校验、构造与发布脚本
│   └── scripts/              # 训练、服务、转换与评估入口脚本
├── rag/                      # 法律 RAG 系统
│   ├── answering/            # 证据包、回答协议与中文渲染
│   ├── chunk/                # 法条结构化切分与索引构建
│   ├── core/                 # 统一结果契约与 RAG 总编排
│   ├── knowledge/            # 只读法条仓库与精确查条
│   ├── query/                # 查询路由、法条引用识别与 Query 增强协议
│   ├── retrieval/            # 稠密检索、BM25、RRF 融合与重排
│   └── eval/                 # 检索与模型评估代码
└── scripts/                  # 项目级辅助脚本
```

## 仓库边界与依赖

本仓库发布源码、训练脚本与评估逻辑，不包含数据与大型生成物。以下资源保存在云端或本地工作区，并由 `.gitignore` 排除：

- 预训练、CPT、SFT、RAG-SFT 与评审数据；
- 法律原始 DOCX、法条 JSONL 索引；
- 模型权重、检查点、训练日志、FAISS/BM25 索引与评测结果；
- 单元测试源码、内部 agents 文档、IDE 与本地协作工具目录。

项目使用 Conda 环境 `minimind`，Python 依赖位于 `minimind/requirements.txt`：

```powershell
conda run -n minimind pip install -r minimind/requirements.txt
```

克隆仓库后，需在云端准备版本匹配的数据、模型与检索产物，并由调用方显式传入对应路径。

## 模块文档

- [RAG 总编排](rag/core/README.md)
- [查询路由与增强](rag/query/README.md)
- [法律知识库](rag/knowledge/README.md)
- [检索与重排](rag/retrieval/README.md)
- [证据与回答协议](rag/answering/README.md)
- [法条切分设计](rag/chunk/DESIGN.md)

## 致谢与来源

- 模型训练基础来自 [MiniMind](https://github.com/jingyaogong/minimind)。引用、修改和再发布相关代码时，应继续遵守上游仓库的许可证要求。
- 法律知识库的原始文本与数据处理流程不随本仓库发布；使用者应自行确认数据来源、版本、授权和适用范围。
