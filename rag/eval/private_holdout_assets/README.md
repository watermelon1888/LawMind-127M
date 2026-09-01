# 项目私有留出集资产

此目录承接原 `C:\Users\18881\PycharmProjects\private-eval` 中用于 RAG 评估的文件，迁移日期为 2026-08-25。迁移采用逐字节复制，原目录保留，未删除或覆盖任何源文件。

## 目录与用途

- `authoring/private-holdout-v2.jsonl`：当前正式私有留出集，共 100 条；RAG-SFT 私有评估使用该文件。
- `authoring/private-holdout-v1.jsonl`、`frozen/v1/holdout.jsonl` 与 `manifests/private-holdout-v1.*`：v1 的历史 authoring、冻结结果和原始冻结 manifest，仅用于复核既有历史记录。
- `migration-manifest.json`：本次迁移的文件身份与外部依赖说明。

v1 原始 manifest 中的绝对路径是其 2026-08-01 冻结时的审计内容，因此保持原样；不得为了目录迁移修改它或其 `.sha256` 文件。

## v2 的正式评估依赖

`rag.eval.rag_sft_v2_private_holdout_evaluation` 除 v2 数据外，还要求正式训练隔离 manifest `evaluation-exclusions-project-rag-v2.json`。该文件不在原 `private-eval` 目录，本次未重建或伪造。历史正式运行绑定的身份为：

- bytes：42,482
- SHA256：`a493b5ac542d50d4bbbf8926ef1d08ea818dfe31176f9cfa16fd6fbf8210b3d4`

恢复正式 v2 私有评估前，必须从权威训练隔离归档取得该文件，并同时校验该文件和 `authoring/private-holdout-v2.jsonl` 的哈希。私有留出集已使用过，结果不得用于继续选 checkpoint、调参、修改训练数据或逐题修复。

