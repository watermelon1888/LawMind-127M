# 法律 RAG FastAPI 入口

`api.main.create_app(application)` 创建法律 RAG HTTP 应用。调用方先通过
`rag.app.build_current_law_rag()` 装配一次 `LegalRAG`，再注入 API 层：

```python
from api import create_app

application = build_current_law_rag(...)  # 步骤 13 的统一装配入口
app = create_app(application)
```

本地完整展示服务使用已经封装好的装配入口：

```powershell
conda run --no-capture-output -n minimind python -m api.server
```

`api.server` 默认加载本地 `rag_epoch_2.pth`、当前 Tokenizer、法律与部门规章
合并索引、统一 Dense/BM25 产物和 `frontend/dist/`。也可以通过
`LAWMIND_WEIGHTS`、`LAWMIND_TOKENIZER_PATH`、`LAWMIND_ARTICLE_INDEX`、
`LAWMIND_ARTIFACT_DIR`、`LAWMIND_FRONTEND_DIR`、`LAWMIND_DEVICE`、
`LAWMIND_HOST` 和 `LAWMIND_PORT` 覆盖默认值。Agentic RAG 默认启用；设置
`LAWMIND_ENABLE_EXTERNAL_LLM=0` 可以显式关闭外部模型。

首版接口：

- `POST /v1/answer`：请求体为 `{"query": "..."}`，返回统一业务 JSON；
- `GET /health/live`：进程存活检查；
- `GET /health/ready`：检查 `LegalRAG` 是否已注入。

React 展示页位于 `frontend/`。执行 `npm.cmd run build` 后，
`create_app()` 会自动检测并挂载 `frontend/dist/`，使 FastAPI 同时提供 API 和页面。
开发模式使用 Vite 自带的代理访问 `127.0.0.1:8000`，无需额外启用 CORS。

`api.main:app` 提供可导入的默认应用，未注入实例时问答和就绪检查返回 `503`；
生产启动应使用已完成装配的实例调用 `create_app`。业务状态直接使用现有状态字符串，
不会把业务结果映射成抽象数字状态码。

问答响应直接返回 `status`、`route`、可选 `answer_mode`、展示答案、`candidate_answer`、证据、诊断信息和完整审计链，不再返回 `task_type`。检索生成回答的 `answer` 只包含通过协议校验的简短归纳，完整法条由结构化 `evidence` 提供，避免页面答案区与可展开的法律依据重复。`candidate_answer` 保存 127M 的原始生成结果；协议校验失败时可供调用方审计，但不表示该内容可以交付。法律检索回答先记录 `query_assessment`；判断为 `clarify` 时直接记录澄清并停止，判断为 `answer` 时继续记录检索、Agent 工具调用、证据构包、回答生成和本地协议校验。生产链路不执行生成后的外部答案审查。
