# LawMind React 前端

该目录是 LawMind-127M 的正式项目展示页，使用 React、TypeScript 和 Vite，通过现有 FastAPI 接口完成单轮法律问答。

## 开发运行

首次运行先安装依赖：

```powershell
cd frontend
npm.cmd install
```

之后回到项目根目录启动完整服务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-lawmind-web.ps1
```

访问 `http://127.0.0.1:8000`。该入口会装配本地 RAG，并由 FastAPI 同时提供
React 页面和 `/v1/answer` API。

需要单独调试前端时执行 `npm.cmd run dev`，访问 `http://127.0.0.1:5173`；
Vite 会把 `/v1` 和 `/health` 代理到 `127.0.0.1:8000`。

## 构建

```powershell
cd frontend
npm.cmd run build
```

产物输出到 `frontend/dist/`。FastAPI 检测到该目录后会自动将其挂载到 `/`。
