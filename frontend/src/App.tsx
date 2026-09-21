import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { answerQuestion } from "./api";
import { AuditFlow } from "./AuditFlow";
import type {
  AnswerResponse,
  HistoryCategory,
  HistoryItem,
} from "./types";

const HISTORY_KEY = "lawmind-query-history-v1";
const MAX_QUERY_CHARACTERS = 1000;

const categoryLabels: Record<HistoryCategory, string> = {
  pending: "处理中",
  exact: "精确查条",
  answer: "法律回答",
  clarify: "需要澄清",
  refuse: "策略拒答",
  general: "通用对话",
  error: "处理失败",
};

const routeLabels: Record<string, string> = {
  answer: "法律回答",
  clarify: "需要澄清",
  refuse: "策略拒答",
  general_chat: "通用对话",
};

const answerModeLabels: Record<string, string> = {
  exact_lookup: "精确查找法条",
  retrieval: "检索生成回答",
};

const statusLabels: Record<string, string> = {
  verified_lookup: "精确查条",
  retrieved_evidence: "法律检索回答",
  clarification_required: "需要补充事实",
  refused: "策略拒答",
  general_chat: "通用对话",
  processing_failed: "处理失败",
};

function readHistory(): HistoryItem[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]");
    return Array.isArray(parsed) ? parsed.slice(0, 30) : [];
  } catch {
    return [];
  }
}

function categoryFor(result: AnswerResponse): HistoryCategory {
  if (result.status === "processing_failed") return "error";
  if (result.route === "refuse" || result.status === "refused") return "refuse";
  if (result.route === "clarify" || result.status === "clarification_required") {
    return "clarify";
  }
  if (result.route === "general_chat" || result.status === "general_chat") {
    return "general";
  }
  if (result.status === "verified_lookup") return "exact";
  if (result.route === "answer") return "answer";
  return "error";
}

function articleLabel(articleNo: string): string {
  const [main, suffix] = articleNo.split("之");
  return suffix ? `第${main}条之${suffix}` : `第${main}条`;
}

function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(timestamp);
}

function answerText(answer: string): string {
  return answer.replace(/^提示：\s*/, "").trim();
}

export default function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<AnswerResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>(readHistory);
  const [view, setView] = useState<"answer" | "audit">("answer");
  const [candidateOpen, setCandidateOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      // 浏览器存储空间不足时仍保留当前会话内的历史结果。
    }
  }, [history]);

  useEffect(() => () => request.current?.abort(), []);

  const status = useMemo(() => {
    if (loading) return "正在分析问题并检索法律依据";
    if (error) return "处理失败";
    if (!result) return "等待输入";
    return statusLabels[result.status] ?? routeLabels[result.route] ?? "已完成";
  }, [error, loading, result]);

  function updateHistory(item: HistoryItem) {
    setHistory((current) => {
      const existing = current.find((entry) => entry.query === item.query);
      const next = item.result || !existing?.result
        ? item
        : { ...item, result: existing.result };
      return [
        next,
        ...current.filter((entry) => entry.query !== item.query),
      ].slice(0, 30);
    });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const cleaned = query.trim();
    if (!cleaned) {
      setError("请输入一个法律问题。");
      return;
    }
    if (cleaned.length > MAX_QUERY_CHARACTERS) {
      setError(`问题最多允许 ${MAX_QUERY_CHARACTERS} 个字符。`);
      return;
    }

    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    setLoading(true);
    setError(null);
    setResult(null);
    setView("answer");
    setCandidateOpen(false);
    updateHistory({ query: cleaned, updatedAt: Date.now(), category: "pending" });
    try {
      const response = await answerQuestion(cleaned, controller.signal);
      setResult(response);
      updateHistory({
        query: cleaned,
        updatedAt: Date.now(),
        category: categoryFor(response),
        result: response,
      });
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "服务暂不可用，请稍后再试。");
      updateHistory({ query: cleaned, updatedAt: Date.now(), category: "error" });
    } finally {
      if (request.current === controller) {
        request.current = null;
        setLoading(false);
      }
    }
  }

  function chooseHistory(item: HistoryItem) {
    request.current?.abort();
    setQuery(item.query);
    setResult(item.result ?? null);
    setError(null);
    setView("answer");
    setCandidateOpen(false);
  }

  return (
    <div className="app-shell">
      <aside className="history-panel" aria-label="提问历史">
        <div className="history-header">
          <div>
            <p>提问历史</p>
            <h2>最近的问题</h2>
          </div>
          <button
            className="text-button"
            type="button"
            disabled={history.length === 0}
            onClick={() => {
              if (window.confirm("确定清空全部历史提问吗？")) setHistory([]);
            }}
          >
            清空
          </button>
        </div>
        <div className="history-list" role="list">
          {history.length === 0 ? (
            <p className="history-empty">提交问题后，记录会保存在当前浏览器中。</p>
          ) : (
            history.map((item) => (
              <article
                className={`history-item history-item--${item.category}`}
                key={item.query}
                role="listitem"
              >
                <button
                  className="history-query"
                  type="button"
                  title={item.query}
                  onClick={() => chooseHistory(item)}
                >
                  {item.query}
                </button>
                <div className="history-meta">
                  <time dateTime={new Date(item.updatedAt).toISOString()}>
                    {formatTime(item.updatedAt)}
                  </time>
                  <span>{categoryLabels[item.category]}</span>
                </div>
                <button
                  className="history-remove"
                  type="button"
                  aria-label="删除这条记录"
                  onClick={() => setHistory((current) => current.filter((entry) => entry.query !== item.query))}
                >
                  ×
                </button>
              </article>
            ))
          )}
        </div>
      </aside>

      <main>
        <header className="app-title">
          <h1>LawMind-127M</h1>
          <p>轻量法律语言模型与可审计 RAG。每个问题独立处理，结果可查看完整决策路径。</p>
        </header>

        <form className="query-form" onSubmit={submit}>
          <label htmlFor="legal-query">法律问题</label>
          <textarea
            id="legal-query"
            value={query}
            maxLength={MAX_QUERY_CHARACTERS}
            rows={3}
            placeholder="例如：民事法律行为有效需要满足什么条件？"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <div className="query-actions">
            <span>{query.length}/{MAX_QUERY_CHARACTERS}</span>
            <div className="query-buttons">
              <button className="primary-button" type="submit" disabled={loading}>
                {loading ? "正在检索…" : "开始检索"}
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={!result || loading}
                onClick={() => setView(view === "answer" ? "audit" : "answer")}
              >
                {view === "answer" ? "查看审计流程" : "返回最终答案"}
              </button>
            </div>
          </div>
        </form>

        {view === "answer" ? (
          <section className="answer-panel" aria-live="polite" aria-busy={loading}>
            <div className="answer-heading">
              <div>
                <span className={`status-dot ${error ? "status-dot--error" : ""}`} />
                <strong>{status}</strong>
              </div>
              {result && (
                <span className="route-summary">
                  {routeLabels[result.route] ?? result.route}
                  {result.answer_mode
                    ? ` / ${answerModeLabels[result.answer_mode] ?? result.answer_mode}`
                    : ""}
                </span>
              )}
            </div>

            {loading ? (
              <div className="loading-block">
                <span /><span /><span />
                <p>系统正在执行路由、检索和回答校验。</p>
              </div>
            ) : error ? (
              <p className="error-message">{error}</p>
            ) : result ? (
              <>
                <div className="answer-copy">{answerText(result.answer)}</div>
                {(result.status === "clarification_required"
                  || result.status === "processing_failed") && (
                  <div className="candidate-section">
                    {result.candidate_answer ? (
                      <button
                        className="secondary-button"
                        type="button"
                        aria-expanded={candidateOpen}
                        onClick={() => setCandidateOpen((current) => !current)}
                      >
                        {candidateOpen ? "收起模型原始输出" : "查看模型原始输出"}
                      </button>
                    ) : (
                      <p className="candidate-empty">模型未产生可展示的原始输出。</p>
                    )}
                    {candidateOpen && result.candidate_answer && (
                      <div className="candidate-copy">
                        <strong>LawMind-127M 模型原始输出</strong>
                        <p>以下内容未通过本地回答协议校验，不作为本次正式回答。</p>
                        <div>{result.candidate_answer}</div>
                      </div>
                    )}
                  </div>
                )}
                {result.status === "retrieved_evidence" && result.evidence.length > 0 && (
                  <details className="evidence-panel">
                    <summary>法律依据 <span>{result.evidence.length} 条</span></summary>
                    <div className="evidence-list">
                      {result.evidence.map((item, index) => (
                        <article key={`${item.law_name}-${item.article_no}-${index}`}>
                          <h3>《{item.law_name}》{articleLabel(item.article_no)}</h3>
                          <p>{item.content}</p>
                        </article>
                      ))}
                    </div>
                  </details>
                )}
              </>
            ) : (
              <p className="empty-copy">请输入问题后开始检索。</p>
            )}
          </section>
        ) : result ? <AuditFlow result={result} /> : null}

        <footer>
          <strong>免责声明：</strong>本服务仅供法律信息检索与学习，不构成法律意见；请以官方公布的现行法律文本及专业人士意见为准。
        </footer>
      </main>
    </div>
  );
}
