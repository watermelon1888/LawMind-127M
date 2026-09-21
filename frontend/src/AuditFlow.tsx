import type { AnswerResponse, AuditEvent } from "./types";

type AuditRow = {
  label: string;
  value: string;
};

type ToolCall = {
  name: string;
  stageLabel: string;
  purpose: string;
  rows: AuditRow[];
  hits?: AuditHit[];
};

type AuditParameter = {
  label: string;
  value: string;
};

type AuditHit = {
  lawName: string;
  articleNo: string;
  sourceType: string;
};

type AuditBudget = {
  reserved: number;
  context: number;
};

type AuditCard = {
  key: string;
  title: string;
  summary: string;
  status: string;
  rawStatus: string;
  rows: AuditRow[];
  toolCalls?: ToolCall[];
  parameters?: AuditParameter[];
  hits?: AuditHit[];
  hitLabel?: string;
  budget?: AuditBudget;
  agentic?: boolean;
  static?: boolean;
};

const routeLabels: Record<string, string> = {
  answer: "法律回答",
  clarify: "需要澄清",
  refuse: "策略拒答",
  general_chat: "通用对话",
};

const answerModeLabels: Record<string, string> = {
  retrieval: "检索后回答",
  exact_lookup: "精确查找法条",
};

const eventStatusLabels: Record<string, string> = {
  succeeded: "已完成",
  failed: "失败",
  skipped: "未执行",
  fallback: "已采用固定处理",
};

const sourceLabels: Record<string, string> = {
  core: "系统",
  deterministic: "程序规则",
  exact_lookup: "法条查找器",
  semantic_retriever: "混合检索器",
  external_llm: "外部语言模型",
  evidence_packager: "证据构建器",
  rag_generator: "LawMind-127M 法律回答模型",
  answer_protocol: "本地回答协议",
  answering: "回答渲染器",
  fallback: "固定处理规则",
};

const reasonLabels: Record<string, string> = {
  sufficient: "现有证据和事实足以支持候选答案",
  missing_key_facts: "缺少作答所需的关键事实",
  input_requires_clarification: "原问题缺少继续处理所需的信息",
  exact_lookup_not_found: "没有找到与指定名称和条号一致的法条",
  no_retrieval_candidates: "没有检索到可用于回答的法条",
  missing_decisive_information: "问题缺少决定适用制度的核心信息",
  query_assessment_failed: "检索前问题判断未能完成",
  agentic_retrieval_failed: "证据助理未能完成证据选择",
  semantic_retrieval_failed: "混合检索未能完成",
  reranking_failed: "Cross-Encoder 精排未能完成",
  evidence_packaging_failed: "证据构建未能完成",
  generation_failed: "候选答案生成失败",
  output_validation_failed: "候选答案未通过本地协议校验",
  general_chat_unavailable: "通用对话服务不可用",
  general_chat_failed: "通用对话生成失败",
};

const sourceTypeLabels: Record<string, string> = {
  department_rule: "部门规章",
  legal_regulation: "法律法规",
};

function detail(event: AuditEvent | undefined, key: string): unknown {
  return event?.details?.[key];
}

function textValue(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) return value;
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  return null;
}

function numberValue(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function stringList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && Boolean(item.trim()))
    : [];
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function hitList(value: unknown): AuditHit[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item) => {
    const hit = objectValue(item);
    const lawName = textValue(hit?.law_name);
    const articleNo = textValue(hit?.article_no);
    const sourceType = textValue(hit?.source_type);
    return lawName && articleNo && sourceType
      ? [{ lawName, articleNo, sourceType }]
      : [];
  });
}

function toolCallList(value: unknown, addedHits: AuditHit[]): ToolCall[] {
  if (!Array.isArray(value)) return [];
  const calls: ToolCall[] = [];
  let searchRound = 0;
  for (const item of value) {
    const call = objectValue(item);
    const name = textValue(call?.name);
    if (!name) continue;
    const chunkIds = stringList(call?.chunk_ids);
    if (name === "search_law") {
      searchRound += 1;
      calls.push({
        name,
        stageLabel: `第 ${searchRound} 轮补充检索`,
        purpose: "仅在已观察候选缺少回答必需证据时，按原问题边界补充候选。",
        rows: [
          { label: "补充问题", value: textValue(call?.query) ?? "未记录" },
          { label: "返回候选", value: `${chunkIds.length} 条法条` },
        ],
      });
      continue;
    }
    if (name === "finalize_evidence") {
      const promoted = stringList(call?.promoted_chunk_ids);
      calls.push({
        name,
        stageLabel: "完成证据检查",
        purpose: "提交证据提升建议，最终 top-5 由程序确定性融合。",
        rows: [
          {
            label: "提交提升建议",
            value: promoted.length ? `${promoted.length} 条候选` : "未提交",
          },
          {
            label: "实际新增证据",
            value: addedHits.length
              ? `${addedHits.length} 条法条`
              : "0 条，初始 Cross-Encoder top-5 已充分",
          },
        ],
        hits: addedHits,
      });
      continue;
    }
    calls.push({
      name,
      stageLabel: "未知工具调用",
      purpose: "系统记录了未识别的工具调用。",
      rows: [],
    });
  }
  return calls;
}

function articleLabel(articleNo: string): string {
  const [main, suffix] = articleNo.split("之");
  return suffix ? `第${main}条之${suffix}` : `第${main}条`;
}

function eventStatus(event: AuditEvent): string {
  return eventStatusLabels[event.status] ?? "已记录";
}

function source(event: AuditEvent): string {
  return event.source ? sourceLabels[event.source] ?? "系统组件" : "系统组件";
}

function reason(event: AuditEvent): string | null {
  return event.reason ? reasonLabels[event.reason] ?? event.reason : null;
}

function route(value: unknown): string {
  const key = textValue(value);
  return key ? routeLabels[key] ?? "其他处理路径" : "未记录";
}

function answerMode(value: unknown): string {
  const key = textValue(value);
  return key ? answerModeLabels[key] ?? "法律回答" : "未记录";
}

function questionClassification(event: AuditEvent): string {
  const selectedRoute = textValue(detail(event, "route"));
  const selectedMode = textValue(detail(event, "answer_mode"));
  if (selectedRoute === "answer" && selectedMode === "exact_lookup") {
    return "明确法条查询";
  }
  if (selectedRoute === "answer" && selectedMode === "retrieval") {
    return "法律检索问题";
  }
  if (selectedRoute === "general_chat") return "非法律问题";
  if (selectedRoute === "clarify") return "输入不完整或范围过宽";
  if (selectedRoute === "refuse") {
    if (event.reason === "time_sensitive") return "历史时点法律问题";
    if (event.reason === "unsupported_legal_source") return "暂不支持的法律来源";
    if (event.reason === "unsupported_legal_task") return "暂不支持的法律任务";
    return "策略受限问题";
  }
  return "未识别";
}

function cardForEvent(
  event: AuditEvent,
  index: number,
  events: AuditEvent[],
  result: AnswerResponse,
): AuditCard | null {
  const key = `${event.stage}-${index}`;
  const base = {
    key,
    status: eventStatus(event),
    rawStatus: event.status,
  };

  if (event.stage === "request_received" || event.stage === "completed") return null;
  if (["answer_review", "answer_retry", "answer_adjustment", "answer_quality"].includes(event.stage)) {
    return null;
  }
  if (
    event.stage === "protocol_validation"
    && events.some((item) => item.stage === "answer_generation")
  ) return null;

  if (event.stage === "route_decided") {
    const selectedRoute = route(detail(event, "route"));
    const selectedMode = textValue(detail(event, "answer_mode"));
    return {
      ...base,
      title: "识别用户问题",
      summary: `系统识别问题类型后，将本次请求交给“${selectedRoute}”流程。`,
      rows: [
        { label: "问题分类", value: questionClassification(event) },
        { label: "处理路径", value: selectedMode ? answerMode(selectedMode) : selectedRoute },
        { label: "分类依据", value: source(event) },
      ],
    };
  }

  if (event.stage === "query_assessment") {
    const decision = textValue(detail(event, "decision"));
    const clarification = events.find((item) => item.stage === "clarification");
    const question = textValue(detail(clarification, "question"));
    const rows: AuditRow[] = [
      {
        label: "判断结论",
        value: decision === "answer"
          ? "问题信息足以进入法律检索"
          : decision === "clarify"
            ? "需要先补充核心信息"
            : eventStatus(event),
      },
      { label: "判断组件", value: source(event) },
    ];
    if (question) rows.push({ label: "澄清问题", value: question });
    const eventReason = reason(event);
    if (eventReason) rows.push({ label: "处理说明", value: eventReason });
    return {
      ...base,
      title: "判断是否需要澄清",
      summary: event.status === "failed"
        ? "外部模型未能完成问题判断，本次请求按错误处理且不默认进入检索。"
        : decision === "clarify"
          ? "问题缺少决定适用制度的核心信息，系统先提出一条必要追问。"
          : "问题可以给出一般性或条件式法律回答，继续进入检索。",
      rows,
    };
  }

  if (event.stage === "exact_lookup") {
    const count = numberValue(detail(event, "article_count")) ?? 0;
    return {
      ...base,
      title: "精确查找法条",
      summary: count ? `已找到 ${count} 条与指定名称和条号一致的法条。` : "未找到可直接交付的指定法条。",
      rows: [
        { label: "查找结果", value: count ? `${count} 条法条` : "未找到" },
        { label: "执行组件", value: source(event) },
      ],
    };
  }

  if (event.stage === "retrieval") {
    const count = numberValue(detail(event, "candidate_count")) ?? 0;
    const parameters = objectValue(detail(event, "parameters"));
    return {
      ...base,
      title: "混合检索",
      summary: event.status === "succeeded"
        ? `已按原问题完成检索，得到 ${count} 条候选法条。`
        : "混合检索未能正常完成。",
      rows: [
        { label: "检索问题", value: result.query },
        { label: "检索结果", value: `${count} 条候选法条` },
      ],
      parameters: [
        { label: "Dense top-k", value: textValue(parameters?.dense_top_k) ?? "未记录" },
        { label: "BM25 top-k", value: textValue(parameters?.bm25_top_k) ?? "未记录" },
        { label: "RRF k", value: textValue(parameters?.rrf_k) ?? "未记录" },
        { label: "候选池", value: textValue(parameters?.candidate_pool) ?? "未记录" },
      ],
      hits: hitList(detail(event, "hits")),
      hitLabel: "Hybrid 候选法条",
    };
  }

  if (event.stage === "reranking") {
    const count = numberValue(detail(event, "candidate_count")) ?? 0;
    return {
      ...base,
      title: "Cross-Encoder 精排",
      summary: event.status === "succeeded"
        ? `已对混合检索候选统一打分，选出初始 top-${count}。`
        : "Cross-Encoder 未能完成候选精排。",
      rows: [
        { label: "精排结果", value: `${count} 条候选法条` },
        { label: "执行组件", value: source(event) },
      ],
      hits: hitList(detail(event, "hits")),
      hitLabel: "初始 Cross-Encoder top-5",
    };
  }

  if (event.stage === "agentic_retrieval") {
    const supplementalCount = numberValue(detail(event, "supplemental_count")) ?? 0;
    const observedCount = numberValue(detail(event, "observed_count"));
    const selectedCount = numberValue(detail(event, "selected_count"));
    const addedHits = hitList(detail(event, "added_hits"));
    const toolCalls = toolCallList(detail(event, "tool_calls"), addedHits);
    const summary = event.status === "failed"
      ? "证据助理未能完成工具调用或证据提交，本次请求按错误处理。"
      : supplementalCount
        ? `证据助理完成 ${supplementalCount} 次补充检索，再由程序融合最终回答证据。`
        : addedHits.length
          ? `证据助理未补充检索，并从 Hybrid top-20 中纠偏新增 ${addedHits.length} 条证据。`
          : "证据助理确认初始 Cross-Encoder top-5 已充分。";
    const rows: AuditRow[] = [
      {
        label: "证据选择",
        value: event.status === "failed"
          ? "选择失败"
          : supplementalCount
            ? "补充检索后确定"
            : addedHits.length ? "top-20 内纠偏" : "初始 top-5 充分",
      },
      { label: "执行组件", value: source(event) },
    ];
    if (observedCount !== null) rows.push({ label: "已观察候选", value: `${observedCount} 条法条` });
    if (selectedCount !== null) rows.push({ label: "最终证据", value: `${selectedCount} 条法条` });
    const eventReason = reason(event);
    if (eventReason) rows.push({ label: "处理说明", value: eventReason });
    return {
      ...base,
      title: "证据助理",
      summary,
      rows,
      toolCalls,
      hits: hitList(detail(event, "selected_hits")),
      hitLabel: "程序融合后的最终证据",
      agentic: true,
    };
  }

  if (event.stage === "evidence_selected" || event.stage === "evidence_packaging") {
    const count = numberValue(
      detail(event, event.stage === "evidence_selected" ? "article_count" : "evidence_count"),
    ) ?? 0;
    const reserved = numberValue(detail(event, "context_budget_tokens"));
    const context = numberValue(detail(event, "context_limit"));
    return {
      ...base,
      title: "构建回答证据",
      summary: event.status === "succeeded"
        ? `已选取并整理 ${count} 条法律依据。`
        : "法律证据未能完成整理。",
      rows: [
        { label: "证据数量", value: `${count} 条` },
        { label: "执行组件", value: source(event) },
      ],
      budget: reserved !== null && context !== null
        ? { reserved, context }
        : undefined,
      hits: hitList(detail(event, "hits")),
      hitLabel: "证据包内容",
    };
  }

  if (event.stage === "answer_generation") {
    const validation = events.find((item) => item.stage === "protocol_validation");
    const citations = numberValue(detail(event, "citation_count"));
    const succeeded = event.status === "succeeded" && validation?.status === "succeeded";
    const rows: AuditRow[] = [
      { label: "生成模型", value: source(event) },
      {
        label: "本地校验",
        value: validation ? eventStatus(validation) : "未执行",
      },
    ];
    if (citations !== null) rows.splice(1, 0, { label: "引用数量", value: `${citations} 条` });
    const failureReason = validation && validation.status !== "succeeded"
      ? reason(validation)
      : reason(event);
    if (failureReason) rows.push({ label: "处理说明", value: failureReason });
    return {
      ...base,
      rawStatus: succeeded ? "succeeded" : event.status === "failed" ? "failed" : validation?.status ?? event.status,
      status: succeeded ? "已通过" : validation ? eventStatus(validation) : eventStatus(event),
      title: "生成并校验法律回答",
      summary: succeeded
        ? "LawMind 回答已生成，并通过结构、引用和证据边界校验，可直接交付。"
        : "法律回答生成或本地协议校验未能通过。",
      rows,
    };
  }

  if (event.stage === "protocol_validation") {
    return {
      ...base,
      title: "校验候选答案",
      summary: event.status === "succeeded"
        ? "候选答案已通过结构、引用和证据边界校验。"
        : "候选答案未通过本地回答协议校验，因此没有交付。",
      rows: [
        { label: "本地校验", value: eventStatus(event) },
        { label: "执行组件", value: source(event) },
        ...(reason(event) ? [{ label: "处理说明", value: reason(event)! }] : []),
      ],
    };
  }

  if (event.stage === "clarification") {
    if (events.some((item) => item.stage === "query_assessment")) return null;
    const question = textValue(detail(event, "question")) ?? result.answer.replace(/^提示：\s*/, "").trim();
    return {
      ...base,
      title: "提出澄清问题",
      summary: "系统需要用户补充事实后才能继续回答。",
      rows: [
        { label: "追问问题", value: question },
        { label: "处理方式", value: source(event) },
      ],
    };
  }

  if (event.stage === "final_render") {
    return {
      ...base,
      title: "输出结果",
      summary: "处理结果已交付到回答区。",
      rows: [],
      static: true,
    };
  }

  if (event.stage === "general_chat_generation") {
    return {
      ...base,
      title: "生成通用回答",
      summary: event.status === "succeeded" ? "外部语言模型已生成通用回答。" : "通用回答未能生成。",
      rows: [
        { label: "生成方", value: source(event) },
        ...(reason(event) ? [{ label: "处理说明", value: reason(event)! }] : []),
      ],
    };
  }

  return {
    ...base,
    title: "处理节点",
    summary: event.status === "succeeded" ? "该处理节点已完成。" : "该处理节点未能正常完成。",
    rows: [
      { label: "执行组件", value: source(event) },
      ...(reason(event) ? [{ label: "处理说明", value: reason(event)! }] : []),
    ],
  };
}

export function AuditFlow({ result }: { result: AnswerResponse }) {
  const events = result.audit_trace.events;
  const cards = events
    .map((event, index) => cardForEvent(event, index, events, result))
    .filter((card): card is AuditCard => card !== null);

  return (
    <section className="audit-panel">
      <div className="audit-header">
        <h2>审计流程</h2>
        <span>共 {cards.length} 个处理节点</span>
      </div>
      <div className="audit-list">
        {cards.map((card, index) => card.static ? (
          <article
            className={`audit-event audit-event--${card.rawStatus} audit-event--static`}
            key={card.key}
          >
            <div className="audit-static-summary">
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{card.title}</strong>
              <small>{card.status}</small>
            </div>
          </article>
        ) : (
          <details
            className={`audit-event audit-event--${card.rawStatus}${card.agentic ? " audit-event--agentic" : ""}`}
            key={card.key}
            open={index === 0 || Boolean(card.toolCalls?.length) || index === cards.length - 1}
          >
            <summary>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{card.title}</strong>
              <small>{card.status}</small>
            </summary>
            <div className="audit-event-body">
              <p>{card.summary}</p>
              {card.parameters && (
                <dl className="audit-parameters" aria-label="检索参数">
                  {card.parameters.map((parameter) => (
                    <div key={`${card.key}-${parameter.label}`}>
                      <dt>{parameter.label}</dt>
                      <dd>{parameter.value}</dd>
                    </div>
                  ))}
                </dl>
              )}
              {card.budget && (
                <div className="audit-budget">
                  <div>
                    <span style={{ width: `${Math.min(100, card.budget.reserved / card.budget.context * 100)}%` }} />
                  </div>
                  <strong>证据包预算 {card.budget.reserved} / {card.budget.context}</strong>
                  <small>给证据包预留的 Token / 模型上下文长度</small>
                </div>
              )}
              {card.rows.length > 0 && (
                <dl className="audit-details">
                  {card.rows.map((row) => (
                    <div key={`${card.key}-${row.label}`}>
                      <dt>{row.label}</dt>
                      <dd>{row.value}</dd>
                    </div>
                  ))}
                </dl>
              )}
              {card.toolCalls && card.toolCalls.length > 0 && (
                <div className="tool-call-list" aria-label="证据助理工具调用">
                  {card.toolCalls.map((call, callIndex) => (
                    <article key={`${call.name}-${callIndex}`}>
                      <span className="tool-call-index">{callIndex + 1}</span>
                      <div className="tool-call-content">
                        <header>
                          <strong>{call.name}</strong>
                          <span>{call.stageLabel}</span>
                        </header>
                        <p>{call.purpose}</p>
                        {call.rows.map((row) => (
                          <dl key={`${call.name}-${callIndex}-${row.label}`}>
                            <dt>{row.label}</dt>
                            <dd>{row.value}</dd>
                          </dl>
                        ))}
                        {call.hits && call.hits.length > 0 && (
                          <ol className="audit-hit-list tool-hit-list" aria-label="实际新增证据">
                            {call.hits.map((hit, hitIndex) => (
                              <li key={`${hit.lawName}-${hit.articleNo}-${hitIndex}`}>
                                <span>《{hit.lawName}》{articleLabel(hit.articleNo)}</span>
                                <small>{sourceTypeLabels[hit.sourceType] ?? "法律法规"}</small>
                              </li>
                            ))}
                          </ol>
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              )}
              {card.hits && card.hits.length > 0 && (
                <div className="audit-hits">
                  {card.hitLabel && <strong>{card.hitLabel}</strong>}
                  <ol className="audit-hit-list" aria-label={`${card.title}命中法条`}>
                    {card.hits.map((hit, hitIndex) => (
                      <li key={`${hit.lawName}-${hit.articleNo}-${hitIndex}`}>
                        <span>《{hit.lawName}》{articleLabel(hit.articleNo)}</span>
                        <small>{sourceTypeLabels[hit.sourceType] ?? "法律法规"}</small>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          </details>
        ))}
      </div>
    </section>
  );
}
