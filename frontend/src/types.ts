export type Evidence = {
  law_name: string;
  article_no: string;
  content: string;
};

export type AuditEvent = {
  stage: string;
  status: string;
  source?: string | null;
  reason?: string | null;
  details?: Record<string, unknown>;
  [key: string]: unknown;
};

export type AuditTrace = {
  trace_id: string;
  events: AuditEvent[];
  final_route?: string | null;
  final_status?: string | null;
};

export type AnswerResponse = {
  request_id: string;
  query: string;
  route: string;
  status: string;
  answer_mode: string | null;
  answer: string;
  candidate_answer: string | null;
  evidence: Evidence[];
  unanswered_reason: string | null;
  diagnostic_code: string | null;
  audit_trace: AuditTrace;
};

export type HistoryCategory =
  | "pending"
  | "exact"
  | "answer"
  | "clarify"
  | "refuse"
  | "general"
  | "error";

export type HistoryItem = {
  query: string;
  updatedAt: number;
  category: HistoryCategory;
  result?: AnswerResponse;
};
