import type { AnswerResponse } from "./types";

type ErrorBody = {
  detail?: string | Array<{ msg?: string }>;
};

function errorMessage(body: ErrorBody | null, status: number): string {
  if (typeof body?.detail === "string") {
    return body.detail;
  }
  if (Array.isArray(body?.detail)) {
    const message = body.detail.find((item) => item.msg)?.msg;
    if (message) return message;
  }
  return `请求失败（HTTP ${status}）`;
}

export async function answerQuestion(
  query: string,
  signal?: AbortSignal,
): Promise<AnswerResponse> {
  const response = await fetch("/v1/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
    signal,
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ErrorBody | null;
    throw new Error(errorMessage(body, response.status));
  }
  return (await response.json()) as AnswerResponse;
}
