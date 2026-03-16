// Use the same host the browser loaded from so phones on LAN hit the right IP
const BASE = `http://${window.location.hostname}:8000`;

async function req<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

// ─── Node CRUD ───────────────────────────────────────────────────────────────

export interface ApiNode {
  id: string;
  title: string;
  hub: boolean;
  status: string;
  created_at: string;
}

export function createNode(title: string, isHub = false, status = "concept"): Promise<ApiNode> {
  return req("/nodes/", {
    method: "POST",
    body: JSON.stringify({ title, is_hub: isHub, status }),
  });
}

export function updateNode(id: string, patch: { title?: string; status?: string }): Promise<ApiNode> {
  return req(`/nodes/${id}`, {
    method: "PATCH",
    body: JSON.stringify(patch),
  });
}

export function deleteNode(id: string): Promise<{ ok: boolean }> {
  return req(`/nodes/${id}`, { method: "DELETE" });
}

// ─── Logs ────────────────────────────────────────────────────────────────────

export interface ApiLog {
  id: number;
  time: string;
  text: string;
  nextStep: string | null;
  state: string;
}

export function addLog(
  nodeId: string,
  text: string,
  state: string,
  nextStep?: string,
): Promise<ApiLog> {
  return req(`/nodes/${nodeId}/logs`, {
    method: "POST",
    body: JSON.stringify({ text, state, next_step: nextStep ?? null }),
  });
}

export function getLogs(nodeId: string): Promise<{ items: ApiLog[] }> {
  return req(`/nodes/${nodeId}/logs`);
}

// ─── Classify ────────────────────────────────────────────────────────────────

export interface ClassifyResult {
  categories: { label: string; score: number }[];
  state: { label: string; score: number };
  nextStep: { template: string; confidence: number };
  linkHints: unknown[];
  uncertain: boolean;
}

export function classifyText(text: string): Promise<ClassifyResult> {
  return req("/classify/", {
    method: "POST",
    body: JSON.stringify({ text }),
  });
}
