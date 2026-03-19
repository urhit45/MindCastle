/**
 * TinyNet API Client
 * Handles communication with the backend reasoning and classification endpoints
 */

export interface ReasonRequest {
  text: string;
  pred: {
    cats: string[];
    state: string;
  };
  context?: {
    recent_nodes?: Array<{id: string; title: string}>;
    last_steps?: string[];
    constraints?: Record<string, any>;
  };
}

export interface ReasonResponse {
  subtasks: string[];
  blockers: string[];
  next_step: {
    template: string;
    slots?: Record<string, any>;
  };
  link_to: Array<{
    nodeId: string;
    reason: string;
  }>;
}

export interface ClassifyResponse {
  categories: Array<{label: string; score: number}>;
  state: {label: string; score: number};
  linkHints: Array<{nodeId: string; title: string; similarity: number}>;
  nextStep: {template: string; slots?: Record<string, any>; confidence: number};
  uncertain: boolean;
  route: 'needs_confirm' | 'suggest_plan' | 'auto_save_ok';
}

export interface ReviewItem {
  id: string;
  type: 'progress_log' | 'todo';
  reason: 'blocked' | 'next_step' | 'stale' | 'overdue' | 'due_soon';
  title: string;
  nodeId?: string | null;
  created_at?: string;
  due_at?: string;
  state?: string;
  next_step?: string | null;
  priority?: 'high' | 'medium' | 'low';
}

export interface ReviewStats {
  total_items: number;
  blocked_count: number;
  next_step_count: number;
  stale_count: number;
  overdue_count: number;
  due_soon_count: number;
  high_priority: number;
  medium_priority: number;
  low_priority: number;
}

export interface ReviewResponse {
  ok: boolean;
  items: ReviewItem[];
  stats: ReviewStats;
  generated_at: string;
}

/**
 * Reason about a text input and get structured planning suggestions
 */
export async function reasonAbout(payload: ReasonRequest): Promise<ReasonResponse> {
  const r = await fetch("http://localhost:8002/reason/", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  
  if (!r.ok) {
    throw new Error(`Reasoning failed: ${r.status} ${r.statusText}`);
  }
  
  return await r.json();
}

/**
 * Classify text and get routing guidance
 */
export async function classifyText(text: string, contextNodeId?: string): Promise<ClassifyResponse> {
  const r = await fetch("http://localhost:8002/classify/", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ text, contextNodeId })
  });
  
  if (!r.ok) {
    throw new Error(`Classification failed: ${r.status} ${r.statusText}`);
  }
  
  return await r.json();
}

/**
 * Get home review items used for the "One Next Task" experience.
 */
export async function getHomeReview(limit = 25): Promise<ReviewResponse> {
  const r = await fetch(`http://localhost:8002/home/review?limit=${limit}`);

  if (!r.ok) {
    throw new Error(`Home review failed: ${r.status} ${r.statusText}`);
  }

  return await r.json();
}

/**
 * Send correction feedback for online learning
 */
export async function correctClassification(payload: {
  text: string;
  categories?: string[] | null;
  state?: string | null;
  linkTo?: string | null;
  nextStepTemplate?: string | null;
}): Promise<{ok: boolean}> {
  const r = await fetch("http://localhost:8002/classify/correct", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  
  if (!r.ok) {
    throw new Error(`Correction failed: ${r.status} ${r.statusText}`);
  }
  
  return await r.json();
}
