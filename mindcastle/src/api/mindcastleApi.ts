import type { StateVector } from '../store/types';

const BASE = 'http://localhost:8000';

export async function classify(text: string) {
  const res = await fetch(`${BASE}/classify/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return res.json() as Promise<{ state: string; categories: string[]; next_step: string }>;
}

export async function getStateVector(): Promise<StateVector> {
  const res = await fetch(`${BASE}/state/`);
  return res.json() as Promise<StateVector>;
}

export async function writeLog(artifactId: string, text: string) {
  const res = await fetch(`${BASE}/nodes/${artifactId}/logs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function getReadout(payload: {
  schedulerOutput: unknown;
  stateVector: StateVector;
}): Promise<{ state: string; task: string }> {
  const res = await fetch(`${BASE}/readout/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return res.json() as Promise<{ state: string; task: string }>;
}

export function startStateVectorPolling(
  onUpdate: (sv: StateVector) => void,
): () => void {
  const interval = setInterval(async () => {
    try {
      const sv = await getStateVector();
      onUpdate(sv);
    } catch {
      // Backend offline — use cached state silently
    }
  }, 30_000);
  return () => clearInterval(interval);
}
