import React, { useMemo, useState } from 'react';
import { getHomeReview, type ReviewItem } from '../api/client';

export type NextTaskCandidate = {
  id: string;
  task: string;
  artifact: string;
  engineLabel: string;
  engineIcon: string;
  reasonLabel: string;
  source: ReviewItem;
};

export type EngineSwitchRequest = {
  fromEngine: string;
  toEngine: string;
  toNextStep: string;
};

const ENGINE_THEME: Record<string, { label: string; icon: string }> = {
  blocked: { label: 'Unblock', icon: '🚧' },
  momentum: { label: 'Momentum', icon: '🏃' },
  stall_guard: { label: 'Stall Guard', icon: '🛡️' },
  suggestion: { label: 'Scout Suggestion', icon: '🧭' },
};

function prettifyReason(reason: ReviewItem['reason']): string {
  if (reason === 'blocked') return 'Blocked item with a next step';
  if (reason === 'next_step') return 'Recent artifact with a clear next step';
  if (reason === 'stale') return 'Stale artifact to revive';
  if (reason === 'overdue') return 'Overdue task';
  return 'Task due soon';
}

function buildCandidates(items: ReviewItem[]): NextTaskCandidate[] {
  const blockedWithNext = items.find(
    (item) => item.reason === 'blocked' && Boolean(item.next_step?.trim()),
  );
  const mostRecent = [...items]
    .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
    .find((item) => item.reason === 'next_step' || item.reason === 'blocked');
  const stale = items.find((item) => item.reason === 'stale');
  const recentSuggestion = [...items]
    .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''))
    .find((item) => Boolean(item.next_step?.trim()));

  const ordered: Array<{ key: string; item?: ReviewItem }> = [
    { key: 'blocked', item: blockedWithNext },
    { key: 'momentum', item: mostRecent },
    { key: 'stall_guard', item: stale },
    { key: 'suggestion', item: recentSuggestion },
  ];

  const seen = new Set<string>();
  const candidates: NextTaskCandidate[] = [];

  ordered.forEach(({ key, item }) => {
    if (!item || seen.has(item.id)) return;
    seen.add(item.id);
    const theme = ENGINE_THEME[key];
    candidates.push({
      id: item.id,
      task: item.next_step?.trim() || item.title,
      artifact: item.title,
      engineLabel: theme.label,
      engineIcon: theme.icon,
      reasonLabel: prettifyReason(item.reason),
      source: item,
    });
  });

  if (candidates.length === 0) {
    candidates.push({
      id: 'fallback',
      task: 'Capture one progress note for your most important artifact.',
      artifact: 'Castle Front Door',
      engineLabel: 'Warm Start',
      engineIcon: '🌱',
      reasonLabel: 'No actionable review items yet',
      source: {
        id: 'fallback',
        type: 'progress_log',
        reason: 'next_step',
        title: 'Warm start fallback',
      },
    });
  }

  return candidates;
}

interface OneNextTaskViewProps {
  onStartFocusMode?: (candidate: NextTaskCandidate) => void;
  onEngineSwitch?: (request: EngineSwitchRequest) => void;
  onCandidateChange?: (candidate: NextTaskCandidate) => void;
  transitionModeEnabled?: boolean;
  onTransitionModeEnabledChange?: (enabled: boolean) => void;
  preferredNodeId?: string | null;
  onClearPreferredNode?: () => void;
}

export default function OneNextTaskView({
  onStartFocusMode,
  onEngineSwitch,
  onCandidateChange,
  transitionModeEnabled = true,
  onTransitionModeEnabledChange,
  preferredNodeId,
  onClearPreferredNode,
}: OneNextTaskViewProps) {
  const [items, setItems] = useState<ReviewItem[]>([]);
  const [index, setIndex] = useState(0);
  const [status, setStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    let mounted = true;
    const load = async () => {
      setStatus('loading');
      setError(null);
      try {
        const response = await getHomeReview();
        if (!mounted) return;
        setItems(response.items);
        setStatus('ready');
      } catch (e) {
        if (!mounted) return;
        setError(e instanceof Error ? e.message : 'Failed to load next task');
        setStatus('error');
      }
    };
    load();
    return () => {
      mounted = false;
    };
  }, []);

  const candidates = useMemo(() => buildCandidates(items), [items]);
  const candidate = candidates[index % candidates.length];

  React.useEffect(() => {
    if (!preferredNodeId || candidates.length === 0) return;
    const preferredIndex = candidates.findIndex((c) => c.source.nodeId === preferredNodeId);
    if (preferredIndex >= 0 && preferredIndex !== index) {
      setIndex(preferredIndex);
    }
  }, [preferredNodeId, candidates, index]);

  React.useEffect(() => {
    onCandidateChange?.(candidate);
  }, [candidate, onCandidateChange]);

  const rotateCandidate = () => {
    const nextIndex = (index + 1) % candidates.length;
    const nextCandidate = candidates[nextIndex];
    if (transitionModeEnabled && nextCandidate && nextCandidate.engineLabel !== candidate.engineLabel) {
      onEngineSwitch?.({
        fromEngine: candidate.engineLabel,
        toEngine: nextCandidate.engineLabel,
        toNextStep: nextCandidate.task,
      });
    }
    setIndex(nextIndex);
  };

  return (
    <section className="next-task-view panel">
      <div className="panel-header">
        <h2>One Next Task</h2>
        <p>What should I do right now?</p>
      </div>

      <div className="next-task-body">
        {status === 'loading' && <p className="next-task-meta">Scanning your castle...</p>}
        {status === 'error' && (
          <p className="next-task-meta">Using fallback suggestion ({error ?? 'Unknown issue'}).</p>
        )}

        <p className="next-task-engine">
          <span>{candidate.engineIcon}</span>
          <span>{candidate.engineLabel}</span>
        </p>
        <h3 className="next-task-title">{candidate.task}</h3>
        <p className="next-task-artifact">{candidate.artifact}</p>
        <p className="next-task-meta">{candidate.reasonLabel}</p>
        {preferredNodeId && (
          <p className="next-task-meta">
            Navigating from mind map node.
            <button type="button" className="next-task-clear" onClick={onClearPreferredNode}>
              Clear
            </button>
          </p>
        )}

        <div className="next-task-actions">
          <button
            type="button"
            className="primary-btn"
            onClick={() => onStartFocusMode?.(candidate)}
          >
            Start Focus Mode
          </button>
          <button type="button" className="pill-btn" onClick={rotateCandidate}>
            Not this - show me another
          </button>
        </div>

        <label className="next-task-toggle">
          <input
            type="checkbox"
            checked={transitionModeEnabled}
            onChange={(event) => onTransitionModeEnabledChange?.(event.target.checked)}
          />
          <span>Enable transition ritual between engine shifts</span>
        </label>
        <p className="next-task-toggle-help">
          Triggers when you switch engine context and when Focus Mode ends without selecting your next engine.
        </p>
      </div>
    </section>
  );
}
