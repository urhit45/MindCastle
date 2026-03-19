import React, { useEffect, useMemo, useState } from 'react';
import type { NextTaskCandidate } from './OneNextTaskView';

interface FocusModeProps {
  candidate: NextTaskCandidate;
  onExit: (reason: 'done' | 'break') => void;
}

function formatElapsed(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export default function FocusMode({ candidate, onExit }: FocusModeProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [logText, setLogText] = useState('');
  const [entries, setEntries] = useState<string[]>([]);
  const [artifactProgress, setArtifactProgress] = useState(12);

  useEffect(() => {
    const timer = window.setInterval(() => setElapsedSeconds((prev) => prev + 1), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onExit('break');
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onExit]);

  const ambientPercent = useMemo(() => ((elapsedSeconds % 90) / 90) * 100, [elapsedSeconds]);
  const sessionProgress = Math.min(100, Math.round((elapsedSeconds / (25 * 60)) * 100));

  const saveEntry = () => {
    const next = logText.trim();
    if (!next) return;
    setEntries((prev) => [next, ...prev].slice(0, 4));
    setLogText('');
    setArtifactProgress((prev) => Math.min(prev + 3, 100));
  };

  return (
    <div className="focus-mode-overlay">
      <div className="focus-mode-backdrop" />
      <section className="focus-mode-panel">
        <header className="focus-mode-header">
          <p className="focus-mode-kicker">{candidate.engineIcon} Focus Mode</p>
          <h2>{candidate.artifact}</h2>
          <p>{candidate.reasonLabel}</p>
        </header>

        <main className="focus-mode-main">
          <p className="focus-mode-label">One next step</p>
          <h3 className="focus-mode-task">{candidate.task}</h3>

          <div className="time-flow">
            <div className="time-flow-orb" style={{ '--ambient-progress': `${ambientPercent}%` } as React.CSSProperties} />
            <p>Time flow {formatElapsed(elapsedSeconds)}</p>
          </div>

          <div className="focus-progress-grid">
            <div>
              <div className="focus-progress-head">
                <span>Artifact progress</span>
                <span>{artifactProgress}%</span>
              </div>
              <div className="focus-progress-track">
                <div className="focus-progress-fill" style={{ width: `${artifactProgress}%` }} />
              </div>
            </div>
            <div>
              <div className="focus-progress-head">
                <span>Session pace (25m)</span>
                <span>{sessionProgress}%</span>
              </div>
              <div className="focus-progress-track">
                <div className="focus-progress-fill focus-progress-fill-alt" style={{ width: `${sessionProgress}%` }} />
              </div>
            </div>
          </div>

          <div className="focus-log">
            <label htmlFor="focus-log-input">Log entry</label>
            <div className="focus-log-row">
              <input
                id="focus-log-input"
                type="text"
                className="text-input"
                value={logText}
                onChange={(event) => setLogText(event.target.value)}
                placeholder="Capture a thought without leaving focus..."
              />
              <button type="button" className="pill-btn" onClick={saveEntry}>
                Save
              </button>
            </div>
            {entries.length > 0 && (
              <ul className="focus-log-list">
                {entries.map((entry) => (
                  <li key={entry}>{entry}</li>
                ))}
              </ul>
            )}
          </div>
        </main>

        <footer className="focus-mode-footer">
          <button type="button" className="primary-btn" onClick={() => onExit('done')}>
            I'm done
          </button>
          <button type="button" className="pill-btn" onClick={() => onExit('break')}>
            I need a break
          </button>
        </footer>
      </section>
    </div>
  );
}
