import React, { useEffect, useState } from 'react';
import type { EngineSwitchRequest } from './OneNextTaskView';

interface TransitionModeProps {
  transition: EngineSwitchRequest;
  onComplete: () => void;
}

type RitualPhase = 'leave' | 'breathe' | 'enter';

const PHASE_DURATIONS_MS: Record<RitualPhase, number> = {
  leave: 15000,
  breathe: 30000,
  enter: 15000,
};

const PHASES: RitualPhase[] = ['leave', 'breathe', 'enter'];
const PHASE_LABELS: Record<RitualPhase, string> = {
  leave: 'Leave',
  breathe: 'Breathe',
  enter: 'Enter',
};

export default function TransitionMode({ transition, onComplete }: TransitionModeProps) {
  const [phase, setPhase] = useState<RitualPhase>('leave');
  const [reflection, setReflection] = useState('');

  useEffect(() => {
    if (phase === 'leave') {
      const timer = window.setTimeout(() => setPhase('breathe'), PHASE_DURATIONS_MS.leave);
      return () => window.clearTimeout(timer);
    }
    if (phase === 'breathe') {
      const timer = window.setTimeout(() => setPhase('enter'), PHASE_DURATIONS_MS.breathe);
      return () => window.clearTimeout(timer);
    }
    const timer = window.setTimeout(onComplete, PHASE_DURATIONS_MS.enter);
    return () => window.clearTimeout(timer);
  }, [phase, onComplete]);

  return (
    <div className="transition-overlay">
      <div className="transition-backdrop" />
      <section className="transition-panel">
        <header className="transition-header">
          <p className="transition-kicker">Transition Mode</p>
          <h2>Context switch ritual</h2>
          <button type="button" className="pill-btn" onClick={onComplete}>
            Skip ritual
          </button>
        </header>
        <div className="transition-phases" aria-label="Transition ritual phases">
          {PHASES.map((p) => {
            const isActive = phase === p;
            const isComplete = PHASES.indexOf(p) < PHASES.indexOf(phase);
            return (
              <div
                key={p}
                className={`transition-phase-pill ${isActive ? 'is-active' : ''} ${isComplete ? 'is-complete' : ''}`.trim()}
              >
                {PHASE_LABELS[p]}
              </div>
            );
          })}
        </div>

        <main className="transition-main">
          {phase === 'leave' && (
            <div className="transition-stage">
              <p className="transition-stage-title">You are leaving {transition.fromEngine}</p>
              <label htmlFor="transition-reflection">One line reflection</label>
              <input
                id="transition-reflection"
                type="text"
                className="text-input"
                value={reflection}
                onChange={(event) => setReflection(event.target.value)}
                placeholder="What matters from this engine?"
              />
            </div>
          )}

          {phase === 'breathe' && (
            <div className="transition-stage">
              <p className="transition-stage-title">Pause and breathe</p>
              <div className="transition-breath-orb" />
              <p className="transition-stage-copy">No timer pressure. Just settle before the next context.</p>
            </div>
          )}

          {phase === 'enter' && (
            <div className="transition-stage">
              <p className="transition-stage-title">You are entering {transition.toEngine}</p>
              <p className="transition-stage-copy">Next step: {transition.toNextStep}</p>
              <button type="button" className="primary-btn" onClick={onComplete}>
                Return to castle
              </button>
            </div>
          )}
        </main>
      </section>
    </div>
  );
}
