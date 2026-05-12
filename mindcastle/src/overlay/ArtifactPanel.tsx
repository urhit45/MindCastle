import { useRef } from 'react';
import { useCastleStore } from '../store/castleStore';
import type { Artifact } from '../store/types';
import styles from './ArtifactPanel.module.css';

declare global {
  interface Window {
    __mindcastleCam?: () => void;
  }
}

const STATUS_LABELS: Record<string, string> = {
  active:   'Active',
  blocked:  'Blocked',
  concept:  'Concept',
  planned:  'Planned',
  live:     'Live',
  planning: 'Planning',
};

function ArtifactCard({ artifact }: { artifact: Artifact }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardHeader}>
        <span className={styles.cardTitle}>{artifact.title}</span>
        <span className={`${styles.badge} ${styles[`badge_${artifact.status}`]}`}>
          {STATUS_LABELS[artifact.status]}
        </span>
      </div>
      {artifact.subtitle && <div className={styles.cardSubtitle}>{artifact.subtitle}</div>}
      {artifact.next && <div className={styles.cardNext}>→ {artifact.next}</div>}
      {artifact.stack && artifact.stack.length > 0 && (
        <div className={styles.stack}>
          {artifact.stack.map(s => <span key={s} className={styles.stackTag}>{s}</span>)}
        </div>
      )}
      {artifact.progress && artifact.progress.map(p => (
        <div key={p.label} className={styles.progressRow}>
          <span className={styles.progressLabel}>{p.label}</span>
          <div className={styles.progressTrack}>
            <div className={styles.progressFill} style={{ width: `${p.val}%` }} />
          </div>
          <span className={styles.progressVal}>{p.val}%</span>
        </div>
      ))}
      {artifact.notes && <div className={styles.notes}>{artifact.notes}</div>}
    </div>
  );
}

export function ArtifactPanel() {
  const { engines, selectedEngineId, setSelectedEngine } = useCastleStore();
  const engine = engines.find(e => e.id === selectedEngineId);
  const panelRef = useRef<HTMLDivElement>(null);

  // Camera reset handled in CastleWorld via selectedEngineId watch
  function close() {
    setSelectedEngine(null);
    // Trigger camera reset via store change — CastleWorld watches this
    // We expose it as a global for direct camera access
    window.__mindcastleCam?.();
  }

  return (
    <div
      ref={panelRef}
      className={`${styles.panel} ${engine ? styles.open : ''}`}
    >
      {engine && (
        <>
          <div className={styles.header}>
            <div className={styles.engineMeta}>
              <span className={styles.engineIcon}>{engine.icon}</span>
              <div>
                <div className={styles.engineName}>{engine.name}</div>
                <div className={styles.engineSubtitle}>{engine.subtitle}</div>
              </div>
            </div>
            <button className={styles.close} onClick={close} aria-label="Close">✕</button>
          </div>

          <div className={styles.activityRow}>
            <span className={styles.activityLabel}>Activity</span>
            <div className={styles.activityTrack}>
              <div
                className={styles.activityFill}
                style={{ width: `${engine.activityLevel * 100}%`, background: engine.color }}
              />
            </div>
          </div>

          <div className={styles.artifacts}>
            {engine.artifacts.length === 0 ? (
              <div className={styles.empty}>No artifacts yet</div>
            ) : (
              engine.artifacts.map(a => <ArtifactCard key={a.id} artifact={a} />)
            )}
          </div>
        </>
      )}
    </div>
  );
}
