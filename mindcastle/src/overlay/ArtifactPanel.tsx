
import { useCastleStore } from '../store/castleStore';
import type { EngineStatus } from '../store/types';
import styles from './ArtifactPanel.module.css';

interface ArtifactPanelProps {
  onClose: () => void;
}

const STATUS_LABELS: Record<EngineStatus, string> = {
  active:   'active',
  blocked:  'blocked',
  concept:  'concept',
  planned:  'planned',
  live:     'live',
  planning: 'planning',
};

export function ArtifactPanel({ onClose }: ArtifactPanelProps) {
  const { engines, selectedEngineId } = useCastleStore();
  const engine = selectedEngineId ? engines.find((e) => e.id === selectedEngineId) : null;

  const isOpen = engine !== null && engine !== undefined;

  return (
    <div className={`${styles.panel} ${isOpen ? styles.open : ''}`}>
      <button className={styles.close} onClick={onClose} aria-label="Close panel">
        ✕
      </button>
      {engine && (
        <>
          <div className={styles.icon}>{engine.icon}</div>
          <div className={styles.name}>{engine.name}</div>
          <div className={styles.subtitle}>{engine.subtitle}</div>
          <div className={styles.artifacts}>
            {engine.artifacts.length === 0 ? (
              <div className={styles.empty}>No artifacts yet.</div>
            ) : (
              engine.artifacts.map((artifact) => (
                <div key={artifact.id} className={styles.card}>
                  <div className={styles.cardTitle}>{artifact.title}</div>
                  {artifact.next && (
                    <div className={styles.cardNext}>{artifact.next}</div>
                  )}
                  <span className={`${styles.badge} ${styles[`status_${artifact.status}`]}`}>
                    {STATUS_LABELS[artifact.status]}
                  </span>
                </div>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
