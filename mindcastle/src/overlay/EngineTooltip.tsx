
import { useCastleStore } from '../store/castleStore';
import styles from './EngineTooltip.module.css';

interface EngineTooltipProps {
  visible: boolean;
  x: number;
  y: number;
  engineId: string | null;
}

export function EngineTooltip({ visible, x, y, engineId }: EngineTooltipProps) {
  const { engines } = useCastleStore();
  const engine = engineId ? engines.find((e) => e.id === engineId) : null;

  if (!engine) return null;

  const firstArtifact = engine.artifacts[0];

  return (
    <div
      className={`${styles.tooltip} ${visible ? styles.visible : ''}`}
      style={{ left: x + 16, top: y - 20 }}
    >
      <div className={styles.name}>{engine.icon}{'  '}{engine.name}</div>
      <div className={styles.subtitle}>{engine.subtitle}</div>
      <div className={styles.next}>
        {firstArtifact ? `→ ${firstArtifact.next ?? ''}` : 'No active tasks'}
      </div>
    </div>
  );
}
