import { useEffect, useState } from 'react';
import { useCastleStore } from '../store/castleStore';
import styles from './EngineTooltip.module.css';

export function EngineTooltip() {
  const { engines, hoveredEngineId } = useCastleStore();
  const [pos, setPos] = useState({ x: 0, y: 0 });

  const engine = engines.find(e => e.id === hoveredEngineId);

  useEffect(() => {
    const onMove = (e: MouseEvent) => setPos({ x: e.clientX + 16, y: e.clientY - 8 });
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, []);

  if (!engine) return null;

  const nextTask = engine.artifacts.find(a => a.next)?.next;

  return (
    <div
      className={styles.tooltip}
      style={{ left: pos.x, top: pos.y }}
    >
      <div className={styles.name}>
        <span className={styles.icon}>{engine.icon}</span>
        {engine.name}
      </div>
      <div className={styles.subtitle}>{engine.subtitle}</div>
      {nextTask && <div className={styles.next}>→ {nextTask}</div>}
    </div>
  );
}
