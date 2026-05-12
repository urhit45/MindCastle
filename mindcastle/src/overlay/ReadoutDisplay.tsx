import { useCastleStore } from '../store/castleStore';
import styles from './ReadoutDisplay.module.css';

const STATE_LABELS: Record<string, string> = {
  flow_available:   'FLOW AVAILABLE',
  low_energy:       'LOW ENERGY',
  transition_ready: 'TRANSITION READY',
  pre_crash:        'PRE-CRASH',
  recovery:         'RECOVERY',
  blocked_pattern:  'BLOCKED PATTERN',
  emergence:        'EMERGENCE',
};

export function ReadoutDisplay() {
  const sv = useCastleStore(s => s.stateVector);
  return (
    <div className={styles.container}>
      <div className={styles.state}>{STATE_LABELS[sv.readoutState] ?? sv.readoutState}</div>
      <div className={styles.task}>{sv.readoutTask}</div>
    </div>
  );
}
