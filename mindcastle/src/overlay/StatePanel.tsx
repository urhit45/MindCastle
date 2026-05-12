import { useCastleStore } from '../store/castleStore';
import styles from './StatePanel.module.css';

const BARS = [
  { key: 'energy',    label: 'Energy',    color: '#4dcc66' },
  { key: 'momentum',  label: 'Momentum',  color: '#6680e6' },
  { key: 'risk',      label: 'Risk',      color: '#e66666' },
  { key: 'readiness', label: 'Readiness', color: '#e6b233' },
] as const;

export function StatePanel() {
  const sv = useCastleStore(s => s.stateVector);

  return (
    <div className={styles.panel}>
      <div className={styles.title}>MIND CASTLE</div>
      {BARS.map(({ key, label, color }) => {
        const val = sv[key];
        return (
          <div key={key} className={styles.row}>
            <span className={styles.label}>{label}</span>
            <div className={styles.track}>
              <div
                className={styles.fill}
                style={{ width: `${val * 100}%`, background: color }}
              />
            </div>
            <span className={styles.value}>{Math.round(val * 100).toString().padStart(3, ' ')}%</span>
          </div>
        );
      })}
    </div>
  );
}
