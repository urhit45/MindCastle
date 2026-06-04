
import { useCastleStore } from '../store/castleStore';
import styles from './StatePanel.module.css';

const BAR_COLORS = {
  energy:    'linear-gradient(90deg, #4a7a5a, #6a9a7a)',
  momentum:  'linear-gradient(90deg, #4a5a7a, #6a7a9a)',
  risk:      'linear-gradient(90deg, #7a4a4a, #9a6a6a)',
  readiness: 'linear-gradient(90deg, #7a6a4a, #9a8a6a)',
} as const;

type BarKey = keyof typeof BAR_COLORS;

const BARS: { key: BarKey; label: string }[] = [
  { key: 'energy',    label: 'Energy' },
  { key: 'momentum',  label: 'Momentum' },
  { key: 'risk',      label: 'Risk' },
  { key: 'readiness', label: 'Readiness' },
];

export function StatePanel() {
  const { stateVector } = useCastleStore();

  return (
    <div className={styles.panel}>
      <div className={styles.castleName}>Your Castle</div>
      {BARS.map(({ key, label }) => (
        <div className={styles.row} key={key}>
          <span className={styles.label}>{label}</span>
          <div className={styles.track}>
            <div
              className={styles.fill}
              style={{
                width: `${stateVector[key] * 100}%`,
                background: BAR_COLORS[key],
              }}
            />
          </div>
          <span className={styles.value}>
            {stateVector[key].toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  );
}
