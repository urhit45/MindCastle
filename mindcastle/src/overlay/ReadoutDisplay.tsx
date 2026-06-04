
import { useCastleStore } from '../store/castleStore';
import styles from './ReadoutDisplay.module.css';

export function ReadoutDisplay() {
  const { stateVector } = useCastleStore();

  return (
    <div className={styles.readout}>
      <div className={styles.state}>
        {stateVector.readoutState.replace(/_/g, ' ')}
      </div>
      <div className={styles.task}>
        {stateVector.readoutTask}
      </div>
    </div>
  );
}
