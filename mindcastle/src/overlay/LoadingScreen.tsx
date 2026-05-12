import { useEffect, useState } from 'react';
import styles from './LoadingScreen.module.css';

export function LoadingScreen({ onDone }: { onDone: () => void }) {
  const [opacity, setOpacity] = useState(1);

  useEffect(() => {
    const timer = setTimeout(() => {
      setOpacity(0);
      setTimeout(onDone, 800);
    }, 1800);
    return () => clearTimeout(timer);
  }, [onDone]);

  return (
    <div className={styles.screen} style={{ opacity, transition: 'opacity 0.8s ease' }}>
      <div className={styles.name}>MIND CASTLE</div>
      <div className={styles.sub}>entering the keep…</div>
    </div>
  );
}
