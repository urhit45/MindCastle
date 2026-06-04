import { useEffect, useRef, useState } from 'react';
import styles from './LoadingScreen.module.css';

interface LoadingScreenProps {
  onComplete: () => void;
}

export function LoadingScreen({ onComplete }: LoadingScreenProps) {
  const [progress, setProgress] = useState(0);
  const [fading, setFading] = useState(false);
  const [removed, setRemoved] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setProgress((prev) => {
        const next = prev + Math.random() * 15;
        if (next >= 100) {
          if (intervalRef.current) clearInterval(intervalRef.current);
          setTimeout(() => {
            setFading(true);
            setTimeout(() => {
              setRemoved(true);
              onComplete();
            }, 1200);
          }, 1800);
          return 100;
        }
        return next;
      });
    }, 80);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [onComplete]);

  if (removed) return null;

  return (
    <div className={`${styles.loading} ${fading ? styles.fade : ''}`}>
      <div className={styles.title}>MINDCASTLE</div>
      <div className={styles.sub}>preparing your castle</div>
      <div className={styles.barTrack}>
        <div className={styles.bar} style={{ width: `${progress}%` }} />
      </div>
    </div>
  );
}
