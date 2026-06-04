import { useState, useCallback } from 'react';
import { CastleWorld } from './world/CastleWorld';
import { StatePanel } from './overlay/StatePanel';
import { ReadoutDisplay } from './overlay/ReadoutDisplay';
import { EngineTooltip } from './overlay/EngineTooltip';
import { ArtifactPanel } from './overlay/ArtifactPanel';
import { LoadingScreen } from './overlay/LoadingScreen';
import { useCastleStore } from './store/castleStore';

interface TooltipInfo {
  visible: boolean;
  x: number;
  y: number;
  engineId: string | null;
}

export default function App() {
  const [loaded, setLoaded] = useState(false);
  const [tooltip, setTooltip] = useState<TooltipInfo>({
    visible: false,
    x: 0,
    y: 0,
    engineId: null,
  });

  const { setSelectedEngine } = useCastleStore();

  const handleEngineHover = useCallback((id: string | null, x: number, y: number) => {
    setTooltip({ visible: id !== null, x, y, engineId: id });
  }, []);

  const handleEngineClick = useCallback((_id: string | null) => {
    // Panel opens reactively via store
  }, []);

  const handlePanelClose = useCallback(() => {
    setSelectedEngine(null);
  }, [setSelectedEngine]);

  return (
    <>
      {/* Custom Tauri drag region — frameless window */}
      <div
        data-tauri-drag-region
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 340,
          height: 28,
          zIndex: 999,
          pointerEvents: 'all',
        }}
      />

      {/* Vignette */}
      <div
        style={{
          position: 'fixed',
          inset: 0,
          background: 'radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.7) 100%)',
          pointerEvents: 'none',
          zIndex: 5,
        }}
      />

      <CastleWorld
        onEngineHover={handleEngineHover}
        onEngineClick={handleEngineClick}
      />

      {loaded && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            pointerEvents: 'none',
            zIndex: 10,
          }}
        >
          <StatePanel />
          <ReadoutDisplay />
          <EngineTooltip
            visible={tooltip.visible}
            x={tooltip.x}
            y={tooltip.y}
            engineId={tooltip.engineId}
          />
        </div>
      )}

      <ArtifactPanel onClose={handlePanelClose} />

      <LoadingScreen onComplete={() => setLoaded(true)} />
    </>
  );
}
