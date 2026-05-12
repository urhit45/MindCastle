import { useState } from 'react';
import { CastleWorld } from './world/CastleWorld';
import { StatePanel } from './overlay/StatePanel';
import { ReadoutDisplay } from './overlay/ReadoutDisplay';
import { EngineTooltip } from './overlay/EngineTooltip';
import { ArtifactPanel } from './overlay/ArtifactPanel';
import { LoadingScreen } from './overlay/LoadingScreen';

export default function App() {
  const [loaded, setLoaded] = useState(false);

  return (
    <>
      <CastleWorld />
      <StatePanel />
      <ReadoutDisplay />
      <EngineTooltip />
      <ArtifactPanel />
      {!loaded && <LoadingScreen onDone={() => setLoaded(true)} />}
    </>
  );
}
