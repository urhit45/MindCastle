import { useEffect, useRef } from 'react';
import * as BABYLON from 'babylonjs';
import { useCastleStore } from '../store/castleStore';
import { buildScene } from './SceneBuilder';
import { buildCamera, focusOnPosition, resetCamera } from './CameraController';
import { startStateVectorPolling } from '../api/mindcastleApi';
import type { StateVector } from '../store/types';

declare global {
  interface Window {
    __mindcastle?: { updateStateVector: (sv: StateVector) => void };
    __mindcastleCam?: () => void;
  }
}

export function CastleWorld() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { engines, stateVector, setSelectedEngine, setHoveredEngine, setStateVector } = useCastleStore();

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    const scene = new BABYLON.Scene(engine);

    const cam = buildCamera(scene, canvas);
    const handles = buildScene(scene, engines);

    handles.updateStateVector(stateVector);

    window.__mindcastle = {
      updateStateVector: (sv) => {
        useCastleStore.getState().setStateVector(sv);
        handles.updateStateVector(sv);
      },
    };

    window.__mindcastleCam = () => resetCamera(cam);

    const stopPolling = startStateVectorPolling((sv) => {
      setStateVector(sv);
      handles.updateStateVector(sv);
    });

    scene.onPointerObservable.add((info) => {
      if (info.type === BABYLON.PointerEventTypes.POINTERMOVE) {
        const hit = scene.pick(scene.pointerX, scene.pointerY);
        const engineId = (hit.pickedMesh?.metadata?.engineId as string) ?? null;
        setHoveredEngine(engineId);
      }

      if (info.type === BABYLON.PointerEventTypes.POINTERPICK) {
        const hit = scene.pick(scene.pointerX, scene.pointerY);
        const engineId = hit.pickedMesh?.metadata?.engineId as string | undefined;
        if (engineId) {
          setSelectedEngine(engineId);
          const pos = handles.structurePositions.get(engineId);
          if (pos) focusOnPosition(cam, pos);
        }
      }
    });

    let tick = 0;

    scene.registerBeforeRender(() => {
      tick += 0.008;
      const sv = useCastleStore.getState().stateVector;

      handles.structureObjects.forEach((objs, engineId) => {
        const eng = useCastleStore.getState().engines.find(e => e.id === engineId);
        if (!eng) return;
        const speed = 1.0 + eng.activityLevel * 0.5;
        const offset = engineId.charCodeAt(0) * 0.3;
        objs.light.intensity = eng.activityLevel * 1.2 + Math.sin(tick * speed + offset) * 0.1;
      });

      handles.atmosphere.keepLight.intensity =
        0.7 + Math.sin(tick * 1.2) * 0.15 * sv.energy;
    });

    engine.runRenderLoop(() => scene.render());

    const onResize = () => engine.resize();
    window.addEventListener('resize', onResize);

    return () => {
      stopPolling();
      window.removeEventListener('resize', onResize);
      engine.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        outline: 'none',
        touchAction: 'none',
      }}
    />
  );
}
