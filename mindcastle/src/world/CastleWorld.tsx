import { useEffect, useRef, useCallback } from 'react';
import * as BABYLON from 'babylonjs';
import { useCastleStore } from '../store/castleStore';
import { useSchedulerStore } from '../store/schedulerStore';
import { SceneBuilder } from './SceneBuilder';
import { CameraController } from './CameraController';
import { AtmosphereSystem } from './AtmosphereSystem';
import { mapStateToVisuals } from './StateMapper';
import { ENGINE_POSITIONS } from '../store/types';
import { startStateVectorPolling, getReadout } from '../api/mindcastleApi';
import { EEVDFScheduler } from '../scheduler/EEVDFScheduler';
import { CognitiveClock } from '../scheduler/CognitiveClock';
import type { Engine, StateVector, ReadoutState } from '../store/types';
import type { EngineSchedulerState, TimeAffinityProfile, SchedulerOutput } from '../scheduler/SchedulerTypes';

interface CastleWorldProps {
  onEngineHover: (id: string | null, x: number, y: number) => void;
  onEngineClick: (id: string | null) => void;
}

export function CastleWorld({ onEngineHover, onEngineClick }: CastleWorldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const babylonEngineRef = useRef<BABYLON.Engine | null>(null);
  const sceneBuilderRef = useRef<SceneBuilder | null>(null);
  const cameraRef = useRef<CameraController | null>(null);
  const atmosphereRef = useRef<AtmosphereSystem | null>(null);
  const schedulerRef = useRef<EEVDFScheduler | null>(null);
  const clockRef = useRef<CognitiveClock | null>(null);

  // Refs to latest store values for use inside stable effect closures
  const enginesRef = useRef<Engine[]>([]);
  const selectedEngineIdRef = useRef<string | null>(null);

  const { engines, stateVector, selectedEngineId } = useCastleStore();
  const { clockState, schedStates, affinityProfiles } = useSchedulerStore();

  useEffect(() => { enginesRef.current = engines; }, [engines]);
  useEffect(() => { selectedEngineIdRef.current = selectedEngineId; }, [selectedEngineId]);

  const runSchedulerPass = useCallback((sv: StateVector): SchedulerOutput | null => {
    if (!schedulerRef.current || !clockRef.current) return null;
    clockRef.current.advance(sv);
    const output = schedulerRef.current.schedule(enginesRef.current, sv, selectedEngineIdRef.current);
    useSchedulerStore.getState().setClockState(clockRef.current.serialize());
    schedulerRef.current.getScheduleStates().forEach((state, id) => {
      useSchedulerStore.getState().setSchedState(id, state);
    });
    return output;
  }, []);

  const applyOutputToReadout = useCallback((sv: StateVector, output: SchedulerOutput) => {
    const readoutState: ReadoutState =
      output.engineId === null
        ? 'recovery'
        : output.reason === 'flow_protect'
        ? 'flow_available'
        : sv.risk > 0.7
        ? 'pre_crash'
        : 'flow_available';

    const fallbackTask = output.diagnosis.split('\n')[0] ?? '';

    getReadout({ schedulerOutput: output, stateVector: sv })
      .then((r) => {
        useCastleStore.getState().setStateVector({
          ...sv,
          readoutState,
          readoutTask: r.task ?? fallbackTask,
        });
      })
      .catch(() => {
        useCastleStore.getState().setStateVector({ ...sv, readoutState, readoutTask: fallbackTask });
      });
  }, []);

  // Main Babylon.js setup — runs once on mount
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const babylonEngine = new BABYLON.Engine(canvas, true, {
      preserveDrawingBuffer: true,
      stencil: true,
      antialias: true,
    });
    babylonEngineRef.current = babylonEngine;

    const scene = new BABYLON.Scene(babylonEngine);
    scene.clearColor = new BABYLON.Color4(0.04, 0.04, 0.07, 1);
    scene.fogMode = BABYLON.Scene.FOGMODE_EXP2;
    scene.fogDensity = 0.02;
    scene.fogColor = new BABYLON.Color3(0.04, 0.04, 0.07);

    const builder = new SceneBuilder(scene);
    sceneBuilderRef.current = builder;
    builder.setupLighting();
    builder.buildScene(enginesRef.current);

    const camCtrl = new CameraController(canvas, scene);
    cameraRef.current = camCtrl;

    const atm = new AtmosphereSystem(scene);
    atmosphereRef.current = atm;

    // Scheduler init from persisted state
    const clock = CognitiveClock.fromSerialized(clockState);
    clockRef.current = clock;
    const schedMap = new Map(Object.entries(schedStates) as [string, EngineSchedulerState][]);
    const affinityMap = new Map(Object.entries(affinityProfiles) as [string, TimeAffinityProfile][]);
    schedulerRef.current = new EEVDFScheduler(clock, schedMap, affinityMap);

    // Apply initial visuals from current store state
    const initialSV = useCastleStore.getState().stateVector;
    const params = mapStateToVisuals(initialSV);
    builder.applyVisuals(params);
    atm.applyVisuals(params);

    const initialOutput = runSchedulerPass(initialSV);
    if (initialOutput) {
      applyOutputToReadout(initialSV, initialOutput);
      builder.applySchedulerVisuals(
        initialOutput,
        useSchedulerStore.getState().schedStates,
        enginesRef.current,
      );
    }

    // ONE beforeRender — all per-frame animation
    let tick = 0;
    scene.registerBeforeRender(() => {
      tick += 0.008;
      builder.tickLights(tick, useCastleStore.getState().stateVector.energy, enginesRef.current);
    });

    // Pointer events
    scene.onPointerObservable.add((info) => {
      const type = info.type;
      if (type === BABYLON.PointerEventTypes.POINTERMOVE) {
        const pick = scene.pick(scene.pointerX, scene.pointerY);
        if (pick.hit && pick.pickedMesh?.metadata?.engineId) {
          const id = pick.pickedMesh.metadata.engineId as string;
          useCastleStore.getState().setHoveredEngine(id);
          onEngineHover(id, scene.pointerX, scene.pointerY);
        } else {
          useCastleStore.getState().setHoveredEngine(null);
          onEngineHover(null, 0, 0);
        }
      }
      if (type === BABYLON.PointerEventTypes.POINTERDOWN) {
        const pick = scene.pick(scene.pointerX, scene.pointerY);
        if (pick.hit && pick.pickedMesh?.metadata?.engineId) {
          const id = pick.pickedMesh.metadata.engineId as string;
          useCastleStore.getState().setSelectedEngine(id);
          onEngineClick(id);
          const p = ENGINE_POSITIONS[id] ?? { x: 0, z: 0 };
          cameraRef.current?.focusOn(p);
        }
      }
    });

    babylonEngine.runRenderLoop(() => scene.render());

    const handleResize = () => babylonEngine.resize();
    window.addEventListener('resize', handleResize);

    // State vector polling (30s interval)
    const stopPolling = startStateVectorPolling((sv) => {
      useCastleStore.getState().setStateVector(sv);
      const vp = mapStateToVisuals(sv);
      builder.applyVisuals(vp);
      atm.applyVisuals(vp);
      const output = runSchedulerPass(sv);
      if (output) {
        applyOutputToReadout(sv, output);
        builder.applySchedulerVisuals(output, useSchedulerStore.getState().schedStates, enginesRef.current);
      }
    });

    // Global Mac Mini API
    type MindCastleGlobal = Window & { __mindcastle?: { updateStateVector: (sv: StateVector) => void } };
    (window as MindCastleGlobal).__mindcastle = {
      updateStateVector: (sv: StateVector) => {
        useCastleStore.getState().setStateVector(sv);
        const vp = mapStateToVisuals(sv);
        builder.applyVisuals(vp);
        atm.applyVisuals(vp);
        const output = runSchedulerPass(sv);
        if (output) applyOutputToReadout(sv, output);
      },
    };

    return () => {
      stopPolling();
      window.removeEventListener('resize', handleResize);
      atm.dispose();
      scene.dispose();
      babylonEngine.dispose();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle panel close → camera returns to center
  useEffect(() => {
    if (!selectedEngineId && cameraRef.current) {
      cameraRef.current.returnToCenter();
    }
  }, [selectedEngineId]);

  // React to store state vector changes (non-polling updates)
  useEffect(() => {
    if (!sceneBuilderRef.current || !atmosphereRef.current) return;
    const params = mapStateToVisuals(stateVector);
    sceneBuilderRef.current.applyVisuals(params);
    atmosphereRef.current.applyVisuals(params);
  }, [stateVector]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100vw', height: '100vh', display: 'block', touchAction: 'none' }}
    />
  );
}
