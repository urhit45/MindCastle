import * as BABYLON from 'babylonjs';
import type { Engine } from '../store/types';
import { buildEngineStructure, buildPath, type EngineStructureObjects } from './EngineStructure';
import { buildAtmosphere, type AtmosphereHandles } from './AtmosphereSystem';
import { mapStateToVisuals } from './StateMapper';
import type { StateVector } from '../store/types';

// 8 positions arranged in a rough circle around the keep
const STRUCTURE_POSITIONS: BABYLON.Vector3[] = [
  new BABYLON.Vector3(0, 0, 7),
  new BABYLON.Vector3(5, 0, 5),
  new BABYLON.Vector3(7, 0, 0),
  new BABYLON.Vector3(5, 0, -5),
  new BABYLON.Vector3(0, 0, -7),
  new BABYLON.Vector3(-5, 0, -5),
  new BABYLON.Vector3(-7, 0, 0),
  new BABYLON.Vector3(-5, 0, 5),
];

export interface SceneHandles {
  structureObjects: Map<string, EngineStructureObjects>;
  structurePositions: Map<string, BABYLON.Vector3>;
  atmosphere: AtmosphereHandles;
  glowLayer: BABYLON.GlowLayer;
  updateStateVector: (sv: StateVector) => void;
}

export function buildScene(
  scene: BABYLON.Scene,
  engines: Engine[]
): SceneHandles {
  // Ground
  const ground = BABYLON.MeshBuilder.CreateGround('ground', { width: 50, height: 50, subdivisions: 1 }, scene);
  const groundMat = new BABYLON.StandardMaterial('groundMat', scene);
  groundMat.diffuseColor = new BABYLON.Color3(0.05, 0.045, 0.04);
  groundMat.specularColor = new BABYLON.Color3(0, 0, 0);
  ground.material = groundMat;

  // Keep (central structure)
  const keep = BABYLON.MeshBuilder.CreateBox('keep', { width: 2.5, height: 4, depth: 2.5 }, scene);
  keep.position.y = 2;
  const keepMat = new BABYLON.StandardMaterial('keepMat', scene);
  keepMat.diffuseColor = new BABYLON.Color3(0.06, 0.055, 0.05);
  keep.material = keepMat;

  const keepRoof = BABYLON.MeshBuilder.CreateCylinder('keepRoof', { height: 1.5, diameterTop: 0, diameterBottom: 3, tessellation: 4 }, scene);
  keepRoof.position.y = 4.75;
  const keepRoofMat = new BABYLON.StandardMaterial('keepRoofMat', scene);
  keepRoofMat.diffuseColor = new BABYLON.Color3(0.08, 0.06, 0.05);
  keepRoof.material = keepRoofMat;

  const atmosphere = buildAtmosphere(scene);

  const glowLayer = new BABYLON.GlowLayer('glow', scene);
  glowLayer.intensity = 0.6;

  const structureObjects = new Map<string, EngineStructureObjects>();
  const structurePositions = new Map<string, BABYLON.Vector3>();

  engines.slice(0, 8).forEach((engine, i) => {
    const pos = STRUCTURE_POSITIONS[i];
    const objs = buildEngineStructure(scene, engine, pos);
    structureObjects.set(engine.id, objs);
    structurePositions.set(engine.id, pos);

    // path from keep to structure
    buildPath(scene, BABYLON.Vector3.Zero(), pos, engine.color, engine.activityLevel, engine.id);
  });

  function updateStateVector(sv: StateVector) {
    const params = mapStateToVisuals(sv);
    atmosphere.update(params);
  }

  return { structureObjects, structurePositions, atmosphere, glowLayer, updateStateVector };
}
