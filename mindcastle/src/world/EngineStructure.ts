import * as BABYLON from 'babylonjs';
import type { Engine, StructureType } from '../store/types';

const FLARE_URL = 'https://assets.babylonjs.com/textures/flare.png';

export interface EngineStructureObjects {
  meshes: BABYLON.Mesh[];
  light: BABYLON.PointLight;
  particles: BABYLON.ParticleSystem[];
}

function hexToColor3(hex: string): BABYLON.Color3 {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return new BABYLON.Color3(r, g, b);
}

function stoneMat(scene: BABYLON.Scene, name: string): BABYLON.StandardMaterial {
  const mat = new BABYLON.StandardMaterial(name, scene);
  const v = 0.05 + Math.random() * 0.03;
  mat.diffuseColor = new BABYLON.Color3(v, v * 0.95, v * 0.9);
  mat.specularColor = new BABYLON.Color3(0.02, 0.02, 0.02);
  return mat;
}

function roofMat(scene: BABYLON.Scene, name: string, color: BABYLON.Color3): BABYLON.StandardMaterial {
  const mat = new BABYLON.StandardMaterial(name, scene);
  mat.diffuseColor = color.scale(0.4);
  mat.specularColor = new BABYLON.Color3(0.05, 0.05, 0.05);
  return mat;
}

function windowMat(scene: BABYLON.Scene, name: string, color: BABYLON.Color3, activity: number): BABYLON.StandardMaterial {
  const mat = new BABYLON.StandardMaterial(name, scene);
  mat.diffuseColor = new BABYLON.Color3(0, 0, 0);
  mat.emissiveColor = color.scale(activity);
  return mat;
}

function addWindow(
  scene: BABYLON.Scene,
  parent: BABYLON.Mesh,
  x: number,
  y: number,
  z: number,
  color: BABYLON.Color3,
  activity: number,
  id: string
): BABYLON.Mesh {
  const win = BABYLON.MeshBuilder.CreateBox(`win_${id}`, { width: 0.15, height: 0.2, depth: 0.05 }, scene);
  win.position.set(x, y, z);
  win.parent = parent;
  win.material = windowMat(scene, `winMat_${id}`, color, activity);
  return win;
}

function buildTower(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.Mesh[] {
  const a = engine.activityLevel;
  const color = hexToColor3(engine.color);
  const meshes: BABYLON.Mesh[] = [];

  const base = BABYLON.MeshBuilder.CreateBox(`base_${engine.id}`, { width: 2, height: 1, depth: 2 }, scene);
  base.position.copyFrom(pos);
  base.material = stoneMat(scene, `baseMat_${engine.id}`);
  base.metadata = { engineId: engine.id };
  meshes.push(base);

  const bodyH = 2 + a * 4;
  const body = BABYLON.MeshBuilder.CreateBox(`body_${engine.id}`, { width: 1.2, height: bodyH, depth: 1.2 }, scene);
  body.position.set(pos.x, pos.y + 0.5 + bodyH / 2, pos.z);
  body.material = stoneMat(scene, `bodyMat_${engine.id}`);
  body.metadata = { engineId: engine.id };
  meshes.push(body);

  const cone = BABYLON.MeshBuilder.CreateCylinder(`roof_${engine.id}`, { height: 1.2, diameterTop: 0, diameterBottom: 1.6, tessellation: 4 }, scene);
  cone.position.set(pos.x, pos.y + 0.5 + bodyH + 0.6, pos.z);
  cone.material = roofMat(scene, `roofMat_${engine.id}`, color);
  meshes.push(cone);

  addWindow(scene, body, 0.6, 0, 0, color, a, `${engine.id}_f`);
  addWindow(scene, body, -0.6, 0, 0, color, a, `${engine.id}_b`);

  return meshes;
}

function buildGarden(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.Mesh[] {
  const a = engine.activityLevel;
  const color = hexToColor3(engine.color);
  const meshes: BABYLON.Mesh[] = [];

  const base = BABYLON.MeshBuilder.CreateBox(`base_${engine.id}`, { width: 2, height: 0.8, depth: 2 }, scene);
  base.position.copyFrom(pos);
  base.material = stoneMat(scene, `baseMat_${engine.id}`);
  base.metadata = { engineId: engine.id };
  meshes.push(base);

  const r = 0.75 + a * 0.5;
  const dome = BABYLON.MeshBuilder.CreateSphere(`dome_${engine.id}`, { diameter: r * 2, segments: 12 }, scene);
  dome.position.set(pos.x, pos.y + 0.4 + r, pos.z);
  const mat = new BABYLON.StandardMaterial(`domeMat_${engine.id}`, scene);
  mat.diffuseColor = color.scale(0.25 + a * 0.2);
  mat.specularColor = new BABYLON.Color3(0.02, 0.02, 0.02);
  dome.material = mat;
  dome.metadata = { engineId: engine.id };
  meshes.push(dome);

  return meshes;
}

function buildVault(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.Mesh[] {
  const a = engine.activityLevel;
  const color = hexToColor3(engine.color);
  const meshes: BABYLON.Mesh[] = [];

  const base = BABYLON.MeshBuilder.CreateBox(`base_${engine.id}`, { width: 2, height: 1, depth: 2 }, scene);
  base.position.copyFrom(pos);
  base.material = stoneMat(scene, `baseMat_${engine.id}`);
  base.metadata = { engineId: engine.id };
  meshes.push(base);

  const w = 3 + a * 1;
  const bodyH = 1.5 + a;
  const body = BABYLON.MeshBuilder.CreateBox(`body_${engine.id}`, { width: w, height: bodyH, depth: 2 }, scene);
  body.position.set(pos.x, pos.y + 0.5 + bodyH / 2, pos.z);
  body.material = stoneMat(scene, `bodyMat_${engine.id}`);
  body.metadata = { engineId: engine.id };
  meshes.push(body);

  // flat top with parapet boxes
  for (let i = -1; i <= 1; i++) {
    const merlon = BABYLON.MeshBuilder.CreateBox(`merlon_${engine.id}_${i}`, { width: 0.3, height: 0.4, depth: 0.3 }, scene);
    merlon.position.set(pos.x + i * 1.2, pos.y + 0.5 + bodyH + 0.2, pos.z + 0.85);
    merlon.material = stoneMat(scene, `merlMat_${engine.id}_${i}`);
    meshes.push(merlon);
    const merlon2 = merlon.clone(`merlon2_${engine.id}_${i}`);
    merlon2.position.z = pos.z - 0.85;
    meshes.push(merlon2);
  }

  addWindow(scene, body, w / 2, 0, 0, color, a, `${engine.id}_r`);
  addWindow(scene, body, -w / 2, 0, 0, color, a, `${engine.id}_l`);

  return meshes;
}

function buildWorkshop(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.Mesh[] {
  const a = engine.activityLevel;
  const color = hexToColor3(engine.color);
  const meshes: BABYLON.Mesh[] = [];

  const base = BABYLON.MeshBuilder.CreateBox(`base_${engine.id}`, { width: 2, height: 0.8, depth: 2 }, scene);
  base.position.copyFrom(pos);
  base.material = stoneMat(scene, `baseMat_${engine.id}`);
  base.metadata = { engineId: engine.id };
  meshes.push(base);

  const bodyH = 1.5 + a * 1.5;
  const body = BABYLON.MeshBuilder.CreateBox(`body_${engine.id}`, { width: 1.8, height: bodyH, depth: 1.8 }, scene);
  body.position.set(pos.x, pos.y + 0.4 + bodyH / 2, pos.z);
  body.material = stoneMat(scene, `bodyMat_${engine.id}`);
  body.metadata = { engineId: engine.id };
  meshes.push(body);

  // pitched roof: two planes
  const roofMaterial = roofMat(scene, `roofMat_${engine.id}`, color);
  for (let side = 0; side < 2; side++) {
    const panel = BABYLON.MeshBuilder.CreateBox(`roofPanel_${engine.id}_${side}`, { width: 2.2, height: 0.08, depth: 1.2 }, scene);
    panel.position.set(pos.x, pos.y + 0.4 + bodyH + 0.3, pos.z + (side === 0 ? 0.5 : -0.5));
    panel.rotation.x = (side === 0 ? -0.5 : 0.5);
    panel.material = roofMaterial;
    meshes.push(panel);
  }

  addWindow(scene, body, 0.9, 0, 0, color, a, `${engine.id}_f`);

  return meshes;
}

function buildStudy(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.Mesh[] {
  const a = engine.activityLevel;
  const color = hexToColor3(engine.color);
  const meshes: BABYLON.Mesh[] = [];

  const base = BABYLON.MeshBuilder.CreateBox(`base_${engine.id}`, { width: 2, height: 0.8, depth: 2 }, scene);
  base.position.copyFrom(pos);
  base.material = stoneMat(scene, `baseMat_${engine.id}`);
  base.metadata = { engineId: engine.id };
  meshes.push(base);

  const bodyH = 1.2 + a * 1.2;
  const body = BABYLON.MeshBuilder.CreateBox(`body_${engine.id}`, { width: 1.6, height: bodyH, depth: 1.6 }, scene);
  body.position.set(pos.x, pos.y + 0.4 + bodyH / 2, pos.z);
  body.material = stoneMat(scene, `bodyMat_${engine.id}`);
  body.metadata = { engineId: engine.id };
  meshes.push(body);

  // gabled roof
  const roofMaterial = roofMat(scene, `roofMat_${engine.id}`, color);
  const ridge = BABYLON.MeshBuilder.CreateBox(`ridge_${engine.id}`, { width: 1.8, height: 0.08, depth: 0.08 }, scene);
  ridge.position.set(pos.x, pos.y + 0.4 + bodyH + 0.55, pos.z);
  ridge.material = roofMaterial;
  meshes.push(ridge);
  for (let side = 0; side < 2; side++) {
    const slope = BABYLON.MeshBuilder.CreateBox(`slope_${engine.id}_${side}`, { width: 1.8, height: 0.08, depth: 1.0 }, scene);
    slope.position.set(pos.x, pos.y + 0.4 + bodyH + 0.25, pos.z + (side === 0 ? 0.45 : -0.45));
    slope.rotation.x = (side === 0 ? -0.55 : 0.55);
    slope.material = roofMaterial;
    meshes.push(slope);
  }

  addWindow(scene, body, 0.8, 0.1, 0, color, a, `${engine.id}_f`);

  return meshes;
}

function buildHall(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.Mesh[] {
  const a = engine.activityLevel;
  const color = hexToColor3(engine.color);
  const meshes: BABYLON.Mesh[] = [];

  const base = BABYLON.MeshBuilder.CreateBox(`base_${engine.id}`, { width: 2, height: 0.8, depth: 2 }, scene);
  base.position.copyFrom(pos);
  base.material = stoneMat(scene, `baseMat_${engine.id}`);
  base.metadata = { engineId: engine.id };
  meshes.push(base);

  const w = 3.5 + a * 0.8;
  const bodyH = 1 + a * 0.8;
  const body = BABYLON.MeshBuilder.CreateBox(`body_${engine.id}`, { width: w, height: bodyH, depth: 2 }, scene);
  body.position.set(pos.x, pos.y + 0.4 + bodyH / 2, pos.z);
  body.material = stoneMat(scene, `bodyMat_${engine.id}`);
  body.metadata = { engineId: engine.id };
  meshes.push(body);

  // crenellated flat top
  for (let i = -2; i <= 2; i++) {
    const merlon = BABYLON.MeshBuilder.CreateBox(`merlonH_${engine.id}_${i}`, { width: 0.28, height: 0.35, depth: 0.28 }, scene);
    merlon.position.set(pos.x + i * (w / 4.5), pos.y + 0.4 + bodyH + 0.175, pos.z + 0.9);
    merlon.material = stoneMat(scene, `merlMatH_${engine.id}_${i}`);
    meshes.push(merlon);
    const m2 = merlon.clone(`merlonH2_${engine.id}_${i}`);
    m2.position.z = pos.z - 0.9;
    meshes.push(m2);
  }

  addWindow(scene, body, w / 2 - 0.3, 0, 0, color, a, `${engine.id}_r`);
  addWindow(scene, body, -(w / 2 - 0.3), 0, 0, color, a, `${engine.id}_l`);

  return meshes;
}

function buildChapel(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.Mesh[] {
  const a = engine.activityLevel;
  const color = hexToColor3(engine.color);
  const meshes: BABYLON.Mesh[] = [];

  const base = BABYLON.MeshBuilder.CreateBox(`base_${engine.id}`, { width: 2, height: 0.8, depth: 2 }, scene);
  base.position.copyFrom(pos);
  base.material = stoneMat(scene, `baseMat_${engine.id}`);
  base.metadata = { engineId: engine.id };
  meshes.push(base);

  const bodyH = 1.5 + a * 2;
  const body = BABYLON.MeshBuilder.CreateBox(`body_${engine.id}`, { width: 1.6, height: bodyH, depth: 1.6 }, scene);
  body.position.set(pos.x, pos.y + 0.4 + bodyH / 2, pos.z);
  body.material = stoneMat(scene, `bodyMat_${engine.id}`);
  body.metadata = { engineId: engine.id };
  meshes.push(body);

  // spire
  const spire = BABYLON.MeshBuilder.CreateCylinder(`spire_${engine.id}`, { height: 2 + a * 1.5, diameterTop: 0, diameterBottom: 0.6, tessellation: 8 }, scene);
  spire.position.set(pos.x, pos.y + 0.4 + bodyH + (1 + a * 0.75), pos.z);
  spire.material = roofMat(scene, `spireMat_${engine.id}`, color);
  meshes.push(spire);

  addWindow(scene, body, 0.8, 0.2, 0, color, a, `${engine.id}_f`);

  return meshes;
}

const STRUCTURE_BUILDERS: Record<StructureType, (scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3) => BABYLON.Mesh[]> = {
  tower:    buildTower,
  garden:   buildGarden,
  vault:    buildVault,
  workshop: buildWorkshop,
  study:    buildStudy,
  hall:     buildHall,
  chapel:   buildChapel,
};

function createEmberParticles(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.ParticleSystem {
  const ps = new BABYLON.ParticleSystem(`embers_${engine.id}`, 200, scene);
  ps.particleTexture = new BABYLON.Texture(FLARE_URL, scene);
  const emitter = new BABYLON.Vector3(pos.x, pos.y + 3, pos.z);
  ps.emitter = emitter;
  ps.minEmitBox = new BABYLON.Vector3(-0.3, 0, -0.3);
  ps.maxEmitBox = new BABYLON.Vector3(0.3, 0, 0.3);
  const c = hexToColor3(engine.color);
  ps.color1 = new BABYLON.Color4(c.r, c.g, c.b, 1.0);
  ps.color2 = new BABYLON.Color4(c.r * 0.6, c.g * 0.6, c.b * 0.6, 0.5);
  ps.colorDead = new BABYLON.Color4(0, 0, 0, 0);
  ps.minSize = 0.05;
  ps.maxSize = 0.2;
  ps.minLifeTime = 0.5;
  ps.maxLifeTime = 1.5;
  ps.emitRate = 30;
  ps.direction1 = new BABYLON.Vector3(-0.2, 1, -0.2);
  ps.direction2 = new BABYLON.Vector3(0.2, 2, 0.2);
  ps.minEmitPower = 0.3;
  ps.maxEmitPower = 0.8;
  ps.updateSpeed = 0.015;
  ps.start();
  return ps;
}

function createSmokeParticles(scene: BABYLON.Scene, engine: Engine, pos: BABYLON.Vector3): BABYLON.ParticleSystem {
  const ps = new BABYLON.ParticleSystem(`smoke_${engine.id}`, 100, scene);
  ps.particleTexture = new BABYLON.Texture(FLARE_URL, scene);
  const emitter = new BABYLON.Vector3(pos.x, pos.y + 2, pos.z);
  ps.emitter = emitter;
  ps.minEmitBox = new BABYLON.Vector3(-0.5, 0, -0.5);
  ps.maxEmitBox = new BABYLON.Vector3(0.5, 0, 0.5);
  ps.color1 = new BABYLON.Color4(0.15, 0.12, 0.2, 0.6);
  ps.color2 = new BABYLON.Color4(0.1, 0.08, 0.15, 0.3);
  ps.colorDead = new BABYLON.Color4(0, 0, 0, 0);
  ps.minSize = 0.3;
  ps.maxSize = 0.8;
  ps.minLifeTime = 1.5;
  ps.maxLifeTime = 3.0;
  ps.emitRate = 15;
  ps.direction1 = new BABYLON.Vector3(-0.1, 1, -0.1);
  ps.direction2 = new BABYLON.Vector3(0.1, 1.5, 0.1);
  ps.minEmitPower = 0.2;
  ps.maxEmitPower = 0.5;
  ps.updateSpeed = 0.01;
  ps.start();
  return ps;
}

export function buildEngineStructure(
  scene: BABYLON.Scene,
  engine: Engine,
  pos: BABYLON.Vector3
): EngineStructureObjects {
  const builder = STRUCTURE_BUILDERS[engine.structure];
  const meshes = builder(scene, engine, pos);

  const color = hexToColor3(engine.color);
  const light = new BABYLON.PointLight(`light_${engine.id}`, new BABYLON.Vector3(pos.x, pos.y + 3, pos.z), scene);
  light.diffuse = color;
  light.specular = color.scale(0.3);
  light.intensity = engine.activityLevel * 1.2;
  light.range = 8;

  const particles: BABYLON.ParticleSystem[] = [];

  const hasBlockedArtifact = engine.artifacts.some(a => a.status === 'blocked');
  if (hasBlockedArtifact) {
    particles.push(createSmokeParticles(scene, engine, pos));
  }

  if (!hasBlockedArtifact && engine.activityLevel > 0.5) {
    particles.push(createEmberParticles(scene, engine, pos));
  }

  return { meshes, light, particles };
}

export function buildPath(
  scene: BABYLON.Scene,
  from: BABYLON.Vector3,
  to: BABYLON.Vector3,
  engineColor: string,
  activity: number,
  id: string
): BABYLON.Mesh {
  const diff = to.subtract(from);
  const len = diff.length();
  const mid = from.add(diff.scale(0.5));

  const path = BABYLON.MeshBuilder.CreateBox(`path_${id}`, { width: 0.3, height: 0.06, depth: len }, scene);
  path.position.copyFrom(mid);
  path.rotation.y = Math.atan2(diff.x, diff.z);

  const color = hexToColor3(engineColor);
  const mat = new BABYLON.StandardMaterial(`pathMat_${id}`, scene);
  mat.diffuseColor = new BABYLON.Color3(0.06, 0.05, 0.04);
  mat.emissiveColor = color.scale(activity * 0.3);
  path.material = mat;

  return path;
}
