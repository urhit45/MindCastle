import * as BABYLON from 'babylonjs';
import type { Engine, StructureType } from '../store/types';
import { hexToColor3 } from './StateMapper';

function stoneMat(name: string, r: number, g: number, b: number, scene: BABYLON.Scene): BABYLON.StandardMaterial {
  const mat = new BABYLON.StandardMaterial(name, scene);
  mat.diffuseColor = new BABYLON.Color3(r, g, b);
  mat.specularColor = new BABYLON.Color3(0.02, 0.02, 0.02);
  return mat;
}

export interface StructureMeshes {
  all: BABYLON.AbstractMesh[];
  light: BABYLON.PointLight | null;
}

export function buildStructure(
  eng: Engine,
  pos: { x: number; z: number },
  scene: BABYLON.Scene,
): StructureMeshes {
  const a = eng.activityLevel;
  const col = hexToColor3(eng.color);
  const meshes: BABYLON.AbstractMesh[] = [];

  const baseH = 0.8;
  const base = BABYLON.MeshBuilder.CreateBox(`${eng.id}_base`, {
    width: 2,
    height: baseH,
    depth: 2,
  }, scene);
  base.position.set(pos.x, baseH / 2, pos.z);
  base.material = stoneMat(`${eng.id}_baseMat`, 0.08 + a * 0.04, 0.07 + a * 0.03, 0.10 + a * 0.03, scene);
  base.metadata = { engineId: eng.id };
  meshes.push(base);

  buildMainBody(eng.structure, eng.id, pos, a, col, scene, meshes);

  // Window glow
  if (a > 0.3) {
    const winH = 0.5;
    const win = BABYLON.MeshBuilder.CreateBox(`${eng.id}_window`, {
      width: 0.3,
      height: 0.4,
      depth: 0.1,
    }, scene);
    win.position.set(pos.x + 0.6, winH + baseH, pos.z + 1.05);
    const winMat = new BABYLON.StandardMaterial(`${eng.id}_winMat`, scene);
    winMat.diffuseColor = new BABYLON.Color3(col.r * 0.8, col.g * 0.8, col.b * 0.6);
    winMat.emissiveColor = new BABYLON.Color3(col.r * a, col.g * a, col.b * a * 0.5);
    winMat.specularColor = new BABYLON.Color3(0, 0, 0);
    win.material = winMat;
    win.metadata = { engineId: eng.id };
    meshes.push(win);
  }

  // Invisible pick plane
  const label = BABYLON.MeshBuilder.CreatePlane(`${eng.id}_label`, { size: 3 }, scene);
  label.position.set(pos.x, 0.1, pos.z);
  label.rotation.x = Math.PI / 2;
  label.isPickable = true;
  label.metadata = { engineId: eng.id };
  label.visibility = 0;
  meshes.push(label);

  // Engine light
  let light: BABYLON.PointLight | null = null;
  if (a > 0.3) {
    light = new BABYLON.PointLight(`${eng.id}_light`, new BABYLON.Vector3(pos.x, 3, pos.z), scene);
    light.intensity = a * 1.2;
    light.range = 4 + a * 3;
    light.diffuse = col.clone();
  }

  // Blocked smoke particles
  if (eng.artifacts.some((ar) => ar.status === 'blocked')) {
    const smoke = new BABYLON.ParticleSystem(`${eng.id}_smoke`, 100, scene);
    smoke.particleTexture = new BABYLON.Texture('https://assets.babylonjs.com/textures/flare.png', scene);
    smoke.emitter = new BABYLON.Vector3(pos.x, 2, pos.z);
    smoke.minEmitBox = new BABYLON.Vector3(-0.5, 0, -0.5);
    smoke.maxEmitBox = new BABYLON.Vector3(0.5, 0, 0.5);
    smoke.color1 = new BABYLON.Color4(0.15, 0.12, 0.2, 0.3);
    smoke.color2 = new BABYLON.Color4(0.1, 0.08, 0.15, 0);
    smoke.minSize = 0.3;
    smoke.maxSize = 0.8;
    smoke.minLifeTime = 2;
    smoke.maxLifeTime = 4;
    smoke.emitRate = 15;
    smoke.direction1 = new BABYLON.Vector3(-0.1, 1, -0.1);
    smoke.direction2 = new BABYLON.Vector3(0.1, 1.5, 0.1);
    smoke.minEmitPower = 0.1;
    smoke.maxEmitPower = 0.3;
    smoke.start();
  }

  // Active + high momentum ember particles
  if (eng.artifacts.some((ar) => ar.status === 'active') && a > 0.5) {
    const fire = new BABYLON.ParticleSystem(`${eng.id}_fire`, 50, scene);
    fire.particleTexture = new BABYLON.Texture('https://assets.babylonjs.com/textures/flare.png', scene);
    fire.emitter = new BABYLON.Vector3(pos.x, 0.3, pos.z);
    fire.color1 = new BABYLON.Color4(col.r, col.g * 0.6, col.b * 0.2, 0.8);
    fire.color2 = new BABYLON.Color4(col.r * 0.5, col.g * 0.3, 0, 0);
    fire.minSize = 0.05;
    fire.maxSize = 0.2;
    fire.minLifeTime = 0.3;
    fire.maxLifeTime = 0.8;
    fire.emitRate = 30;
    fire.direction1 = new BABYLON.Vector3(-0.05, 1, -0.05);
    fire.direction2 = new BABYLON.Vector3(0.05, 1.5, 0.05);
    fire.minEmitPower = 0.5;
    fire.maxEmitPower = 1;
    fire.start();
  }

  meshes.forEach((m) => {
    if (!m.metadata) m.metadata = { engineId: eng.id };
  });

  return { all: meshes, light };
}

function buildMainBody(
  structure: StructureType,
  id: string,
  pos: { x: number; z: number },
  a: number,
  col: BABYLON.Color3,
  scene: BABYLON.Scene,
  meshes: BABYLON.AbstractMesh[],
): void {
  const baseH = 0.8;

  switch (structure) {
    case 'tower':
    case 'library': {
      // Base: 2×2 box h=1, Main: 1.2×1.2 h=2+(a×4), Roof: 4-sided cone
      const towerH = 2 + a * 4;
      const tower = BABYLON.MeshBuilder.CreateBox(`${id}_tower`, {
        width: 1.2, height: towerH, depth: 1.2,
      }, scene);
      tower.position.set(pos.x, baseH + towerH / 2, pos.z);
      tower.material = stoneMat(`${id}_towerMat`, 0.09 + a * 0.03, 0.08 + a * 0.02, 0.11 + a * 0.04, scene);
      tower.metadata = { engineId: id };
      meshes.push(tower);

      const roof = BABYLON.MeshBuilder.CreateCylinder(`${id}_roof`, {
        diameterTop: 0,
        diameterBottom: 1.6,
        height: 1.2,
        tessellation: 4,
      }, scene);
      roof.position.set(pos.x, baseH + towerH + 0.6, pos.z);
      const roofMat = new BABYLON.StandardMaterial(`${id}_roofMat`, scene);
      roofMat.diffuseColor = new BABYLON.Color3(col.r * 0.4, col.g * 0.4, col.b * 0.4);
      roofMat.specularColor = new BABYLON.Color3(0, 0, 0);
      roof.material = roofMat;
      roof.metadata = { engineId: id };
      meshes.push(roof);
      break;
    }

    case 'garden': {
      // Sphere d=1.5+(a×1), no roof — shrinks/darkens with neglect
      const garden = BABYLON.MeshBuilder.CreateSphere(`${id}_garden`, {
        diameter: 1.5 + a,
        segments: 6,
      }, scene);
      garden.position.set(pos.x, baseH + 0.5 + a * 0.3, pos.z);
      garden.scaling.y = 0.5;
      const gardenMat = new BABYLON.StandardMaterial(`${id}_gardenMat`, scene);
      gardenMat.diffuseColor = new BABYLON.Color3(
        0.1 + a * 0.1,
        0.2 + a * 0.4,
        0.1 + a * 0.1,
      );
      gardenMat.specularColor = new BABYLON.Color3(0, 0, 0);
      garden.material = gardenMat;
      garden.metadata = { engineId: id };
      meshes.push(garden);
      break;
    }

    case 'vault': {
      // 3×2 box h=1.5+(a×1), flat with parapet — widens
      const vaultH = 1.5 + a;
      const vault = BABYLON.MeshBuilder.CreateBox(`${id}_vault`, {
        width: 3 + a,
        height: vaultH,
        depth: 2,
      }, scene);
      vault.position.set(pos.x, baseH + vaultH / 2, pos.z);
      vault.material = stoneMat(`${id}_vaultMat`, 0.09, 0.08, 0.11, scene);
      vault.metadata = { engineId: id };
      meshes.push(vault);

      // Parapet
      const parapet = BABYLON.MeshBuilder.CreateBox(`${id}_parapet`, {
        width: 3 + a + 0.2,
        height: 0.3,
        depth: 2.2,
      }, scene);
      parapet.position.set(pos.x, baseH + vaultH + 0.15, pos.z);
      parapet.material = stoneMat(`${id}_parapetMat`, 0.08, 0.07, 0.10, scene);
      parapet.metadata = { engineId: id };
      meshes.push(parapet);

      const roofMat = new BABYLON.StandardMaterial(`${id}_roofMat`, scene);
      roofMat.diffuseColor = new BABYLON.Color3(col.r * 0.4, col.g * 0.4, col.b * 0.4);
      roofMat.specularColor = new BABYLON.Color3(0, 0, 0);
      parapet.material = roofMat;
      break;
    }

    case 'workshop': {
      // 1.8×1.8 h=1.5+(a×1.5), pitched roof (2 planes)
      const bodyH = 1.5 + a * 1.5;
      const body = BABYLON.MeshBuilder.CreateBox(`${id}_body`, {
        width: 1.8, height: bodyH, depth: 1.8,
      }, scene);
      body.position.set(pos.x, baseH + bodyH / 2, pos.z);
      body.material = stoneMat(`${id}_bodyMat`, 0.09 + a * 0.03, 0.08, 0.12, scene);
      body.metadata = { engineId: id };
      meshes.push(body);

      // Pitched roof: two triangular wedges
      const roofMat = new BABYLON.StandardMaterial(`${id}_roofMat`, scene);
      roofMat.diffuseColor = new BABYLON.Color3(col.r * 0.4, col.g * 0.4, col.b * 0.4);
      roofMat.specularColor = new BABYLON.Color3(0, 0, 0);
      const ridge = BABYLON.MeshBuilder.CreateCylinder(`${id}_ridge`, {
        diameterTop: 0,
        diameterBottom: 2.2,
        height: 0.8,
        tessellation: 4,
      }, scene);
      ridge.position.set(pos.x, baseH + bodyH + 0.4, pos.z);
      ridge.rotation.y = Math.PI / 4;
      ridge.material = roofMat;
      ridge.metadata = { engineId: id };
      meshes.push(ridge);
      break;
    }

    case 'study': {
      // 1.6×1.6 h=1.2+(a×1.2), gabled
      const bodyH = 1.2 + a * 1.2;
      const body = BABYLON.MeshBuilder.CreateBox(`${id}_body`, {
        width: 1.6, height: bodyH, depth: 1.6,
      }, scene);
      body.position.set(pos.x, baseH + bodyH / 2, pos.z);
      body.material = stoneMat(`${id}_bodyMat`, 0.09, 0.08, 0.12, scene);
      body.metadata = { engineId: id };
      meshes.push(body);

      const roofMat = new BABYLON.StandardMaterial(`${id}_roofMat`, scene);
      roofMat.diffuseColor = new BABYLON.Color3(col.r * 0.4, col.g * 0.4, col.b * 0.4);
      roofMat.specularColor = new BABYLON.Color3(0, 0, 0);
      const gable = BABYLON.MeshBuilder.CreateCylinder(`${id}_gable`, {
        diameterTop: 0,
        diameterBottom: 2.0,
        height: 0.7,
        tessellation: 4,
      }, scene);
      gable.position.set(pos.x, baseH + bodyH + 0.35, pos.z);
      gable.rotation.y = Math.PI / 4;
      gable.material = roofMat;
      gable.metadata = { engineId: id };
      meshes.push(gable);
      break;
    }

    case 'hall': {
      // 3.5×2 h=1+(a×0.8), crenellated flat — widens
      const hallH = 1 + a * 0.8;
      const hall = BABYLON.MeshBuilder.CreateBox(`${id}_hall`, {
        width: 3.5 + a,
        height: hallH,
        depth: 2,
      }, scene);
      hall.position.set(pos.x, baseH + hallH / 2, pos.z);
      hall.material = stoneMat(`${id}_hallMat`, 0.09, 0.08, 0.11, scene);
      hall.metadata = { engineId: id };
      meshes.push(hall);

      // Crenellations
      const roofMat = new BABYLON.StandardMaterial(`${id}_roofMat`, scene);
      roofMat.diffuseColor = new BABYLON.Color3(col.r * 0.4, col.g * 0.4, col.b * 0.4);
      roofMat.specularColor = new BABYLON.Color3(0, 0, 0);
      for (let i = 0; i < 5; i++) {
        const crenel = BABYLON.MeshBuilder.CreateBox(`${id}_crenel${i}`, {
          width: 0.3, height: 0.3, depth: 0.3,
        }, scene);
        crenel.position.set(pos.x - 1.5 + i * 0.75, baseH + hallH + 0.15, pos.z + 1);
        crenel.material = roofMat;
        crenel.metadata = { engineId: id };
        meshes.push(crenel);
      }
      break;
    }

    case 'chapel': {
      // 1.6×1.6 h=1.5+(a×2), spire — grows tall
      const bodyH = 1.5 + a * 2;
      const body = BABYLON.MeshBuilder.CreateBox(`${id}_body`, {
        width: 1.6, height: bodyH, depth: 1.6,
      }, scene);
      body.position.set(pos.x, baseH + bodyH / 2, pos.z);
      body.material = stoneMat(`${id}_bodyMat`, 0.09 + a * 0.03, 0.08, 0.12, scene);
      body.metadata = { engineId: id };
      meshes.push(body);

      const spire = BABYLON.MeshBuilder.CreateCylinder(`${id}_spire`, {
        diameterTop: 0,
        diameterBottom: 0.8,
        height: 1.5 + a,
        tessellation: 6,
      }, scene);
      spire.position.set(pos.x, baseH + bodyH + (1.5 + a) / 2, pos.z);
      const spireMat = new BABYLON.StandardMaterial(`${id}_spireMat`, scene);
      spireMat.diffuseColor = new BABYLON.Color3(col.r * 0.4, col.g * 0.4, col.b * 0.4);
      spireMat.emissiveColor = new BABYLON.Color3(col.r * 0.1, col.g * 0.1, col.b * 0.15);
      spireMat.specularColor = new BABYLON.Color3(0, 0, 0);
      spire.material = spireMat;
      spire.metadata = { engineId: id };
      meshes.push(spire);
      break;
    }
  }
}
