import * as BABYLON from 'babylonjs';
import type { Engine } from '../store/types';
import type { VisualParams } from './StateMapper';
import type { SchedulerOutput } from '../scheduler/SchedulerTypes';
import type { EngineSchedulerState } from '../scheduler/SchedulerTypes';
import { buildStructure } from './EngineStructure';
import { ENGINE_POSITIONS } from '../store/types';
import { hexToColor3 } from './StateMapper';

export class SceneBuilder {
  private scene: BABYLON.Scene;
  private engineLights = new Map<string, BABYLON.PointLight>();
  private pathMaterials = new Map<string, BABYLON.StandardMaterial>();
  private keepLight: BABYLON.PointLight | null = null;
  private ambientLight: BABYLON.HemisphericLight | null = null;
  private glowLayer: BABYLON.GlowLayer;
  private edgeGlowMeshes = new Map<string, BABYLON.AbstractMesh>();

  constructor(scene: BABYLON.Scene) {
    this.scene = scene;
    this.glowLayer = new BABYLON.GlowLayer('glow', scene);
    this.glowLayer.intensity = 0.6;
  }

  buildScene(engines: Engine[]): void {
    this.buildGround();
    this.buildKeep();
    engines.forEach((eng) => this.buildEngineStructure(eng));
    this.buildPathways(engines);
  }

  private buildGround(): void {
    const ground = BABYLON.MeshBuilder.CreateGround('ground', {
      width: 40, height: 40, subdivisions: 8,
    }, this.scene);
    const mat = new BABYLON.StandardMaterial('groundMat', this.scene);
    mat.diffuseColor = new BABYLON.Color3(0.06, 0.055, 0.07);
    mat.specularColor = new BABYLON.Color3(0, 0, 0);
    ground.material = mat;

    const grid = BABYLON.MeshBuilder.CreateGround('groundGrid', {
      width: 40, height: 40, subdivisions: 20,
    }, this.scene);
    const gridMat = new BABYLON.StandardMaterial('gridMat', this.scene);
    gridMat.wireframe = true;
    gridMat.alpha = 0.04;
    gridMat.diffuseColor = new BABYLON.Color3(0.8, 0.7, 0.5);
    grid.material = gridMat;
    grid.position.y = 0.01;
  }

  private buildKeep(): void {
    const keepBase = BABYLON.MeshBuilder.CreateBox('keep', {
      width: 3, height: 2.5, depth: 3,
    }, this.scene);
    keepBase.position.y = 1.25;
    this.applyStone(keepBase, 0.12, 0.11, 0.14);

    const keepTower = BABYLON.MeshBuilder.CreateBox('keepTower', {
      width: 1.5, height: 4, depth: 1.5,
    }, this.scene);
    keepTower.position.y = 4;
    this.applyStone(keepTower, 0.10, 0.09, 0.12);

    for (let i = 0; i < 4; i++) {
      const crenel = BABYLON.MeshBuilder.CreateBox(`keepCrenel${i}`, {
        width: 0.3, height: 0.4, depth: 0.3,
      }, this.scene);
      crenel.position.y = 6.2;
      const angle = (i / 4) * Math.PI * 2;
      crenel.position.x = Math.cos(angle) * 0.6;
      crenel.position.z = Math.sin(angle) * 0.6;
      this.applyStone(crenel, 0.10, 0.09, 0.12);
    }

    this.keepLight = new BABYLON.PointLight('keepLight', new BABYLON.Vector3(0, 3, 0), this.scene);
    this.keepLight.intensity = 0.8;
    this.keepLight.range = 8;
    this.keepLight.diffuse = new BABYLON.Color3(0.9, 0.85, 0.6);
  }

  private buildEngineStructure(eng: Engine): void {
    const pos = ENGINE_POSITIONS[eng.id] ?? { x: 0, z: 0 };
    const result = buildStructure(eng, pos, this.scene);
    if (result.light) {
      this.engineLights.set(eng.id, result.light);
    }
  }

  private buildPathways(engines: Engine[]): void {
    engines.forEach((eng) => {
      const pos = ENGINE_POSITIONS[eng.id] ?? { x: 0, z: 0 };
      const from = new BABYLON.Vector3(0, 0.02, 0);
      const to = new BABYLON.Vector3(pos.x * 0.6, 0.02, pos.z * 0.6);
      const path = BABYLON.MeshBuilder.CreateTube(`path_${eng.id}`, {
        path: [from, to],
        radius: 0.12,
        tessellation: 4,
      }, this.scene);
      const mat = new BABYLON.StandardMaterial(`pathMat_${eng.id}`, this.scene);
      const w = eng.activityLevel * 0.5;
      mat.diffuseColor = new BABYLON.Color3(0.08 + w, 0.07 + w, 0.09 + w);
      mat.specularColor = new BABYLON.Color3(0, 0, 0);
      if (eng.activityLevel > 0.5) {
        const col = hexToColor3(eng.color);
        mat.emissiveColor = new BABYLON.Color3(col.r * w * 0.3, col.g * w * 0.3, col.b * w * 0.3);
      }
      path.material = mat;
      path.isPickable = false;
      this.pathMaterials.set(eng.id, mat);
    });
  }

  setupLighting(): void {
    const ambient = new BABYLON.HemisphericLight('ambient', new BABYLON.Vector3(0, 1, 0), this.scene);
    ambient.intensity = 0.12;
    ambient.groundColor = new BABYLON.Color3(0.05, 0.04, 0.08);
    ambient.diffuse = new BABYLON.Color3(0.4, 0.35, 0.3);
    this.ambientLight = ambient;

    const moon = new BABYLON.DirectionalLight('moon', new BABYLON.Vector3(-0.5, -1, -0.3), this.scene);
    moon.intensity = 0.3;
    moon.diffuse = new BABYLON.Color3(0.6, 0.65, 0.8);
  }

  applyVisuals(params: VisualParams): void {
    if (this.keepLight) {
      this.keepLight.diffuse = params.keepLightColor;
      this.keepLight.intensity = params.keepLightIntensity;
    }
    if (this.ambientLight) {
      this.ambientLight.intensity = params.ambientIntensity;
    }
  }

  // Called per-frame from the animation loop
  tickLights(tick: number, energy: number, engines: Engine[]): void {
    if (this.keepLight) {
      this.keepLight.intensity = 0.7 + Math.sin(tick * 1.2) * 0.15 * energy;
    }
    engines.forEach((eng) => {
      const light = this.engineLights.get(eng.id);
      if (light) {
        const pos = ENGINE_POSITIONS[eng.id] ?? { x: 0, z: 0 };
        const breathe = Math.sin(tick * (0.8 + eng.activityLevel) + pos.x) * 0.1;
        light.intensity = eng.activityLevel * 1.2 + breathe;
      }
    });
  }

  applySchedulerVisuals(
    output: SchedulerOutput | null,
    schedStates: Record<string, EngineSchedulerState>,
    engines: Engine[],
  ): void {
    engines.forEach((eng) => {
      const pathMat = this.pathMaterials.get(eng.id);
      if (!pathMat) return;

      const col = hexToColor3(eng.color);
      const isScheduled = output?.engineId === eng.id;
      const isEligible = output ? !(eng.id in output.ineligibleReasons) : false;
      const sched = schedStates[eng.id];

      if (isScheduled) {
        // Fully lit gold path
        pathMat.emissiveColor = new BABYLON.Color3(
          col.r * 0.5 + 0.3,
          col.g * 0.5 + 0.25,
          col.b * 0.3 + 0.1,
        );
        pathMat.diffuseColor = new BABYLON.Color3(0.3, 0.25, 0.1);
      } else if (isEligible) {
        // Faintly lit
        pathMat.emissiveColor = new BABYLON.Color3(col.r * 0.1, col.g * 0.1, col.b * 0.1);
      }

      // Red edge glow for starved engines (lag < -8)
      if (sched && sched.lag < -8) {
        this.addEdgeGlow(eng);
      } else {
        this.removeEdgeGlow(eng.id);
      }

      // Recovery: dim keep light
      if (output?.reason === 'recovery' && this.keepLight) {
        this.keepLight.intensity = 0.2;
      }
    });
  }

  private addEdgeGlow(eng: Engine): void {
    if (this.edgeGlowMeshes.has(eng.id)) return;
    const pos = ENGINE_POSITIONS[eng.id] ?? { x: 0, z: 0 };
    const glow = BABYLON.MeshBuilder.CreateBox(`${eng.id}_edgeGlow`, {
      width: 2.4, height: 0.05, depth: 2.4,
    }, this.scene);
    glow.position.set(pos.x, 0.85, pos.z);
    glow.isPickable = false;
    const mat = new BABYLON.StandardMaterial(`${eng.id}_edgeGlowMat`, this.scene);
    mat.emissiveColor = new BABYLON.Color3(0.8, 0.1, 0.1);
    mat.alpha = 0.5;
    glow.material = mat;
    this.edgeGlowMeshes.set(eng.id, glow);
  }

  private removeEdgeGlow(engineId: string): void {
    const mesh = this.edgeGlowMeshes.get(engineId);
    if (mesh) {
      mesh.dispose();
      this.edgeGlowMeshes.delete(engineId);
    }
  }

  private applyStone(mesh: BABYLON.AbstractMesh, r: number, g: number, b: number): void {
    const mat = new BABYLON.StandardMaterial(`${mesh.name}Mat`, this.scene);
    mat.diffuseColor = new BABYLON.Color3(r, g, b);
    mat.specularColor = new BABYLON.Color3(0.02, 0.02, 0.02);
    mesh.material = mat;
  }
}
