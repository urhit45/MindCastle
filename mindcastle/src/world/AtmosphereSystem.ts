import * as BABYLON from 'babylonjs';
import type { VisualParams } from './StateMapper';

export class AtmosphereSystem {
  private mist: BABYLON.ParticleSystem;
  private stars: BABYLON.ParticleSystem;
  private scene: BABYLON.Scene;

  constructor(scene: BABYLON.Scene) {
    this.scene = scene;
    this.stars = this.buildStars();
    this.mist = this.buildMist();
  }

  private buildStars(): BABYLON.ParticleSystem {
    const ps = new BABYLON.ParticleSystem('stars', 500, this.scene);
    ps.particleTexture = new BABYLON.Texture('https://assets.babylonjs.com/textures/flare.png', this.scene);
    ps.emitter = new BABYLON.Vector3(0, 20, 0);
    ps.minEmitBox = new BABYLON.Vector3(-30, -2, -30);
    ps.maxEmitBox = new BABYLON.Vector3(30, 2, 30);
    ps.color1 = new BABYLON.Color4(0.9, 0.9, 1, 0.6);
    ps.color2 = new BABYLON.Color4(0.7, 0.8, 1, 0.3);
    ps.minSize = 0.02;
    ps.maxSize = 0.06;
    ps.minLifeTime = 999;
    ps.maxLifeTime = 999;
    ps.emitRate = 500;
    ps.gravity = new BABYLON.Vector3(0, 0, 0);
    ps.direction1 = new BABYLON.Vector3(0, 0, 0);
    ps.direction2 = new BABYLON.Vector3(0, 0, 0);
    ps.minEmitPower = 0;
    ps.maxEmitPower = 0;
    ps.start();
    return ps;
  }

  private buildMist(): BABYLON.ParticleSystem {
    const ps = new BABYLON.ParticleSystem('mist', 200, this.scene);
    ps.particleTexture = new BABYLON.Texture('https://assets.babylonjs.com/textures/flare.png', this.scene);
    ps.emitter = new BABYLON.Vector3(0, 0.2, 0);
    ps.minEmitBox = new BABYLON.Vector3(-15, 0, -15);
    ps.maxEmitBox = new BABYLON.Vector3(15, 0.1, 15);
    ps.color1 = new BABYLON.Color4(0.5, 0.5, 0.7, 0.05);
    ps.color2 = new BABYLON.Color4(0.4, 0.4, 0.6, 0);
    ps.minSize = 2;
    ps.maxSize = 5;
    ps.minLifeTime = 6;
    ps.maxLifeTime = 10;
    ps.emitRate = 30;
    ps.direction1 = new BABYLON.Vector3(-0.2, 0.1, -0.2);
    ps.direction2 = new BABYLON.Vector3(0.2, 0.3, 0.2);
    ps.minEmitPower = 0.1;
    ps.maxEmitPower = 0.3;
    ps.start();
    return ps;
  }

  applyVisuals(params: VisualParams): void {
    this.scene.fogDensity = params.fogDensity;
    this.mist.emitRate = params.mistEmitRate;
    this.mist.color1 = new BABYLON.Color4(0.5, 0.5, 0.7, params.mistAlpha);
  }

  dispose(): void {
    this.mist.dispose();
    this.stars.dispose();
  }
}
