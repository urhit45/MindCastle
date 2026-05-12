import * as BABYLON from 'babylonjs';
import type { VisualParams } from './StateMapper';

const FLARE_URL = 'https://assets.babylonjs.com/textures/flare.png';

export interface AtmosphereHandles {
  mist: BABYLON.ParticleSystem;
  stars: BABYLON.ParticleSystem;
  keepLight: BABYLON.DirectionalLight;
  ambientLight: BABYLON.HemisphericLight;
  update: (params: VisualParams) => void;
}

export function buildAtmosphere(scene: BABYLON.Scene): AtmosphereHandles {
  scene.clearColor = new BABYLON.Color4(0.04, 0.03, 0.06, 1);
  scene.fogMode = BABYLON.Scene.FOGMODE_EXP2;
  scene.fogColor = new BABYLON.Color3(0.04, 0.035, 0.06);
  scene.fogDensity = 0.02;

  // Ambient
  const ambientLight = new BABYLON.HemisphericLight('ambient', new BABYLON.Vector3(0, 1, 0), scene);
  ambientLight.diffuse = new BABYLON.Color3(0.12, 0.10, 0.15);
  ambientLight.groundColor = new BABYLON.Color3(0.04, 0.03, 0.06);
  ambientLight.intensity = 0.12;

  // Keep (directional)
  const keepLight = new BABYLON.DirectionalLight('keep', new BABYLON.Vector3(-0.3, -1, -0.5), scene);
  keepLight.diffuse = new BABYLON.Color3(0.9, 0.85, 0.6);
  keepLight.intensity = 0.8;

  // Ground mist
  const mist = new BABYLON.ParticleSystem('mist', 500, scene);
  mist.particleTexture = new BABYLON.Texture(FLARE_URL, scene);
  mist.emitter = new BABYLON.Vector3(0, 0.1, 0);
  mist.minEmitBox = new BABYLON.Vector3(-18, 0, -18);
  mist.maxEmitBox = new BABYLON.Vector3(18, 0.2, 18);
  mist.color1 = new BABYLON.Color4(0.3, 0.35, 0.5, 0.05);
  mist.color2 = new BABYLON.Color4(0.25, 0.3, 0.45, 0.03);
  mist.colorDead = new BABYLON.Color4(0, 0, 0, 0);
  mist.minSize = 1.5;
  mist.maxSize = 3.5;
  mist.minLifeTime = 4.0;
  mist.maxLifeTime = 8.0;
  mist.emitRate = 30;
  mist.direction1 = new BABYLON.Vector3(-0.2, 0.05, -0.2);
  mist.direction2 = new BABYLON.Vector3(0.2, 0.1, 0.2);
  mist.minEmitPower = 0.05;
  mist.maxEmitPower = 0.15;
  mist.updateSpeed = 0.005;
  mist.start();

  // Stars
  const stars = new BABYLON.ParticleSystem('stars', 500, scene);
  stars.particleTexture = new BABYLON.Texture(FLARE_URL, scene);
  stars.emitter = new BABYLON.Vector3(0, 20, 0);
  stars.minEmitBox = new BABYLON.Vector3(-25, 0, -25);
  stars.maxEmitBox = new BABYLON.Vector3(25, 0.01, 25);
  stars.color1 = new BABYLON.Color4(0.8, 0.85, 1.0, 0.7);
  stars.color2 = new BABYLON.Color4(0.6, 0.7, 0.9, 0.5);
  stars.colorDead = new BABYLON.Color4(0.6, 0.7, 0.9, 0.4);
  stars.minSize = 0.05;
  stars.maxSize = 0.15;
  stars.minLifeTime = 999;
  stars.maxLifeTime = 1000;
  stars.emitRate = 2;
  stars.direction1 = new BABYLON.Vector3(0, -0.001, 0);
  stars.direction2 = new BABYLON.Vector3(0, -0.001, 0);
  stars.minEmitPower = 0;
  stars.maxEmitPower = 0;
  stars.updateSpeed = 0.001;
  stars.start();

  function update(params: VisualParams) {
    scene.fogDensity = params.fogDensity;
    ambientLight.intensity = params.ambientIntensity;
    keepLight.diffuse = params.keepLightColor;
    keepLight.intensity = params.keepLightIntensity;
    mist.emitRate = params.mistEmitRate;
    mist.color1.a = params.mistAlpha + 0.03;
    mist.color2.a = params.mistAlpha;
    stars.color1.a = params.starIntensity;
    stars.color2.a = params.starIntensity * 0.7;
  }

  return { mist, stars, keepLight, ambientLight, update };
}
