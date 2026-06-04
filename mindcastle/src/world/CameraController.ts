import * as BABYLON from 'babylonjs';

export class CameraController {
  private camera: BABYLON.ArcRotateCamera;

  constructor(canvas: HTMLCanvasElement, scene: BABYLON.Scene) {
    this.camera = new BABYLON.ArcRotateCamera(
      'cam',
      -Math.PI / 2,
      Math.PI / 3.5,
      22,
      BABYLON.Vector3.Zero(),
      scene,
    );
    this.camera.lowerRadiusLimit = 10;
    this.camera.upperRadiusLimit = 40;
    this.camera.upperBetaLimit = Math.PI / 2.2;
    this.camera.lowerBetaLimit = 0.3;
    this.camera.attachControl(canvas, true);
    this.camera.wheelPrecision = 50;
  }

  focusOn(position: { x: number; z: number }): void {
    const target = new BABYLON.Vector3(position.x * 0.5, 0, position.z * 0.5);
    BABYLON.Animation.CreateAndStartAnimation(
      'camFocus',
      this.camera,
      'target',
      30,
      40,
      this.camera.target.clone(),
      target,
      BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT,
    );
  }

  returnToCenter(): void {
    BABYLON.Animation.CreateAndStartAnimation(
      'camHome',
      this.camera,
      'target',
      30,
      40,
      this.camera.target.clone(),
      BABYLON.Vector3.Zero(),
      BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT,
    );
  }

  get instance(): BABYLON.ArcRotateCamera {
    return this.camera;
  }
}
