import * as BABYLON from 'babylonjs';

export function buildCamera(scene: BABYLON.Scene, canvas: HTMLCanvasElement): BABYLON.ArcRotateCamera {
  const cam = new BABYLON.ArcRotateCamera('cam', -Math.PI / 2, Math.PI / 3.5, 22, BABYLON.Vector3.Zero(), scene);
  cam.attachControl(canvas, true);
  cam.lowerRadiusLimit = 10;
  cam.upperRadiusLimit = 40;
  cam.lowerBetaLimit = 0.3;
  cam.upperBetaLimit = Math.PI / 2.2;
  cam.wheelPrecision = 5;
  cam.panningSensibility = 0;
  return cam;
}

export function focusOnPosition(
  cam: BABYLON.ArcRotateCamera,
  targetPos: BABYLON.Vector3
): void {
  const dest = targetPos.scale(0.5);
  BABYLON.Animation.CreateAndStartAnimation(
    'camFocus', cam, 'target', 30, 40,
    cam.target.clone(), dest,
    BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT
  );
}

export function resetCamera(cam: BABYLON.ArcRotateCamera): void {
  BABYLON.Animation.CreateAndStartAnimation(
    'camReset', cam, 'target', 30, 40,
    cam.target.clone(), BABYLON.Vector3.Zero(),
    BABYLON.Animation.ANIMATIONLOOPMODE_CONSTANT
  );
}
