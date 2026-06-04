import * as BABYLON from 'babylonjs';
import type { StateVector, ReadoutState } from '../store/types';

export interface VisualParams {
  fogDensity: number;
  ambientIntensity: number;
  keepLightColor: BABYLON.Color3;
  keepLightIntensity: number;
  mistEmitRate: number;
  mistAlpha: number;
  starIntensity: number;
}

const STATE_COLORS: Record<ReadoutState, BABYLON.Color3> = {
  flow_available:   new BABYLON.Color3(0.9, 0.85, 0.6),
  low_energy:       new BABYLON.Color3(0.3, 0.3, 0.5),
  pre_crash:        new BABYLON.Color3(0.8, 0.3, 0.2),
  blocked_pattern:  new BABYLON.Color3(0.6, 0.4, 0.2),
  recovery:         new BABYLON.Color3(0.4, 0.5, 0.7),
  transition_ready: new BABYLON.Color3(0.7, 0.8, 0.6),
  emergence:        new BABYLON.Color3(0.6, 0.7, 0.9),
};

export function mapStateToVisuals(sv: StateVector): VisualParams {
  return {
    fogDensity:         0.01 + (1 - sv.energy) * 0.03,
    ambientIntensity:   0.08 + sv.energy * 0.12,
    keepLightColor:     STATE_COLORS[sv.readoutState] ?? STATE_COLORS.flow_available,
    keepLightIntensity: 0.6 + sv.momentum * 0.6,
    mistEmitRate:       Math.floor(10 + sv.risk * 80),
    mistAlpha:          sv.risk * 0.18,
    starIntensity:      0.3 + sv.energy * 0.4,
  };
}

export function hexToColor3(hex: string): BABYLON.Color3 {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return new BABYLON.Color3(r, g, b);
}
