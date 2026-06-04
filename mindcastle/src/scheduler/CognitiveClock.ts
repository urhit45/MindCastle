import type { StateVector } from '../store/types';
import type { CognitiveClockState } from './SchedulerTypes';

// The virtual clock slows when energy is low.
// A depleted hour is worth less cognitive time than a peak hour.
export class CognitiveClock {
  private state: CognitiveClockState;

  constructor(initialVirtualTime = 0) {
    this.state = {
      virtualTime: initialVirtualTime,
      wallTimeAnchor: Date.now(),
      lastEnergy: 0.5,
    };
  }

  advance(sv: StateVector): number {
    const wallNow = Date.now();
    const wallElapsedMinutes = (wallNow - this.state.wallTimeAnchor) / 60_000;

    const avgEnergy = (this.state.lastEnergy + sv.energy) / 2;
    const energyCoeff = 0.3 + avgEnergy * 0.7;

    const cognitiveMinutesElapsed = wallElapsedMinutes * energyCoeff;

    this.state.virtualTime += cognitiveMinutesElapsed / 60;
    this.state.wallTimeAnchor = wallNow;
    this.state.lastEnergy = sv.energy;

    return this.state.virtualTime;
  }

  get current(): number {
    return this.state.virtualTime;
  }

  serialize(): CognitiveClockState {
    return { ...this.state };
  }

  static fromSerialized(s: CognitiveClockState): CognitiveClock {
    const clock = new CognitiveClock(s.virtualTime);
    clock.state = { ...s };
    return clock;
  }
}
