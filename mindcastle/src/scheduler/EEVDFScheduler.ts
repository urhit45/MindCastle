import type { Engine, StateVector } from '../store/types';
import type {
  EngineSchedulerState,
  SchedulerOutput,
  IneligibleReason,
  TimeAffinityProfile,
} from './SchedulerTypes';
import { CognitiveClock } from './CognitiveClock';
import {
  DEFAULT_ENGINE_WEIGHTS,
  energyThresholdForWeight,
  targetIntervalForWeight,
} from './EngineWeightConfig';

export class EEVDFScheduler {
  private clock: CognitiveClock;
  private schedStates: Map<string, EngineSchedulerState>;
  private affinityProfiles: Map<string, TimeAffinityProfile>;

  constructor(
    clock: CognitiveClock,
    schedStates: Map<string, EngineSchedulerState>,
    affinityProfiles: Map<string, TimeAffinityProfile>,
  ) {
    this.clock = clock;
    this.schedStates = schedStates;
    this.affinityProfiles = affinityProfiles;
  }

  schedule(
    engines: Engine[],
    sv: StateVector,
    currentEngineId: string | null,
  ): SchedulerOutput {
    const now = this.clock.current;
    const hour = new Date().getHours();

    const eligibilityMap = new Map<string, IneligibleReason | null>();
    engines.forEach((eng) => {
      const reason = this.ineligibilityReason(eng, sv, now, hour);
      eligibilityMap.set(eng.id, reason);
    });

    const eligible = engines.filter((e) => eligibilityMap.get(e.id) === null);

    if (eligible.length === 0) {
      return {
        engineId: null,
        reason: 'recovery',
        sliceMinutes: 0,
        vd: Infinity,
        eligibleCount: 0,
        ineligibleReasons: this.buildIneligibleMap(eligibilityMap),
        diagnosis: 'No engines are eligible right now. Rest is the correct output.',
      };
    }

    if (currentEngineId && sv.momentum > 0.75) {
      const currentEng = engines.find((e) => e.id === currentEngineId);
      if (currentEng && eligibilityMap.get(currentEngineId) === null) {
        const sched = this.getOrInitState(currentEngineId);
        return {
          engineId: currentEngineId,
          reason: 'flow_protect',
          sliceMinutes: this.computeSlice(sched, sv) + 20,
          vd: this.computeVD(sched),
          eligibleCount: eligible.length,
          ineligibleReasons: this.buildIneligibleMap(eligibilityMap),
          diagnosis: `Momentum detected in ${currentEng.name}. Extending session.`,
        };
      }
    }

    const scheduled = eligible.reduce((best, eng) => {
      const vdEng = this.computeVD(this.getOrInitState(eng.id));
      const vdBest = this.computeVD(this.getOrInitState(best.id));
      return vdEng < vdBest ? eng : best;
    });

    const schedState = this.getOrInitState(scheduled.id);
    const slice = this.computeSlice(schedState, sv);
    const vd = this.computeVD(schedState);

    this.updateLag(scheduled.id, eligible);

    schedState.ve = now + slice / 60;
    schedState.lastScheduled = new Date().toISOString();
    schedState.sliceMinutes = slice;

    return {
      engineId: scheduled.id,
      reason: 'eevdf',
      sliceMinutes: slice,
      vd,
      eligibleCount: eligible.length,
      ineligibleReasons: this.buildIneligibleMap(eligibilityMap),
      diagnosis: this.buildDiagnosis(scheduled, schedState, slice, eligible.length),
    };
  }

  private ineligibilityReason(
    eng: Engine,
    sv: StateVector,
    now: number,
    hour: number,
  ): IneligibleReason | null {
    const sched = this.getOrInitState(eng.id);

    if (now < sched.ve) return 've_not_reached';

    const threshold = energyThresholdForWeight(sched.weight);
    if (sv.energy < threshold) return 'energy_too_low';

    const affinity = this.getAffinityScore(eng.id, hour);
    if (affinity < 0.15) return 'time_affinity_miss';

    if (eng.artifacts.every((a) => a.status === 'blocked' && !a.next)) {
      return 'blocked_no_next';
    }

    return null;
  }

  private computeVD(sched: EngineSchedulerState): number {
    const interval = targetIntervalForWeight(sched.weight);
    const base = sched.ve + interval;
    const lagCorrection = -sched.lag * 0.5;
    const blockedUrgency = sched.blockedDays > 3 ? -(sched.blockedDays - 3) * 2 : 0;
    return base + lagCorrection + blockedUrgency;
  }

  private computeSlice(sched: EngineSchedulerState, sv: StateVector): number {
    const base = 20 + sched.weight * 17.5;
    const energyMod = 0.5 + sv.energy * 0.5;
    const momentumMod = sv.momentum > 0.7 ? 1.25 : 1.0;
    const lagMod = sched.lag < -10 ? 0.7 : 1.0;
    const raw = base * energyMod * momentumMod * lagMod;
    return Math.round(Math.min(90, Math.max(15, raw)));
  }

  private updateLag(scheduledId: string, eligible: Engine[]): void {
    const scheduled = this.getOrInitState(scheduledId);
    scheduled.lag += 1.0;

    eligible
      .filter((e) => e.id !== scheduledId)
      .forEach((e) => {
        const s = this.getOrInitState(e.id);
        s.lag -= 1.0 / eligible.length;
      });
  }

  private getAffinityScore(engineId: string, hour: number): number {
    const profile = this.affinityProfiles.get(engineId);
    if (!profile || profile.sampleCount < 10) return 0.5;
    return profile.hourlyScores[hour] ?? 0.5;
  }

  updateAffinity(engineId: string, hour: number): void {
    if (!this.affinityProfiles.has(engineId)) {
      this.affinityProfiles.set(engineId, {
        engineId,
        hourlyScores: new Array(24).fill(0) as number[],
        sampleCount: 0,
        lastUpdated: new Date().toISOString(),
      });
    }
    const profile = this.affinityProfiles.get(engineId)!;
    profile.hourlyScores[hour] = profile.hourlyScores[hour] * 0.85 + 1.0 * 0.15;
    for (let h = 0; h < 24; h++) {
      if (h !== hour) profile.hourlyScores[h] *= 0.99;
    }
    profile.sampleCount++;
    profile.lastUpdated = new Date().toISOString();
  }

  getScheduleStates(): Map<string, EngineSchedulerState> {
    return this.schedStates;
  }

  private getOrInitState(engineId: string): EngineSchedulerState {
    if (!this.schedStates.has(engineId)) {
      this.schedStates.set(engineId, {
        engineId,
        ve: 0,
        vd: 0,
        lag: 0,
        weight: DEFAULT_ENGINE_WEIGHTS[engineId] ?? 1.0,
        sliceMinutes: 45,
        eligible: false,
        blockedDays: 0,
        lastScheduled: null,
        lastActive: null,
        lastUpdated: new Date().toISOString(),
      });
    }
    return this.schedStates.get(engineId)!;
  }

  private buildIneligibleMap(
    map: Map<string, IneligibleReason | null>,
  ): Record<string, IneligibleReason> {
    const result: Record<string, IneligibleReason> = {};
    map.forEach((reason, id) => {
      if (reason !== null) result[id] = reason;
    });
    return result;
  }

  private buildDiagnosis(
    eng: Engine,
    sched: EngineSchedulerState,
    slice: number,
    eligibleCount: number,
  ): string {
    const lines = [
      `Scheduled: ${eng.name} (weight: ${sched.weight}, lag: ${sched.lag.toFixed(1)})`,
      `Slice: ${slice} minutes`,
      `Virtual deadline: ${this.computeVD(sched).toFixed(2)} CTU`,
      `Eligible engines: ${eligibleCount} of 8`,
    ];
    if (sched.blockedDays > 3) {
      lines.push(`Warning: blocked for ${sched.blockedDays} days — deadline pulled forward`);
    }
    if (sched.lag < -5) {
      lines.push(`Note: starvation correction applied (lag: ${sched.lag.toFixed(1)})`);
    }
    return lines.join('\n');
  }
}
