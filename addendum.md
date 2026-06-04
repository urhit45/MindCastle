# MindCastle — EEVDF Scheduler Addendum

## Addendum to: MindCastle_ClaudeCode_Brief.md

## Implement this alongside the main brief. It extends, never replaces.

-----

## WHAT THIS IS

MindCastle uses a cognitive adaptation of the Linux EEVDF (Earliest Eligible Virtual
Deadline First) scheduler — introduced in kernel 6.6 — as its core task scheduling
algorithm.

The scheduler answers one question: **which engine should run next, for how long, and why.**

It is not a priority queue. It is not a to-do list sorter. It is a deadline-aware,
eligibility-gated, lag-correcting scheduler that treats your attention as a finite
CPU and your life domains as competing processes.

Key properties inherited from EEVDF:

- Tasks only compete when **eligible** (ve <= current_time)
- Among eligible tasks, pick the one with the **earliest virtual deadline**
- **Lag correction** prevents starvation — neglected engines get earlier deadlines
- The virtual clock advances proportionally to **cognitive availability**, not wall time

-----

## NEW FILES — ADD TO PROJECT STRUCTURE

```
src/
└── scheduler/
    ├── EEVDFScheduler.ts       # Core algorithm — main file
    ├── CognitiveClock.ts       # Virtual time advancement
    ├── EngineWeightConfig.ts   # Per-engine weight (user-configurable)
    └── SchedulerTypes.ts       # All scheduler-specific types
```

Add to existing store:

```
src/store/
└── schedulerStore.ts           # Zustand slice for scheduler state
```

-----

## TYPES — schedulerTypes.ts

```typescript
export interface EngineSchedulerState {
  engineId: string;

  // EEVDF core values — in Cognitive Time Units (CTU)
  // 1 CTU = 1 hour of wall time × energy coefficient
  ve: number;           // virtual eligible time — when engine becomes schedulable
  vd: number;           // virtual deadline — when neglect becomes costly
  lag: number;          // debt: negative = starved, positive = well-served

  // Session config
  weight: number;       // importance coefficient 0.1–2.0, default 1.0 (user-set)
  sliceMinutes: number; // recommended session duration (computed)

  // Derived flags
  eligible: boolean;    // ve <= currentCognitiveTime
  blockedDays: number;  // days since last active log

  // Timestamps
  lastScheduled: string | null;  // ISO
  lastActive: string | null;     // ISO — last progress log
  lastUpdated: string;           // ISO
}

export interface SchedulerOutput {
  engineId: string | null;
  reason: SchedulerReason;
  sliceMinutes: number;
  vd: number;
  eligibleCount: number;
  ineligibleReasons: Record<string, IneligibleReason>;
  // Human-readable — fed to LLM readout interpreter
  diagnosis: string;
}

export type SchedulerReason =
  | 'eevdf'           // normal scheduling
  | 'recovery'        // no eligible engines — rest is the output
  | 'flow_protect'    // current engine has momentum, extend slice
  | 'emergency';      // vd overdue by threshold — must schedule now

export type IneligibleReason =
  | 've_not_reached'       // too soon since last session
  | 'energy_too_low'       // energy below engine weight threshold
  | 'time_affinity_miss'   // wrong time of day for this engine (learned)
  | 'blocked_no_next'      // blocked with no defined next step
  | 'manually_deferred';   // user explicitly deferred

export interface TimeAffinityProfile {
  engineId: string;
  // 24 hourly buckets — 0.0 (never active) to 1.0 (always active)
  hourlyScores: number[];
  sampleCount: number;
  lastUpdated: string;
}

export interface CognitiveClockState {
  virtualTime: number;      // current CTU value
  wallTimeAnchor: number;   // Date.now() at last update
  lastEnergy: number;       // energy at last update
}
```

-----

## COGNITIVE CLOCK — CognitiveClock.ts

```typescript
import { StateVector } from '../store/types';
import { CognitiveClockState } from './SchedulerTypes';

// The virtual clock slows when energy is low.
// A depleted hour is worth less cognitive time than a peak hour.
// This makes the scheduler forgiving by design — deadlines don't
// advance as fast when you're not at capacity.

export class CognitiveClock {
  private state: CognitiveClockState;

  constructor(initialVirtualTime = 0) {
    this.state = {
      virtualTime: initialVirtualTime,
      wallTimeAnchor: Date.now(),
      lastEnergy: 0.5,
    };
  }

  // Call this whenever state vector updates
  advance(sv: StateVector): number {
    const wallNow = Date.now();
    const wallElapsedMinutes = (wallNow - this.state.wallTimeAnchor) / 60_000;

    // Energy coefficient: 0.3 at zero energy, 1.0 at full energy
    // Smoothed between last known energy and current
    const avgEnergy = (this.state.lastEnergy + sv.energy) / 2;
    const energyCoeff = 0.3 + avgEnergy * 0.7;

    const cognitiveMinutesElapsed = wallElapsedMinutes * energyCoeff;

    this.state.virtualTime += cognitiveMinutesElapsed / 60; // store in CTU (hours)
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
```

-----

## ENGINE WEIGHTS — EngineWeightConfig.ts

```typescript
// Default weights — user can adjust in settings (future UI)
// Higher weight = shorter target interval = engine scheduled more frequently

export const DEFAULT_ENGINE_WEIGHTS: Record<string, number> = {
  tech:     1.2,   // core building work — high frequency
  health:   1.4,   // biology is non-negotiable — highest frequency
  finance:  0.8,   // important but not daily
  creative: 1.0,   // balanced
  learning: 1.0,   // balanced
  writing:  0.9,   // slightly less frequent than core work
  social:   0.7,   // lower frequency but starvation penalty is high
  music:    0.8,   // regular but not urgent
};

// Minimum energy required to be eligible, per weight class
// Heavier engines require more energy to be eligible
export function energyThresholdForWeight(weight: number): number {
  return 0.2 + weight * 0.12; // range: 0.21–0.44
}

// Target session interval in cognitive hours
// How often should this engine ideally be scheduled?
export function targetIntervalForWeight(weight: number): number {
  return 48 / weight; // range: 24–480 CTU hours
}
```

-----

## CORE SCHEDULER — EEVDFScheduler.ts

```typescript
import { Engine, StateVector } from '../store/types';
import {
  EngineSchedulerState,
  SchedulerOutput,
  SchedulerReason,
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
    affinityProfiles: Map<string, TimeAffinityProfile>
  ) {
    this.clock = clock;
    this.schedStates = schedStates;
    this.affinityProfiles = affinityProfiles;
  }

  // ── Main scheduling pass ──────────────────────────────────────────────────

  schedule(
    engines: Engine[],
    sv: StateVector,
    currentEngineId: string | null
  ): SchedulerOutput {

    const now = this.clock.current;
    const hour = new Date().getHours();

    // Step 1: evaluate eligibility for all engines
    const eligibilityMap = new Map<string, IneligibleReason | null>();
    engines.forEach(eng => {
      const reason = this.ineligibilityReason(eng, sv, now, hour);
      eligibilityMap.set(eng.id, reason);
    });

    const eligible = engines.filter(e => eligibilityMap.get(e.id) === null);

    // No eligible engines — rest
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

    // Step 2: flow protection — extend current engine if in momentum
    if (currentEngineId && sv.momentum > 0.75) {
      const currentEng = engines.find(e => e.id === currentEngineId);
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

    // Step 3: EEVDF — pick eligible engine with earliest virtual deadline
    const scheduled = eligible.reduce((best, eng) => {
      const vdEng = this.computeVD(this.getOrInitState(eng.id));
      const vdBest = this.computeVD(this.getOrInitState(best.id));
      return vdEng < vdBest ? eng : best;
    });

    const schedState = this.getOrInitState(scheduled.id);
    const slice = this.computeSlice(schedState, sv);
    const vd = this.computeVD(schedState);

    // Step 4: update lag for all engines
    this.updateLag(scheduled.id, eligible);

    // Step 5: update ve for scheduled engine
    schedState.ve = now + (slice / 60); // next eligible after this slice
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

  // ── Eligibility ───────────────────────────────────────────────────────────

  private ineligibilityReason(
    eng: Engine,
    sv: StateVector,
    now: number,
    hour: number
  ): IneligibleReason | null {

    const sched = this.getOrInitState(eng.id);

    // Virtual time not reached
    if (now < sched.ve) return 've_not_reached';

    // Energy too low for this engine's weight
    const threshold = energyThresholdForWeight(sched.weight);
    if (sv.energy < threshold) return 'energy_too_low';

    // Time of day affinity mismatch (learned)
    const affinity = this.getAffinityScore(eng.id, hour);
    if (affinity < 0.15) return 'time_affinity_miss';

    // Blocked with no next step defined
    if (eng.artifacts.every(a => a.status === 'blocked' && !a.next)) {
      return 'blocked_no_next';
    }

    return null;
  }

  // ── Virtual deadline ──────────────────────────────────────────────────────

  private computeVD(sched: EngineSchedulerState): number {
    const interval = targetIntervalForWeight(sched.weight);

    // Base deadline: when this engine was last eligible + target interval
    const base = sched.ve + interval;

    // Lag correction: starved engines get earlier deadlines
    // This is the core EEVDF starvation prevention mechanism
    const lagCorrection = -sched.lag * 0.5;

    // Blocked urgency: engines blocked > 3 days get deadline pulled forward
    const blockedUrgency = sched.blockedDays > 3
      ? -(sched.blockedDays - 3) * 2
      : 0;

    return base + lagCorrection + blockedUrgency;
  }

  // ── Timeslice ─────────────────────────────────────────────────────────────

  private computeSlice(sched: EngineSchedulerState, sv: StateVector): number {
    // Base: 20–55 minutes depending on weight
    const base = 20 + sched.weight * 17.5;

    // Energy: low energy → shorter sessions
    const energyMod = 0.5 + sv.energy * 0.5;

    // Momentum: high momentum → extend (protect flow)
    const momentumMod = sv.momentum > 0.7 ? 1.25 : 1.0;

    // Starvation: heavily starved → shorter but more frequent
    const lagMod = sched.lag < -10 ? 0.7 : 1.0;

    const raw = base * energyMod * momentumMod * lagMod;

    // Clamp: minimum 15 min, maximum 90 min
    return Math.round(Math.min(90, Math.max(15, raw)));
  }

  // ── Lag update ────────────────────────────────────────────────────────────

  private updateLag(scheduledId: string, eligible: Engine[]): void {
    // Scheduled engine: gains lag (it's being served)
    const scheduled = this.getOrInitState(scheduledId);
    scheduled.lag += 1.0;

    // Other eligible engines: lose lag (they're being starved this round)
    eligible
      .filter(e => e.id !== scheduledId)
      .forEach(e => {
        const s = this.getOrInitState(e.id);
        s.lag -= 1.0 / eligible.length;
      });
  }

  // ── Time affinity ─────────────────────────────────────────────────────────

  private getAffinityScore(engineId: string, hour: number): number {
    const profile = this.affinityProfiles.get(engineId);
    if (!profile || profile.sampleCount < 10) return 0.5; // neutral if no data
    return profile.hourlyScores[hour] ?? 0.5;
  }

  // Update affinity when a session occurs
  updateAffinity(engineId: string, hour: number): void {
    if (!this.affinityProfiles.has(engineId)) {
      this.affinityProfiles.set(engineId, {
        engineId,
        hourlyScores: new Array(24).fill(0),
        sampleCount: 0,
        lastUpdated: new Date().toISOString(),
      });
    }
    const profile = this.affinityProfiles.get(engineId)!;
    // Exponential moving average — recent sessions weighted more
    profile.hourlyScores[hour] =
      profile.hourlyScores[hour] * 0.85 + 1.0 * 0.15;
    // Decay all other hours slightly
    for (let h = 0; h < 24; h++) {
      if (h !== hour) profile.hourlyScores[h] *= 0.99;
    }
    profile.sampleCount++;
    profile.lastUpdated = new Date().toISOString();
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

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
    map: Map<string, IneligibleReason | null>
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
    eligibleCount: number
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
```

-----

## ZUSTAND STORE SLICE — schedulerStore.ts

```typescript
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { EngineSchedulerState, CognitiveClockState, TimeAffinityProfile } from '../scheduler/SchedulerTypes';

interface SchedulerStore {
  // Scheduler states per engine
  schedStates: Record<string, EngineSchedulerState>;
  // Cognitive clock persistence
  clockState: CognitiveClockState;
  // Time affinity profiles
  affinityProfiles: Record<string, TimeAffinityProfile>;

  setSchedState: (engineId: string, state: EngineSchedulerState) => void;
  setClockState: (clock: CognitiveClockState) => void;
  setAffinityProfile: (engineId: string, profile: TimeAffinityProfile) => void;
  resetScheduler: () => void;
}

export const useSchedulerStore = create<SchedulerStore>()(
  persist(
    (set) => ({
      schedStates: {},
      clockState: { virtualTime: 0, wallTimeAnchor: Date.now(), lastEnergy: 0.5 },
      affinityProfiles: {},

      setSchedState: (engineId, state) =>
        set(s => ({ schedStates: { ...s.schedStates, [engineId]: state } })),

      setClockState: (clock) => set({ clockState: clock }),

      setAffinityProfile: (engineId, profile) =>
        set(s => ({ affinityProfiles: { ...s.affinityProfiles, [engineId]: profile } })),

      resetScheduler: () => set({
        schedStates: {},
        clockState: { virtualTime: 0, wallTimeAnchor: Date.now(), lastEnergy: 0.5 },
        affinityProfiles: {},
      }),
    }),
    { name: 'mindcastle_scheduler_v1' }
  )
);
```

-----

## DATABASE SCHEMA ADDITION

Add this table to `mindcastle.db` alongside the existing schema:

```sql
-- Scheduler state per engine
CREATE TABLE IF NOT EXISTS scheduler_state (
    engine_id        TEXT PRIMARY KEY,
    ve               REAL    DEFAULT 0.0,
    vd               REAL    DEFAULT 0.0,
    lag              REAL    DEFAULT 0.0,
    weight           REAL    DEFAULT 1.0,
    slice_minutes    INTEGER DEFAULT 45,
    blocked_days     INTEGER DEFAULT 0,
    last_scheduled   TIMESTAMP,
    last_active      TIMESTAMP,
    last_updated     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Time affinity profiles (hourly activation patterns per engine)
CREATE TABLE IF NOT EXISTS affinity_profiles (
    engine_id        TEXT PRIMARY KEY,
    hourly_scores    TEXT    NOT NULL DEFAULT '[]', -- JSON array[24]
    sample_count     INTEGER DEFAULT 0,
    last_updated     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Cognitive clock persistence
CREATE TABLE IF NOT EXISTS cognitive_clock (
    id               INTEGER PRIMARY KEY CHECK (id = 1), -- singleton
    virtual_time     REAL    DEFAULT 0.0,
    wall_time_anchor INTEGER DEFAULT 0, -- Unix ms
    last_energy      REAL    DEFAULT 0.5,
    last_updated     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

-----

## HOW SCHEDULER OUTPUT CONNECTS TO THE CASTLE

The scheduler output replaces the hardcoded `readoutState` and `readoutTask`
in the state vector. Wire it like this:

```typescript
// In CastleWorld.tsx, after state vector poll:

const output = scheduler.schedule(engines, stateVector, currentEngineId);

// Map scheduler output → readout display
const readoutState: ReadoutState = output.engineId === null
  ? 'recovery'
  : output.reason === 'flow_protect'
  ? 'flow_available'
  : stateVector.risk > 0.7
  ? 'pre_crash'
  : 'flow_available';

// Feed to LLM interpreter (localhost:8000/readout/)
// which returns one sentence and one task
const readout = await fetch('http://localhost:8000/readout/', {
  method: 'POST',
  body: JSON.stringify({ schedulerOutput: output, stateVector })
});
```

### Castle visual additions for scheduler state

When scheduler output is available, add these visual cues to the existing scene:

|Condition                                 |Visual                                              |
|------------------------------------------|----------------------------------------------------|
|Engine is scheduled (current)             |Path from keep to structure is fully lit + gold tint|
|Engine is eligible but not scheduled      |Path faintly lit                                    |
|Engine is ineligible: `ve_not_reached`    |Structure gate visually closed (dark doorway box)   |
|Engine is ineligible: `energy_too_low`    |Structure slightly dimmer than base                 |
|Engine is ineligible: `time_affinity_miss`|No change — silently ineligible                     |
|Engine has `lag < -8` (starved)           |Faint red edge glow on structure walls              |
|Engine has `blockedDays > 3`              |Existing smoke particles + subtle wall crack decal  |
|`reason === 'recovery'`                   |Keep light dims to minimum, mist increases          |

-----

## COMPLETION CRITERIA ADDITIONS

Add these to the 11 criteria in the main brief:

1. `EEVDFScheduler.schedule()` returns a valid `SchedulerOutput` given mock engines and state vector
1. Cognitive clock advances slower at low energy than high energy (unit testable)
1. Scheduled engine’s path to keep is visually brighter than other paths
1. Starved engine (lag < -8) shows red edge glow on structure
1. Scheduler state persists across app restarts via Zustand persist middleware

-----

## IMPLEMENTATION ORDER

Build in this sequence after the main brief is complete:

1. `SchedulerTypes.ts` — types only, no logic
1. `CognitiveClock.ts` — pure class, unit testable
1. `EngineWeightConfig.ts` — constants only
1. `EEVDFScheduler.ts` — core algorithm
1. `schedulerStore.ts` — Zustand persistence
1. Wire scheduler into `CastleWorld.tsx` — replace hardcoded state
1. Add scheduler visual cues to `SceneBuilder.ts`
1. Add database tables to migration

-----

*Addendum version 1.0 — EEVDF Cognitive Scheduler*
*Implements Linux kernel EEVDF (Earliest Eligible Virtual Deadline First)*
*adapted for cognitive load scheduling over life domains.*
*Read alongside MindCastle_ClaudeCode_Brief.md*