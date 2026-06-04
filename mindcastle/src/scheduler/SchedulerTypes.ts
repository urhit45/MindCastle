export interface EngineSchedulerState {
  engineId: string;

  // EEVDF core values — in Cognitive Time Units (CTU)
  // 1 CTU = 1 hour of wall time × energy coefficient
  ve: number;
  vd: number;
  lag: number;

  weight: number;
  sliceMinutes: number;

  eligible: boolean;
  blockedDays: number;

  lastScheduled: string | null;
  lastActive: string | null;
  lastUpdated: string;
}

export interface SchedulerOutput {
  engineId: string | null;
  reason: SchedulerReason;
  sliceMinutes: number;
  vd: number;
  eligibleCount: number;
  ineligibleReasons: Record<string, IneligibleReason>;
  diagnosis: string;
}

export type SchedulerReason =
  | 'eevdf'
  | 'recovery'
  | 'flow_protect'
  | 'emergency';

export type IneligibleReason =
  | 've_not_reached'
  | 'energy_too_low'
  | 'time_affinity_miss'
  | 'blocked_no_next'
  | 'manually_deferred';

export interface TimeAffinityProfile {
  engineId: string;
  hourlyScores: number[];
  sampleCount: number;
  lastUpdated: string;
}

export interface CognitiveClockState {
  virtualTime: number;
  wallTimeAnchor: number;
  lastEnergy: number;
}
