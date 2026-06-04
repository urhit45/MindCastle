import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  EngineSchedulerState,
  CognitiveClockState,
  TimeAffinityProfile,
} from '../scheduler/SchedulerTypes';

interface SchedulerStore {
  schedStates: Record<string, EngineSchedulerState>;
  clockState: CognitiveClockState;
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
        set((s) => ({ schedStates: { ...s.schedStates, [engineId]: state } })),

      setClockState: (clock) => set({ clockState: clock }),

      setAffinityProfile: (engineId, profile) =>
        set((s) => ({ affinityProfiles: { ...s.affinityProfiles, [engineId]: profile } })),

      resetScheduler: () =>
        set({
          schedStates: {},
          clockState: { virtualTime: 0, wallTimeAnchor: Date.now(), lastEnergy: 0.5 },
          affinityProfiles: {},
        }),
    }),
    { name: 'mindcastle_scheduler_v1' },
  ),
);
