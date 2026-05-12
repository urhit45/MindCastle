import { create } from 'zustand';
import type { CastleStore, Engine, StateVector } from './types';
import { DEFAULT_ENGINES } from './defaultEngines';

const STORAGE_KEY = 'mindcastle_v1';

const DEFAULT_STATE_VECTOR: StateVector = {
  energy: 0.7,
  momentum: 0.6,
  risk: 0.2,
  readiness: 0.8,
  readoutState: 'flow_available',
  readoutTask: 'Choose what to work on next',
  lastUpdated: new Date().toISOString(),
};

function loadEngines(): Engine[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Engine[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {
    // fall through to default
  }
  return DEFAULT_ENGINES;
}

export const useCastleStore = create<CastleStore>((set) => ({
  engines: loadEngines(),
  stateVector: DEFAULT_STATE_VECTOR,
  selectedEngineId: null,
  hoveredEngineId: null,

  setEngines: (engines) => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(engines));
    } catch {
      // storage full — continue in-memory
    }
    set({ engines });
  },

  setStateVector: (sv) => set({ stateVector: sv }),
  setSelectedEngine: (id) => set({ selectedEngineId: id }),
  setHoveredEngine: (id) => set({ hoveredEngineId: id }),
}));
