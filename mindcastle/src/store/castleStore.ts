import { create } from 'zustand';
import type { Engine, StateVector, CastleStore } from './types';
import { DEFAULT_ENGINES, DEFAULT_STATE_VECTOR } from './types';

const STORAGE_KEY = 'mindcastle_v1';

function loadEngines(): Engine[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw) as Engine[];
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch {
    // ignore parse errors
  }
  return DEFAULT_ENGINES;
}

function persistEngines(engines: Engine[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(engines));
  } catch {
    // ignore storage errors
  }
}

export const useCastleStore = create<CastleStore>()((set) => ({
  engines: loadEngines(),
  stateVector: DEFAULT_STATE_VECTOR,
  selectedEngineId: null,
  hoveredEngineId: null,

  setEngines: (engines) => {
    persistEngines(engines);
    set({ engines });
  },

  setStateVector: (sv: StateVector) => set({ stateVector: sv }),

  setSelectedEngine: (id) => set({ selectedEngineId: id }),

  setHoveredEngine: (id) => set({ hoveredEngineId: id }),
}));
