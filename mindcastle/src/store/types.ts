export type EngineStatus = 'concept' | 'active' | 'planned' | 'live' | 'blocked' | 'planning';

export type ReadoutState =
  | 'flow_available'
  | 'low_energy'
  | 'transition_ready'
  | 'pre_crash'
  | 'recovery'
  | 'blocked_pattern'
  | 'emergence';

export type StructureType =
  | 'tower'
  | 'library'  // same geometry as tower — Learning engine
  | 'garden'
  | 'vault'
  | 'workshop'
  | 'study'
  | 'hall'
  | 'chapel';

export interface StateVector {
  energy: number;
  momentum: number;
  risk: number;
  readiness: number;
  readoutState: ReadoutState;
  readoutTask: string;
  lastUpdated: string;
}

export interface ProgressBar {
  label: string;
  val: number;
}

export interface Artifact {
  id: string;
  node_id?: string;
  title: string;
  subtitle?: string;
  status: EngineStatus;
  next?: string;
  stack?: string[];
  progress?: ProgressBar[];
  notes?: string;
}

export interface Engine {
  id: string;
  name: string;
  subtitle: string;
  description?: string;
  color: string;
  icon: string;
  context?: string;
  structure: StructureType;
  activityLevel: number;
  artifacts: Artifact[];
}

export interface CastleStore {
  engines: Engine[];
  stateVector: StateVector;
  selectedEngineId: string | null;
  hoveredEngineId: string | null;
  setEngines: (engines: Engine[]) => void;
  setStateVector: (sv: StateVector) => void;
  setSelectedEngine: (id: string | null) => void;
  setHoveredEngine: (id: string | null) => void;
}

export const DEFAULT_STATE_VECTOR: StateVector = {
  energy: 0.65,
  momentum: 0.72,
  risk: 0.28,
  readiness: 0.58,
  readoutState: 'flow_available',
  readoutTask: 'Your Tech Builder tower is lit. One deep session awaits.',
  lastUpdated: new Date().toISOString(),
};

export const DEFAULT_ENGINES: Engine[] = [
  {
    id: 'tech',
    name: 'Tech Builder',
    subtitle: 'Code, systems & side projects',
    color: '#3ecfcf',
    icon: '◈',
    structure: 'tower',
    activityLevel: 0.85,
    artifacts: [
      {
        id: 'art-1',
        title: 'TinyNet API',
        status: 'active',
        next: 'Write integration tests for /classify/ endpoint',
      },
    ],
  },
  {
    id: 'health',
    name: 'Health',
    subtitle: 'Fitness, sleep & nutrition',
    color: '#4dcc66',
    icon: '◎',
    structure: 'garden',
    activityLevel: 0.15,
    artifacts: [
      {
        id: 'art-2',
        title: 'Morning run habit',
        status: 'blocked',
        next: 'Start with 10 minutes, not 30',
      },
    ],
  },
  {
    id: 'finance',
    name: 'Finance',
    subtitle: 'Budgets, savings & investing',
    color: '#e6b233',
    icon: '⬡',
    structure: 'vault',
    activityLevel: 0.35,
    artifacts: [],
  },
  {
    id: 'creative',
    name: 'Creative',
    subtitle: 'Art, design & making things',
    color: '#cc4db3',
    icon: '◇',
    structure: 'workshop',
    activityLevel: 0.6,
    artifacts: [],
  },
  {
    id: 'learning',
    name: 'Learning',
    subtitle: 'Books, courses & deep dives',
    color: '#6680e6',
    icon: '⟁',
    structure: 'library',
    activityLevel: 0.7,
    artifacts: [],
  },
  {
    id: 'writing',
    name: 'Writing',
    subtitle: 'Essays, docs & storytelling',
    color: '#e6993d',
    icon: '✦',
    structure: 'study',
    activityLevel: 0.25,
    artifacts: [],
  },
  {
    id: 'social',
    name: 'Social',
    subtitle: 'Relationships & community',
    color: '#e66666',
    icon: '△',
    structure: 'hall',
    activityLevel: 0.2,
    artifacts: [],
  },
  {
    id: 'music',
    name: 'Music',
    subtitle: 'Practice, theory & recording',
    color: '#80ccee',
    icon: '⊕',
    structure: 'chapel',
    activityLevel: 0.45,
    artifacts: [],
  },
];

// Engine positions in the 3D world
export const ENGINE_POSITIONS: Record<string, { x: number; z: number }> = {
  tech:     { x: -3, z: -3 },
  health:   { x:  3, z: -3 },
  finance:  { x:  3, z:  3 },
  creative: { x: -3, z:  3 },
  learning: { x:  0, z: -5 },
  writing:  { x:  0, z:  5 },
  social:   { x: -5, z:  0 },
  music:    { x:  5, z:  0 },
};
