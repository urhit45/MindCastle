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
