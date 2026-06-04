# MindCastle — Claude Code Implementation Brief

## For: Claude Code (agentic execution)

## Status: POC Phase 1 — Castle World in Tauri + Babylon.js

-----

## WHAT YOU ARE BUILDING

A macOS desktop application called **MindCastle**. It is a personal operating system for neurodivergent thinkers. The UI is a living 3D castle rendered in Babylon.js inside a Tauri 2.0 shell. The castle is not decoration — it IS the interface. Data drives geometry, lighting, particles, and atmosphere.

Do not build a productivity app with 3D decoration. Build a 3D world that contains productivity data.

-----

## TECH STACK — NON-NEGOTIABLE

|Layer           |Technology                                          |Notes                                       |
|----------------|----------------------------------------------------|--------------------------------------------|
|Desktop shell   |Tauri 2.0                                           |Rust backend, macOS only for POC            |
|3D engine       |Babylon.js 6.x                                      |Loaded via npm, not CDN                     |
|Frontend        |React 18 + TypeScript                               |Vite 5                                      |
|3D-React bridge |react-babylonjs OR direct BabylonEngine in useEffect|Prefer direct engine control                |
|State           |Zustand                                             |Single store, no prop drilling              |
|Styling         |CSS Modules                                         |Only for overlay UI panels, not the 3D world|
|Data persistence|localStorage (mindcastle_v1)                        |Existing schema, do not change              |
|Backend comms   |fetch to localhost:8000                             |FastAPI on Mac Mini / MacBook               |
|Fonts           |Cinzel (headings) + Crimson Pro (body)              |Google Fonts or local                       |

-----

## PROJECT STRUCTURE

```
mindcastle/
├── src-tauri/                  # Tauri Rust shell
│   ├── Cargo.toml
│   ├── tauri.conf.json         # macOS only, no updater for POC
│   └── src/
│       └── main.rs             # Minimal — just window + IPC stubs
│
├── src/                        # React + Babylon frontend
│   ├── main.tsx
│   ├── App.tsx                 # Route: /castle (world) | /onboarding
│   │
│   ├── world/                  # Babylon.js castle world
│   │   ├── CastleWorld.tsx     # Canvas mount, engine init
│   │   ├── SceneBuilder.ts     # All mesh/light/particle creation
│   │   ├── EngineStructure.ts  # Per-engine 3D structure logic
│   │   ├── AtmosphereSystem.ts # Fog, stars, mist, weather
│   │   ├── StateMapper.ts      # state vector → visual params
│   │   └── CameraController.ts # Arc rotate, transitions, enter-room
│   │
│   ├── overlay/                # React UI panels (not 3D)
│   │   ├── StatePanel.tsx      # Top-left state vector bars
│   │   ├── ReadoutDisplay.tsx  # Bottom-center readout state + task
│   │   ├── EngineTooltip.tsx   # Hover tooltip on structures
│   │   ├── ArtifactPanel.tsx   # Slide-in panel on structure click
│   │   └── LoadingScreen.tsx   # Initial castle load screen
│   │
│   ├── store/
│   │   ├── castleStore.ts      # Zustand: engines, artifacts, state vector
│   │   └── types.ts            # All TypeScript interfaces
│   │
│   ├── api/
│   │   └── mindcastleApi.ts    # fetch wrappers for localhost:8000
│   │
│   └── styles/
│       ├── global.css          # Base reset, body, canvas
│       └── tokens.css          # CSS variables: colors, fonts, spacing
│
└── package.json
```

-----

## DATA TYPES — USE EXACTLY THESE

```typescript
// types.ts

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
  | 'tower'    // Tech Builder, Learning
  | 'garden'   // Health
  | 'vault'    // Finance
  | 'workshop' // Creative
  | 'study'    // Writing
  | 'hall'     // Social
  | 'chapel';  // Music

export interface StateVector {
  energy: number;     // 0.0–1.0
  momentum: number;   // 0.0–1.0
  risk: number;       // 0.0–1.0
  readiness: number;  // 0.0–1.0
  readoutState: ReadoutState;
  readoutTask: string;
  lastUpdated: string; // ISO timestamp
}

export interface ProgressBar {
  label: string;
  val: number; // 0–100
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
  color: string;           // hex
  icon: string;            // unicode glyph
  context?: string;        // private AI instruction
  structure: StructureType;
  activityLevel: number;   // 0.0–1.0, computed from recent logs
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
```

-----

## CASTLE WORLD — VISUAL RULES

### Structure geometry per type

Every engine structure follows these geometry rules. Heights and sizes scale with `engine.activityLevel` (0.0–1.0).

|Structure|Base          |Main body                 |Roof/top          |Activity effect              |
|---------|--------------|--------------------------|------------------|-----------------------------|
|tower    |2×2 box, h=1  |1.2×1.2 box, h=2+(a×4)    |4-sided cone      |Grows taller                 |
|garden   |2×2 box, h=0.8|Sphere d=1.5+(a×1)        |None              |Shrinks, darkens with neglect|
|vault    |2×2 box, h=1  |3×2 box, h=1.5+(a×1)      |Flat with parapet |Widens                       |
|workshop |2×2 box, h=0.8|1.8×1.8 box, h=1.5+(a×1.5)|Pitched (2 planes)|Grows                        |
|study    |2×2 box, h=0.8|1.6×1.6 box, h=1.2+(a×1.2)|Gabled            |Grows                        |
|hall     |2×2 box, h=0.8|3.5×2 box, h=1+(a×0.8)    |Crenellated flat  |Widens                       |
|chapel   |2×2 box, h=0.8|1.6×1.6 box, h=1.5+(a×2)  |Spire             |Grows tall                   |

### Color system

Each engine has a hex color. Apply it to:

- Roof/spire material (at 40% brightness)
- Window emissive glow (at activityLevel brightness)
- Point light diffuse color
- Path emissive (at activityLevel × 0.3 brightness)

Stone base and walls are always near-black with slight variation. Do NOT apply engine color to stone.

### State vector → visual mapping

```typescript
// StateMapper.ts

interface VisualParams {
  fogDensity: number;
  ambientIntensity: number;
  keepLightColor: BABYLON.Color3;
  keepLightIntensity: number;
  mistEmitRate: number;
  mistAlpha: number;
  starIntensity: number;
}

function mapStateToVisuals(sv: StateVector): VisualParams {
  const stateColors: Record<ReadoutState, BABYLON.Color3> = {
    flow_available:   new BABYLON.Color3(0.9, 0.85, 0.6),
    low_energy:       new BABYLON.Color3(0.3, 0.3, 0.5),
    pre_crash:        new BABYLON.Color3(0.8, 0.3, 0.2),
    blocked_pattern:  new BABYLON.Color3(0.6, 0.4, 0.2),
    recovery:         new BABYLON.Color3(0.4, 0.5, 0.7),
    transition_ready: new BABYLON.Color3(0.7, 0.8, 0.6),
    emergence:        new BABYLON.Color3(0.6, 0.7, 0.9),
  };

  return {
    fogDensity:         0.01 + (1 - sv.energy) * 0.03,
    ambientIntensity:   0.08 + sv.energy * 0.12,
    keepLightColor:     stateColors[sv.readoutState],
    keepLightIntensity: 0.6 + sv.momentum * 0.6,
    mistEmitRate:       Math.floor(10 + sv.risk * 80),
    mistAlpha:          sv.risk * 0.18,
    starIntensity:      0.3 + sv.energy * 0.4,
  };
}
```

### Particle systems — required

|Condition                                   |System     |Parameters                                            |
|--------------------------------------------|-----------|------------------------------------------------------|
|`status === 'blocked'`                      |Dark smoke |Color: rgba(0.15,0.12,0.2), emitRate: 15, upward drift|
|`status === 'active' && activityLevel > 0.5`|Ember fire |Engine color, emitRate: 30, small size 0.05–0.2       |
|Always                                      |Ground mist|Blue-grey, emitRate driven by risk                    |
|Always                                      |Stars      |White-blue, static, 500 particles at y=20             |

Use `https://assets.babylonjs.com/textures/flare.png` for all particle textures.

### Animation loop requirements

Register ONE `scene.registerBeforeRender` callback. Inside it:

- Keep light pulses: `0.7 + sin(tick × 1.2) × 0.15 × energy`
- Engine lights breathe: `activityLevel × 1.2 + sin(tick × speed + offset) × 0.1`
- Do NOT animate geometry (no bobbing structures) — atmosphere only

### Camera

- ArcRotateCamera
- Initial: alpha=-π/2, beta=π/3.5, radius=22, target=Vector3.Zero()
- Limits: radius 10–40, beta 0.3–π/2.2
- On engine click: animate target toward selected engine position × 0.5
- On panel close: animate target back to Vector3.Zero()
- Transition duration: 40 frames at 30fps

-----

## OVERLAY UI — VISUAL RULES

All overlay elements use `position: fixed`, `pointer-events: none` except the artifact panel.

### Fonts

```css
--font-display: 'Cinzel', serif;      /* engine names, labels */
--font-body: 'Crimson Pro', serif;    /* all other text */
```

### Color tokens

```css
--color-bg:         #0a0a0f;
--color-stone-dark: #08070a;
--color-text-dim:   #3a3028;
--color-text-mid:   #5a4e3a;
--color-text-warm:  #7a6a52;
--color-text-light: #c8b89a;
--color-border:     rgba(120, 100, 70, 0.15);
--color-border-hover: rgba(120, 100, 70, 0.3);
```

### State panel (top-left)

- Castle name in Cinzel, 11px, letter-spacing 0.3em
- 4 state bars: Energy / Momentum / Risk / Readiness
- Bar track: 80px wide, 2px tall, background rgba(255,255,255,0.06)
- Bar fill: transitions over 2s ease
- Bar colors: energy=green, momentum=blue, risk=red, readiness=amber
- Value: 4 chars, tabular-nums, right-aligned

### Readout display (bottom-center)

- State name: Cinzel 10px, letter-spacing 0.4em, uppercase
- Task: Crimson Pro italic 16px, max-width 400px
- Both transition color over 2s when state changes

### Tooltip (follows cursor)

- Appears on structure hover
- Engine icon + name in Cinzel 11px
- Subtitle in italic 12px
- Next task prefixed with “→” in 11px
- Background: rgba(10,9,8,0.85), border: 1px solid var(–color-border)
- Fade in/out: opacity transition 0.3s

### Artifact panel (right side, slides in)

- Width: 340px, full height
- Transform: translateX(100%) → translateX(0) on open
- Transition: 0.5s cubic-bezier(0.16, 1, 0.3, 1)
- pointer-events: all (only panel that accepts input)
- Artifact cards: border 1px solid var(–color-border), hover state
- Status badges: color-coded (active=green, blocked=red, concept=dim, planned=blue)
- Close button: top-right, ✕, hover brightens

-----

## LOCAL STORAGE SCHEMA

Read and write engines from `localStorage['mindcastle_v1']`. Parse as `Engine[]`. If empty or missing, load DEFAULT_ENGINES (see below).

```typescript
// Default engines if localStorage empty
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
        next: 'Write integration tests for /classify/ endpoint'
      }
    ]
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
        next: 'Start with 10 minutes, not 30'
      }
    ]
  },
  {
    id: 'finance',
    name: 'Finance',
    subtitle: 'Budgets, savings & investing',
    color: '#e6b233',
    icon: '⬡',
    structure: 'vault',
    activityLevel: 0.35,
    artifacts: []
  },
  {
    id: 'creative',
    name: 'Creative',
    subtitle: 'Art, design & making things',
    color: '#cc4db3',
    icon: '◇',
    structure: 'workshop',
    activityLevel: 0.6,
    artifacts: []
  },
  {
    id: 'learning',
    name: 'Learning',
    subtitle: 'Books, courses & deep dives',
    color: '#6680e6',
    icon: '⟁',
    structure: 'library',
    activityLevel: 0.7,
    artifacts: []
  },
  {
    id: 'writing',
    name: 'Writing',
    subtitle: 'Essays, docs & storytelling',
    color: '#e6993d',
    icon: '✦',
    structure: 'study',
    activityLevel: 0.25,
    artifacts: []
  },
  {
    id: 'social',
    name: 'Social',
    subtitle: 'Relationships & community',
    color: '#e66666',
    icon: '△',
    structure: 'hall',
    activityLevel: 0.2,
    artifacts: []
  },
  {
    id: 'music',
    name: 'Music',
    subtitle: 'Practice, theory & recording',
    color: '#80ccee',
    icon: '⊕',
    structure: 'chapel',
    activityLevel: 0.45,
    artifacts: []
  }
];
```

-----

## BACKEND API — localhost:8000

```typescript
// mindcastleApi.ts

const BASE = 'http://localhost:8000';

// Classify a log entry text
export async function classify(text: string) {
  const res = await fetch(`${BASE}/classify/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return res.json();
  // Returns: { state, categories, next_step }
}

// Get state vector from signal collector
export async function getStateVector(): Promise<StateVector> {
  const res = await fetch(`${BASE}/state/`);
  return res.json();
}

// Write a progress log
export async function writeLog(artifactId: string, text: string) {
  const res = await fetch(`${BASE}/nodes/${artifactId}/logs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return res.json();
}

// Poll state vector every 30 seconds
export function startStateVectorPolling(
  onUpdate: (sv: StateVector) => void
): () => void {
  const interval = setInterval(async () => {
    try {
      const sv = await getStateVector();
      onUpdate(sv);
    } catch {
      // Backend offline — use cached state, no error shown to user
    }
  }, 30_000);
  return () => clearInterval(interval);
}
```

If backend is offline, the app works fully with localStorage data and a default state vector. Never show error messages to user for backend failures — degrade silently.

-----

## TAURI CONFIGURATION

```json
// tauri.conf.json (key fields)
{
  "build": {
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build",
    "devPath": "http://localhost:5173",
    "distDir": "../dist"
  },
  "tauri": {
    "windows": [{
      "title": "MindCastle",
      "width": 1440,
      "height": 900,
      "minWidth": 1024,
      "minHeight": 700,
      "decorations": false,
      "transparent": true,
      "resizable": true,
      "fullscreenable": true
    }],
    "macOSPrivateApi": true,
    "bundle": {
      "identifier": "io.mindcastle.app",
      "icon": ["icons/icon.icns"]
    },
    "allowlist": {
      "fs": { "all": true, "scope": ["$HOME/mindcastle/**"] },
      "http": { "all": true, "scope": ["http://localhost:8000/**"] }
    }
  }
}
```

`decorations: false` + `transparent: true` enables the frameless window. Implement a custom drag region using `data-tauri-drag-region` on a top bar element.

-----

## WHAT NOT TO BUILD (YET)

Do not build these in this phase. Stub them if needed but do not implement:

- Onboarding wizard (3 paths)
- Mind map / graph view
- Focus mode (full-screen single artifact)
- Voice capture
- Theme selector (Minecraft/Dark Souls/Journey/Myst — future onboarding step)
- TinyNet training UI
- Apple Watch companion
- iPhone companion
- HealthKit bridge

-----

## WHAT MUST WORK ON COMPLETION

1. `npm run tauri dev` launches the app on macOS
1. Castle renders with all 8 engine structures
1. Each structure reflects its engine’s activityLevel (height, light, particles)
1. Blocked engines emit dark smoke
1. Active high-momentum engines emit ember particles
1. Hover any structure → tooltip appears
1. Click any structure → artifact panel slides in from right
1. Close panel → camera returns to center
1. State vector bars update in overlay
1. App works fully with no backend running (offline mode)
1. `window.__mindcastle.updateStateVector(sv)` updates the scene live (for Mac Mini integration)

-----

## IMPLEMENTATION NOTES

- Babylon.js engine must be initialised inside a `useEffect` with the canvas ref. Clean up `engine.dispose()` on unmount.
- All scene objects that need interaction must have `mesh.metadata = { engineId: string }`.
- Use a single `GlowLayer` with intensity 0.6.
- Register ONE `beforeRender` callback total. Don’t register multiple.
- The `tick` variable increments by 0.008 per frame.
- Do not use `@babylonjs/core` (ES module version) — use the UMD build via npm package `babylonjs` imported as `import * as BABYLON from 'babylonjs'`.
- All particle textures must use the flare URL: `https://assets.babylonjs.com/textures/flare.png`
- Do not use any UI component library. Raw HTML/CSS only for overlays.
- TypeScript strict mode on.

-----

## REFERENCE IMPLEMENTATION

A working HTML prototype exists at: `mindcastle-world.html`

It contains the full visual logic in a single file. Use it as the reference for:

- Scene setup and camera configuration
- Structure geometry per engine type
- Particle system parameters
- State vector → visual mapping
- Overlay styling and layout
- Interaction handling

The Tauri + React implementation should reproduce this exactly, split into the file structure above.

-----

*Brief version 1.0 — POC Phase 1*
*MindCastle — a place your mind can come home to*