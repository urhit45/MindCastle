# ADHD Mind Castle

> **A personal operating system for focus, life, and mental clarity.**
> *"A second brain that actually understands you."*

---

## The Problem

### Society Assumes. Reality Differs.

| What the world assumes | ADHD reality |
|---|---|
| Stable attention spans | Overwhelm leads to paralysis |
| Linear, orderly thinking | Hyperfocus leads to burnout |
| Consistent day-to-day motivation | Rigid to-do lists cause anxiety |
| Endless working memory | Time feels abstract and slippery |

### Why Existing Tools Fail

- **Punishing streaks** — Missing one day feels like failing completely.
- **Information hoarding** — Cloud apps that capture data but never resurface it.
- **Guilt-based UX** — Notifications that shame you into productivity.
- **What's missing** — Forgiveness, visual time, and reduced cognitive friction.

> ADHD is not a lack of attention. It is a lack of *attention regulation.*

---

## The Vision

ADHD Mind Castle is a **local-first personal OS** that helps you:

- Know what to do next without overwhelm.
- Start tasks with zero friction.
- Transition without mental whiplash.

**Not an app. A cognitive prosthetic.**

---

## Core Design Principles

1. **Local-first** — Your sensitive thoughts live on your device, not a server.
2. **Forgiveness over streaks** — There is no streak to break. Every session is a fresh start.
3. **One next task** — No giant lists. See only the single smallest step to prevent paralysis.
4. **Zero friction capture** — Voice, text, or photo. Offload your brain in seconds.
5. **No guilt UX** — No shame notifications. No manipulative algorithms. No data mining.
6. **Visual time** — A tangible sense of passing time to combat time-blindness.
7. **Human control** — Export your data anytime. Your mind belongs to you.

---

## Who It's For

- **Neurodivergent thinkers** — People whose brains don't fit into standard boxes.
- **Creatives & builders** — Who thrive on hyperfocus but fear the burnout.
- **Burned-out professionals** — Anyone tired of being told to "just be more disciplined."

> This isn't about productivity. It's about dignity.

---

## The Castle Metaphor

MindCastle organises your life into a physical mental space you can return to.

### Engines

An **Engine** is a domain of your life — a room in your castle.

```
◈  Tech Builder    — Code, systems & side projects
✦  Writer          — Essays, docs & storytelling
◎  Health          — Fitness, sleep & nutrition
⬡  Finance         — Budgets, savings & investing
⟁  Learning        — Books, courses & deep dives
◇  Creative        — Art, design & making things
⊕  Music           — Practice, theory & recording
△  Social          — Relationships & community
```

Each engine has:
- A **name** and **subtitle** (what this domain is)
- A **color** and **icon** (visual identity)
- A **context** block (private AI instructions — what you want the AI to know about this domain)
- A list of **Artifacts**

### Artifacts

An **Artifact** is a project, goal, or task living inside an engine.

| Field | Description |
|---|---|
| `title` | The name of the project or task |
| `subtitle` | One-line description |
| `status` | `concept` · `active` · `planned` · `live` · `blocked` · `planning` |
| `next` | The single next step (prevents paralysis) |
| `stack` | Tags or technologies associated |
| `progress` | Named progress bars (e.g. "Feature A: 45%") |
| `notes` | Free-form working notes |

### Nodes (Backend)

When an artifact is first saved, it creates a **Node** in the backend database — a persistent, searchable record that accumulates **Progress Logs** over time. Each log entry is classified by the TinyNet AI to detect your current state, suggest your next step, and surface patterns in your work.

---

## Signature Features

### Current (v0.1)

| Feature | Description |
|---|---|
| Engine management | Create, edit, delete life-domain containers with custom icons and colors |
| Artifact tracking | Projects and tasks with status, next step, progress bars, and notes |
| Progress logging | Free-text log entries classified by AI (state + category + next step) |
| TinyNet classifier | On-device multi-task model: detects 20 categories, 6 states, 12 next-step templates |
| Context per engine | Private AI instruction block injected per domain |
| Local-only storage | All data in `localStorage` — nothing leaves your machine |
| Backend sync | Optional FastAPI backend for persistent node history and log retrieval |

### Planned

| Feature | Description |
|---|---|
| **Time-Flow** | A visual sense of passing time — a flowing display to combat time-blindness |
| **Transition Mode** | 60-second ritual to switch gears between engines without burnout |
| **Focus Mode** | Deep-work sanctuary — one artifact, full screen, visual time flow |
| **Mind Map** | Interactive graph of your nodes and their connections |
| **Review Panel** | Surfaces blocked, stale, and next-step items automatically |
| **Voice capture** | Zero-friction entry via microphone |

---

## Onboarding Paths

When you open MindCastle for the first time (or reset it), you are shown three paths to populate your castle:

### Path 1 — Bring Your Stuff Over (AI Import)

Designed for users coming from ChatGPT, Claude, Gemini, or any AI tool.

1. MindCastle shows you a **copyable prompt** to paste into your AI tool.
2. The AI reviews your conversation history and exports your projects, goals, and tasks as a structured JSON block.
3. You paste the response back into MindCastle.
4. MindCastle parses it, shows you a **preview of the engines and artifacts** it found, and imports them in one click.

The generated prompt looks like:

```
I'm migrating to MindCastle — a local personal productivity tool that organises
my work into "Engines" (life domains) and "Artifacts" (projects/tasks inside them).

Please review our conversation history and extract every project, goal, task, or
ongoing topic we've discussed. Group them into logical Engines (e.g. Work, Learning,
Health, Finance, Creative). Then output ONLY a JSON block in this exact format...
```

### Path 2 — Build Your Palace (Guided Wizard)

An animated, step-by-step castle builder.

1. **Select your engines** — Pick from 8 preset life-domain templates (multi-select grid).
2. **Customise names** — Rename or retitle each engine to match your life.
3. **Castle rises** — An animated celebration as your palace is constructed.

You arrive in a fully set-up castle, ready to add your first artifacts.

### Path 3 — Text Import

For the "I have notes everywhere" user.

1. Paste any raw text — notes, tasks, journal entries, brain dumps.
2. MindCastle sends each line to the TinyNet classifier.
3. Lines are grouped by detected category and become engines, with each text chunk as an artifact.
4. Preview the structure, adjust if needed, and import.

*Falls back to a single "General" engine if the backend is offline.*

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Browser (local)                    │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              tinynet-ui  (React 18 + TS)         │  │
│  │                                                  │  │
│  │  App.tsx ─── EngineView ─── ArtifactCard        │  │
│  │      └───── Onboarding ─── PathSelection        │  │
│  │                         ├── ChatGPTWizard        │  │
│  │                         ├── PalaceWizard         │  │
│  │                         └── TextWizard           │  │
│  │                                                  │  │
│  │  localStorage["mindcastle_v1"]  ← Engine[]       │  │
│  └────────────────┬─────────────────────────────────┘  │
│                   │ HTTP (localhost:8000)                │
└───────────────────┼─────────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────────┐
│              tinynet-api  (FastAPI + Python)             │
│                                                         │
│  POST /classify/      ← text → categories, state, next  │
│  POST /nodes/         ← create artifact node            │
│  PATCH /nodes/{id}    ← update node title/status        │
│  POST /nodes/{id}/logs ← append progress log            │
│  GET  /nodes/{id}/logs ← retrieve log history           │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │           TinyNet  (PyTorch, CPU-only)           │  │
│  │                                                  │  │
│  │  HashingVectorizer512  (word + char n-grams)     │  │
│  │  Trunk: Linear(512→64→32) + ReLU + Dropout      │  │
│  │  Head 1: 20-label categories  (BCE)              │  │
│  │  Head 2: 6-class state        (CrossEntropy)     │  │
│  │  Head 3: 12-class next step   (CrossEntropy)     │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
│  SQLite  (tinynet.db)  — Nodes, Links, ProgressLogs     │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript, Vite 5 |
| State | `useState` + `localStorage` (Zustand planned) |
| Graph view | `react-force-graph-2d` |
| Backend | FastAPI 0.104, SQLAlchemy 2 async, Pydantic v2 |
| Database | SQLite (async via aiosqlite) |
| ML model | PyTorch 2.0 (CPU), scikit-learn (vectorizer) |
| Migrations | Alembic |
| Config | PyYAML (`backend/config/labels.yaml`) |

---

## Data Model

### Engine (localStorage)

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Tech Builder",
  "subtitle": "Code, systems & side projects",
  "description": "Everything I'm building and shipping",
  "color": "#3ecfcf",
  "icon": "◈",
  "context": "I'm a senior engineer focusing on Rust + React. Prefer practical over theoretical.",
  "contextLabel": "Context",
  "artifacts": [
    {
      "id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
      "node_id": "a87ff679-a2f5-4fec-bad8-0e3e9e5a6d63",
      "title": "TinyNet API",
      "subtitle": "Local ML classification service",
      "status": "active",
      "next": "Write integration tests for /classify/ endpoint",
      "stack": ["Python", "FastAPI", "PyTorch"],
      "progress": [
        { "label": "Core API", "val": 85 },
        { "label": "ML model", "val": 70 }
      ],
      "notes": "Model trains in < 1s on CPU from JSONL. Need to wire in online learning."
    }
  ]
}
```

### Node (backend DB)

```json
{
  "id": "a87ff679-a2f5-4fec-bad8-0e3e9e5a6d63",
  "title": "TinyNet API",
  "is_hub": false,
  "status": "active",
  "user_id": 1,
  "created_at": "2026-03-11T09:00:00Z",
  "updated_at": "2026-03-11T12:30:00Z"
}
```

### Progress Log (backend DB)

```json
{
  "id": 42,
  "node_id": "a87ff679-a2f5-4fec-bad8-0e3e9e5a6d63",
  "text": "Wired classify router to model service. All imports green.",
  "state": "continue",
  "next_step": "WriteIntegrationTests",
  "created_at": "2026-03-11T12:30:00Z"
}
```

### TinyNet Categories (20)

`Fitness` `Running` `Strength` `Music` `Guitar` `Learning` `AI` `Admin` `Finance` `Social`
`Health` `Cooking` `Travel` `Work` `SideProject` `Design` `Reading` `Writing` `Mindfulness` `Household`

### TinyNet States (6)

| State | Meaning |
|---|---|
| `start` | Beginning a new activity |
| `continue` | Resuming ongoing work |
| `pause` | Temporarily stepping back |
| `end` | Completed or closing |
| `blocked` | Stuck, waiting, or uncertain |
| `idea` | Brainstorm or concept |

### Next Step Templates (12)

`PracticeForDuration` · `RepeatTask` · `IncreaseVolume` · `ScheduleFollowUp` · `AttachLink`
`ReviewNotes` · `OutlineThreeBullets` · `BookAppointment` · `BuySupplies` · `CreateSubtasks`
`SetReminder` · `LogReflection`

---

## Import JSON Schema

Used by the AI Import onboarding path. Any AI tool can produce this format.

```json
{
  "version": "1",
  "engines": [
    {
      "name": "Engine name",
      "subtitle": "One-line description (optional)",
      "description": "Longer description (optional)",
      "color": "#3ecfcf",
      "icon": "◈",
      "artifacts": [
        {
          "title": "Task or project title",
          "subtitle": "One-liner (optional)",
          "notes": "Any context or details (optional)",
          "status": "concept"
        }
      ]
    }
  ]
}
```

**Valid `status` values:** `concept` · `active` · `planned` · `live` · `blocked` · `planning`

`color` (hex) and `icon` (any Unicode glyph) are optional — defaults are assigned automatically.

---

## Privacy & Trust

| Principle | Implementation |
|---|---|
| Local-first | All engine and artifact data lives in browser `localStorage`. Nothing is sent to any external server. |
| Optional backend | The FastAPI backend runs on `localhost:8000`. It never communicates with the internet. |
| No telemetry | Zero analytics, zero crash reporting, zero usage tracking. |
| No accounts | No sign-up, no email, no cloud sync. User ID 1 is always `local@mindcastle`. |
| Data portability | Export your castle at any time via the import JSON schema. Your data is a plain JSON file. |
| Open source | All code is readable and auditable. No black boxes. |
| No lock-in | Export your data anytime. Delete the app and nothing is left behind except what's in your browser. |

> User data is treated as a personal archive, not raw material. Trust is the core feature.

---

## Getting Started

### Requirements

- Node 18+ and npm/pnpm
- Python 3.10+
- (Optional) `make`

### Run the frontend

```bash
cd tinynet-ui
npm install
npm run dev
# → http://localhost:5173
```

### Run the backend

```bash
cd tinynet-api
pip install -r requirements.txt
make db-upgrade    # run Alembic migrations
make api           # starts uvicorn on :8000
```

### Bootstrap the ML model

The TinyNet model trains automatically from `data/train.jsonl` on first API startup.
To regenerate training data from raw markdown notes:

```bash
cd tinynet-api
make bootstrap    # parses data/raw/*.md → data/train.jsonl
make api          # model retrains on next start (< 1s)
```

### Directory structure

```
MindCastle/
├── tinynet-ui/                 # React frontend
│   └── src/
│       ├── App.tsx             # Main application
│       ├── Onboarding.tsx      # First-run onboarding wizard
│       └── api.ts              # Backend API client
├── tinynet-api/                # FastAPI backend
│   ├── app/
│   │   ├── routers/
│   │   │   ├── classify.py     # POST /classify/
│   │   │   ├── nodes.py        # Node CRUD + logs
│   │   │   └── home.py         # Review queue
│   │   ├── ml/
│   │   │   ├── tinynet.py      # Multi-task neural net
│   │   │   ├── vectorizer.py   # HashingVectorizer512
│   │   │   ├── bootstrap.py    # Auto-label from markdown
│   │   │   └── model_service.py # Singleton, trains on startup
│   │   └── models.py           # SQLAlchemy models
│   └── data/
│       ├── train.jsonl         # Training data (JSONL)
│       └── raw/                # Source markdown notes
└── backend/
    └── config/
        ├── labels.yaml         # Categories, states, templates
        └── settings.py         # Config loader
```

---

## Roadmap

### v0.1 — Local Release (current)

- [x] Engine + Artifact CRUD
- [x] TinyNet on-device classifier
- [x] Progress log with AI state detection
- [x] Per-engine context (AI instructions)
- [x] 3-path onboarding wizard (AI import, palace builder, text import)
- [x] Local-only, zero-server, privacy-first

### v0.2 — Focus & Flow

- [ ] **Focus Mode** — Full-screen single-artifact deep work session
- [ ] **Time-Flow display** — Visual time passing to fight time-blindness
- [ ] **One Next Task view** — Global "what do I do right now?" screen
- [ ] Zustand state management migration (refactor localStorage sync)

### v0.3 — Transition & Rhythm

- [ ] **Transition Mode** — 60-second gear-change ritual between engines
- [ ] **Review Panel** — Surface blocked, stale, and overdue artifacts
- [ ] Keyboard-first navigation
- [ ] Customisable status labels per engine

### v0.4 — Memory & Graph

- [ ] **Mind Map view** — react-force-graph-2d node graph
- [ ] Node linking (artifact → artifact relationships)
- [ ] Pattern surfacing — "You tend to block on Finance tasks on Mondays"
- [ ] TinyNet online learning (corrections improve the model)

### v1.0 — Release

- [ ] Packaged desktop app (Tauri or Electron)
- [ ] Voice capture
- [ ] Mobile companion (PWA)
- [ ] End-to-end encrypted optional sync

---

## Design Manifesto

> I've lived the overwhelm. I know the cost of burnout. I believe tools should be humane.

The ADHD brain is not broken. It is differently shaped. It needs:

- **Forgiveness** over accountability theatre.
- **Visual feedback** over abstract calendars.
- **One thing** over infinite lists.
- **Momentum** over perfection.
- **A place to land** after the chaos.

MindCastle is not a productivity app. It is a place your mind can come home to.

---

## Contact

Reach out: `contact@adhdmindcastle.io`

---

*Built with care for the non-linear mind.*
