# Mind Castle Demo Build Plan (MD Split)

This folder contains the full `MindCastle Demo Build Plan` converted into structured Markdown files.

## Source

- `MindCastle Demo Build Plan.pdf`

## Document Map

- `PHASE_1_QUICK_WINS.md`
- `PHASE_2_GRAPH_FOUNDATION.md`
- `PHASE_3_NEMOTRON_REASONING.md`
- `PHASE_4_OPENCLAW_NEMOCLAW_SIMULATED.md`
- `SUMMARY_AND_OPEN_QUESTIONS.md`

## Project Goal

Personal proof-of-concept covering the full product vision.

Baseline:

- Frontend: React 18 + TypeScript
- Backend: TinyNet API (FastAPI + SQLite)
- Local-only constraint: Ollama for Nemotron, nothing leaves the machine.

## What Already Works (v0.1)

- Engine + Artifact CRUD
- TinyNet classification (state, category, next step)
- Progress log with AI state detection
- Per-engine context block
- 3-path onboarding wizard
- Local-only storage (localStorage + SQLite)
