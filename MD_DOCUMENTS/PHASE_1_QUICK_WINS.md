# Phase 1 - Quick Wins, Visible Progress

**Target:** Week 1

## 1.1 One Next Task View

### What it is

A dedicated full-screen view that answers one question:

> What should I do right now?

It surfaces the single most important next step across the entire castle (global, not per engine). The intent is to reduce overwhelm and reduce startup friction for ADHD cognition.

### Priority logic (in order)

1. Artifacts with `status=blocked` that have a `next` step defined.
2. Most recently active engine (resume momentum).
3. Lowest progress percent artifact (prevent silent stalling).
4. TinyNet `next_step` suggestion from the most recent log.

### UI should show

- One single next task (large and calm presentation)
- Engine context (color + icon)
- Artifact context
- One-tap `Start Focus Mode`
- Escape hatch: `Not this - show me another` (no guilt, no explanation required)

### Placement

Home screen when no engine is selected (castle front door).

---

## 1.2 Focus Mode

### What it is

A full-screen deep-work sanctuary around a single artifact.

One task visible. Everything else hidden.

### UI should show

- Artifact title and subtitle
- Single next step (large, centered)
- Time-Flow visual (ambient, non-alarming time sense)
- Artifact progress bars
- Log entry field (capture thoughts without exiting)
- Soft exits:
  - `I'm done`
  - `I need a break`

### Entry points

- From One Next Task: `Start Focus Mode`
- From any artifact card directly

### Design principle

Entering Focus Mode is intention, not commitment. Exiting is never framed as failure.

---

## 1.3 Transition Mode

### What it is

A 60-second ritual between engines to reduce context-switch whiplash.

### Ritual flow

1. "You're leaving [Engine A]" + 1-line reflection prompt.
2. 30-second breathing/stillness display (ambient, no countdown pressure).
3. "You're entering [Engine B]" + last next-step from that engine.
4. Return to castle in the new engine, oriented.

### Trigger conditions

- User navigates from one engine to another (optional and skippable)
- Focus Mode ends and no next engine is chosen

### Design principle

The transition is the feature. Context switching is a cognitive cost that should be ritualized, not ignored.
