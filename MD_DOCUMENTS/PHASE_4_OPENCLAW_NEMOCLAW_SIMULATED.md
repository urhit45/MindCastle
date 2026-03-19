# Phase 4 - OpenClaw + NemoClaw (Simulated)

**Target:** Week 4

## 4.1 Per-Engine Policy Files

### What they are

Local JSON policy files that define exactly what an agent can touch per engine.

One file per engine.

### Policy structure (conceptual)

```json
{
  "engine_id": "tech-builder-uuid",
  "engine_name": "Tech Builder",
  "allowed_paths": ["~/code/projects/", "~/notes/tech/"],
  "allowed_actions": ["read_file", "write_file", "run_script"],
  "blocked_domains": ["social", "finance"],
  "max_duration_seconds": 300
}
```

### Principle

The policy file is the sandbox boundary.

An agent attached to one engine must not access other domains outside explicit permissions.

---

## 4.2 Push Flow UI

### What it is

A 4-step delegation flow initiated from artifact cards.

### Flow

1. User marks artifact `next` step as `delegate`.
2. MindCastle composes task packet:
   - artifact title
   - next step
   - engine context
   - policy id
3. Packet is dispatched to OpenClaw agent (simulated for demo).
4. Agent result is written back as progress log on artifact node and appears in UI.

### Simulation behavior for demo

No real agent execution in personal demo.

System writes clearly-labeled simulated result, for example:

"Searched ~/code/projects/ for integration test scaffold. Found 2 relevant files. Draft written to notes."

### Design principle

The castle remains a sanctuary. Delegation is explicit user intent ("open the gate"), and results return directly to the user's existing workflow surface.
