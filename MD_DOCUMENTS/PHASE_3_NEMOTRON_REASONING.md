# Phase 3 - Nemotron Reasoning

**Target:** Week 3

## 3.1 Ollama Setup

### Runtime

- Model served locally via Ollama on `localhost:11434`
- Model: Nemotron variant chosen for structured reasoning over complex context

### Why Nemotron

The model should reason over full castle state and return coherent diagnosis and actionable next-step guidance, not generic advice.

---

## 3.2 `/reason/` Endpoint (FastAPI)

### What it does

Accepts full castle snapshot and returns a reasoned diagnosis.

### Input snapshot includes

- Engines
- Artifact statuses
- Recent logs
- Graph weights
- User free-text reflection

### Prompt intent

"Reason over the graph. Explain the pattern causing current state. Identify root engine of drag. Recommend the single best next action."

### Example structured payload (shape)

```json
{
  "engines": [
    {
      "id": "uuid",
      "name": "Tech Builder",
      "status_summary": "blocked",
      "days_since_last_log": 3,
      "artifacts": [
        { "title": "TinyNet API", "status": "blocked", "next": "Write integration tests" }
      ],
      "graph_edges": [
        { "target": "Finance", "weight": 0.82 },
        { "target": "Health", "weight": 0.34 }
      ]
    }
  ],
  "user_log": "Can't seem to start anything today. Everything feels stuck."
}
```

### Output expectations

- Plain-language diagnosis of systemic pattern
- Single recommended next action
- Root engine identified as likely source of drag

---

## 3.3 Reasoning Surfaces in UI

Nemotron is invoked deliberately, not always-on.

### Trigger points

1. `Why am I stuck?` button in One Next Task view when surfaced task is in blocked engine.
2. Review Panel `Diagnose` option when cross-engine stalling is detected.

### Result presentation

Show result as calm conversational card (not alert spam, not noisy notification).
