# Demo Summary and Open Questions

## Phase Delivery Summary

| Phase Item | Deliverable | Week |
|---|---|---|
| 1.1 | One Next Task view | 1 |
| 1.2 | Focus Mode | 1 |
| 1.3 | Transition Mode | 1 |
| 2.1 | GraphEdge schema + live weight learning | 2 |
| 2.2 | Mind Map weighted view | 2 |
| 3.1 | Ollama + Nemotron setup | 3 |
| 3.2 | `/reason/` endpoint | 3 |
| 3.3 | Reasoning surfaces in UI | 3 |
| 4.1 | Per-engine policy files | 4 |
| 4.2 | Push flow UI (simulated) | 4 |

## Open Questions (Resolve Before Build)

1. One Next Task priority logic sign-off:
   blocked first -> recent engine -> lowest progress -> TinyNet suggestion.
2. Nemotron model size selection:
   which variant fits available RAM comfortably?
3. GraphEdge update cadence:
   every log entry vs batched updates (for example session-end).
4. OpenShell policy authoring model:
   user-authored manually vs app-generated draft at engine creation.
5. OpenClaw execution mode:
   confirm demo remains simulated and real execution deferred to v1.0.

## Capture Date and Timeline

- Captured: March 2026
- Personal demo target: 4 weeks from baseline
