# Phase 2 - Graph Foundation

**Target:** Week 2

## 2.1 GraphEdge Schema (SQLite)

### What it is

Add a new `GraphEdge` table via Alembic migration to store learned relationships between engines.

### Schema shape (conceptual)

```json
{
  "source_engine_id": "uuid",
  "target_engine_id": "uuid",
  "weight": 0.82,
  "sample_count": 47,
  "last_updated": "ISO timestamp",
  "signal_breakdown": {
    "temporal_co_activation": 0.41,
    "blocked_propagation": 0.28,
    "progress_correlation": 0.13
  }
}
```

### Weight learning signals (incremental on every log)

- Temporal co-activation: Engine A then Engine B in short windows repeatedly.
- Blocked propagation: A goes blocked, B stalls later.
- Hyperfocus spillover: deep focus in one engine correlates with neglect in another.
- State transition sequences: recurring transition patterns across engines.
- Progress correlation: progress in one domain correlates with activity in another.

### Update model

No retraining required. Weights update live with each new log entry.

---

## 2.2 Mind Map View

### What it is

Interactive castle graph using `react-force-graph-2d`.

Engines are nodes. Learned relationships are weighted edges.

### UI behavior

- Node = engine (color + icon from engine config)
- Edge thickness = relationship strength (`weight`)
- Edge color = dominant signal type (temporal, blocked, progress)
- Clicking a node opens that engine
- Clicking an edge reveals "why connected" signal breakdown

### Product value

Reveals cognitive topology:

- Which engines pull on each other
- Where blockages propagate from
- Where momentum lives

### Design principle

Not a productivity dashboard. A mirror that helps the user understand underlying pattern mechanics.
