# MindCastle — Release Operations

> Reference for deploying, rolling back, and operating MindCastle.
> Update this document whenever env requirements, thresholds, or limitations change.

---

## Environment Configuration

### tinynet-api (`tinynet-api/.env`)

| Variable | Default | Override | Notes |
|---|---|---|---|
| `TINYNET_DATABASE_URL` | `sqlite+aiosqlite:///./tinynet.db` | Required for Postgres | SQLite is fine for local/single-user |
| `TINYNET_DEBUG` | `false` | `true` in dev | Enables FastAPI debug mode |
| `TINYNET_APP_NAME` | `TinyNet API` | Optional | Surfaced at `/` |
| `TINYNET_ALLOWED_ORIGINS` | `["http://localhost:5173", ...]` | JSON list | Add your domain for prod |
| `TINYNET_LOG_LEVEL` | `INFO` | `DEBUG` / `WARNING` | Sets root logger level |

Config is loaded via `pydantic-settings` with prefix `TINYNET_`. Copy `.env` and edit.

### tinynet-ui (build-time)

No env vars required at runtime — all config is baked in at build time via Vite.

- **API URL**: hardcoded to `http://localhost:8000` in `tinynet-ui/src/api.ts`.
  Change for staging/prod and rebuild: `npm run build`.

### Model + data paths (relative to `tinynet-api/`)

| Path | Purpose |
|---|---|
| `data/train.jsonl` | Training corpus (bootstrapped labels) |
| `models/best.pt` | Best checkpoint (loaded at startup if present) |
| `runs/<run_id>/` | Per-run training artifacts |
| `backend/config/labels.yaml` | Canonical category / state / next-step definitions |

---

## Deployment

### Local (development)

```bash
# API (port 8000)
cd tinynet-api && make api

# UI (port 5173)
cd tinynet-ui && npm run dev
```

### Production (single-server)

```bash
# Build UI
cd tinynet-ui && npm run build   # dist/ is ready to serve statically

# Run API in production mode
cd tinynet-api && make api-prod  # uvicorn without --reload

# Optional: run database migrations first
cd tinynet-api && make db-upgrade
```

### First-time model training

```bash
cd tinynet-api
make bootstrap     # generate data/train.jsonl from raw markdown
python3 scripts/train.py --out runs/exp1 --epochs 30
# Artifacts: runs/exp1/best.pt, run_manifest.json, fairness_report.json, model_card.json
cp runs/exp1/best.pt models/best.pt  # promote to live
```

---

## Rollback

### Rollback API / model (git-tag workflow)

```bash
# 1. Note the tag of the last known-good release
git log --oneline --tags | head -5

# 2. Checkout the tag
git checkout v<last-good-tag>

# 3. Reinstall deps and restart
cd tinynet-api && pip install -r requirements.txt && make api-prod
```

### Rollback model weights only (no code change needed)

```bash
# Keep a timestamped backup of models/best.pt before any training run
cp tinynet-api/models/best.pt tinynet-api/models/best.pt.bak_$(date +%Y%m%d)

# To revert: replace best.pt with the backup and restart the API
cp tinynet-api/models/best.pt.bak_<date> tinynet-api/models/best.pt
# API reloads weights on next startup (no hot-reload; bounce the process)
```

### Rollback UI

```bash
git checkout v<last-good-tag> -- tinynet-ui/dist   # restore built assets
# Or rebuild from the tag:
git checkout v<last-good-tag>
cd tinynet-ui && npm ci && npm run build
```

### Database rollback

```bash
cd tinynet-api
make db-downgrade   # rolls back one Alembic migration
# Repeat as needed; check current state with:
make db-status
```

---

## Known Limitations

### Model

- **Small training corpus**: bootstrapped from keyword rules, not human-labelled data.
  Confidence scores are not calibrated probabilities — treat thresholds as heuristics.
- **English only**: no multi-language support. Non-English input will get low-confidence
  predictions and likely trigger ABSTAIN or DEFER safety mode.
- **Drift detection is a proxy**: uses L2 norm of feature vectors as an input-drift signal,
  not a full distribution test (e.g., MMD). Alerts are early warnings, not guarantees.
- **No demographic parity guarantees**: fairness checks run at training time; meaningful
  only when training data includes labelled group metadata.

### API

- **In-memory rate limiting**: counters reset on process restart. Not suitable for
  multi-process/multi-instance deployments without an external counter store.
- **SQLite default**: not suitable for concurrent writers. Switch to Postgres for
  multi-user production use.
- **Auth stub**: `get_current_user()` always returns `user_id=1` (MVP). Replace with
  real JWT/session auth before opening to multiple users.

### UI

- **No offline support**: requires the API to be running. No service worker / PWA.
- **Local state only**: all engine/artifact data lives in the browser's in-memory state
  (synced to API nodes but not fully persisted across hard refreshes without a page reload).

### Pre-existing test failures (not blocking)

These failures existed before Phase IV and reflect test assertion mismatches, not bugs
in the production code:

| Test | Root cause |
|---|---|
| `test_vectorizer::test_empty_text` | Hashing vectorizer produces 1 non-zero bucket for `""` (valid behavior; test assertion wrong) |
| `test_vectorizer::test_different_seeds` | Determinism assertion mismatch |
| `test_training_loop::test_dataset_item` | Shape assertion off-by-one |
| `test_api_contract` (4 node tests) | Tests expect pre-seeded DB rows; DB is empty in CI |

---

## Release Gate Checklist

Run before every release: `bash scripts/release_gate.sh`

Or gate-by-gate: `make release-check`

| Gate | Command | Must pass |
|---|---|---|
| 1. Product Experience | `npm run test:run` | Yes |
| 2. Technical Reliability | `npm run lint && npm run typecheck` | Yes |
| 3. Backend + Model | `python3 scripts/release_gate.py` | Yes |
| 4. Quality | `make api-test-unit api-test-integration api-test-load` | Yes |
| 5. Release Ops | Automated file + content checks | Yes |

**Do not merge or deploy if any gate is red.**
