## Summary
<!-- 1–3 bullets: what changed and why -->
-

## Test Gate Checklist

All CI jobs must be green before merge. Confirm each suite locally first:

### Frontend
- [ ] `npm run lint` — 0 warnings
- [ ] `npm run typecheck` — 0 errors
- [ ] `npm run test:run` — all tests pass

### Backend
- [ ] `pytest -m unit` — all pass
- [ ] `pytest -m integration` — all pass
- [ ] `pytest -m load` — p95 SLO met (< 200 ms)

## Governance Artifacts *(model changes only)*

- [ ] `run_manifest.json` generated and reviewed
- [ ] `fairness_report.json` shows `"overall_pass": true`
- [ ] `model_card.json` updated if architecture changed

## Notes
<!-- Edge cases, trade-offs, follow-ups, or reviewer focus areas -->
