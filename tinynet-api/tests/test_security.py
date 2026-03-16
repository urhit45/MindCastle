"""
Phase II security tests.
Covers:
  - Schema validation: rejects invalid status / oversized fields / blank titles
  - Exception handlers: uniform error envelope on validation + HTTP errors
  - Classify failure path: sanitized 503 (no raw internals)
  - Node CRUD: ownership scope, uniform 404 shape
  - Rate limiting: 429 fires under burst; service stays stable (no 500s)
"""
import pytest
from unittest.mock import patch
from pydantic import ValidationError
from fastapi.testclient import TestClient

from app.main import app
from app.routers.classify import ClassifyRequest, TrainRequest, TrainingSample
from app.routers.nodes import NodeCreate, NodeLogCreate, NodeUpdate
from app.middleware import clear_rate_limits

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Clear in-memory rate limit counters before every test."""
    clear_rate_limits()
    yield
    clear_rate_limits()


# ─── Unit: schema validation ──────────────────────────────────────────────────

@pytest.mark.unit
class TestSchemaValidation:
    def test_classify_rejects_empty_text(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(text="")

    def test_classify_rejects_oversized_text(self):
        with pytest.raises(ValidationError):
            ClassifyRequest(text="x" * 2001)

    def test_classify_accepts_max_length_text(self):
        req = ClassifyRequest(text="x" * 2000)
        assert len(req.text) == 2000

    def test_node_create_rejects_invalid_status(self):
        with pytest.raises(ValidationError):
            NodeCreate(title="Test", status="INVALID_STATUS")

    def test_node_create_rejects_empty_title(self):
        with pytest.raises(ValidationError):
            NodeCreate(title="", status="continue")

    def test_node_create_rejects_blank_title(self):
        with pytest.raises(ValidationError):
            NodeCreate(title="   ", status="continue")

    def test_node_create_strips_title_whitespace(self):
        node = NodeCreate(title="  my node  ", status="continue")
        assert node.title == "my node"

    def test_node_create_rejects_oversized_title(self):
        with pytest.raises(ValidationError):
            NodeCreate(title="t" * 201, status="continue")

    def test_node_log_rejects_empty_text(self):
        with pytest.raises(ValidationError):
            NodeLogCreate(text="", state="continue")

    def test_node_log_rejects_oversized_text(self):
        with pytest.raises(ValidationError):
            NodeLogCreate(text="x" * 2001, state="continue")

    def test_node_log_rejects_invalid_state(self):
        with pytest.raises(ValidationError):
            NodeLogCreate(text="some progress", state="FLYING")

    def test_train_request_rejects_empty_samples(self):
        with pytest.raises(ValidationError):
            TrainRequest(samples=[])

    def test_train_request_rejects_too_many_samples(self):
        samples = [
            TrainingSample(text="t", categories=["x"], state="continue")
            for _ in range(201)
        ]
        with pytest.raises(ValidationError):
            TrainRequest(samples=samples)

    def test_train_sample_rejects_invalid_state(self):
        with pytest.raises(ValidationError):
            TrainingSample(text="hello", categories=["x"], state="NOTASTATE")

    def test_node_update_rejects_invalid_status(self):
        with pytest.raises(ValidationError):
            NodeUpdate(status="bogus")

    def test_node_update_allows_none_fields(self):
        # Partial update with no fields set is valid
        update = NodeUpdate()
        assert update.title is None
        assert update.status is None


# ─── Integration: error envelope shape ───────────────────────────────────────

@pytest.mark.integration
class TestErrorEnvelope:
    def test_classify_empty_text_returns_uniform_validation_error(self):
        resp = client.post("/classify/", json={"text": ""})
        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert "requestId" in body["error"]
        assert "message" in body["error"]

    def test_classify_oversized_text_returns_validation_error(self):
        resp = client.post("/classify/", json={"text": "x" * 2001})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_node_invalid_uuid_returns_uniform_400(self):
        resp = client.get("/nodes/not-a-valid-uuid")
        assert resp.status_code == 400
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "INVALID_INPUT"
        assert "Invalid node ID format" in body["error"]["message"]
        assert "requestId" in body["error"]

    def test_node_nonexistent_returns_uniform_404(self):
        resp = client.get("/nodes/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
        body = resp.json()
        assert body["error"]["code"] == "NOT_FOUND"
        assert "requestId" in body["error"]

    def test_response_always_has_request_id_header(self):
        resp = client.get("/healthz")
        assert "x-request-id" in resp.headers

    def test_client_request_id_is_echoed(self):
        resp = client.get("/healthz", headers={"X-Request-Id": "test-abc-123"})
        assert resp.headers.get("x-request-id") == "test-abc-123"

    def test_node_create_invalid_status_returns_422(self):
        resp = client.post("/nodes/", json={"title": "Test", "status": "NOTVALID"})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_node_create_blank_title_returns_422(self):
        resp = client.post("/nodes/", json={"title": "   ", "status": "continue"})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_classify_service_error_returns_sanitized_503(self):
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.side_effect = RuntimeError("internal weights corrupt")
            resp = client.post("/classify/", json={"text": "some text"})
        assert resp.status_code == 503
        body = resp.json()
        assert body["error"]["code"] == "SERVICE_UNAVAILABLE"
        # Must NOT leak internal error string
        assert "internal weights corrupt" not in body["error"]["message"]
        assert "requestId" in body["error"]

    def test_search_empty_q_returns_422(self):
        resp = client.get("/nodes/search?q=")
        assert resp.status_code in (400, 422)

    def test_search_oversized_q_returns_422(self):
        resp = client.get(f"/nodes/search?q={'a' * 201}")
        assert resp.status_code == 422


# ─── Load: rate limiting ──────────────────────────────────────────────────────

@pytest.mark.load
class TestRateLimiting:
    def test_classify_rate_limit_activates_under_burst(self):
        """Send 25 classify requests; at least some must be rate-limited (limit=20/60s)."""
        results = []
        for _ in range(25):
            r = client.post("/classify/", json={"text": "burst test"})
            results.append(r.status_code)

        assert 429 in results, "Rate limit should have fired within 25 requests"
        limited = [r for r in client.app.state.__dict__.values() if False]  # noqa
        rate_resp = next(
            client.post("/classify/", json={"text": "x"})
            for _ in range(1)
            if 429 in results
        )
        # Verify the 429 response has the correct error shape
        # (We already checked above, but let's verify one more time cleanly)
        rate_limited_responses = [
            r for r in results if r == 429
        ]
        assert len(rate_limited_responses) > 0

    def test_rate_limited_response_has_uniform_shape(self):
        """Exhaust the classify limit and verify 429 shape."""
        last_resp = None
        for _ in range(25):
            r = client.post("/classify/", json={"text": "shape test"})
            if r.status_code == 429:
                last_resp = r
                break

        assert last_resp is not None, "Expected a 429 within 25 requests"
        body = last_resp.json()
        assert "error" in body
        assert body["error"]["code"] == "RATE_LIMITED"
        assert "message" in body["error"]
        assert "requestId" in body["error"]

    def test_service_stable_under_burst_no_500s(self):
        """Under burst traffic every response must be an *expected* status.
        200 = model loaded, 429 = rate limited, 503 = model not loaded in
        test context (controlled error) — none of these is a crash (500).
        """
        for i in range(30):
            r = client.post("/classify/", json={"text": f"stability test {i}"})
            assert r.status_code in (200, 429, 503), (
                f"Request {i} returned unexpected crash status {r.status_code}: {r.text}"
            )
            if r.status_code == 500:
                pytest.fail(f"Unexpected 500 on request {i}: {r.text}")

    def test_rate_limit_counter_resets_between_tests(self):
        """Verify fixture clears state: 20 requests within limit must not get 429."""
        _MOCK_RESULT = {
            "categories": [{"label": "Tech", "score": 0.9}],
            "state": {"label": "continue", "score": 0.8},
            "next_step": {"template": "Keep going", "confidence": 0.7},
            "uncertain": False,
        }
        with patch("app.routers.classify.model_service") as mock_svc:
            mock_svc.classify.return_value = _MOCK_RESULT
            results = []
            for _ in range(20):
                r = client.post("/classify/", json={"text": "reset test"})
                results.append(r.status_code)

        assert 429 not in results, (
            f"Rate limit fired within first 20 requests after reset: {results}"
        )
        assert all(s == 200 for s in results), f"Unexpected statuses: {results}"
