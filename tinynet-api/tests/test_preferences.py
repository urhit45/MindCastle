"""
Integration tests for GET/PATCH /users/me/preferences (theme sync).
Requires migrated DB (users.preferences column).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.integration

client = TestClient(app)


class TestUserPreferences:
    def test_get_preferences_shape(self):
        r = client.get("/users/me/preferences")
        assert r.status_code == 200, r.text
        data = r.json()
        assert "theme" in data
        assert data["theme"] is None or data["theme"] in (
            "tsushima",
            "transylvania",
            "frieren",
            "lofi",
        )

    def test_patch_theme_round_trip(self):
        r = client.patch("/users/me/preferences", json={"theme": "tsushima"})
        assert r.status_code == 200, r.text
        assert r.json()["theme"] == "tsushima"

        r2 = client.get("/users/me/preferences")
        assert r2.status_code == 200
        assert r2.json()["theme"] == "tsushima"

        client.patch("/users/me/preferences", json={"theme": "lofi"})

    def test_patch_invalid_theme_422(self):
        r = client.patch("/users/me/preferences", json={"theme": "neon_punk"})
        assert r.status_code == 422
