"""
Test the real inference endpoint using TinyNet model.
"""

import pytest
from fastapi.testclient import TestClient
import json

from app.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


def test_classify_endpoint_smoke(client):
    """Test that the /classify endpoint returns the expected structure."""
    # Test with the example text from the task
    response = client.post(
        "/classify",
        json={"text": "Ran 2 miles, shin tight", "contextNodeId": None}
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    data = response.json()
    
    # Check that all required keys exist
    assert "categories" in data, "Response missing 'categories' key"
    assert "state" in data, "Response missing 'state' key"
    assert "linkHints" in data, "Response missing 'linkHints' key"
    assert "nextStep" in data, "Response missing 'nextStep' key"
    assert "uncertain" in data, "Response missing 'uncertain' key"
    
    # Check types
    assert isinstance(data["categories"], list), "categories should be a list"
    assert isinstance(data["state"], dict), "state should be a dict"
    assert isinstance(data["linkHints"], list), "linkHints should be a list"
    assert isinstance(data["nextStep"], dict), "nextStep should be a dict"
    assert isinstance(data["uncertain"], bool), "uncertain should be a boolean"
    
    # Check state structure
    state = data["state"]
    assert "label" in state, "state missing 'label' key"
    assert "score" in state, "state missing 'score' key"
    assert isinstance(state["label"], str), "state.label should be a string"
    assert isinstance(state["score"], (int, float)), "state.score should be a number"
    
    # Check nextStep structure
    next_step = data["nextStep"]
    assert "template" in next_step, "nextStep missing 'template' key"
    assert "slots" in next_step, "nextStep missing 'slots' key"
    assert "confidence" in next_step, "nextStep missing 'confidence' key"
    assert isinstance(next_step["template"], str), "nextStep.template should be a string"
    assert isinstance(next_step["slots"], dict), "nextStep.slots should be a dict"
    assert isinstance(next_step["confidence"], (int, float)), "nextStep.confidence should be a number"
    
    # Check categories structure (if any exist)
    if data["categories"]:
        for cat in data["categories"]:
            assert "label" in cat, "category missing 'label' key"
            assert "score" in cat, "category missing 'score' key"
            assert isinstance(cat["label"], str), "category.label should be a string"
            assert isinstance(cat["score"], (int, float)), "category.score should be a number"
    
    # Check linkHints structure (if any exist)
    if data["linkHints"]:
        for hint in data["linkHints"]:
            assert "nodeId" in hint, "linkHint missing 'nodeId' key"
            assert "title" in hint, "linkHint missing 'title' key"
            assert "similarity" in hint, "linkHint missing 'similarity' key"
            assert isinstance(hint["nodeId"], str), "linkHint.nodeId should be a string"
            assert isinstance(hint["title"], str), "linkHint.title should be a string"
            assert isinstance(hint["similarity"], (int, float)), "linkHint.similarity should be a number"


def test_classify_endpoint_different_texts(client):
    """Test classification with different types of text."""
    test_cases = [
        "Guitar practice today",
        "Learning Python programming",
        "Admin: submit invoice",
        "Workout: 3 sets of squats"
    ]
    
    for text in test_cases:
        response = client.post(
            "/classify",
            json={"text": text, "contextNodeId": None}
        )
        
        assert response.status_code == 200, f"Failed for text: {text}"
        data = response.json()
        
        # Basic structure checks
        assert all(key in data for key in ["categories", "state", "linkHints", "nextStep", "uncertain"])
        assert isinstance(data["uncertain"], bool)


def test_classify_endpoint_with_context(client):
    """Test classification with context node ID."""
    response = client.post(
        "/classify",
        json={"text": "Ran 2 miles, shin tight", "contextNodeId": "test-node-123"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should still return the same structure
    assert all(key in data for key in ["categories", "state", "linkHints", "nextStep", "uncertain"])


def test_classify_endpoint_empty_text(client):
    """Test classification with empty text."""
    response = client.post(
        "/classify",
        json={"text": "", "contextNodeId": None}
    )
    
    # Should handle gracefully (either 400 or 200 with empty results)
    assert response.status_code in [200, 400]


def test_classify_endpoint_missing_text(client):
    """Test classification with missing text."""
    response = client.post(
        "/classify",
        json={"contextNodeId": None}
    )
    
    assert response.status_code == 422  # Validation error


def test_classify_endpoint_large_text(client):
    """Test classification with longer text."""
    long_text = "This is a much longer text that should still work with the vectorizer. " * 10
    
    response = client.post(
        "/classify",
        json={"text": long_text, "contextNodeId": None}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should still return valid structure
    assert all(key in data for key in ["categories", "state", "linkHints", "nextStep", "uncertain"])


def test_classify_endpoint_unicode_text(client):
    """Test classification with unicode text."""
    unicode_text = "🎯 Running 🏃‍♂️ 5 miles with 🎵 music"
    
    response = client.post(
        "/classify",
        json={"text": unicode_text, "contextNodeId": None}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should handle unicode gracefully
    assert all(key in data for key in ["categories", "state", "linkHints", "nextStep", "uncertain"])


if __name__ == "__main__":
    # Run basic smoke test
    client = TestClient(app)
    test_classify_endpoint_smoke(client)
    print("✅ Basic inference endpoint test passed!")
