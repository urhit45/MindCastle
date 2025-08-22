"""
Tests for TinyNet API Contract
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestAPIContract:
    """Test suite for API contract compliance"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/healthz")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert data["status"] == "healthy"
        assert data["service"] == "tinynet-api"
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "openapi" in data
        assert data["message"] == "Welcome to TinyNet API"
    
    def test_openapi_spec(self):
        """Test OpenAPI specification endpoint"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        spec = response.json()
        assert "openapi" in spec
        assert "info" in spec
        assert "paths" in spec
        assert spec["info"]["title"] == "TinyNet API"
    
    def test_classify_endpoint(self):
        """Test POST /classify returns expected keys and types"""
        request_data = {
            "text": "Ran 2 miles, shin tight",
            "contextNodeId": None
        }
        
        response = client.post("/classify", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required keys
        required_keys = ["categories", "state", "linkHints", "nextStep", "uncertain"]
        for key in required_keys:
            assert key in data, f"Missing required key: {key}"
        
        # Check types
        assert isinstance(data["categories"], list)
        assert isinstance(data["state"], dict)
        assert isinstance(data["linkHints"], list)
        assert isinstance(data["nextStep"], dict)
        assert isinstance(data["uncertain"], bool)
        
        # Check category structure
        if data["categories"]:
            category = data["categories"][0]
            assert "label" in category
            assert "score" in category
            assert isinstance(category["label"], str)
            assert isinstance(category["score"], (int, float))
        
        # Check state structure
        assert "label" in data["state"]
        assert "score" in data["state"]
        assert isinstance(data["state"]["label"], str)
        assert isinstance(data["state"]["score"], (int, float))
        
        # Check link hints structure
        if data["linkHints"]:
            link_hint = data["linkHints"][0]
            assert "nodeId" in link_hint
            assert "title" in link_hint
            assert "similarity" in link_hint
            assert isinstance(link_hint["nodeId"], str)
            assert isinstance(link_hint["title"], str)
            assert isinstance(link_hint["similarity"], (int, float))
        
        # Check next step structure
        assert "template" in data["nextStep"]
        assert "confidence" in data["nextStep"]
        assert isinstance(data["nextStep"]["template"], str)
        assert isinstance(data["nextStep"]["confidence"], (int, float))
    
    def test_correct_endpoint(self):
        """Test POST /correct endpoint"""
        request_data = {
            "text": "Ran 2 miles, shin tight",
            "categories": ["Fitness", "Running"],
            "state": "blocked",
            "linkTo": None,
            "nextStepTemplate": "PracticeForDuration"
        }
        
        response = client.post("/correct", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "ok" in data
        assert data["ok"] is True
    
    def test_train_endpoint(self):
        """Test POST /train endpoint"""
        request_data = {
            "samples": [
                {
                    "text": "Ran 2 miles, shin tight",
                    "categories": ["Fitness", "Running"],
                    "state": "continue"
                },
                {
                    "text": "Guitar practice session",
                    "categories": ["Music", "Guitar"],
                    "state": "continue"
                }
            ]
        }
        
        response = client.post("/train", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "ok" in data
        assert data["ok"] is True
        assert "metrics" in data
        
        metrics = data["metrics"]
        assert "samplesProcessed" in metrics
        assert "accuracy" in metrics
        assert metrics["samplesProcessed"] == 2
    
    def test_nodes_search(self):
        """Test GET /nodes/search endpoint"""
        response = client.get("/nodes/search?q=run&limit=20")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) > 0
        
        # Check item structure
        item = data["items"][0]
        assert "id" in item
        assert "title" in item
        assert "hub" in item
        assert "score" in item
        
        assert isinstance(item["id"], str)
        assert isinstance(item["title"], str)
        assert isinstance(item["hub"], bool)
        assert isinstance(item["score"], (int, float))
    
    def test_nodes_search_with_limit(self):
        """Test nodes search with different limit values"""
        # Test default limit
        response = client.get("/nodes/search?q=run")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["items"]) <= 10  # Default limit
        
        # Test custom limit
        response = response = client.get("/nodes/search?q=run&limit=5")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["items"]) <= 5
    
    def test_node_details(self):
        """Test GET /nodes/{id} endpoint"""
        # Use a valid UUID
        node_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.get(f"/nodes/{node_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert "title" in data
        assert "hub" in data
        assert "status" in data
        assert "related" in data
        
        assert data["id"] == node_id
        assert isinstance(data["title"], str)
        assert isinstance(data["hub"], bool)
        assert isinstance(data["status"], str)
        assert isinstance(data["related"], list)
    
    def test_node_details_invalid_uuid(self):
        """Test node details with invalid UUID"""
        response = client.get("/nodes/invalid-uuid")
        assert response.status_code == 400
        
        data = response.json()
        assert "detail" in data
        assert "Invalid node ID format" in data["detail"]
    
    def test_node_logs(self):
        """Test GET /nodes/{id}/logs endpoint"""
        node_id = "123e4567-e89b-12d3-a456-426614174000"
        
        response = client.get(f"/nodes/{node_id}/logs?limit=20")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) > 0
        
        # Check log item structure
        log_item = data["items"][0]
        assert "id" in log_item
        assert "time" in log_item
        assert "text" in log_item
        assert "state" in log_item
        
        assert isinstance(log_item["id"], int)
        assert isinstance(log_item["time"], str)
        assert isinstance(log_item["text"], str)
        assert isinstance(log_item["state"], str)
    
    def test_node_logs_with_limit(self):
        """Test node logs with different limit values"""
        node_id = "123e4567-e89b-12d3-a456-426614174000"
        
        # Test default limit
        response = client.get(f"/nodes/{node_id}/logs")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["items"]) <= 50  # Default limit
        
        # Test custom limit
        response = client.get(f"/nodes/{node_id}/logs?limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["items"]) <= 10
    
    def test_home_review(self):
        """Test GET /home/review returns array with at least 1 item"""
        response = client.get("/home/review")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert isinstance(data["items"], list)
        assert len(data["items"]) >= 1
        
        # Check item structure
        item = data["items"][0]
        assert "id" in item
        assert "title" in item
        assert "reason" in item
        assert "nodeId" in item
        
        assert isinstance(item["id"], str)
        assert isinstance(item["title"], str)
        assert isinstance(item["reason"], str)
        assert isinstance(item["nodeId"], str)
        
        # Check reason enum values
        valid_reasons = ["blocked", "nextStep", "stale"]
        assert item["reason"] in valid_reasons
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        response = client.get("/")
        assert response.status_code == 200
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-credentials" in response.headers
        
        # Should allow localhost origins
        origin = response.headers["access-control-allow-origin"]
        assert origin in ["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"]
    
    def test_docs_endpoint(self):
        """Test that /docs endpoint is accessible"""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_endpoint(self):
        """Test that /openapi.json endpoint is accessible"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
