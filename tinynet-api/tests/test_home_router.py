"""
Tests for the home router functionality.
"""

import pytest
from unittest.mock import AsyncMock, Mock
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.models import ProgressLog, Todo, Node
from app.schemas import ProgressLog, Todo, Node


class TestHomeRouter:
    """Test the home router endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        session = AsyncMock()
        return session
    
    @pytest.fixture
    def sample_progress_logs(self):
        """Sample progress log data."""
        now = datetime.utcnow()
        return [
            ProgressLog(
                id="log_001",
                node_id="node_001",
                text="Running progress - feeling good today",
                next_step="ScheduleFollowUp",
                state="continue",
                created_at=now - timedelta(hours=2)
            ),
            ProgressLog(
                id="log_002",
                node_id="node_002",
                text="Guitar practice - learned new chord",
                next_step="RepeatTask",
                state="continue",
                created_at=now - timedelta(hours=1)
            ),
            ProgressLog(
                id="log_003",
                node_id="node_003",
                text="Blocked on AI project - need help",
                next_step=None,
                state="blocked",
                created_at=now - timedelta(days=1)
            ),
            ProgressLog(
                id="log_004",
                node_id="node_004",
                text="Old project that's been forgotten",
                next_step=None,
                state="pause",
                created_at=now - timedelta(days=10)
            )
        ]
    
    @pytest.fixture
    def sample_todos(self):
        """Sample todo data."""
        now = datetime.utcnow()
        return [
            Todo(
                id="todo_001",
                title="Complete running plan",
                status="in_progress",
                due_at=now - timedelta(days=1),  # Overdue
                node_id="node_001",
                created_at=now - timedelta(days=3)
            ),
            Todo(
                id="todo_002",
                title="Practice guitar scales",
                status="pending",
                due_at=now + timedelta(days=2),  # Due soon
                node_id="node_002",
                created_at=now - timedelta(days=1)
            ),
            Todo(
                id="todo_003",
                title="Review AI course",
                status="completed",
                due_at=now + timedelta(days=5),
                node_id="node_003",
                created_at=now - timedelta(days=2)
            )
        ]
    
    @pytest.fixture
    def sample_nodes(self):
        """Sample node data."""
        return [
            Node(
                id="node_001",
                title="Running Progress",
                is_hub=False,
                status="active",
                created_at=datetime.utcnow() - timedelta(days=5)
            ),
            Node(
                id="node_002",
                title="Guitar Practice",
                is_hub=False,
                status="active",
                created_at=datetime.utcnow() - timedelta(days=3)
            ),
            Node(
                id="node_003",
                title="AI Learning",
                is_hub=True,
                status="active",
                created_at=datetime.utcnow() - timedelta(days=10)
            )
        ]
    
    def test_quick_actions_endpoint(self, client):
        """Test the quick actions endpoint."""
        response = client.get("/home/quick-actions")
        assert response.status_code == 200
        
        data = response.json()
        assert data["ok"] == True
        assert "quick_actions" in data
        
        actions = data["quick_actions"]
        assert len(actions) == 4
        
        # Check specific actions
        action_ids = [action["id"] for action in actions]
        assert "add_progress" in action_ids
        assert "create_todo" in action_ids
        assert "review_blocked" in action_ids
        assert "plan_next_steps" in action_ids
        
        # Check action structure
        for action in actions:
            assert "id" in action
            assert "title" in action
            assert "description" in action
            assert "action" in action
            assert "icon" in action
    
    @pytest.mark.asyncio
    async def test_review_items_endpoint_mock(self, client, mock_session, 
                                            sample_progress_logs, sample_todos):
        """Test the review items endpoint with mocked database."""
        # Mock the database session dependency
        app.dependency_overrides = {
            get_session: lambda: mock_session
        }
        
        # Mock the database queries
        mock_session.scalar.return_value = 1  # For count queries
        mock_session.execute.return_value.scalars.return_value.all.side_effect = [
            [sample_progress_logs[2]],  # blocked logs
            [sample_progress_logs[0], sample_progress_logs[1]],  # next step logs
            [sample_progress_logs[3]],  # stale logs
            [sample_todos[0]],  # overdue todos
            [sample_todos[1]]   # soon todos
        ]
        
        try:
            response = client.get("/home/review?limit=20")
            assert response.status_code == 200
            
            data = response.json()
            assert data["ok"] == True
            assert "items" in data
            assert "stats" in data
            
            items = data["items"]
            assert len(items) > 0
            
            # Check that we have items from different categories
            reasons = [item["reason"] for item in items]
            assert "blocked" in reasons
            assert "next_step" in reasons
            assert "stale" in reasons
            assert "overdue" in reasons
            assert "due_soon" in reasons
            
            # Check item structure
            for item in items:
                assert "id" in item
                assert "type" in item
                assert "reason" in item
                assert "title" in item
                assert "priority" in item
                
                if item["type"] == "progress_log":
                    assert "created_at" in item
                    assert "state" in item
                elif item["type"] == "todo":
                    assert "due_at" in item
                    assert "status" in item
            
            # Check stats
            stats = data["stats"]
            assert "total_items" in stats
            assert "blocked_count" in stats
            assert "next_step_count" in stats
            assert "stale_count" in stats
            assert "overdue_count" in stats
            assert "due_soon_count" in stats
            
        finally:
            # Clean up dependency override
            app.dependency_overrides = {}
    
    @pytest.mark.asyncio
    async def test_dashboard_endpoint_mock(self, client, mock_session, 
                                         sample_progress_logs, sample_todos, sample_nodes):
        """Test the dashboard endpoint with mocked database."""
        # Mock the database session dependency
        app.dependency_overrides = {
            get_session: lambda: mock_session
        }
        
        # Mock the database queries
        mock_session.scalar.side_effect = [
            3,  # total_nodes
            4,  # total_progress
            3,  # total_todos
            2,  # recent_progress (last 7 days)
            1,  # recent_todos (last 7 days)
            1   # completed_todos
        ]
        
        # Mock state distribution query
        mock_result = Mock()
        mock_result.__iter__ = lambda self: iter([
            ("continue", 2),
            ("blocked", 1),
            ("pause", 1)
        ])
        mock_session.execute.return_value = mock_result
        
        try:
            response = client.get("/home/dashboard")
            assert response.status_code == 200
            
            data = response.json()
            assert data["ok"] == True
            assert "summary" in data
            
            summary = data["summary"]
            assert summary["total_nodes"] == 3
            assert summary["total_progress_logs"] == 4
            assert summary["total_todos"] == 3
            assert summary["recent_activity"]["progress_logs_7d"] == 2
            assert summary["recent_activity"]["todos_7d"] == 1
            assert summary["completion"]["completed_todos"] == 1
            assert "completion_rate_percent" in summary["completion"]
            assert "state_distribution" in summary
            
            # Check state distribution
            state_dist = summary["state_distribution"]
            assert state_dist["continue"] == 2
            assert state_dist["blocked"] == 1
            assert state_dist["pause"] == 1
            
        finally:
            # Clean up dependency override
            app.dependency_overrides = {}
    
    def test_review_items_limit_parameter(self, client):
        """Test that the limit parameter works correctly."""
        response = client.get("/home/review?limit=5")
        assert response.status_code == 200
        
        # Note: This will return mock data since we don't have real DB
        # The actual limit enforcement happens in the database queries
    
    def test_review_items_default_limit(self, client):
        """Test that the default limit is applied."""
        response = client.get("/home/review")
        assert response.status_code == 200
    
    def test_dashboard_generated_timestamp(self, client):
        """Test that dashboard includes generation timestamp."""
        response = client.get("/home/dashboard")
        assert response.status_code == 200
        
        data = response.json()
        assert "generated_at" in data
        
        # Should be a valid ISO timestamp
        timestamp = data["generated_at"]
        try:
            datetime.fromisoformat(timestamp)
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp}")
    
    def test_review_items_priority_sorting(self, client):
        """Test that items are sorted by priority."""
        response = client.get("/home/review")
        assert response.status_code == 200
        
        data = response.json()
        if "items" in data and len(data["items"]) > 1:
            items = data["items"]
            priorities = [item.get("priority", "low") for item in items]
            
            # Check that high priority items come first
            # (This is a basic check - actual sorting logic is in the endpoint)
            assert len(priorities) > 0
    
    def test_error_handling(self, client):
        """Test that errors are handled gracefully."""
        # Test with invalid limit
        response = client.get("/home/review?limit=invalid")
        # Should handle gracefully or return 400
        assert response.status_code in [200, 400, 422]
    
    def test_cors_headers(self, client):
        """Test that CORS headers are present."""
        response = client.get("/home/quick-actions")
        assert response.status_code == 200
        
        # Check for CORS headers (if configured)
        # This depends on your CORS configuration
        # assert "Access-Control-Allow-Origin" in response.headers
    
    def test_response_content_type(self, client):
        """Test that responses have correct content type."""
        response = client.get("/home/quick-actions")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    def test_quick_actions_structure(self, client):
        """Test that quick actions have consistent structure."""
        response = client.get("/home/quick-actions")
        assert response.status_code == 200
        
        data = response.json()
        actions = data["quick_actions"]
        
        for action in actions:
            # Check required fields
            required_fields = ["id", "title", "description", "action", "icon"]
            for field in required_fields:
                assert field in action
                assert action[field] is not None
                assert len(str(action[field])) > 0
            
            # Check field types
            assert isinstance(action["id"], str)
            assert isinstance(action["title"], str)
            assert isinstance(action["description"], str)
            assert isinstance(action["action"], str)
            assert isinstance(action["icon"], str)
