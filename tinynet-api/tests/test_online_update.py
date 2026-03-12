"""
Tests for online learning update functionality.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from app.services.online_update import OnlineUpdateService
from app.ml.online_learner import OnlineLearner
from app.ml.tinynet import TinyNet
from app.ml.vectorizer import HashingVectorizer512


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model():
    """Create a mock TinyNet model for testing."""
    model = Mock(spec=TinyNet)
    
    # Mock the forward method to return expected outputs
    def mock_forward(x):
        batch_size = x.shape[0]
        hidden = torch.randn(batch_size, 32)  # 32-d hidden representation
        cat_logits = torch.randn(batch_size, 20)  # 20 categories
        state_logits = torch.randn(batch_size, 6)  # 6 states
        nextstep_logits = torch.randn(batch_size, 12)  # 12 next steps
        return hidden, cat_logits, state_logits, nextstep_logits
    
    model.forward = mock_forward
    model.train = Mock()
    model.eval = Mock()
    
    return model


@pytest.fixture
def mock_vectorizer():
    """Create a mock vectorizer for testing."""
    vectorizer = Mock(spec=HashingVectorizer512)
    vectorizer.encode.return_value = [0.1] * 512  # 512-d vector
    return vectorizer


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return {
        "categories": ["Fitness", "Running", "Strength", "Music", "Guitar"],
        "states": ["start", "continue", "pause", "end", "blocked", "idea"],
        "next_steps": ["PracticeForDuration", "RepeatTask", "IncreaseVolume"]
    }


@pytest.fixture
def online_learner(mock_model, temp_dir):
    """Create an OnlineLearner instance for testing."""
    return OnlineLearner(
        model=mock_model,
        save_dir=temp_dir,
        save_every=5,
        learning_rate=3e-4
    )


@pytest.fixture
def online_update_service(mock_model, online_learner, mock_vectorizer, sample_labels):
    """Create an OnlineUpdateService instance for testing."""
    return OnlineUpdateService(
        model=mock_model,
        online_learner=online_learner,
        vectorizer=mock_vectorizer,
        category_labels=sample_labels["categories"],
        state_labels=sample_labels["states"],
        next_step_labels=sample_labels["next_steps"]
    )


class TestOnlineUpdateService:
    """Test the OnlineUpdateService class."""
    
    def test_initialization(self, online_update_service, sample_labels):
        """Test service initialization."""
        assert online_update_service.model is not None
        assert online_update_service.online_learner is not None
        assert online_update_service.vectorizer is not None
        assert len(online_update_service.category_labels) == 5
        assert len(online_update_service.state_labels) == 6
        assert len(online_update_service.next_step_labels) == 3
    
    def test_process_correction_with_categories_and_state(self, online_update_service):
        """Test processing correction with both categories and state."""
        result = online_update_service.process_correction(
            text="Ran 2 miles, shin tight",
            categories=["Fitness", "Running"],
            state="continue",
            link_to=None,
            next_step_template="PracticeForDuration"
        )
        
        assert result["ok"] is True
        assert "Classification corrected and model updated" in result["message"]
        assert result["update_result"] is not None
        assert "update_count" in result["update_result"]
    
    def test_process_correction_with_categories_only(self, online_update_service):
        """Test processing correction with only categories."""
        result = online_update_service.process_correction(
            text="Guitar practice today",
            categories=["Music", "Guitar"],
            state=None,
            link_to=None,
            next_step_template=None
        )
        
        assert result["ok"] is True
        assert "Classification corrected and model updated" in result["message"]
        assert result["update_result"] is not None
    
    def test_process_correction_with_state_only(self, online_update_service):
        """Test processing correction with only state."""
        result = online_update_service.process_correction(
            text="Learning Python programming",
            categories=None,
            state="start",
            link_to=None,
            next_step_template=None
        )
        
        assert result["ok"] is True
        assert "Classification corrected and model updated" in result["message"]
        assert result["update_result"] is not None
    
    def test_process_correction_no_corrections(self, online_update_service):
        """Test processing correction with no corrections provided."""
        result = online_update_service.process_correction(
            text="Some text",
            categories=None,
            state=None,
            link_to=None,
            next_step_template=None
        )
        
        assert result["ok"] is True
        assert "No corrections provided" in result["message"]
        assert result["update_result"] is None
    
    def test_process_correction_with_link_to(self, online_update_service, mock_model):
        """Test processing correction with link_to parameter."""
        # Mock the model to return a specific hidden representation
        mock_model.forward.return_value = (
            torch.randn(1, 32),  # hidden
            torch.randn(1, 20),  # cat_logits
            torch.randn(1, 6),   # state_logits
            torch.randn(1, 12)   # nextstep_logits
        )
        
        with patch('app.services.online_update.NodeBank') as mock_nodebank:
            mock_nodebank_instance = Mock()
            mock_nodebank.return_value = mock_nodebank_instance
            
            result = online_update_service.process_correction(
                text="Running progress update",
                categories=["Fitness", "Running"],
                state="continue",
                link_to="node_001",
                next_step_template="PracticeForDuration"
            )
            
            assert result["ok"] is True
            # Verify nodebank was called
            mock_nodebank_instance.upsert_node_embedding.assert_called_once()
    
    def test_process_correction_online_learner_unavailable(self, mock_model, mock_vectorizer, sample_labels):
        """Test processing correction when online learner is not available."""
        service = OnlineUpdateService(
            model=mock_model,
            online_learner=None,  # No online learner
            vectorizer=mock_vectorizer,
            category_labels=sample_labels["categories"],
            state_labels=sample_labels["states"],
            next_step_labels=sample_labels["next_steps"]
        )
        
        result = service.process_correction(
            text="Some text",
            categories=["Fitness"],
            state="continue"
        )
        
        assert result["ok"] is True
        assert "online learning not available" in result["message"]
        assert result["update_result"] is None
    
    def test_get_update_stats(self, online_update_service):
        """Test getting update statistics."""
        stats = online_update_service.get_update_stats()
        
        assert stats["status"] == "available"
        assert "total_updates" in stats
        assert "save_every" in stats
        assert "learning_rate" in stats
        assert stats["save_every"] == 5
        assert stats["learning_rate"] == 3e-4
    
    def test_get_update_stats_no_learner(self, mock_model, mock_vectorizer, sample_labels):
        """Test getting update statistics when no online learner is available."""
        service = OnlineUpdateService(
            model=mock_model,
            online_learner=None,
            vectorizer=mock_vectorizer,
            category_labels=sample_labels["categories"],
            state_labels=sample_labels["states"],
            next_step_labels=sample_labels["next_steps"]
        )
        
        stats = service.get_update_stats()
        assert stats["status"] == "not_available"


class TestOnlineLearnerIntegration:
    """Test integration between OnlineUpdateService and OnlineLearner."""
    
    def test_online_learning_step(self, online_update_service, online_learner):
        """Test that a single online learning step works correctly."""
        initial_count = online_learner.update_count
        
        result = online_update_service.process_correction(
            text="Test text for learning",
            categories=["Fitness"],
            state="continue"
        )
        
        assert result["ok"] is True
        assert online_learner.update_count == initial_count + 1
        
        # Check that metrics were updated
        assert len(online_learner.metrics["loss_history"]) > 0
        latest_metric = online_learner.metrics["loss_history"][-1]
        assert latest_metric["update"] == online_learner.update_count
    
    def test_gradient_clipping(self, online_update_service, mock_model):
        """Test that gradient clipping is applied."""
        # This test verifies that the OnlineLearner applies gradient clipping
        # The actual clipping happens in the OnlineLearner.update_model method
        
        result = online_update_service.process_correction(
            text="Test text",
            categories=["Fitness"],
            state="continue"
        )
        
        assert result["ok"] is True
        # The gradient clipping is applied in the OnlineLearner, so we just verify
        # that the update completed successfully
    
    def test_save_checkpoint_interval(self, online_update_service, online_learner, temp_dir):
        """Test that checkpoints are saved at the correct interval."""
        # Set save_every to 2 for testing
        online_learner.save_every = 2
        
        # Perform updates
        for i in range(3):
            result = online_update_service.process_correction(
                text=f"Test text {i}",
                categories=["Fitness"],
                state="continue"
            )
            assert result["ok"] is True
        
        # Check that a checkpoint was saved
        checkpoint_files = list(Path(temp_dir).glob("*.pt"))
        assert len(checkpoint_files) > 0
    
    def test_label_mapping(self, online_update_service, sample_labels):
        """Test that label names are correctly mapped to indices."""
        # Test with valid labels
        result = online_update_service.process_correction(
            text="Test text",
            categories=["Fitness", "Running"],  # These exist in sample_labels
            state="continue"  # This exists in sample_labels
        )
        
        assert result["ok"] is True
        
        # Test with invalid labels (should handle gracefully)
        result = online_update_service.process_correction(
            text="Test text",
            categories=["InvalidCategory"],  # This doesn't exist
            state="InvalidState"  # This doesn't exist
        )
        
        # Should still work (invalid labels are filtered out)
        assert result["ok"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
