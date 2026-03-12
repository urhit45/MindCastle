"""
Tests for online learning functionality.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from app.ml.online_learner import OnlineLearner
from app.ml.tinynet import TinyNet


class TestOnlineLearner:
    """Test the OnlineLearner class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_model(self):
        """Create a real TinyNet model for testing."""
        # Create a real TinyNet model with small dimensions for testing
        model = TinyNet()
        model.eval()
        return model
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing."""
        # Use the default TinyNet dimensions (20 categories, 6 states, 12 next steps)
        category_labels = [
            "Fitness", "Running", "Strength", "Music", "Guitar", "Learning", "AI", 
            "Admin", "Finance", "Social", "Health", "Cooking", "Travel", "Work", 
            "SideProject", "Design", "Reading", "Writing", "Mindfulness", "Household"
        ]
        state_labels = ["start", "continue", "pause", "end", "blocked", "idea"]
        nextstep_labels = [
            "PracticeForDuration", "RepeatTask", "IncreaseVolume", "ScheduleFollowUp",
            "AttachLink", "ReviewNotes", "OutlineThreeBullets", "BookAppointment",
            "BuySupplies", "CreateSubtasks", "SetReminder", "LogReflection"
        ]
        
        return {
            "text_vector": torch.randn(512),
            "categories": ["Fitness", "Running"],
            "state": "continue",
            "nextstep": "ScheduleFollowUp",
            "category_labels": category_labels,
            "state_labels": state_labels,
            "nextstep_labels": nextstep_labels
        }
    
    def test_initialization(self, mock_model, temp_dir):
        """Test OnlineLearner initialization."""
        learner = OnlineLearner(
            model=mock_model,
            save_dir=temp_dir,
            save_every=5,
            learning_rate=1e-4
        )
        
        assert learner.model == mock_model
        assert learner.save_dir == Path(temp_dir)
        assert learner.save_every == 5
        assert learner.learning_rate == 1e-4
        assert learner.update_count == 0
        assert Path(temp_dir).exists()
    
    def test_initialization_creates_save_dir(self, mock_model):
        """Test that save directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "new_dir"
            learner = OnlineLearner(model=mock_model, save_dir=str(save_path))
            assert save_path.exists()
    
    def test_load_metrics_new_file(self, mock_model, temp_dir):
        """Test loading metrics when no file exists."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        expected_metrics = {
            "total_updates": 0,
            "last_save": None,
            "loss_history": [],
            "accuracy_history": []
        }
        
        assert learner.metrics == expected_metrics
    
    def test_load_metrics_existing_file(self, mock_model, temp_dir):
        """Test loading metrics from existing file."""
        import json
        
        # Create existing metrics file
        metrics_data = {
            "total_updates": 5,
            "last_save": "2024-01-01T00:00:00",
            "loss_history": [{"update": 1, "loss": 0.5}],
            "accuracy_history": []
        }
        
        metrics_file = Path(temp_dir) / "online_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f)
        
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        assert learner.metrics["total_updates"] == 5
        assert len(learner.metrics["loss_history"]) == 1
    
    def test_update_model_basic(self, mock_model, temp_dir, sample_data):
        """Test basic model update functionality."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        # Mock the loss computation
        with patch('torch.nn.BCEWithLogitsLoss') as mock_bce, \
             patch('torch.nn.CrossEntropyLoss') as mock_ce:
            
            mock_bce.return_value.return_value = torch.tensor(0.5)
            mock_ce.return_value.return_value = torch.tensor(0.3)
            
            result = learner.update_model(
                text_vector=sample_data["text_vector"],
                categories_target=sample_data["categories"],
                state_target=sample_data["state"],
                nextstep_target=sample_data["nextstep"],
                category_labels=sample_data["category_labels"],
                state_labels=sample_data["state_labels"],
                nextstep_labels=sample_data["nextstep_labels"]
            )
        
        assert result["update_count"] == 1
        assert "total_loss" in result
        assert "cat_accuracy" in result
        assert "state_accuracy" in result
        assert "nextstep_accuracy" in result
        assert result["saved"] == False  # First update, not saved yet
    
    def test_update_model_with_save(self, mock_model, temp_dir, sample_data):
        """Test model update with periodic saving."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir, save_every=1)
        
        # Mock the loss computation
        with patch('torch.nn.BCEWithLogitsLoss') as mock_bce, \
             patch('torch.nn.CrossEntropyLoss') as mock_ce:
            
            mock_bce.return_value.return_value = torch.tensor(0.5)
            mock_ce.return_value.return_value = torch.tensor(0.3)
            
            result = learner.update_model(
                text_vector=sample_data["text_vector"],
                categories_target=sample_data["categories"],
                state_target=sample_data["state"],
                nextstep_target=sample_data["nextstep"],
                category_labels=sample_data["category_labels"],
                state_labels=sample_data["state_labels"],
                nextstep_labels=sample_data["nextstep_labels"]
            )
        
        assert result["saved"] == True
        assert (Path(temp_dir) / "online_latest.pt").exists()
    
    def test_update_model_validation(self, mock_model, temp_dir):
        """Test validation of required parameters."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        with pytest.raises(ValueError, match="Category and state labels must be provided"):
            learner.update_model(
                text_vector=torch.randn(512),
                categories_target=["Fitness"],
                state_target="continue",
                category_labels=None,
                state_labels=["start", "continue"]
            )
    
    def test_update_model_handles_missing_nextstep(self, mock_model, temp_dir, sample_data):
        """Test that missing nextstep is handled gracefully."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        # Mock the loss computation
        with patch('torch.nn.BCEWithLogitsLoss') as mock_bce, \
             patch('torch.nn.CrossEntropyLoss') as mock_ce:
            
            mock_bce.return_value.return_value = torch.tensor(0.5)
            mock_ce.return_value.return_value = torch.tensor(0.3)
            
            result = learner.update_model(
                text_vector=sample_data["text_vector"],
                categories_target=sample_data["categories"],
                state_target=sample_data["state"],
                nextstep_target=None,  # Missing nextstep
                category_labels=sample_data["category_labels"],
                state_labels=sample_data["state_labels"],
                nextstep_labels=sample_data["nextstep_labels"]
            )
        
        assert result["update_count"] == 1
    
    def test_metrics_history_limiting(self, mock_model, temp_dir, sample_data):
        """Test that metrics history is limited to prevent memory bloat."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        # Mock the loss computation
        with patch('torch.nn.BCEWithLogitsLoss') as mock_bce, \
             patch('torch.nn.CrossEntropyLoss') as mock_ce:
            
            mock_bce.return_value.return_value = torch.tensor(0.5)
            mock_ce.return_value.return_value = torch.tensor(0.3)
            
            # Perform more than 100 updates
            for i in range(105):
                learner.update_model(
                    text_vector=sample_data["text_vector"],
                    categories_target=sample_data["categories"],
                    state_target=sample_data["state"],
                    nextstep_target=sample_data["nextstep"],
                    category_labels=sample_data["category_labels"],
                    state_labels=sample_data["state_labels"],
                    nextstep_labels=sample_data["nextstep_labels"]
                )
        
        # Should keep only last 100 entries
        assert len(learner.metrics["loss_history"]) <= 100
        assert learner.update_count == 105
    
    def test_get_metrics_summary_no_updates(self, mock_model, temp_dir):
        """Test metrics summary when no updates have been performed."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        summary = learner.get_metrics_summary()
        assert summary["message"] == "No updates performed yet"
    
    def test_get_metrics_summary_with_updates(self, mock_model, temp_dir, sample_data):
        """Test metrics summary with update history."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        # Mock the loss computation
        with patch('torch.nn.BCEWithLogitsLoss') as mock_bce, \
             patch('torch.nn.CrossEntropyLoss') as mock_ce:
            
            mock_bce.return_value.return_value = torch.tensor(0.5)
            mock_ce.return_value.return_value = torch.tensor(0.3)
            
            # Perform a few updates
            for i in range(3):
                learner.update_model(
                    text_vector=sample_data["text_vector"],
                    categories_target=sample_data["categories"],
                    state_target=sample_data["state"],
                    nextstep_target=sample_data["nextstep"],
                    category_labels=sample_data["category_labels"],
                    state_labels=sample_data["state_labels"],
                    nextstep_labels=sample_data["nextstep_labels"]
                )
        
        summary = learner.get_metrics_summary()
        assert summary["total_updates"] == 3
        assert "recent_avg_loss" in summary
        assert "recent_avg_cat_accuracy" in summary
        assert "learning_rate" in summary
        assert "save_every" in summary
    
    def test_checkpoint_cleanup(self, mock_model, temp_dir, sample_data):
        """Test that old checkpoints are cleaned up."""
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir, save_every=1)
        
        # Mock the loss computation
        with patch('torch.nn.BCEWithLogitsLoss') as mock_bce, \
             patch('torch.nn.CrossEntropyLoss') as mock_ce:
            
            mock_bce.return_value.return_value = torch.tensor(0.5)
            mock_ce.return_value.return_value = torch.tensor(0.3)
            
            # Perform 7 updates (should create 7 checkpoints, keep only last 5)
            for i in range(7):
                learner.update_model(
                    text_vector=sample_data["text_vector"],
                    categories_target=sample_data["categories"],
                    state_target=sample_data["state"],
                    nextstep_target=sample_data["nextstep"],
                    category_labels=sample_data["category_labels"],
                    state_labels=sample_data["state_labels"],
                    nextstep_labels=sample_data["nextstep_labels"]
                )
        
        # Check that only 5 checkpoints remain
        checkpoints = list(Path(temp_dir).glob("online_checkpoint_*.pt"))
        assert len(checkpoints) <= 5
        
        # Latest checkpoint should exist
        assert (Path(temp_dir) / "online_latest.pt").exists()
    
    def test_gradient_clipping(self, mock_model, temp_dir, sample_data):
        """Test that gradient clipping is applied."""
        # Mock the forward pass
        mock_model.return_value = (
            torch.randn(1, 32),  # hidden
            torch.randn(1, 4),   # cat_logits
            torch.randn(1, 6),   # state_logits
            torch.randn(1, 3)    # nextstep_logits
        )
        
        learner = OnlineLearner(model=mock_model, save_dir=temp_dir)
        
        # Mock the loss computation and gradient clipping
        with patch('torch.nn.BCEWithLogitsLoss') as mock_bce, \
             patch('torch.nn.CrossEntropyLoss') as mock_ce, \
             patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            
            mock_bce.return_value.return_value = torch.tensor(0.5)
            mock_ce.return_value.return_value = torch.tensor(0.3)
            
            learner.update_model(
                text_vector=sample_data["text_vector"],
                categories_target=sample_data["categories"],
                state_target=sample_data["state"],
                nextstep_target=sample_data["nextstep"],
                category_labels=sample_data["category_labels"],
                state_labels=sample_data["state_labels"],
                nextstep_labels=sample_data["nextstep_labels"]
            )
        
        # Verify gradient clipping was called
        mock_clip.assert_called_once()
        args, kwargs = mock_clip.call_args
        assert kwargs["max_norm"] == 1.0
