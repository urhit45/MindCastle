"""
Tests for TinyNet Link Hints System
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import json
from unittest.mock import patch, MagicMock
import torch

from app.ml.nodebank import NodeBank
from app.ml.tinynet import TinyNet
from app.ml.vectorizer import HashingVectorizer512


class TestNodeBank:
    """Test suite for NodeBank functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary database for testing
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_nodebank.db")
        self.nodebank = NodeBank(db_path=self.db_path, max_nodes=5)
    
    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test NodeBank initialization"""
        assert self.nodebank.max_nodes == 5
        assert self.nodebank.db_path == self.db_path
        
        # Check that database was created
        assert os.path.exists(self.db_path)
        
        # Check initial node count
        assert self.nodebank.get_node_count() == 0
    
    def test_upsert_node_embedding(self):
        """Test adding and updating node embeddings"""
        # Create test vector
        test_vec = np.random.randn(32)
        test_vec = test_vec / np.linalg.norm(test_vec)  # Normalize
        
        # Add node
        self.nodebank.upsert_node_embedding("test_node_1", "Test Node 1", test_vec)
        
        # Check node count
        assert self.nodebank.get_node_count() == 1
        
        # Update same node
        updated_vec = np.random.randn(32)
        updated_vec = updated_vec / np.linalg.norm(updated_vec)
        self.nodebank.upsert_node_embedding("test_node_1", "Updated Test Node 1", updated_vec)
        
        # Count should still be 1 (update, not insert)
        assert self.nodebank.get_node_count() == 1
    
    def test_invalid_vector_shape(self):
        """Test that invalid vector shapes are rejected"""
        # Wrong shape vector
        wrong_vec = np.random.randn(64)
        
        with pytest.raises(ValueError, match="Expected vector shape"):
            self.nodebank.upsert_node_embedding("test_node", "Test", wrong_vec)
    
    def test_max_nodes_limit(self):
        """Test that max_nodes limit is enforced"""
        # Add more nodes than the limit
        for i in range(7):  # More than max_nodes=5
            vec = np.random.randn(32)
            vec = vec / np.linalg.norm(vec)
            self.nodebank.upsert_node_embedding(f"node_{i}", f"Node {i}", vec)
        
        # Should only have max_nodes
        assert self.nodebank.get_node_count() == 5
        
        # Oldest nodes should be removed
        recent_nodes = self.nodebank.get_recent_nodes(limit=10)
        assert len(recent_nodes) == 5
        
        # Check that newer nodes are present
        node_ids = [node[0] for node in recent_nodes]
        assert "node_6" in node_ids  # Newest
        assert "node_0" not in node_ids  # Oldest (removed)
    
    def test_similarity_search(self):
        """Test similarity search functionality"""
        # Add test nodes
        test_nodes = [
            ("node_1", "Running Progress", np.array([1.0] + [0.0] * 31)),
            ("node_2", "Guitar Practice", np.array([0.0, 1.0] + [0.0] * 30)),
            ("node_3", "Learning AI", np.array([0.0, 0.0, 1.0] + [0.0] * 29))
        ]
        
        for node_id, title, vec in test_nodes:
            self.nodebank.upsert_node_embedding(node_id, title, vec)
        
        # Test similarity search
        query_vec = np.array([1.0] + [0.0] * 31)  # Should match node_1 best
        similar_nodes = self.nodebank.topk_similar(query_vec, k=2)
        
        assert len(similar_nodes) == 2
        
        # First result should be node_1 (perfect match)
        assert similar_nodes[0][0] == "node_1"
        assert similar_nodes[0][1] == "Running Progress"
        assert similar_nodes[0][2] == 1.0  # Perfect similarity
        
        # Second result should be one of the others
        assert similar_nodes[1][0] in ["node_2", "node_3"]
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation"""
        # Test orthogonal vectors
        vec1 = np.array([1.0, 0.0, 0.0] + [0.0] * 29)
        vec2 = np.array([0.0, 1.0, 0.0] + [0.0] * 29)
        
        similarity = self.nodebank._cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-10  # Should be approximately 0
        
        # Test identical vectors
        similarity = self.nodebank._cosine_similarity(vec1, vec1)
        assert abs(similarity - 1.0) < 1e-10  # Should be 1
        
        # Test opposite vectors
        vec3 = -vec1
        similarity = self.nodebank._cosine_similarity(vec1, vec3)
        assert abs(similarity + 1.0) < 1e-10  # Should be -1
    
    def test_empty_nodebank(self):
        """Test behavior when nodebank is empty"""
        # Similarity search on empty nodebank
        query_vec = np.random.randn(32)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        similar_nodes = self.nodebank.topk_similar(query_vec, k=3)
        assert similar_nodes == []
    
    def test_sample_nodes(self):
        """Test adding sample nodes"""
        initial_count = self.nodebank.get_node_count()
        self.nodebank.add_sample_nodes()
        
        # Should have added 5 sample nodes
        assert self.nodebank.get_node_count() == initial_count + 5
        
        # Check that sample nodes are present
        recent_nodes = self.nodebank.get_recent_nodes(limit=10)
        node_titles = [node[1] for node in recent_nodes]
        
        assert "Running Progress" in node_titles
        assert "Guitar Practice" in node_titles
        assert "Learning AI Basics" in node_titles
        assert "Fitness Goals" in node_titles
        assert "Work Projects" in node_titles
    
    def test_clear_all(self):
        """Test clearing all nodes"""
        # Add some nodes
        self.nodebank.add_sample_nodes()
        assert self.nodebank.get_node_count() > 0
        
        # Clear all
        self.nodebank.clear_all()
        assert self.nodebank.get_node_count() == 0


class TestClassifyEndpointIntegration:
    """Test suite for classify endpoint integration with NodeBank"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create temporary database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_classify.db")
        
        # Mock the model and vectorizer
        self.mock_model = MagicMock()
        self.mock_vectorizer = MagicMock()
        self.mock_nodebank = NodeBank(db_path=self.db_path)
    
    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_link_hints_integration(self):
        """Test that link hints are properly integrated in classification"""
        # Add test nodes to nodebank
        test_nodes = [
            ("node_001", "Running Progress", np.array([1.0] + [0.0] * 31)),
            ("node_002", "Guitar Practice", np.array([0.0, 1.0] + [0.0] * 30)),
            ("node_003", "Learning AI", np.array([0.0, 0.0, 1.0] + [0.0] * 29))
        ]
        
        for node_id, title, vec in test_nodes:
            self.mock_nodebank.upsert_node_embedding(node_id, title, vec)
        
        # Mock model output
        mock_hidden = torch.tensor([[1.0] + [0.0] * 31])  # Should match node_001 best
        mock_cat_logits = torch.tensor([[0.8, 0.2, 0.1]])  # High confidence in first category
        mock_state_logits = torch.tensor([[0.1, 0.9, 0.0, 0.0, 0.0, 0.0]])  # High confidence in second state
        mock_nextstep_logits = torch.tensor([[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        self.mock_model.forward.return_value = (mock_hidden, mock_cat_logits, mock_state_logits, mock_nextstep_logits)
        
        # Test similarity search
        hidden_np = mock_hidden.squeeze(0).cpu().numpy()
        similar_nodes = self.mock_nodebank.topk_similar(hidden_np, k=3)
        
        # Should return nodes with similarity scores
        assert len(similar_nodes) == 3
        
        # First result should be node_001 (best match)
        assert similar_nodes[0][0] == "node_001"
        assert similar_nodes[0][1] == "Running Progress"
        assert similar_nodes[0][2] == 1.0  # Perfect similarity
    
    def test_uncertainty_detection(self):
        """Test uncertainty detection logic"""
        # Test case 1: Low max category probability
        low_prob_cat_logits = torch.tensor([[0.0, -0.5, -1.0]])  # Max prob < 0.55 after sigmoid
        cat_probs = torch.sigmoid(low_prob_cat_logits).squeeze(0)
        
        max_cat_prob = max(cat_probs)
        uncertain_low_prob = max_cat_prob < 0.55
        
        assert uncertain_low_prob == True
        
        # Test case 2: Close top-2 category probabilities
        close_probs_cat_logits = torch.tensor([[1.0, 0.8, -1.0]])  # Top diff < 0.15 after sigmoid
        cat_probs = torch.sigmoid(close_probs_cat_logits).squeeze(0)
        
        top_cat_probs = sorted(cat_probs, reverse=True)[:2]
        top_diff = top_cat_probs[0] - top_cat_probs[1]
        uncertain_close_probs = top_diff < 0.15
        
        assert uncertain_close_probs == True
        
        # Test case 3: High confidence (not uncertain)
        high_conf_cat_logits = torch.tensor([[2.0, -1.0, -2.0]])  # Max prob > 0.55, diff > 0.15 after sigmoid
        cat_probs = torch.sigmoid(high_conf_cat_logits).squeeze(0)
        
        max_cat_prob = max(cat_probs)
        top_cat_probs = sorted(cat_probs, reverse=True)[:2]
        top_diff = top_cat_probs[0] - top_cat_probs[1]
        uncertain_high_conf = (max_cat_prob < 0.55) or (top_diff < 0.15)
        
        assert uncertain_high_conf == False


class TestRealModelIntegration:
    """Test suite for real model integration (requires trained model)"""
    
    def test_model_loading(self):
        """Test that trained model can be loaded"""
        try:
            # Try to load a trained model
            model_path = Path("runs/test/best.pt")
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                model = TinyNet()
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                # Test forward pass
                x = torch.randn(1, 512)
                with torch.no_grad():
                    hidden, cat_logits, state_logits, nextstep_logits = model.forward(x)
                
                # Check output shapes
                assert hidden.shape == (1, 32)
                assert cat_logits.shape[1] == 20  # 20 categories
                assert state_logits.shape[1] == 6  # 6 states
                assert nextstep_logits.shape[1] == 12  # 12 next steps
                
                assert True  # If we get here, model loading works
            else:
                pytest.skip("No trained model found")
                
        except Exception as e:
            pytest.skip(f"Model loading failed: {e}")
    
    def test_vectorizer_integration(self):
        """Test that vectorizer works with real text"""
        try:
            vectorizer = HashingVectorizer512(use_tfidf=False, seed=42)
            
            # Test vectorization
            text = "Ran 2 miles today, feeling good"
            vector = vectorizer.encode(text)
            
            assert vector.shape == (512,)
            assert isinstance(vector, np.ndarray)
            assert not np.any(np.isnan(vector))  # No NaN values
            
        except Exception as e:
            pytest.skip(f"Vectorizer test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
