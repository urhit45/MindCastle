"""
Tests for TinyNet PyTorch Model
"""

import pytest
import torch
import torch.nn as nn
from app.ml.tinynet import TinyNet


class TestTinyNet:
    """Test suite for TinyNet PyTorch model"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        self.batch_size = 4
        self.input_dim = 512
        
        # Create model
        self.model = TinyNet()
        
        # Create sample input
        self.sample_input = torch.randn(self.batch_size, self.input_dim)
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        assert isinstance(self.model, nn.Module)
        assert hasattr(self.model, 'trunk')
        assert hasattr(self.model, 'categories_head')
        assert hasattr(self.model, 'state_head')
        assert hasattr(self.model, 'next_step_head')
        
        # Check architecture components
        assert len(self.model.trunk) == 5  # Linear + ReLU + Dropout + Linear + ReLU
        assert isinstance(self.model.trunk[0], nn.Linear)  # First Linear layer
        assert isinstance(self.model.trunk[1], nn.ReLU)    # First ReLU
        assert isinstance(self.model.trunk[2], nn.Dropout)  # Dropout
        assert isinstance(self.model.trunk[3], nn.Linear)   # Second Linear layer
        assert isinstance(self.model.trunk[4], nn.ReLU)    # Second ReLU
    
    def test_forward_pass_shapes(self):
        """Test that forward pass returns correct shapes"""
        hidden, cat_logits, state_logits, nextstep_logits = self.model.forward(self.sample_input)
        
        # Check hidden representation shape
        assert hidden.shape == (self.batch_size, 32)
        
        # Check category logits shape
        assert cat_logits.shape == (self.batch_size, self.model.num_categories)
        
        # Check state logits shape
        assert state_logits.shape == (self.batch_size, 6)  # 6 states
        
        # Check next step logits shape
        assert nextstep_logits.shape == (self.batch_size, 12)  # 12 templates
    
    def test_parameter_count_range(self):
        """Test that parameter count is within expected range (20k-80k)"""
        total_params = sum(p.numel() for p in self.model.parameters())
        
        assert 20000 <= total_params <= 80000, f"Parameter count {total_params} outside expected range (20k-80k)"
        
        # Print actual count for reference
        print(f"Actual parameter count: {total_params:,}")
    
    def test_parameter_breakdown(self):
        """Test parameter breakdown by component"""
        trunk_params = sum(p.numel() for p in self.model.trunk.parameters())
        categories_params = sum(p.numel() for p in self.model.categories_head.parameters())
        state_params = sum(p.numel() for p in self.model.state_head.parameters())
        next_step_params = sum(p.numel() for p in self.model.next_step_head.parameters())
        
        total_params = trunk_params + categories_params + state_params + next_step_params
        
        # Verify breakdown matches total
        assert total_params == sum(p.numel() for p in self.model.parameters())
        
        # Check individual component parameter counts
        assert trunk_params > 0, "Trunk should have parameters"
        assert categories_params > 0, "Categories head should have parameters"
        assert state_params > 0, "State head should have parameters"
        assert next_step_params > 0, "Next step head should have parameters"
    
    def test_input_validation(self):
        """Test input shape validation"""
        # Test wrong input dimension
        wrong_input = torch.randn(self.batch_size, 256)  # Wrong dimension
        with pytest.raises(ValueError, match="Expected input shape"):
            self.model.forward(wrong_input)
        
        # Test wrong batch dimension
        wrong_batch = torch.randn(2, 3, self.input_dim)  # 3D tensor
        with pytest.raises(ValueError, match="Expected input shape"):
            self.model.forward(wrong_batch)
    
    def test_deterministic_eval_mode(self):
        """Test that eval() mode produces deterministic outputs for same input"""
        # Set to evaluation mode
        self.model.eval()
        
        # First forward pass
        hidden1, cat_logits1, state_logits1, nextstep_logits1 = self.model.forward(self.sample_input)
        
        # Second forward pass with same input
        hidden2, cat_logits2, state_logits2, nextstep_logits2 = self.model.forward(self.sample_input)
        
        # All outputs should be identical in eval mode
        torch.testing.assert_close(hidden1, hidden2)
        torch.testing.assert_close(cat_logits1, cat_logits2)
        torch.testing.assert_close(state_logits1, state_logits2)
        torch.testing.assert_close(nextstep_logits1, nextstep_logits2)
    
    def test_training_mode_variability(self):
        """Test that training mode produces different outputs due to dropout"""
        # Set to training mode
        self.model.train()
        
        # First forward pass
        hidden1, cat_logits1, state_logits1, nextstep_logits1 = self.model.forward(self.sample_input)
        
        # Second forward pass with same input
        hidden2, cat_logits2, state_logits2, nextstep_logits2 = self.model.forward(self.sample_input)
        
        # In training mode, outputs should be different due to dropout
        # (Note: this test might occasionally fail due to randomness, but should generally pass)
        hidden_different = not torch.allclose(hidden1, hidden2, atol=1e-6)
        cat_different = not torch.allclose(cat_logits1, cat_logits2, atol=1e-6)
        state_different = not torch.allclose(state_logits1, state_logits2, atol=1e-6)
        nextstep_different = not torch.allclose(nextstep_logits1, nextstep_logits2, atol=1e-6)
        
        # At least some outputs should be different
        assert any([hidden_different, cat_different, state_different, nextstep_different]), \
            "Training mode should produce different outputs due to dropout"
    
    def test_loss_computation(self):
        """Test loss computation for all tasks"""
        # Create sample targets
        cat_targets = torch.randint(0, 2, (self.batch_size, self.model.num_categories)).float()
        state_targets = torch.randint(0, 6, (self.batch_size,))
        nextstep_targets = torch.randint(0, 12, (self.batch_size,))
        
        # Forward pass
        hidden, cat_logits, state_logits, nextstep_logits = self.model.forward(self.sample_input)
        
        # Compute losses
        losses = self.model.compute_losses(
            cat_logits, state_logits, nextstep_logits,
            cat_targets, state_targets, nextstep_targets
        )
        
        # Check that all loss components are present
        assert 'total' in losses
        assert 'categories' in losses
        assert 'state' in losses
        assert 'next_step' in losses
        
        # Check that all losses are scalar tensors
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.dim() == 0, f"{loss_name} loss should be scalar"
            assert loss_value.item() > 0, f"{loss_name} loss should be positive"
        
        # Check that total loss is sum of individual losses
        expected_total = losses['categories'] + losses['state'] + losses['next_step']
        torch.testing.assert_close(losses['total'], expected_total)
    
    def test_predict_method(self):
        """Test the predict method"""
        # Set to evaluation mode
        self.model.eval()
        
        # Make predictions
        predictions = self.model.predict(self.sample_input)
        
        # Check structure
        assert 'hidden' in predictions
        assert 'categories' in predictions
        assert 'state' in predictions
        assert 'next_step' in predictions
        
        # Check hidden representation
        assert predictions['hidden'].shape == (self.batch_size, 32)
        
        # Check categories predictions
        cat_preds = predictions['categories']
        assert 'logits' in cat_preds
        assert 'probabilities' in cat_preds
        assert 'predictions' in cat_preds
        assert cat_preds['logits'].shape == (self.batch_size, self.model.num_categories)
        assert cat_preds['probabilities'].shape == (self.batch_size, self.model.num_categories)
        assert cat_preds['predictions'].shape == (self.batch_size, self.model.num_categories)
        
        # Check state predictions
        state_preds = predictions['state']
        assert 'logits' in state_preds
        assert 'probabilities' in state_preds
        assert 'predictions' in state_preds
        assert state_preds['logits'].shape == (self.batch_size, 6)
        assert state_preds['probabilities'].shape == (self.batch_size, 6)
        assert state_preds['predictions'].shape == (self.batch_size,)
        
        # Check next step predictions
        nextstep_preds = predictions['next_step']
        assert 'logits' in nextstep_preds
        assert 'probabilities' in nextstep_preds
        assert 'predictions' in nextstep_preds
        assert nextstep_preds['logits'].shape == (self.batch_size, 12)
        assert nextstep_preds['probabilities'].shape == (self.batch_size, 12)
        assert nextstep_preds['predictions'].shape == (self.batch_size,)
        
        # Check probability ranges
        for task_name in ['categories', 'state', 'next_step']:
            probs = predictions[task_name]['probabilities']
            assert torch.all(probs >= 0) and torch.all(probs <= 1), f"{task_name} probabilities should be in [0,1]"
    
    def test_model_info(self):
        """Test the get_model_info method"""
        info = self.model.get_model_info()
        
        expected_keys = ['input_dim', 'hidden_dim', 'num_categories', 'num_states', 
                        'num_next_steps', 'total_params', 'trainable_params']
        
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
        
        # Check values
        assert info['input_dim'] == 512
        assert info['hidden_dim'] == 32
        assert info['num_categories'] == self.model.num_categories
        assert info['num_states'] == 6
        assert info['num_next_steps'] == 12
        assert info['total_params'] > 0
        assert info['trainable_params'] > 0
        assert info['total_params'] == info['trainable_params']  # All params should be trainable
    
    def test_different_batch_sizes(self):
        """Test that model works with different batch sizes"""
        batch_sizes = [1, 2, 8, 16]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, self.input_dim)
            hidden, cat_logits, state_logits, nextstep_logits = self.model.forward(input_tensor)
            
            # Check shapes
            assert hidden.shape == (batch_size, 32)
            assert cat_logits.shape == (batch_size, self.model.num_categories)
            assert state_logits.shape == (batch_size, 6)
            assert nextstep_logits.shape == (batch_size, 12)
    
    def test_gradient_flow(self):
        """Test that gradients can flow through the model"""
        # Set to training mode
        self.model.train()
        
        # Forward pass
        hidden, cat_logits, state_logits, nextstep_logits = self.model.forward(self.sample_input)
        
        # Create dummy targets
        cat_targets = torch.randint(0, 2, (self.batch_size, self.model.num_categories)).float()
        state_targets = torch.randint(0, 6, (self.batch_size,))
        nextstep_targets = torch.randint(0, 12, (self.batch_size,))
        
        # Compute loss
        losses = self.model.compute_losses(
            cat_logits, state_logits, nextstep_logits,
            cat_targets, state_targets, nextstep_targets
        )
        
        # Backward pass
        total_loss = losses['total']
        total_loss.backward()
        
        # Check that gradients exist for key parameters
        for name, param in self.model.named_parameters():
            assert param.grad is not None, f"Gradient missing for {name}"
            assert param.grad.shape == param.shape, f"Gradient shape mismatch for {name}"
    
    def test_model_save_load(self):
        """Test that model can be saved and loaded"""
        import tempfile
        import os
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
            torch.save(self.model.state_dict(), tmp_file.name)
            save_path = tmp_file.name
        
        try:
            # Load model
            new_model = TinyNet()
            new_model.load_state_dict(torch.load(save_path))
            
            # Test that loaded model produces same output
            self.model.eval()
            new_model.eval()
            
            with torch.no_grad():
                hidden1, cat_logits1, state_logits1, nextstep_logits1 = self.model.forward(self.sample_input)
                hidden2, cat_logits2, state_logits2, nextstep_logits2 = new_model.forward(self.sample_input)
                
                # All outputs should be identical
                torch.testing.assert_close(hidden1, hidden2)
                torch.testing.assert_close(cat_logits1, cat_logits2)
                torch.testing.assert_close(state_logits1, state_logits2)
                torch.testing.assert_close(nextstep_logits1, nextstep_logits2)
        
        finally:
            # Clean up
            if os.path.exists(save_path):
                os.unlink(save_path)
