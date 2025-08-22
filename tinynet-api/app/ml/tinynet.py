"""
TinyNet PyTorch Model
Shared trunk + 3 heads for multi-task learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import yaml
from pathlib import Path


class TinyNet(nn.Module):
    """
    TinyNet: Shared trunk + 3 heads for multi-task learning
    
    Architecture:
    - Input: 512-d float vector (from HashingVectorizer512)
    - Trunk: Linear(512->64) + ReLU + Dropout(0.1) + Linear(64->32) + ReLU
    - Heads:
      * categories: Linear(32 -> K) with K from config.labels.yaml
      * state: Linear(32 -> 6) for 6 states
      * next_step: Linear(32 -> 12) for 12 templates
    """
    
    def __init__(self, config_path: str = "backend/config/labels.yaml"):
        """
        Initialize TinyNet model.
        
        Args:
            config_path: Path to labels.yaml configuration file
        """
        super(TinyNet, self).__init__()
        
        # Load configuration to get category count
        self.num_categories = self._load_category_count(config_path)
        self.num_states = 6  # start, continue, pause, end, blocked, idea
        self.num_next_steps = 12  # Number of next step templates
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.categories_head = nn.Linear(32, self.num_categories)
        self.state_head = nn.Linear(32, self.num_states)
        self.next_step_head = nn.Linear(32, self.num_next_steps)
        
        # Loss functions
        self.categories_loss = nn.BCEWithLogitsLoss()
        self.state_loss = nn.CrossEntropyLoss()
        self.next_step_loss = nn.CrossEntropyLoss()
        
        # Print parameter count for sanity check
        self._print_param_count()
    
    def _load_category_count(self, config_path: str) -> int:
        """
        Load category count from labels.yaml configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Number of categories
        """
        try:
            # Try relative path first
            config_file = Path(config_path)
            if not config_file.exists():
                # Try from project root
                config_file = Path(__file__).parent.parent.parent / config_path
            
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    categories = config.get('categories', [])
                    return len(categories)
            else:
                print(f"Warning: Config file not found at {config_path}, using default category count")
                return 20  # Default fallback
                
        except Exception as e:
            print(f"Warning: Error loading config, using default category count: {e}")
            return 20  # Default fallback
    
    def _print_param_count(self):
        """Print total parameter count for sanity check."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 TinyNet Parameter Count: {total_params:,}")
        
        # Check if within expected range (20k-80k)
        if 20000 <= total_params <= 80000:
            print(f"✅ Parameter count within expected range (20k-80k)")
        else:
            print(f"⚠️  Parameter count outside expected range (20k-80k)")
        
        # Breakdown by component
        trunk_params = sum(p.numel() for p in self.trunk.parameters())
        categories_params = sum(p.numel() for p in self.categories_head.parameters())
        state_params = sum(p.numel() for p in self.state_head.parameters())
        next_step_params = sum(p.numel() for p in self.next_step_head.parameters())
        
        print(f"  Trunk: {trunk_params:,}")
        print(f"  Categories head: {categories_params:,}")
        print(f"  State head: {state_params:,}")
        print(f"  Next step head: {next_step_params:,}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through TinyNet.
        
        Args:
            x: Input tensor of shape (batch_size, 512)
            
        Returns:
            Tuple of (hidden, cat_logits, state_logits, nextstep_logits)
            - hidden: Shared representation (batch_size, 32)
            - cat_logits: Category logits (batch_size, num_categories)
            - state_logits: State logits (batch_size, num_states)
            - nextstep_logits: Next step logits (batch_size, num_next_steps)
        """
        # Input validation
        if x.dim() != 2 or x.size(1) != 512:
            raise ValueError(f"Expected input shape (batch_size, 512), got {x.shape}")
        
        # Shared trunk
        hidden = self.trunk(x)  # (batch_size, 32)
        
        # Task-specific heads
        cat_logits = self.categories_head(hidden)      # (batch_size, num_categories)
        state_logits = self.state_head(hidden)         # (batch_size, num_states)
        nextstep_logits = self.next_step_head(hidden)  # (batch_size, num_next_steps)
        
        return hidden, cat_logits, state_logits, nextstep_logits
    
    def compute_losses(self, 
                       cat_logits: torch.Tensor, 
                       state_logits: torch.Tensor, 
                       nextstep_logits: torch.Tensor,
                       cat_targets: torch.Tensor, 
                       state_targets: torch.Tensor, 
                       nextstep_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute losses for all tasks.
        
        Args:
            cat_logits: Category logits (batch_size, num_categories)
            state_logits: State logits (batch_size, num_states)
            nextstep_logits: Next step logits (batch_size, num_next_steps)
            cat_targets: Category targets (batch_size, num_categories) - binary
            state_targets: State targets (batch_size,) - class indices
            nextstep_targets: Next step targets (batch_size,) - class indices
            
        Returns:
            Dictionary with loss values for each task
        """
        # Categories: BCE loss for multi-label classification
        cat_loss = self.categories_loss(cat_logits, cat_targets.float())
        
        # State: Cross-entropy loss for single-label classification
        state_loss = self.state_loss(state_logits, state_targets)
        
        # Next step: Cross-entropy loss for single-label classification
        nextstep_loss = self.next_step_loss(nextstep_logits, nextstep_targets)
        
        # Total loss (simple sum, could be weighted)
        total_loss = cat_loss + state_loss + nextstep_loss
        
        return {
            'total': total_loss,
            'categories': cat_loss,
            'state': state_loss,
            'next_step': nextstep_loss
        }
    
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Make predictions for input data.
        
        Args:
            x: Input tensor of shape (batch_size, 512)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            hidden, cat_logits, state_logits, nextstep_logits = self.forward(x)
            
            # Convert logits to probabilities
            cat_probs = torch.sigmoid(cat_logits)  # Multi-label
            state_probs = F.softmax(state_logits, dim=1)  # Single-label
            nextstep_probs = F.softmax(nextstep_logits, dim=1)  # Single-label
            
            # Get predictions
            cat_preds = (cat_probs > 0.5).float()  # Threshold for multi-label
            state_preds = torch.argmax(state_probs, dim=1)
            nextstep_preds = torch.argmax(nextstep_probs, dim=1)
            
            return {
                'hidden': hidden,
                'categories': {
                    'logits': cat_logits,
                    'probabilities': cat_probs,
                    'predictions': cat_preds
                },
                'state': {
                    'logits': state_logits,
                    'probabilities': state_probs,
                    'predictions': state_preds
                },
                'next_step': {
                    'logits': nextstep_logits,
                    'probabilities': nextstep_probs,
                    'predictions': nextstep_preds
                }
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            'input_dim': 512,
            'hidden_dim': 32,
            'num_categories': self.num_categories,
            'num_states': self.num_states,
            'num_next_steps': self.num_next_steps,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
