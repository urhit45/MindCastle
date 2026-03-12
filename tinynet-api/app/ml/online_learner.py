"""
Online learning utilities for TinyNet model updates.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class OnlineLearner:
    """Handles online learning updates for TinyNet model."""
    
    def __init__(self, model: nn.Module, save_dir: str = "runs/online", 
                 save_every: int = 10, learning_rate: float = 1e-4):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.learning_rate = learning_rate
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer with very low learning rate for safety
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # Loss functions
        self.categories_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.state_loss = nn.CrossEntropyLoss(reduction='mean')
        self.nextstep_loss = nn.CrossEntropyLoss(reduction='mean')
        
        # Track update count
        self.update_count = 0
        self.metrics_file = self.save_dir / "online_metrics.json"
        
        # Load existing metrics if available
        self.metrics = self._load_metrics()
        
        logger.info(f"OnlineLearner initialized with LR={learning_rate}, save_every={save_every}")
    
    def _load_metrics(self) -> Dict:
        """Load existing online learning metrics."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
        return {
            "total_updates": 0,
            "last_save": None,
            "loss_history": [],
            "accuracy_history": []
        }
    
    def _save_metrics(self):
        """Save current metrics."""
        self.metrics["total_updates"] = self.update_count
        self.metrics["last_save"] = datetime.now().isoformat()
        
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def update_model(self, 
                    text_vector: torch.Tensor,
                    categories_target: List[str],
                    state_target: str,
                    nextstep_target: Optional[str] = None,
                    category_labels: List[str] = None,
                    state_labels: List[str] = None,
                    nextstep_labels: List[str] = None) -> Dict:
        """
        Perform a single online learning update.
        
        Args:
            text_vector: 512-d input vector
            categories_target: List of correct category names
            state_target: Correct state name
            nextstep_target: Correct next step template name (optional)
            category_labels: Available category labels
            state_labels: Available state labels
            nextstep_labels: Available next step labels
        """
        if category_labels is None or state_labels is None:
            raise ValueError("Category and state labels must be provided")
        
        # Convert targets to indices
        category_indices = [category_labels.index(cat) for cat in categories_target if cat in category_labels]
        state_index = state_labels.index(state_target) if state_target in state_labels else 0
        
        # Convert to tensors
        category_target = torch.zeros(len(category_labels), dtype=torch.float32)
        category_target[category_indices] = 1.0
        
        state_target_tensor = torch.tensor([state_index], dtype=torch.long)
        
        if nextstep_target and nextstep_labels:
            nextstep_index = nextstep_labels.index(nextstep_target) if nextstep_target in nextstep_labels else 0
            nextstep_target_tensor = torch.tensor([nextstep_index], dtype=torch.long)
        else:
            nextstep_target_tensor = torch.tensor([0], dtype=torch.long)  # Default to first template
        
        # Ensure input is the right shape
        if text_vector.dim() == 1:
            text_vector = text_vector.unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        self.model.train()
        hidden, cat_logits, state_logits, nextstep_logits = self.model(text_vector)
        
        # Compute losses
        cat_loss = self.categories_loss(cat_logits, category_target.unsqueeze(0))
        state_loss = self.state_loss(state_logits, state_target_tensor)
        nextstep_loss = self.nextstep_loss(nextstep_logits, nextstep_target_tensor)
        
        total_loss = cat_loss + state_loss + nextstep_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping for safety
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        # Track metrics
        self.update_count += 1
        
        # Compute accuracy metrics
        with torch.no_grad():
            cat_probs = torch.sigmoid(cat_logits)
            cat_pred = (cat_probs > 0.5).float()
            cat_accuracy = (cat_pred == category_target.unsqueeze(0)).float().mean().item()
            
            state_pred = torch.argmax(state_logits, dim=1)
            state_accuracy = (state_pred == state_target_tensor).float().mean().item()
            
            nextstep_pred = torch.argmax(nextstep_logits, dim=1)
            nextstep_accuracy = (nextstep_pred == nextstep_target_tensor).float().mean().item()
        
        # Store metrics
        self.metrics["loss_history"].append({
            "update": self.update_count,
            "total_loss": total_loss.item(),
            "cat_loss": cat_loss.item(),
            "state_loss": state_loss.item(),
            "nextstep_loss": nextstep_loss.item(),
            "cat_accuracy": cat_accuracy,
            "state_accuracy": state_accuracy,
            "nextstep_accuracy": nextstep_accuracy,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 entries to prevent memory bloat
        if len(self.metrics["loss_history"]) > 100:
            self.metrics["loss_history"] = self.metrics["loss_history"][-100:]
        
        # Save model periodically
        if self.update_count % self.save_every == 0:
            self._save_checkpoint()
            self._save_metrics()
        
        # Return to eval mode
        self.model.eval()
        
        return {
            "update_count": self.update_count,
            "total_loss": total_loss.item(),
            "cat_accuracy": cat_accuracy,
            "state_accuracy": state_accuracy,
            "nextstep_accuracy": nextstep_accuracy,
            "saved": self.update_count % self.save_every == 0
        }
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "timestamp": datetime.now().isoformat(),
            "learning_rate": self.learning_rate
        }
        
        checkpoint_path = self.save_dir / f"online_checkpoint_{self.update_count}.pt"
        backup_path = self.save_dir / "online_latest.pt"
        
        try:
            # Save numbered checkpoint
            torch.save(checkpoint, checkpoint_path)
            
            # Save latest checkpoint
            torch.save(checkpoint, backup_path)
            
            # Keep only last 5 checkpoints to save disk space
            checkpoints = sorted(self.save_dir.glob("online_checkpoint_*.pt"))
            if len(checkpoints) > 5:
                for old_checkpoint in checkpoints[:-5]:
                    old_checkpoint.unlink()
            
            logger.info(f"Saved online checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def get_metrics_summary(self) -> Dict:
        """Get summary of online learning metrics."""
        if not self.metrics["loss_history"]:
            return {"message": "No updates performed yet"}
        
        recent = self.metrics["loss_history"][-10:]  # Last 10 updates
        
        return {
            "total_updates": self.update_count,
            "last_update": self.metrics["loss_history"][-1]["timestamp"] if self.metrics["loss_history"] else None,
            "recent_avg_loss": sum(entry["total_loss"] for entry in recent) / len(recent),
            "recent_avg_cat_accuracy": sum(entry["cat_accuracy"] for entry in recent) / len(recent),
            "recent_avg_state_accuracy": sum(entry["state_accuracy"] for entry in recent) / len(recent),
            "learning_rate": self.learning_rate,
            "save_every": self.save_every
        }
