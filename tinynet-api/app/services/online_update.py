"""
Online learning service for TinyNet model updates.
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from ..ml.vectorizer import HashingVectorizer512
from ..ml.thresholds import get_category_threshold, get_state_threshold

logger = logging.getLogger(__name__)


class OnlineUpdateService:
    """Service for handling online learning updates."""
    
    def __init__(self, model, online_learner, vectorizer, category_labels, state_labels, next_step_labels):
        self.model = model
        self.online_learner = online_learner
        self.vectorizer = vectorizer
        self.category_labels = category_labels
        self.state_labels = state_labels
        self.next_step_labels = next_step_labels
        
        logger.info(f"OnlineUpdateService initialized with {len(category_labels)} categories, {len(state_labels)} states")
    
    def process_correction(self, 
                          text: str, 
                          categories: Optional[List[str]], 
                          state: Optional[str],
                          link_to: Optional[str] = None,
                          next_step_template: Optional[str] = None) -> Dict:
        """
        Process a correction and perform online learning update.
        
        Args:
            text: Input text
            categories: Correct category labels (can be None)
            state: Correct state label (can be None)
            link_to: Node ID to link to (optional)
            next_step_template: Correct next step template (optional)
            
        Returns:
            Update result dictionary
        """
        # Safety check: if no corrections provided, just return success
        if not categories and not state:
            logger.info("No corrections provided, skipping update")
            return {
                "ok": True,
                "message": "No corrections provided, no update performed",
                "update_result": None
            }
        
        try:
            # Vectorize the text
            x = self.vectorizer.encode(text)
            x_tensor = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
            
            # Perform online learning update
            if self.online_learner:
                update_result = self.online_learner.update_model(
                    text_vector=x_tensor,
                    categories_target=categories or [],
                    state_target=state or "continue",  # Default to continue if not specified
                    nextstep_target=next_step_template,
                    category_labels=self.category_labels,
                    state_labels=self.state_labels,
                    nextstep_labels=self.next_step_labels
                )
                
                # Update nodebank embedding if link_to is provided
                if link_to:
                    self._update_nodebank_embedding(link_to, text, x_tensor)
                
                logger.info(f"Online learning update completed: {update_result}")
                
                return {
                    "ok": True,
                    "message": "Classification corrected and model updated",
                    "update_result": update_result
                }
            else:
                logger.warning("Online learner not available")
                return {
                    "ok": True,
                    "message": "Classification corrected (online learning not available)",
                    "update_result": None
                }
                
        except Exception as e:
            logger.error(f"Error in online update: {e}")
            raise RuntimeError(f"Online update failed: {str(e)}")
    
    def _update_nodebank_embedding(self, node_id: str, title: str, text_tensor: torch.Tensor):
        """Update nodebank embedding for the linked node."""
        try:
            # Get hidden representation from the model
            with torch.no_grad():
                self.model.eval()
                h, _, _, _ = self.model(text_tensor)
                hidden_np = h.squeeze(0).cpu().numpy()
            
            # Try to update nodebank if available
            try:
                from ..ml.nodebank import NodeBank
                nodebank = NodeBank()
                nodebank.upsert_node_embedding(node_id, title, hidden_np)
                logger.info(f"Updated nodebank embedding for node {node_id}")
            except Exception as e:
                logger.warning(f"Failed to update nodebank: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to get hidden representation for nodebank update: {e}")
    
    def get_update_stats(self) -> Dict:
        """Get statistics about online learning updates."""
        if not self.online_learner:
            return {"status": "not_available"}
        
        return {
            "status": "available",
            "total_updates": self.online_learner.update_count,
            "save_every": self.online_learner.save_every,
            "learning_rate": self.online_learner.learning_rate,
            "last_save": self.online_learner.metrics.get("last_save")
        }
