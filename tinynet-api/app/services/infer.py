"""
Inference service for TinyNet classification.
"""

import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
from datetime import datetime

from ..ml.vectorizer import HashingVectorizer512
from ..ml.tinynet import TinyNet
from ..ml.nodebank import NodeBank
from ..ml.thresholds import is_uncertain, determine_route

logger = logging.getLogger(__name__)

# Global instances (singleton-like)
_model: Optional[TinyNet] = None
_vectorizer: Optional[HashingVectorizer512] = None
_nodebank: Optional[NodeBank] = None
_category_labels: List[str] = []
_state_labels: List[str] = []
_next_step_labels: List[str] = []


def load_labels() -> Tuple[List[str], List[str], List[str]]:
    """Load labels from backend/config/labels.yaml."""
    try:
        config_path = Path("../backend/config/labels.yaml")
        if not config_path.exists():
            # Try alternative paths
            alt_paths = [
                Path(__file__).parent.parent.parent / "backend/config/labels.yaml",
                Path("backend/config/labels.yaml"),
                Path("../../backend/config/labels.yaml")
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    config_path = alt_path
                    break
            else:
                logger.warning("Config file not found, using default labels")
                return _get_default_labels()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            categories = config.get('categories', [])
            states = config.get('states', [])
            next_steps = config.get('next_step_templates', [])
            
        logger.info(f"Loaded labels: {len(categories)} categories, {len(states)} states, {len(next_steps)} next steps")
        return categories, states, next_steps
        
    except Exception as e:
        logger.error(f"Failed to load labels: {e}")
        return _get_default_labels()


def _get_default_labels() -> Tuple[List[str], List[str], List[str]]:
    """Get default labels if config file is not available."""
    categories = [
        "Fitness", "Running", "Strength", "Music", "Guitar", "Learning", "AI", 
        "Admin", "Finance", "Social", "Health", "Cooking", "Travel", "Work", 
        "SideProject", "Design", "Reading", "Writing", "Mindfulness", "Household"
    ]
    states = ["start", "continue", "pause", "end", "blocked", "idea"]
    next_steps = [
        "PracticeForDuration", "RepeatTask", "IncreaseVolume", "ScheduleFollowUp",
        "AttachLink", "ReviewNotes", "OutlineThreeBullets", "BookAppointment",
        "BuySupplies", "CreateSubtasks", "SetReminder", "LogReflection"
    ]
    return categories, states, next_steps


def load_model() -> TinyNet:
    """Load TinyNet model from checkpoint or initialize with random weights."""
    global _model
    
    if _model is not None:
        return _model
    
    try:
        # Try to load trained model
        model_path = Path("runs/exp1/best.pt")
        if not model_path.exists():
            # Try to find any best.pt file
            best_files = list(Path("runs").rglob("best.pt"))
            if best_files:
                model_path = best_files[0]
                logger.info(f"Using model from: {model_path}")
            else:
                raise FileNotFoundError("No trained model found")
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model = TinyNet()
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        
        logger.info(f"Loaded trained model from {model_path}")
        
    except Exception as e:
        logger.warning(f"Failed to load model checkpoint: {e}")
        logger.info("Initializing model with random weights (deterministic)")
        
        # Initialize with random weights (deterministic)
        torch.manual_seed(42)
        model = TinyNet()
        model.eval()
        
        logger.warning("Using randomly initialized model - predictions will not be meaningful")
    
    _model = model
    return model


def load_vectorizer() -> HashingVectorizer512:
    """Load HashingVectorizer512."""
    global _vectorizer
    
    if _vectorizer is not None:
        return _vectorizer
    
    _vectorizer = HashingVectorizer512(n_features=512, use_tfidf=False, seed=13)
    logger.info("Initialized HashingVectorizer512")
    return _vectorizer


def load_nodebank() -> Optional[NodeBank]:
    """Load NodeBank if available."""
    global _nodebank
    
    if _nodebank is not None:
        return _nodebank
    
    try:
        _nodebank = NodeBank()
        # Add sample nodes for testing
        _nodebank.add_sample_nodes()
        logger.info("Initialized NodeBank with sample nodes")
        return _nodebank
    except Exception as e:
        logger.warning(f"Failed to initialize NodeBank: {e}")
        return None


def initialize_inference_system():
    """Initialize all components for inference."""
    global _category_labels, _state_labels, _next_step_labels
    
    # Load labels
    _category_labels, _state_labels, _next_step_labels = load_labels()
    
    # Load model
    load_model()
    
    # Load vectorizer
    load_vectorizer()
    
    # Load nodebank (optional)
    load_nodebank()
    
    logger.info("Inference system initialized successfully")


def classify_text(text: str, context_node_id: Optional[str] = None) -> Dict:
    """
    Classify text using TinyNet model.
    
    Args:
        text: Input text to classify
        context_node_id: Optional context node ID
        
    Returns:
        Classification results matching OpenAPI schema
    """
    # Ensure system is initialized
    if _model is None or _vectorizer is None:
        initialize_inference_system()
    
    try:
        # Vectorize text
        x = _vectorizer.encode(text, ts=int(datetime.now().timestamp()))
        x_tensor = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            h, cat_logits, state_logits, ns_logits = _model(x_tensor)
        
        # Process categories (multi-label)
        cat_probs = torch.sigmoid(cat_logits).squeeze(0)
        category_results = []
        for i, prob in enumerate(cat_probs):
            if prob >= 0.5:  # TAU_CAT threshold
                category_results.append({
                    "label": _category_labels[i],
                    "score": float(prob)
                })
        
        # Sort categories by score (descending)
        category_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Process state (single-label)
        state_probs = torch.softmax(state_logits, dim=1).squeeze(0)
        state_idx = torch.argmax(state_probs).item()
        state_result = {
            "label": _state_labels[state_idx],
            "score": float(state_probs[state_idx])
        }
        
        # Process next step (single-label)
        nextstep_probs = torch.softmax(ns_logits, dim=1).squeeze(0)
        nextstep_idx = torch.argmax(nextstep_probs).item()
        nextstep_result = {
            "template": _next_step_labels[nextstep_idx],
            "slots": {},  # Placeholder for future enhancement
            "confidence": float(nextstep_probs[nextstep_idx])
        }
        
        # Get link hints using hidden representation
        link_hints = []
        if _nodebank is not None:
            try:
                hidden_np = h.squeeze(0).cpu().numpy()
                similar_nodes = _nodebank.topk_similar(hidden_np, k=3)
                
                for node_id, title, similarity in similar_nodes:
                    link_hints.append({
                        "nodeId": node_id,
                        "title": title,
                        "similarity": float(similarity)
                    })
            except Exception as e:
                logger.warning(f"Failed to get link hints: {e}")
        
        # Determine uncertainty
        max_cat_prob = max(cat_probs) if len(cat_probs) > 0 else 0.0
        top_cat_probs = sorted(cat_probs, reverse=True)[:2]
        top_diff = top_cat_probs[0] - top_cat_probs[1] if len(top_cat_probs) > 1 else 1.0
        
        uncertain = is_uncertain(
            cat_probs=cat_probs,
            margin=top_diff,
            state_conf=state_result["score"]
        )
        
        # Determine routing hint for UI
        route = determine_route(uncertain, state_result["label"])
        
        return {
            "categories": category_results,
            "state": state_result,
            "linkHints": link_hints,
            "nextStep": nextstep_result,
            "uncertain": uncertain,
            "route": route
        }
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise RuntimeError(f"Classification failed: {str(e)}")


def get_model_info() -> Dict:
    """Get information about the loaded model."""
    if _model is None:
        return {"status": "not_loaded"}
    
    return {
        "status": "loaded",
        "total_params": sum(p.numel() for p in _model.parameters()),
        "category_count": len(_category_labels),
        "state_count": len(_state_labels),
        "next_step_count": len(_next_step_labels),
        "vectorizer_ready": _vectorizer is not None,
        "nodebank_ready": _nodebank is not None
    }
