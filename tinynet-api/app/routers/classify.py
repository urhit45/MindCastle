"""
Classification router for TinyNet
Real inference using trained TinyNet model
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import logging

from ..services.infer import classify_text, get_model_info
from ..services.online_update import OnlineUpdateService
from ..ml.online_learner import OnlineLearner

router = APIRouter(prefix="/classify", tags=["classify"])

# Global instances
online_learner: Optional[OnlineLearner] = None
online_update_service: Optional[OnlineUpdateService] = None


class ClassifyRequest(BaseModel):
    text: str
    contextNodeId: Optional[str] = None


class LabelScore(BaseModel):
    label: str
    score: float


class NextStep(BaseModel):
    template: str
    slots: Optional[dict] = None
    confidence: float


class ClassifyResponse(BaseModel):
    categories: List[LabelScore]
    state: LabelScore
    linkHints: List[dict]
    nextStep: NextStep
    uncertain: bool
    route: str = Field(..., description="UI routing hint: needs_confirm | suggest_plan | auto_save_ok")


@router.on_event("startup")
async def startup_event():
    """Initialize online learner and update service on startup."""
    global online_learner, online_update_service
    try:
        # First, initialize the inference system
        from ..services.infer import initialize_inference_system
        initialize_inference_system()
        logging.info("Inference system initialized during startup")
        
        # Get model info to verify initialization
        model_info = get_model_info()
        logging.info(f"Model info: {model_info}")
        
        # Initialize online learner if model is available
        if model_info.get("status") == "loaded":
            try:
                from ..ml.tinynet import TinyNet
                from ..services.infer import _model, _vectorizer, _category_labels, _state_labels, _next_step_labels
                
                # Check if all required components are available
                if _model is not None and _vectorizer is not None:
                    online_learner = OnlineLearner(
                        model=_model,
                        save_dir="runs/exp1",  # Save to main experiment directory
                        save_every=50,  # Save every 50 updates as requested
                        learning_rate=3e-4  # Small LR as requested
                    )
                    logging.info("Online learner initialized successfully")
                    
                    # Initialize online update service
                    online_update_service = OnlineUpdateService(
                        model=_model,
                        online_learner=online_learner,
                        vectorizer=_vectorizer,
                        category_labels=_category_labels or [],
                        state_labels=_state_labels or [],
                        next_step_labels=_next_step_labels or []
                    )
                    logging.info("Online update service initialized successfully")
                else:
                    logging.warning("Model or vectorizer not available, online learning disabled")
            except ImportError as e:
                logging.warning(f"Failed to import required modules: {e}")
                logging.warning("Online learning will be disabled")
        else:
            logging.warning("Model not loaded, online learning disabled")
            
    except Exception as e:
        logging.error(f"Failed to initialize online learning system: {e}")
        logging.warning("Online learning will be disabled")


@router.post("/", response_model=ClassifyResponse)
async def classify_text_endpoint(request: ClassifyRequest):
    """
    Classify text using trained TinyNet model.
    
    Args:
        request: Classification request with text and optional context
        
    Returns:
        Classification results with categories, state, link hints, and next step
    """
    try:
        # Use the inference service
        result = classify_text(request.text, request.contextNodeId)
        
        # Convert to Pydantic models
        categories = [LabelScore(label=cat["label"], score=cat["score"]) for cat in result["categories"]]
        state = LabelScore(label=result["state"]["label"], score=result["state"]["score"])
        next_step = NextStep(
            template=result["nextStep"]["template"],
            slots=result["nextStep"]["slots"],
            confidence=result["nextStep"]["confidence"]
        )
        
        return ClassifyResponse(
            categories=categories,
            state=state,
            linkHints=result["linkHints"],
            nextStep=next_step,
            uncertain=result["uncertain"],
            route=result["route"]
        )
        
    except Exception as e:
        logging.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/correct")
async def correct_classification(request: dict):
    """Correct a classification and perform online learning update."""
    try:
        # Validate request
        if not request.get("text"):
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Extract correction data
        text = request["text"]
        categories = request.get("categories")
        state = request.get("state")
        link_to = request.get("linkTo")
        next_step_template = request.get("nextStepTemplate")
        
        # Use the online update service
        if online_update_service:
            result = online_update_service.process_correction(
                text=text,
                categories=categories,
                state=state,
                link_to=link_to,
                next_step_template=next_step_template
            )
            return result
        else:
            return {
                "ok": True,
                "message": "Classification corrected (online learning not available)",
                "update_result": None
            }
            
    except Exception as e:
        logging.error(f"Error in correction endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Correction failed: {str(e)}")


@router.post("/train")
async def train_model():
    """Placeholder for training endpoint."""
    return {"ok": True, "message": "Training endpoint - to be implemented"}


@router.get("/metrics")
async def get_online_learning_metrics():
    """Get online learning metrics and progress."""
    if online_learner is None:
        raise HTTPException(status_code=500, detail="Online learner not initialized")
    
    try:
        metrics = online_learner.get_metrics_summary()
        return {
            "ok": True,
            "metrics": metrics
        }
    except Exception as e:
        logging.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")
