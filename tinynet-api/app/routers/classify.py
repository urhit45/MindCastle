"""
Classification router for TinyNet API
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime

router = APIRouter(prefix="/classify", tags=["classify"])


# Request/Response models
class ClassifyRequest(BaseModel):
    text: str
    contextNodeId: Optional[str] = None


class CategoryScore(BaseModel):
    label: str
    score: float


class StateScore(BaseModel):
    label: str
    score: float


class LinkHint(BaseModel):
    nodeId: str
    title: str
    similarity: float


class NextStep(BaseModel):
    template: str
    slots: Optional[dict] = None
    confidence: float


class ClassifyResponse(BaseModel):
    categories: List[CategoryScore]
    state: StateScore
    linkHints: List[LinkHint]
    nextStep: NextStep
    uncertain: bool


class CorrectRequest(BaseModel):
    text: str
    categories: Optional[List[str]] = None
    state: Optional[str] = None
    linkTo: Optional[str] = None
    nextStepTemplate: Optional[str] = None


class CorrectResponse(BaseModel):
    ok: bool


class TrainingSample(BaseModel):
    text: str
    categories: List[str]
    state: str


class TrainRequest(BaseModel):
    samples: List[TrainingSample]


class TrainResponse(BaseModel):
    ok: bool
    metrics: Optional[dict] = None


@router.post("/", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):
    """
    Classify text into categories, states, and suggest next steps.
    
    This is a mock implementation that returns deterministic results.
    """
    text = request.text.lower()
    
    # Mock classification logic
    categories = []
    if any(word in text for word in ["run", "miles", "gym", "workout"]):
        categories.append(CategoryScore(label="Fitness", score=0.95))
    if any(word in text for word in ["run", "mile", "5k"]):
        categories.append(CategoryScore(label="Running", score=0.87))
    if any(word in text for word in ["guitar", "music", "practice"]):
        categories.append(CategoryScore(label="Music", score=0.92))
    if any(word in text for word in ["read", "chapter", "course"]):
        categories.append(CategoryScore(label="Learning", score=0.89))
    
    # Default category if none found
    if not categories:
        categories.append(CategoryScore(label="Misc", score=0.5))
    
    # Determine state
    if any(word in text for word in ["start", "begin", "new"]):
        state = StateScore(label="start", score=0.95)
    elif any(word in text for word in ["pain", "stuck", "blocked", "shin"]):
        state = StateScore(label="blocked", score=0.92)
    elif any(word in text for word in ["finish", "done", "complete"]):
        state = StateScore(label="end", score=0.88)
    else:
        state = StateScore(label="continue", score=0.85)
    
    # Mock link hints
    link_hints = [
        LinkHint(
            nodeId=str(uuid.uuid4()),
            title="Running Progress",
            similarity=0.78
        )
    ]
    
    # Mock next step
    next_step = NextStep(
        template="PracticeForDuration",
        slots={"duration": "30 minutes", "activity": "stretching"},
        confidence=0.82
    )
    
    return ClassifyResponse(
        categories=categories,
        state=state,
        linkHints=link_hints,
        nextStep=next_step,
        uncertain=False
    )


@router.post("/correct", response_model=CorrectResponse)
async def correct_classification(request: CorrectRequest):
    """
    Provide corrections for text classification.
    
    This is a mock implementation that accepts corrections.
    """
    # In a real implementation, this would store corrections for model training
    print(f"Correction received for: {request.text}")
    if request.categories:
        print(f"  Categories: {request.categories}")
    if request.state:
        print(f"  State: {request.state}")
    if request.linkTo:
        print(f"  Link to: {request.linkTo}")
    if request.nextStepTemplate:
        print(f"  Next step: {request.nextStepTemplate}")
    
    return CorrectResponse(ok=True)


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """
    Provide training samples to improve classification.
    
    This is a mock implementation that accepts training data.
    """
    # In a real implementation, this would update the model
    print(f"Training request received with {len(request.samples)} samples")
    
    # Mock metrics
    metrics = {
        "samplesProcessed": len(request.samples),
        "accuracy": 0.85,
        "timestamp": datetime.now().isoformat()
    }
    
    return TrainResponse(ok=True, metrics=metrics)
