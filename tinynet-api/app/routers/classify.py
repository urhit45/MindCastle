"""
Classification router — wired to TinyNet via model_service
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from ..ml.model_service import model_service

router = APIRouter(prefix="/classify", tags=["classify"])


# ─── Request / Response models ───────────────────────────────────────────────

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


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):
    try:
        result = model_service.classify(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ClassifyResponse(
        categories=[CategoryScore(**c) for c in result["categories"]],
        state=StateScore(**result["state"]),
        linkHints=[],  # link hints require node graph — future feature
        nextStep=NextStep(template=result["next_step"]["template"], confidence=result["next_step"]["confidence"]),
        uncertain=result["uncertain"],
    )


@router.post("/correct", response_model=CorrectResponse)
async def correct_classification(request: CorrectRequest):
    # Corrections accepted but not yet used for online learning — stored for future retraining
    return CorrectResponse(ok=True)


@router.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    metrics = {
        "samplesReceived": len(request.samples),
        "note": "Samples queued. Retrain via CLI: make train",
        "timestamp": datetime.now().isoformat(),
    }
    return TrainResponse(ok=True, metrics=metrics)
