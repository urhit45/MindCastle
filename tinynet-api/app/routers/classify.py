"""
Classification router — wired to TinyNet via model_service.
"""
import asyncio
import logging
from datetime import datetime
from typing import Annotated, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..dependencies import get_request_id
from ..ml.model_service import model_service

log = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["classify"])

# ─── Shared type aliases ──────────────────────────────────────────────────────

ValidState = Literal[
    "start", "continue", "pause", "end", "idea",
    "active", "live", "concept", "blocked", "planned", "planning",
]

BoundedText = Annotated[str, Field(min_length=1, max_length=2000)]

# ─── Request / Response models ────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    text: BoundedText
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
    categories:   List[CategoryScore]
    state:        StateScore
    linkHints:    List[LinkHint]
    nextStep:     NextStep
    uncertain:    bool
    # Phase III — governance metadata
    decisionMode: str             = "normal"
    reasonCodes:  List[str]       = []
    driftAlert:   bool            = False
    inferenceMsec: Optional[float] = None


class CorrectRequest(BaseModel):
    text: BoundedText
    categories: Optional[List[str]] = None
    state: Optional[ValidState] = None
    linkTo: Optional[str] = None
    nextStepTemplate: Optional[Annotated[str, Field(max_length=500)]] = None


class CorrectResponse(BaseModel):
    ok: bool


class TrainingSample(BaseModel):
    text: BoundedText
    categories: List[str]
    state: ValidState


class TrainRequest(BaseModel):
    samples: Annotated[List[TrainingSample], Field(min_length=1, max_length=200)]


class TrainResponse(BaseModel):
    ok: bool
    metrics: Optional[dict] = None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/", response_model=ClassifyResponse)
async def classify_text(
    request: ClassifyRequest,
    request_id: str = Depends(get_request_id),
):
    log.info("classify_request request_id=%s text_len=%d", request_id, len(request.text))
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, model_service.classify, request.text),
            timeout=10.0,
        )
    except asyncio.TimeoutError:
        log.error("classify_timeout request_id=%s", request_id)
        raise HTTPException(status_code=503, detail="Classification service timed out.")
    except RuntimeError:
        log.error("classify_runtime_error request_id=%s", request_id, exc_info=True)
        raise HTTPException(status_code=503, detail="Classification service is temporarily unavailable.")

    log.info(
        "classify_ok request_id=%s state=%s decision=%s drift=%s ms=%.1f",
        request_id,
        result["state"]["label"],
        result.get("decision_mode", "normal"),
        result.get("drift_alert", False),
        result.get("inference_ms", 0),
    )
    return ClassifyResponse(
        categories=[CategoryScore(**c) for c in result["categories"]],
        state=StateScore(**result["state"]),
        linkHints=[],
        nextStep=NextStep(
            template=result["next_step"]["template"],
            confidence=result["next_step"]["confidence"],
        ),
        uncertain=result["uncertain"],
        decisionMode=result.get("decision_mode", "normal"),
        reasonCodes=result.get("reason_codes", []),
        driftAlert=result.get("drift_alert", False),
        inferenceMsec=result.get("inference_ms"),
    )


@router.post("/correct", response_model=CorrectResponse)
async def correct_classification(
    request: CorrectRequest,
    request_id: str = Depends(get_request_id),
):
    log.info("correct_request request_id=%s state=%s", request_id, request.state)
    return CorrectResponse(ok=True)


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    request_id: str = Depends(get_request_id),
):
    n = len(request.samples)
    log.info("train_request request_id=%s samples=%d", request_id, n)
    metrics = {
        "samplesReceived": n,
        "note": "Samples queued. Retrain via CLI: make train",
        "timestamp": datetime.now().isoformat(),
    }
    return TrainResponse(ok=True, metrics=metrics)
