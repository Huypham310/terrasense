from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.deps import get_classifier, get_db
from app.inference import Classifier
from app.models import Prediction
from app.schemas import Health, LabelCount, ModelInfo, Stats

router = APIRouter(tags=["meta"])


@router.get("/health", response_model=Health)
def health(request: Request, db: Annotated[Session, Depends(get_db)]):
    """Liveness + readiness in one: is the model loaded and can we reach the database?"""
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    model_ok = getattr(request.app.state, "classifier", None) is not None
    return Health(status="ok" if db_ok and model_ok else "degraded", model_loaded=model_ok, database=db_ok)


@router.get("/model", response_model=ModelInfo)
def model_info(classifier: Annotated[Classifier, Depends(get_classifier)]):
    return ModelInfo(
        model_name=classifier.model_name,
        class_names=classifier.class_names,
        image_size=classifier.image_size,
        test_accuracy=classifier.test_accuracy,
    )


@router.get("/stats", response_model=Stats)
def stats(db: Annotated[Session, Depends(get_db)]):
    """Counts and average confidence per predicted label, for the dashboard."""
    rows = db.execute(
        select(Prediction.label, func.count(Prediction.id), func.avg(Prediction.confidence))
        .group_by(Prediction.label)
        .order_by(func.count(Prediction.id).desc())
    ).all()
    total = sum(count for _, count, _ in rows)
    avg_latency = db.scalar(select(func.avg(Prediction.latency_ms)))
    return Stats(
        total_predictions=total,
        by_label=[LabelCount(label=l, count=c, avg_confidence=float(a)) for l, c, a in rows],
        avg_latency_ms=float(avg_latency) if avg_latency is not None else None,
    )
