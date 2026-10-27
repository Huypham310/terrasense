from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.deps import get_classifier, get_db
from app.inference import Classification, Classifier, InvalidImageError
from app.models import Prediction
from app.schemas import PredictionOut, PredictionPage

router = APIRouter(tags=["predictions"])

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/tiff", "image/webp"}


async def _read_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'",
        )
    return await file.read()


def _save(
    db: Session, filename: str, result: Classification, model_name: str, latency_ms: float
) -> Prediction:
    row = Prediction(
        filename=filename,
        label=result.label,
        confidence=result.confidence,
        top_k=[{"label": s.label, "probability": s.probability} for s in result.top_k],
        model_name=model_name,
        latency_ms=latency_ms,
    )
    db.add(row)
    return row


@router.post("/predict", response_model=PredictionOut, status_code=status.HTTP_201_CREATED)
async def predict(
    file: Annotated[UploadFile, File(description="Satellite tile (PNG/JPEG/TIFF)")],
    db: Annotated[Session, Depends(get_db)],
    classifier: Annotated[Classifier, Depends(get_classifier)],
):
    """Classify one image tile and store the result in the prediction history."""
    data = await _read_upload(file)
    try:
        (result,), latency = classifier.predict([data])
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    row = _save(db, file.filename or "upload", result, classifier.model_name, latency)
    db.commit()
    db.refresh(row)
    return row


@router.post("/predict/batch", response_model=list[PredictionOut], status_code=status.HTTP_201_CREATED)
async def predict_batch(
    files: Annotated[list[UploadFile], File(description="Up to max_batch_size tiles")],
    db: Annotated[Session, Depends(get_db)],
    classifier: Annotated[Classifier, Depends(get_classifier)],
):
    """Classify several tiles in one ONNX call. Latency is split evenly across the batch."""
    limit = get_settings().max_batch_size
    if len(files) > limit:
        raise HTTPException(status_code=422, detail=f"At most {limit} files per batch")
    blobs = [await _read_upload(f) for f in files]
    try:
        results, latency = classifier.predict(blobs)
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    per_image = latency / len(results)
    rows = [
        _save(db, f.filename or "upload", r, classifier.model_name, per_image)
        for f, r in zip(files, results, strict=True)
    ]
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


@router.get("/predictions", response_model=PredictionPage)
def list_predictions(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    label: str | None = None,
):
    """Prediction history, newest first, optionally filtered by predicted label."""
    query = select(Prediction)
    count_query = select(func.count(Prediction.id))
    if label:
        query = query.where(Prediction.label == label)
        count_query = count_query.where(Prediction.label == label)
    total = db.scalar(count_query) or 0
    # sort thêm theo id để thứ tự ổn định khi created_at trùng nhau
    items = db.scalars(
        query.order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit).offset(offset)
    ).all()
    return PredictionPage(items=items, total=total, limit=limit, offset=offset)


@router.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: int, db: Annotated[Session, Depends(get_db)]):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return row


@router.delete("/predictions/{prediction_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_prediction(prediction_id: int, db: Annotated[Session, Depends(get_db)]):
    row = db.get(Prediction, prediction_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(row)
    db.commit()
