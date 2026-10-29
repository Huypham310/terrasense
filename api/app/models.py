from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Prediction(Base):
    """One classified image tile. Images themselves aren't stored, only the result."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    label: Mapped[str] = mapped_column(String(64), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    # top-k dạng [{"label": ..., "probability": ...}], JSON chạy được cả sqlite lẫn postgres
    top_k: Mapped[list[dict]] = mapped_column(JSON)
    model_name: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, index=True
    )
