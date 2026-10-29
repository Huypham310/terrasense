from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, field_validator


class ClassScoreOut(BaseModel):
    label: str
    probability: float


class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    label: str
    confidence: float
    top_k: list[ClassScoreOut]
    model_name: str
    latency_ms: float
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, value: datetime) -> datetime:
        # sqlite trả về datetime không có tzinfo -> browser hiểu nhầm là giờ local
        # values are always written as UTC, so tag them back before serializing
        return value if value.tzinfo else value.replace(tzinfo=UTC)


class PredictionPage(BaseModel):
    items: list[PredictionOut]
    total: int
    limit: int
    offset: int


class ModelInfo(BaseModel):
    model_name: str
    class_names: list[str]
    image_size: int
    test_accuracy: float | None


class LabelCount(BaseModel):
    label: str
    count: int
    avg_confidence: float


class Stats(BaseModel):
    total_predictions: int
    by_label: list[LabelCount]
    avg_latency_ms: float | None


class Health(BaseModel):
    status: str
    model_loaded: bool
    database: bool
