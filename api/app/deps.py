from collections.abc import Iterator

from fastapi import HTTPException, Request
from sqlalchemy.orm import Session

from app.db import session_scope
from app.inference import Classifier


def get_db(request: Request) -> Iterator[Session]:
    yield from session_scope(request.app.state.session_factory)


def get_classifier(request: Request) -> Classifier:
    classifier = getattr(request.app.state, "classifier", None)
    if classifier is None:
        # model chưa export thì API vẫn chạy được, chỉ các route predict trả 503
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return classifier
