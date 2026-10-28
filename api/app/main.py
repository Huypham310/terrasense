import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.db import Base, make_engine, make_session_factory
from app.inference import Classifier
from app.routers import meta, predictions

logger = logging.getLogger("terrasense")


def create_app(settings: Settings | None = None) -> FastAPI:
    """App factory so tests can pass their own settings (temp db, fake model)."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        # chưa dùng alembic, project nhỏ nên create_all là đủ
        Base.metadata.create_all(engine)
        app.state.session_factory = make_session_factory(engine)
        app.state.classifier = None
        if settings.model_path.exists() and settings.meta_path.exists():
            app.state.classifier = Classifier(settings.model_path, settings.meta_path)
            logger.info("loaded model %s", app.state.classifier.model_name)
        else:
            logger.warning("model not found at %s, /predict will return 503", settings.model_path)
        yield
        engine.dispose()

    app = FastAPI(title="TerraSense API", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(meta.router)
    app.include_router(predictions.router)
    return app


app = create_app()
