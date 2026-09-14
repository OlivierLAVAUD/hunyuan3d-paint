"""Shared service singletons, exposed as FastAPI dependencies.

The engine and the store are expensive to build (the engine holds the GPU
weights), so they are created once behind ``lru_cache`` and injected instead of
being constructed inside the routes. Tests override them with
``app.dependency_overrides``.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Request

from .config import Settings, get_settings
from .services.paint import PaintingEngine
from .services.pipeline import TexturePipeline
from .storage import JobStore

__all__ = [
    "get_app_settings",
    "get_engine",
    "get_job_semaphore",
    "get_pipeline",
    "get_settings",
    "get_store",
]


@lru_cache(maxsize=1)
def get_store() -> JobStore:
    return JobStore(get_settings())


@lru_cache(maxsize=1)
def get_engine() -> PaintingEngine:
    return PaintingEngine(get_settings())


@lru_cache(maxsize=1)
def get_pipeline() -> TexturePipeline:
    return TexturePipeline(get_settings(), get_engine(), get_store())


def get_app_settings() -> Settings:
    return get_settings()


def get_job_semaphore(request: Request):
    """Bound the number of in-flight paint jobs.

    Built in the lifespan handler so it belongs to the running event loop; a
    request arriving before startup (impossible through uvicorn, but possible
    with an ASGI test client) falls back to no limiting.
    """
    return getattr(request.app.state, "job_semaphore", None)
