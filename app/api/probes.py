"""Liveness and readiness probes.

``/health`` is a liveness probe: 200 as soon as the process can answer HTTP, so
Docker/orchestrators do not restart the container while the ~7.5 GB checkpoint
is still loading.

``/ready`` is a readiness probe: 503 until the paint model can serve traffic, so
a load balancer keeps the instance out of rotation during the load and after a
failed load.

``cuda`` is reported on its own because the paint model has no CPU path: on a
host without a visible GPU the service will *never* be ready, and a 503 that
never clears is much harder to diagnose than a field that says why.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from ..config import Settings
from ..deps import get_app_settings, get_engine
from ..schemas import HealthResponse, ReadyResponse
from ..services.cuda import cuda_ready
from ..services.paint import PaintingEngine

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    return HealthResponse(name=settings.app_name, version=settings.version)


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness probe",
    responses={503: {"description": "The paint model is not ready yet."}},
)
def ready(
    response: Response,
    engine: PaintingEngine = Depends(get_engine),
) -> ReadyResponse:
    if engine.is_available():
        status = "ready"
    elif engine.state == "error":
        status = "error"
    else:
        status = "loading"
    if status != "ready":
        response.status_code = 503
    return ReadyResponse(
        status=status,
        model_loaded=engine.is_loaded,
        device=engine.device,
        cuda=cuda_ready(),
        error=engine.error,
    )
