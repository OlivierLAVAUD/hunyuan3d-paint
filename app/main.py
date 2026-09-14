"""FastAPI application.

``create_app()`` is the single place where routers, middleware and error
handlers are wired; ``app`` at the bottom is what uvicorn imports
(``uvicorn app.main:app``).

Startup deliberately does not block on the model: the container starts
answering ``/health`` immediately and loads the ~7.5 GB checkpoint in a worker
thread, reporting progress through ``/ready``. That keeps the Docker healthcheck
meaningful instead of declaring a slow download a crash-loop.

Unlike the shape service, preloading is **off by default**: the paint model is
the only thing this container does, but a ~7.5 GB read costs minutes and most
deployments send their first job long after the container came up. It is a
one-line change (``H3D_PRELOAD_MODEL=1``) when the opposite is wanted.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from .api import probes
from .api.errors import register_exception_handlers
from .api.v1.router import router as v1_router
from .config import Settings, get_settings
from .deps import get_engine, get_store

logger = logging.getLogger("app")


def configure_logging(settings: Settings) -> None:
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(settings.log_level.upper())
    # These libs log one line per weight file at INFO level.
    for noisy in ("httpx", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _preload(engine) -> None:
    try:
        await run_in_threadpool(engine.load)
    except Exception:
        # Logged with the traceback: a failed preload must be diagnosable from
        # `docker compose logs` alone. /ready (and /info) reports it too.
        logger.exception("paint model preload failed")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Without this the root logger stays at WARNING and every INFO line this
    # service writes - the load's progress, the resolution and seed a job is
    # actually using - is dropped before it reaches the container log, which is
    # what makes a 13-minute run that ends in a 500 so hard to read.
    configure_logging(settings)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = get_store()
        store.ensure_dirs()
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        app.state.job_semaphore = asyncio.Semaphore(max(1, settings.max_concurrent_jobs))
        preload_tasks: list[asyncio.Task] = []
        if settings.preload_model:
            preload_tasks.append(asyncio.create_task(_preload(get_engine())))
        logger.info(
            "%s %s ready - outputs=%s uploads=%s cache=%s",
            settings.app_name,
            settings.version,
            settings.output_dir,
            settings.upload_dir,
            settings.model_cache_dir,
        )
        try:
            yield
        finally:
            for task in preload_tasks:
                if not task.done():
                    task.cancel()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        summary="Paint a 3D mesh with the colours of a reference image.",
        description=(
            "Takes a 3D file and an image, and returns the mesh with Tencent's "
            "Hunyuan3D-2 paint model's texture baked in. The geometry is "
            "cleaned, smoothed, decimated and UV-unwrapped first when it needs "
            "it, so a raw marching-cubes mesh works as well as a hand-made GLB."
            "\n\n"
            "This is the *texture* half of the Hunyuan3D pipeline, split out of "
            "the image-to-mesh service: that one produces geometry, this one "
            "colours geometry, and neither needs the other.\n\n"
            "Artifacts are served from `/api/v1/texture/jobs/{job_id}/files/...`."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    origins = settings.cors_origin_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        # Starlette refuses the "*" + credentials combination.
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Job-Id", "X-Model-Device"],
    )

    register_exception_handlers(app)
    app.include_router(probes.router)
    app.include_router(v1_router, prefix=settings.api_prefix)
    return app


app = create_app()
