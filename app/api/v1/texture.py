"""Texture routes.

``POST /texture`` is synchronous by design: it returns once the coloured GLB and
the prepared geometry are on disk. A paint run takes a couple of minutes, so
clients should use a generous timeout (the compose file raises uvicorn's
keep-alive accordingly). Jobs are capped by ``H3D_MAX_CONCURRENT_JOBS`` and,
because there is one GPU, extra requests queue instead of failing.

Two uploads, not one: ``mesh`` is the geometry to colour and ``image`` the
reference photo whose colours it should take. They are separate form fields
rather than a zip so a browser can show two pickers, and so the size limits can
differ (see ``Settings.max_mesh_upload_mb``).
"""
from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from ...config import Settings
from ...deps import (
    get_app_settings,
    get_engine,
    get_job_semaphore,
    get_pipeline,
    get_store,
)
from ...exceptions import InvalidParameter, JobNotFound, UnsupportedImage, UnsupportedMesh
from ...schemas import (
    ErrorResponse,
    JobListResponse,
    JobSummary,
    PaintParameters,
    TextureResponse,
)
from ...services.paint import PaintingEngine
from ...services.pipeline import TexturePipeline, file_url
from ...storage import JobStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/texture", tags=["texture"])

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "Unknown job or artifact."},
    413: {"model": ErrorResponse, "description": "One of the uploads is too large."},
    415: {"model": ErrorResponse, "description": "Unsupported mesh or image format."},
    422: {"model": ErrorResponse, "description": "Invalid parameter or unreadable mesh."},
    500: {"model": ErrorResponse, "description": "Texturing failed."},
    503: {"model": ErrorResponse, "description": "The paint model is unavailable."},
}


def base_url_for(request: Request) -> str:
    """Absolute base URL to build artifact links from.

    Honours ``X-Forwarded-Proto``/``X-Forwarded-Host`` when uvicorn runs with
    ``--proxy-headers`` (the default in the container entrypoint).
    """
    return str(request.base_url).rstrip("/")


def refresh_urls(response: TextureResponse, base_url: str) -> TextureResponse:
    """Re-point every artifact URL at ``base_url``.

    Job metadata is stored with absolute URLs, which go stale when the service
    is reached through a different host or port. Rebuilding them on read keeps
    the API usable behind a proxy without persisting host state.
    """
    key = response.job_key

    def fix(artifact):
        return artifact.model_copy(
            update={"url": file_url(base_url, key, artifact.filename)}
        )

    return response.model_copy(
        update={
            "input_mesh": fix(response.input_mesh),
            "mesh": fix(response.mesh),
            "source_image": fix(response.source_image),
            "processed_image": fix(response.processed_image),
            "textured_mesh": fix(response.textured_mesh),
            "files": {name: file_url(base_url, key, name) for name in response.files},
        }
    )


def _resolve_parameters(
    settings: Settings,
    *,
    texture_size: int | None,
    target_faces: int | None,
    taubin_steps: int | None,
    taubin_lambda: float | None,
    taubin_mu: float | None,
    uv_unwrap: bool | None,
    seed: int | None,
    reuse_cached: bool,
) -> PaintParameters:
    """Fill in the settings defaults, then range-check everything.

    Omitted fields fall back to ``H3D_DEFAULT_*``, so the environment stays the
    single source of truth for the defaults instead of the route signature.
    """
    try:
        resolved = PaintParameters(
            texture_size=(
                settings.default_texture_size if texture_size is None else texture_size
            ),
            target_faces=(
                settings.default_target_faces if target_faces is None else target_faces
            ),
            taubin_steps=(
                settings.default_taubin_steps if taubin_steps is None else taubin_steps
            ),
            taubin_lambda=(
                settings.default_taubin_lambda if taubin_lambda is None else taubin_lambda
            ),
            taubin_mu=settings.default_taubin_mu if taubin_mu is None else taubin_mu,
            uv_unwrap=(
                settings.default_uv_unwrap if uv_unwrap is None else uv_unwrap
            ),
            seed=settings.default_seed if seed is None else seed,
            reuse_cached=reuse_cached,
        )
    except ValidationError as exc:
        raise InvalidParameter(
            "invalid paint parameters",
            details={
                "errors": [
                    {
                        "loc": list(error.get("loc", ())),
                        "msg": error.get("msg"),
                        "type": error.get("type"),
                    }
                    for error in exc.errors()
                ]
            },
        ) from exc

    checks = [
        (
            settings.min_texture_size <= resolved.texture_size <= settings.max_texture_size,
            f"texture_size must be between {settings.min_texture_size} and "
            f"{settings.max_texture_size}",
        ),
        (resolved.target_faces >= 0, "target_faces must be >= 0"),
        (
            0 <= resolved.taubin_steps <= settings.taubin_steps_max,
            f"taubin_steps must be between 0 and {settings.taubin_steps_max}",
        ),
        (-1.0 <= resolved.taubin_lambda <= 1.0, "taubin_lambda must be between -1 and 1"),
        (-1.0 <= resolved.taubin_mu <= 1.0, "taubin_mu must be between -1 and 1"),
        (
            resolved.seed == -1 or 0 <= resolved.seed < 2**31,
            "seed must be -1 (draw a fresh one) or between 0 and 2^31 - 1",
        ),
    ]
    for ok, message in checks:
        if not ok:
            raise InvalidParameter(message)
    return resolved


@router.post(
    "",
    response_model=TextureResponse,
    summary="Texture a mesh with the colours of an image",
    responses=_ERROR_RESPONSES,
)
async def texture(
    request: Request,
    response: Response,
    mesh: UploadFile = File(
        ...,
        description="3D file to paint (glb/gltf/obj/ply/stl). A GLB or OBJ with "
        "UVs is painted as is; anything else gets a UV atlas generated first.",
    ),
    image: UploadFile = File(
        ..., description="Reference image whose colours the mesh should take."
    ),
    texture_size: int | None = Form(
        None, description="Resolution of the baked UV atlas (the main VRAM "
        "knob); falls back to H3D_DEFAULT_TEXTURE_SIZE."
    ),
    target_faces: int | None = Form(
        None, description="Decimation budget applied before painting (0 = keep all)."
    ),
    taubin_steps: int | None = Form(None, description="Taubin smoothing iterations."),
    taubin_lambda: float | None = Form(None, description="Taubin lambda."),
    taubin_mu: float | None = Form(None, description="Taubin mu."),
    uv_unwrap: bool | None = Form(
        None,
        description="Generate a UV atlas when the uploaded mesh has none. Leave on "
        "for marching-cubes meshes (the shape service's raw GLB, a bare STL); "
        "falls back to H3D_DEFAULT_UV_UNWRAP when omitted.",
    ),
    seed: int | None = Form(
        None,
        description="Multiview diffusion seed; -1 draws a fresh one per job "
        "(not reproducible), any other value is repeatable. Falls back to "
        "H3D_DEFAULT_SEED.",
    ),
    reuse_cached: bool = Form(
        True, description="Reuse artifacts already on disk for this mesh+image pair."
    ),
    settings: Settings = Depends(get_app_settings),
    engine: PaintingEngine = Depends(get_engine),
    store: JobStore = Depends(get_store),
    pipeline: TexturePipeline = Depends(get_pipeline),
    semaphore=Depends(get_job_semaphore),
) -> TextureResponse:
    mesh_name = (mesh.filename or "").strip()
    image_name = (image.filename or "").strip()
    if not mesh_name:
        raise UnsupportedMesh("the mesh upload needs a file name with a known extension")
    if not image_name:
        raise UnsupportedImage("the image upload needs a file name with a known extension")
    if not settings.is_allowed_mesh(mesh_name):
        raise UnsupportedMesh(
            f"unsupported mesh type {Path(mesh_name).suffix or '(none)'!r}",
            details={"allowed": list(settings.allowed_mesh_suffixes)},
        )
    if not settings.is_allowed_image(image_name):
        raise UnsupportedImage(
            f"unsupported image type {Path(image_name).suffix or '(none)'!r}",
            details={"allowed": list(settings.allowed_image_suffixes)},
        )

    params = _resolve_parameters(
        settings,
        texture_size=texture_size,
        target_faces=target_faces,
        taubin_steps=taubin_steps,
        taubin_lambda=taubin_lambda,
        taubin_mu=taubin_mu,
        uv_unwrap=uv_unwrap,
        seed=seed,
        reuse_cached=reuse_cached,
    )
    base_url = base_url_for(request)

    def _work() -> TextureResponse:
        # Runs in the threadpool: the upload copies, the UV unwrap and the GPU
        # call are all blocking and CPU/GPU-bound, and the event loop must stay
        # free.
        mesh_stem, mesh_path = store.save_upload(mesh_name, mesh.file, role="mesh")
        _, image_path = store.save_upload(image_name, image.file, role="image")
        job_key = store.job_key_for(mesh_name, image_name)
        try:
            return pipeline.run(
                job_key=job_key,
                mesh_path=mesh_path,
                mesh_filename=mesh_name,
                image_path=image_path,
                image_filename=image_name,
                params=params,
                base_url=base_url,
            )
        finally:
            # Both were copied into the job directory by the pipeline; the
            # stems are only kept to make the upload directory readable.
            del mesh_stem
            mesh_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)

    if semaphore is None:  # pragma: no cover - only without the lifespan hooks
        result = await run_in_threadpool(_work)
    else:
        async with semaphore:
            result = await run_in_threadpool(_work)

    response.headers["X-Job-Id"] = result.job_id
    response.headers["X-Model-Device"] = engine.device or "unknown"
    return result


@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List recent jobs",
    responses=_ERROR_RESPONSES,
)
def list_jobs(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: JobStore = Depends(get_store),
) -> JobListResponse:
    base_url = base_url_for(request)
    total, page = store.list_jobs(limit=limit, offset=offset)
    jobs = []
    for entry in page:
        payload = entry.get("response") or {}
        key = payload.get("job_key") or payload.get("job_id") or entry.get("job_key")
        if not key:
            continue
        mesh = payload.get("mesh") or {}
        textured = payload.get("textured_mesh") or {}
        jobs.append(
            JobSummary(
                job_id=key,
                job_key=key,
                status=payload.get("status", "succeeded"),
                created_at=payload.get("created_at") or entry.get("created_at"),
                duration_s=payload.get("duration_s"),
                faces=(textured.get("stats") or {}).get("faces"),
                mesh_url=(
                    file_url(base_url, key, mesh.get("filename", ""))
                    if mesh.get("filename")
                    else None
                ),
                textured_mesh_url=(
                    file_url(base_url, key, textured.get("filename", ""))
                    if textured.get("filename")
                    else None
                ),
            )
        )
    return JobListResponse(total=total, limit=limit, offset=offset, jobs=jobs)


@router.get(
    "/jobs/{job_id}",
    response_model=TextureResponse,
    summary="Fetch a job result",
    responses=_ERROR_RESPONSES,
)
def get_job(
    request: Request,
    job_id: str,
    store: JobStore = Depends(get_store),
) -> TextureResponse:
    entry = store.require_metadata(job_id)
    payload = entry.get("response")
    if not payload:
        raise JobNotFound(f"job {job_id!r} has no recorded result")
    return refresh_urls(TextureResponse.model_validate(payload), base_url_for(request))


@router.get(
    "/jobs/{job_id}/files/{filename}",
    summary="Download a job artifact",
    response_class=FileResponse,
    responses=_ERROR_RESPONSES,
)
def get_job_file(
    job_id: str,
    filename: str,
    store: JobStore = Depends(get_store),
) -> FileResponse:
    path = store.artifact_path(job_id, filename)
    media_type, _encoding = mimetypes.guess_type(filename)
    return FileResponse(
        path, filename=filename, media_type=media_type or "application/octet-stream"
    )


@router.delete(
    "/jobs/{job_id}",
    status_code=204,
    summary="Delete a job and its artifacts",
    responses=_ERROR_RESPONSES,
)
def delete_job(job_id: str, store: JobStore = Depends(get_store)) -> Response:
    if not store.delete_job(job_id):
        raise JobNotFound(f"unknown job {job_id!r}")
    return Response(status_code=204)
