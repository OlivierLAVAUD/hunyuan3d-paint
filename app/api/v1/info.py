"""``GET /info`` - what this service accepts, with which defaults."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ...config import Settings
from ...deps import get_app_settings, get_engine
from ...schemas import InfoResponse, Limits, ModelInfo, PaintParameters
from ...services.paint import PaintingEngine

router = APIRouter(tags=["info"])


@router.get("/info", response_model=InfoResponse, summary="Service metadata")
def info(
    settings: Settings = Depends(get_app_settings),
    engine: PaintingEngine = Depends(get_engine),
) -> InfoResponse:
    status = engine.status()
    return InfoResponse(
        name=settings.app_name,
        version=settings.version,
        api_prefix=settings.api_prefix,
        allowed_image_suffixes=list(settings.allowed_image_suffixes),
        allowed_mesh_suffixes=list(settings.allowed_mesh_suffixes),
        model=ModelInfo(
            model_id=settings.model_id,
            subfolder=settings.subfolder,
            # The paint VAE ships as a pickle .bin upstream: there is no
            # safetensors variant to ask for.
            use_safetensors=False,
            device=status["device"],
            state=status["state"],
            loaded=status["loaded"],
            error=status["error"],
            low_vram=status["low_vram"],
            vram_required_gb=status["vram_required_gb"],
        ),
        defaults=PaintParameters(
            texture_size=settings.default_texture_size,
            target_faces=settings.default_target_faces,
            taubin_steps=settings.default_taubin_steps,
            taubin_lambda=settings.default_taubin_lambda,
            taubin_mu=settings.default_taubin_mu,
            uv_unwrap=settings.default_uv_unwrap,
            seed=settings.default_seed,
            reuse_cached=True,
        ),
        limits=Limits(
            max_upload_mb=settings.max_upload_mb,
            max_mesh_upload_mb=settings.max_mesh_upload_mb,
            min_texture_size=settings.min_texture_size,
            max_texture_size=settings.max_texture_size,
            taubin_steps_max=settings.taubin_steps_max,
            max_concurrent_jobs=settings.max_concurrent_jobs,
        ),
    )
