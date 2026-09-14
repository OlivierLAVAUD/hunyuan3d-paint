"""Process-wide configuration, read from ``H3D_*`` environment variables.

This is the *paint* half of the split: where the shape repo turns an image into
geometry, this one takes geometry somebody else produced and paints it with the
colours of a reference image. Nothing here generates a mesh, so the whole shape
model block (``model_id``, ``octree_resolution``, ``num_chunks``, the rembg
settings) is gone and the paint block is promoted to the main model.

The parsing helpers are the shape repo's, unchanged: one env convention for the
whole family, so an operator moving between the two services sees the same
``H3D_`` names and the same dotenv behaviour.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ENV_PREFIX = "H3D_"

DEFAULT_ALLOWED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
# What a caller may hand us to paint. GLB first: it is what the shape service
# exports, and the only one of the four that can carry UVs and a texture.
DEFAULT_ALLOWED_MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply", ".stl")

_TRUTHY = {"1", "true", "yes", "on"}


def load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from ``path`` into ``os.environ`` (no override).

    Called before the settings are built so a local ``.env`` behaves like the
    ``environment:`` block of the compose file. Existing environment variables
    always win, so the container can still override a file that slipped in.
    """
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.split("#", 1)[0].strip().strip('"').strip("'")
        os.environ[key] = value


def _str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:  # a typo in the compose file must not be silent
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in _TRUTHY


def _csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the service configuration."""

    # -- service ---------------------------------------------------------
    app_name: str = "Hunyuan3D Texture (Paint) API"
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8082
    log_level: str = "INFO"
    cors_origins: str = "*"

    # -- storage ---------------------------------------------------------
    output_dir: Path = Path("/data/outputs")
    upload_dir: Path = Path("/data/uploads")
    model_cache_dir: Path = Path("/data/models")
    max_upload_mb: int = 32
    # Meshes are heavier than the images the shape service takes: a decimated
    # 50k-face STL is a few MB, but a raw marching-cubes GLB can be 30-60 MB.
    max_mesh_upload_mb: int = 256

    # -- Hunyuan3D paint (the model this service exists for) --------------
    # The multiview texture model sits next to `hunyuan3d-delight-v2-0` in
    # `tencent/Hunyuan3D-2`, not in the shape-only `Hunyuan3D-2mini`.
    model_id: str = "tencent/Hunyuan3D-2"
    subfolder: str = "hunyuan3d-paint-v2-0-turbo"
    # The paint model *requires* CUDA: its rasterizer is a CUDA extension with
    # no CPU implementation, so `auto` resolving to cpu is a hard failure.
    device: str = "auto"
    preload_model: bool = False
    # Host RAM the load needs, in GB. The checkpoints are ~7.5 GB of fp16
    # weights (the download is twice that: the repo ships each file both as
    # an fp32 .bin and fp16 safetensors), they are read into *system* memory
    # before anything reaches the GPU, and in low_vram mode they stay there -
    # so a container with less than this is OOM-killed during the load and
    # the client only ever sees a dropped socket. Checked before loading
    # (see services/paint.py).
    min_host_ram_gb: float = 14.0
    # VRAM the full-resident load needs, in GB. The whole resident set is
    # ~8 GB of fp16 weights (multiview pipeline ~5.3 + delight ~2.2 +
    # renderer and CUDA context ~0.7), so 9 leaves a little margin; a 6 GB
    # card cannot hold it however much system RAM the host has. Checked up
    # front so the caller gets "your GPU is too small" instead of a CUDA
    # driver error from the middle of the load. 0 disables the check.
    min_vram_gb: float = 9.0
    # Take the GPU for ourselves and hand it back: the shape service's swap
    # knob. Meaningless when this is the only GPU process, kept because the
    # engine code is shared and a future co-hosted deployment wants it.
    swap_models: bool = False
    # Keep the weights in *system* RAM and stream them onto the GPU one
    # sub-module at a time, instead of holding the whole ~8 GB resident set
    # on the card. This is what makes the paint service usable on a 6 GB
    # GPU: the resident VRAM drops to the renderer, the CUDA context, the
    # activations and the largest single block (~2-2.5 GB). Two things make
    # it work (see `PaintingEngine._build_pipeline` and `_enable_low_vram`):
    # the pipelines are built CPU-first, and the offload is *sequential* -
    # upstream's model-level `enable_model_cpu_offload` still requires each
    # whole model to fit, which the 5.3 GB multiview pipeline does not on
    # 6 GB. The price is speed (every forward pass shuttles weights over
    # PCIe) and host RAM, since the checkpoint stays resident somewhere.
    low_vram: bool = False
    # VRAM that `low_vram` needs, in GB. Only one sub-module is resident at a
    # time, but the activations, the renderer and the CUDA context are not
    # offloaded, so this is not zero: the peak is ~2-2.5 GB and 3 leaves a
    # little margin. 0 disables the check.
    min_vram_low_gb: float = 3.0

    # -- texture defaults ------------------------------------------------
    # Resolution of the baked UV atlas. Upstream hardcodes it to 2048 in
    # Hunyuan3DTexGenConfig - and its pipeline takes no resolution argument at
    # all - so the engine pushes it onto the renderer before each call (see
    # PaintingEngine._apply_texture_size). The atlas, its cos map and its trust
    # map are all allocated at that size, which makes this the memory knob that
    # matters most on a small GPU; 768 is deliberately below upstream's 2048.
    default_texture_size: int = 768
    min_texture_size: int = 256
    max_texture_size: int = 2048
    # Taubin smoothing + decimation applied *before* painting, for callers who
    # hand over a raw marching-cubes mesh with no UVs.
    default_target_faces: int = 50_000
    default_taubin_steps: int = 20
    default_taubin_lambda: float = 0.5
    default_taubin_mu: float = -0.53
    taubin_steps_max: int = 200
    # 1 = run the UV unwrap (xatlas) before painting when the mesh has none.
    default_uv_unwrap: bool = True
    # RNG seed for the multiview diffusion pass. The engine forwards it to the
    # vendored multiview model, where upstream pinned it to 0; -1 draws a fresh
    # seed per job, so a run started from -1 cannot be reproduced.
    default_seed: int = 0

    # -- concurrency -----------------------------------------------------
    max_concurrent_jobs: int = 1

    # -- inputs ----------------------------------------------------------
    allowed_image_suffixes: tuple[str, ...] = DEFAULT_ALLOWED_IMAGE_SUFFIXES
    allowed_mesh_suffixes: tuple[str, ...] = DEFAULT_ALLOWED_MESH_SUFFIXES

    @classmethod
    def from_env(cls, dotenv_path: Path | str | None = None) -> Settings:
        if dotenv_path is not None:
            load_dotenv(Path(dotenv_path))
        return cls(
            app_name=_str("H3D_APP_NAME", cls.app_name),
            version=_str("H3D_VERSION", cls.version),
            api_prefix=_str("H3D_API_PREFIX", cls.api_prefix),
            host=_str("H3D_HOST", cls.host),
            port=_int("H3D_PORT", cls.port),
            log_level=_str("H3D_LOG_LEVEL", cls.log_level),
            cors_origins=_str("H3D_CORS_ORIGINS", cls.cors_origins),
            output_dir=Path(_str("H3D_OUTPUT_DIR", str(cls.output_dir))),
            upload_dir=Path(_str("H3D_UPLOAD_DIR", str(cls.upload_dir))),
            model_cache_dir=Path(_str("H3D_MODEL_CACHE_DIR", str(cls.model_cache_dir))),
            max_upload_mb=_int("H3D_MAX_UPLOAD_MB", cls.max_upload_mb),
            max_mesh_upload_mb=_int(
                "H3D_MAX_MESH_UPLOAD_MB", cls.max_mesh_upload_mb
            ),
            model_id=_str("H3D_MODEL_ID", cls.model_id),
            subfolder=_str("H3D_MODEL_SUBFOLDER", cls.subfolder),
            device=_str("H3D_DEVICE", cls.device),
            preload_model=_bool("H3D_PRELOAD_MODEL", cls.preload_model),
            min_host_ram_gb=_float("H3D_MIN_HOST_RAM_GB", cls.min_host_ram_gb),
            min_vram_gb=_float("H3D_MIN_VRAM_GB", cls.min_vram_gb),
            swap_models=_bool("H3D_SWAP_MODELS", cls.swap_models),
            low_vram=_bool("H3D_LOW_VRAM", cls.low_vram),
            min_vram_low_gb=_float("H3D_MIN_VRAM_LOW_GB", cls.min_vram_low_gb),
            default_texture_size=_int(
                "H3D_DEFAULT_TEXTURE_SIZE", cls.default_texture_size
            ),
            min_texture_size=_int("H3D_MIN_TEXTURE_SIZE", cls.min_texture_size),
            max_texture_size=_int("H3D_MAX_TEXTURE_SIZE", cls.max_texture_size),
            default_target_faces=_int(
                "H3D_DEFAULT_TARGET_FACES", cls.default_target_faces
            ),
            default_taubin_steps=_int(
                "H3D_DEFAULT_TAUBIN_STEPS", cls.default_taubin_steps
            ),
            default_taubin_lambda=_float(
                "H3D_DEFAULT_TAUBIN_LAMBDA", cls.default_taubin_lambda
            ),
            default_taubin_mu=_float("H3D_DEFAULT_TAUBIN_MU", cls.default_taubin_mu),
            taubin_steps_max=_int("H3D_TAUBIN_STEPS_MAX", cls.taubin_steps_max),
            default_uv_unwrap=_bool("H3D_DEFAULT_UV_UNWRAP", cls.default_uv_unwrap),
            default_seed=_int("H3D_DEFAULT_SEED", cls.default_seed),
            max_concurrent_jobs=_int("H3D_MAX_CONCURRENT_JOBS", cls.max_concurrent_jobs),
            allowed_image_suffixes=_csv(
                "H3D_ALLOWED_IMAGE_SUFFIXES", DEFAULT_ALLOWED_IMAGE_SUFFIXES
            ),
            allowed_mesh_suffixes=_csv(
                "H3D_ALLOWED_MESH_SUFFIXES", DEFAULT_ALLOWED_MESH_SUFFIXES
            ),
        )

    # -- derived ---------------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_mesh_upload_bytes(self) -> int:
        return self.max_mesh_upload_mb * 1024 * 1024

    @property
    def vram_required_gb(self) -> float:
        """The VRAM budget the load path enforces, given the current mode.

        The two modes are not two values of one knob: offloading changes what
        has to be resident, so `min_vram_gb` (whole model on the card) and
        `min_vram_low_gb` (one block at a time) are separate settings rather
        than one threshold scaled by a factor.
        """
        return self.min_vram_low_gb if self.low_vram else self.min_vram_gb

    def is_allowed_image(self, filename: str | None) -> bool:
        suffix = Path(filename or "").suffix.lower()
        return suffix in self.allowed_image_suffixes

    def is_allowed_mesh(self, filename: str | None) -> bool:
        suffix = Path(filename or "").suffix.lower()
        return suffix in self.allowed_mesh_suffixes

    def apply_runtime_env(self) -> None:
        """Export the environment variables Hugging Face / CUDA read at import.

        Must run before ``torch`` or ``huggingface_hub`` is imported, otherwise
        the checkpoint cache lands in the home directory instead of the mounted
        volume and is re-downloaded on every container start.
        """
        cache = Path(self.model_cache_dir)
        os.environ.setdefault("HF_HOME", str(cache))
        os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hub"))
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings (built once, from the environment)."""
    settings = Settings.from_env(Path(".env"))
    settings.apply_runtime_env()
    return settings
