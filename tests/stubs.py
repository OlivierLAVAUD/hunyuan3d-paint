"""Shared test helpers: a paint-engine stub and a fully wired test app.

Kept out of ``conftest.py`` so the pytest suite and any ad-hoc script build the
exact same application.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

OUTPUT_SUBDIR = "outputs"
UPLOAD_SUBDIR = "uploads"
MODEL_SUBDIR = "models"


def configure_env(root: Path, *, preload_model: bool = False, max_upload_mb: int = 1,
                  max_mesh_upload_mb: int = 4) -> None:
    """Point the service at ``root`` - must run before ``app.config`` is imported."""
    os.environ["H3D_OUTPUT_DIR"] = str(root / OUTPUT_SUBDIR)
    os.environ["H3D_UPLOAD_DIR"] = str(root / UPLOAD_SUBDIR)
    os.environ["H3D_MODEL_CACHE_DIR"] = str(root / MODEL_SUBDIR)
    os.environ["H3D_PRELOAD_MODEL"] = "1" if preload_model else "0"
    os.environ["H3D_MAX_UPLOAD_MB"] = str(max_upload_mb)
    os.environ["H3D_MAX_MESH_UPLOAD_MB"] = str(max_mesh_upload_mb)
    os.environ.setdefault("H3D_LOG_LEVEL", "WARNING")


def get_settings_for(root: Path):
    """Build a ``Settings`` pointed at ``root`` for one test.

    ``get_settings()`` is ``lru_cache``d and built from the process environment,
    so a per-test data directory cannot go through it. The cached object is
    copied with the three directories (and the guards the suite disables)
    overridden - the rest of the configuration stays exactly what the app uses.
    """
    from dataclasses import replace

    from app.config import get_settings

    return replace(
        get_settings(),
        output_dir=root / OUTPUT_SUBDIR,
        upload_dir=root / UPLOAD_SUBDIR,
        model_cache_dir=root / MODEL_SUBDIR,
        min_vram_gb=0.0,
        min_host_ram_gb=0.0,
    )


class FakePaintEngine:
    """Stand-in for ``PaintingEngine``: 'textures' by tagging the mesh back.

    A real paint run returns a *new* trimesh carrying a baked texture. The tests
    only care that the stage ran, that its artifact is written, served and
    listed, and that the UV atlas travelled through — so the box comes back with
    a marker colour and the calls are recorded.
    """

    def __init__(self) -> None:
        from app.services.cuda import STATE_READY

        self.device = "cuda"
        self.state = STATE_READY
        self.error: str | None = None
        self.calls: list[dict[str, Any]] = []

    @property
    def is_loaded(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def load(self) -> None:  # pragma: no cover - a stub must never load weights
        raise AssertionError("the fake paint engine must not be loaded")

    def ensure_loaded(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def status(self) -> dict[str, Any]:
        # Mirrors `PaintingEngine.status()`. A stub that omits a key turns a
        # present field into a 500 in `/info`, which is how the low-VRAM mode
        # would have shipped invisible.
        return {
            "state": self.state,
            "loaded": True,
            "device": self.device,
            "error": None,
            "low_vram": False,
            "vram_required_gb": 9.0,
        }

    def paint(self, mesh, image, *, texture_size=None, seed=None):
        import numpy as np
        import trimesh

        self.calls.append(
            {
                "size": image.size,
                "mode": image.mode,
                "faces": len(mesh.faces),
                "has_uv": getattr(mesh.visual, "uv", None) is not None,
                "texture_size": texture_size,
                "seed": seed,
            }
        )
        painted = mesh.copy()
        # A real paint run bakes a texture; the stub marks the vertices instead
        # so the test can tell the painted mesh from the input.
        rgba = np.asarray(trimesh.visual.color.hex_to_rgba("#ff0000"), dtype=np.uint8)
        painted.visual.vertex_colors = np.tile(rgba, (len(painted.vertices), 1))
        return painted


def build_app(engine: FakePaintEngine | None = None, root: Path | None = None):
    """Build the real application with only the GPU model swapped out.

    ``root`` is the data directory. Tests pass a fresh one per test: the default
    ``_TMP_ROOT`` is shared for the whole session, and the job cache is keyed on
    the *upload names* only - two tests posting ``cube_mesh.stl`` + ``brick.png``
    would collide, and the second would be served the first one's artifacts
    instead of exercising the pipeline (``fake_paint.calls`` stays empty).
    """
    from app.config import get_settings
    from app.deps import get_app_settings, get_engine, get_pipeline, get_store
    from app.main import create_app
    from app.services.pipeline import TexturePipeline
    from app.storage import JobStore

    settings = get_settings_for(root) if root is not None else get_settings()
    engine = engine or FakePaintEngine()
    store = JobStore(settings)
    pipeline = TexturePipeline(settings, engine, store)

    application = create_app(settings)
    application.dependency_overrides.update(
        {
            get_app_settings: lambda: settings,
            get_engine: lambda: engine,
            get_store: lambda: store,
            get_pipeline: lambda: pipeline,
        }
    )
    return application


def png_bytes(size: int = 16, color: tuple[int, int, int, int] = (0, 128, 255, 255)) -> bytes:
    """A tiny RGBA PNG to stand in for the reference image."""
    import io

    from PIL import Image

    image = Image.new("RGBA", (size, size), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def stl_bytes() -> bytes:
    """A small binary STL (a 1x2x3 box), with **no** UVs.

    This is the shape service's typical output after decimation: geometry only,
    which is exactly the case the UV unwrap exists for.
    """
    import io

    import trimesh

    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    buffer = io.BytesIO()
    box.export(buffer, file_type="stl")
    return buffer.getvalue()


def glb_bytes(*, with_uvs: bool = False) -> bytes:
    """A small binary GLB, optionally carrying a UV atlas."""
    import io

    import numpy as np
    import trimesh

    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    if with_uvs:
        box.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(box.vertices), 2), dtype=np.float64)
        )
    buffer = io.BytesIO()
    box.export(buffer, file_type="glb")
    return buffer.getvalue()


def upload_payload(
    mesh: bytes | None = None,
    image: bytes | None = None,
    *,
    mesh_filename: str = "cube_mesh.stl",
    image_filename: str = "brick.png",
) -> dict[str, Any]:
    """A two-file multipart payload for ``POST /api/v1/texture``."""
    mesh_bytes = mesh if mesh is not None else stl_bytes()
    image_bytes = image if image is not None else png_bytes()
    mesh_type = "model/stl" if mesh_filename.endswith(".stl") else "model/gltf-binary"
    return {
        "mesh": (mesh_filename, mesh_bytes, mesh_type),
        "image": (image_filename, image_bytes, "image/png"),
    }
