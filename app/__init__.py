"""Hunyuan3D texture (paint) API package.

``app.main:app`` is the ASGI application; see ``README.md`` for the endpoint
reference and the module docstrings for the internals:

    app/config.py            settings (H3D_* environment variables)
    app/schemas.py           the public request/response contract
    app/storage.py           job directory layout and upload handling
    app/services/paint.py    Hunyuan3D-2 paint model (singleton)
    app/services/mesh.py     clean / Taubin / decimate / UV unwrap
    app/services/pipeline.py the orchestration the API exposes
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
