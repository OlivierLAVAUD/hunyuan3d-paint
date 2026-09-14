"""Gradio front-end for the Hunyuan3D image-to-mesh API.

A pure API client: it holds no model, imports nothing from ``app.services`` and
never needs a GPU. It is meant to run as its own container next to the API
(``docker compose up``), or locally against a remote API::

    H3D_UI_API_URL=http://gpu-box:8081 python -m ui.app

Modules:

    ui/config.py     settings (H3D_UI_* environment variables)
    ui/client.py     the HTTP client, with the API error envelope decoded
    ui/viewers.py    trimesh helpers behind the Model3D viewers
    ui/app.py        the Gradio UI and its event handlers
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
