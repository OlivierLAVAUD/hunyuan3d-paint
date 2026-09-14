"""Pytest fixtures.

The environment is rewritten *before* ``app.main`` is imported: the settings are
built once and cached, so pointing the service at temporary directories has to
happen first. ``H3D_PRELOAD_MODEL=0`` keeps the suite free of torch — the engine
is replaced by ``tests.stubs.FakePaintEngine``.

The VRAM guard is disabled here (``H3D_MIN_VRAM_GB=0``): it shells out to
``nvidia-smi``, and a test host with a small card would otherwise refuse a load
the fake engine never performs anyway. The guard has its own tests, which call
it directly rather than going through the app.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from tests import stubs

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="h3dpaint-tests-"))
stubs.configure_env(_TMP_ROOT)
os.environ["H3D_MIN_VRAM_GB"] = "0"


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP_ROOT


@pytest.fixture()
def job_root() -> Path:
    """A data directory private to one test.

    The job cache is keyed on the upload *names*, so a shared root would let one
    test's ``cube_mesh.stl`` + ``brick.png`` be served from cache to the next
    one - the pipeline would never run and the engine call log would stay empty.
    """
    return Path(tempfile.mkdtemp(prefix="h3dpaint-job-", dir=_TMP_ROOT))


@pytest.fixture()
def fake_paint() -> stubs.FakePaintEngine:
    return stubs.FakePaintEngine()


@pytest.fixture()
def app(fake_paint, job_root):
    return stubs.build_app(fake_paint, root=job_root)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    # The context manager runs the lifespan (directories + concurrency gate).
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def image_bytes() -> bytes:
    return stubs.png_bytes()


@pytest.fixture()
def mesh_bytes() -> bytes:
    """An STL: geometry with no UVs, the case the unwrap exists for."""
    return stubs.stl_bytes()


@pytest.fixture()
def upload(mesh_bytes, image_bytes) -> dict:
    return stubs.upload_payload(mesh_bytes, image_bytes)
