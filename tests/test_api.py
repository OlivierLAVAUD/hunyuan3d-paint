"""API contract of the texture service."""
from __future__ import annotations

import hashlib

import pytest

from tests import stubs


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["model_loaded"] is True


def test_info_describes_the_paint_model(client):
    payload = client.get("/api/v1/info").json()
    assert payload["model"]["model_id"] == "tencent/Hunyuan3D-2"
    assert payload["model"]["subfolder"] == "hunyuan3d-paint-v2-0-turbo"
    assert payload["defaults"]["texture_size"] == 768
    assert payload["defaults"]["uv_unwrap"] is True
    assert payload["limits"]["max_mesh_upload_mb"] == 4
    assert ".stl" in payload["allowed_mesh_suffixes"]
    assert ".png" in payload["allowed_image_suffixes"]
    # The memory mode is part of the contract: a client that only sees
    # "texturing is slow" deserves to be able to find out why.
    assert payload["model"]["low_vram"] is False
    assert payload["model"]["vram_required_gb"] == 9.0


def test_texture_produces_and_serves_artifacts(client, upload, fake_paint):
    response = client.post("/api/v1/texture", files=upload)
    assert response.status_code == 200, response.text
    payload = response.json()

    # The job key is derived from *both* upload names: the same mesh painted
    # with another photo is a different job.
    assert payload["job_key"] == "cube_mesh_brick"
    assert payload["job_id"] == "cube_mesh_brick"
    assert payload["status"] == "succeeded"
    assert payload["reused_cache"] is False
    assert response.headers["x-job-id"] == "cube_mesh_brick"

    # The stub box is painted as the prepared mesh, so the stats are known: the
    # 12 faces survive and the geometry is still a solid, not a collapsed point.
    assert payload["mesh"]["stats"]["faces"] == 12
    assert payload["mesh"]["stats"]["vertices"] > 0
    # A Taubin pass must not *shrink* the mesh - that is the whole point of the
    # filter. A sign error on `mu` makes both passes shrink, and on an 8-vertex
    # box that collapses it to a point (extents ~1e-4 instead of ~0.4). The
    # precise volume-preservation property is asserted on real geometry in
    # tests/test_mesh.py, where there are interior vertices to hold the shape.
    assert all(extent > 0.1 for extent in payload["mesh"]["stats"]["extents"]), (
        payload["mesh"]["stats"]["extents"]
    )
    # ...and it is reported, so a client can tell the geometry was re-indexed.
    assert payload["uv"]["faces_before"] == 12
    assert payload["uv"]["vertices_before"] == 8
    # Taubin ran for real (scipy is pinned in requirements.txt; a missing scipy
    # would make this False while still returning 200).
    assert payload["uv"]["smoothed"] is True
    # Names mirror the uploads plus a suffix, per role. Every stored file is
    # prefixed with the job key (mesh *and* image name), so two photos painted
    # onto the same mesh never share a directory entry.
    assert payload["input_mesh"]["filename"] == "cube_mesh_brick_mesh_src.stl"
    assert payload["source_image"]["filename"] == "cube_mesh_brick_image_src.png"
    assert payload["processed_image"]["filename"] == "cube_mesh_brick_image_processed.png"
    assert payload["mesh"]["filename"] == "cube_mesh_brick_mesh_prepared.glb"
    assert payload["textured_mesh"]["filename"] == "cube_mesh_brick_mesh_textured.glb"
    assert len(fake_paint.calls) == 1

    for artifact in (
        "input_mesh",
        "mesh",
        "source_image",
        "processed_image",
        "textured_mesh",
    ):
        raw = client.get(payload[artifact]["url"])
        assert raw.status_code == 200, artifact
        assert len(raw.content) == payload[artifact]["size_bytes"]
        assert hashlib.sha256(raw.content).hexdigest() == payload[artifact]["sha256"]


def test_a_mesh_without_uvs_gets_an_atlas_before_painting(client, upload, fake_paint):
    """An STL has no UVs; the paint model cannot texture it without one."""
    payload = client.post("/api/v1/texture", files=upload).json()
    assert payload["uv"]["had_uvs"] is False
    assert payload["uv"]["unwrapped"] is True
    assert payload["uv"]["faces_after"] > 0
    # The engine was handed a mesh that *does* carry the generated atlas.
    assert fake_paint.calls[0]["has_uv"] is True


def test_a_mesh_that_already_has_uvs_keeps_them(client, fake_paint):
    """A hand-made GLB with an atlas must not be re-unwrapped."""
    payload = client.post(
        "/api/v1/texture",
        files=stubs.upload_payload(
            stubs.glb_bytes(with_uvs=True),
            stubs.png_bytes(),
            mesh_filename="uvready.glb",
        ),
    ).json()
    assert payload["uv"]["had_uvs"] is True
    assert payload["uv"]["unwrapped"] is False


def test_uv_unwrap_can_be_disabled(client, fake_paint):
    # The bytes must match the extension: a .glb name over STL bytes is
    # rejected as unreadable before the parameters are ever considered.
    payload = client.post(
        "/api/v1/texture",
        files=stubs.upload_payload(
            stubs.glb_bytes(), stubs.png_bytes(), mesh_filename="nounwrap.glb"
        ),
        data={"uv_unwrap": "false"},
    ).json()
    assert payload["parameters"]["uv_unwrap"] is False
    assert payload["uv"]["unwrapped"] is False


def test_texture_size_is_echoed_and_passed_to_the_model(client, upload, fake_paint):
    payload = client.post(
        "/api/v1/texture", files=upload, data={"texture_size": "1024"}
    ).json()
    assert payload["parameters"]["texture_size"] == 1024
    assert fake_paint.calls[0]["texture_size"] == 1024


def test_seed_is_echoed_and_passed_to_the_model(client, upload, fake_paint):
    """The seed has to reach the model: upstream pins the multiview pass to 0."""
    payload = client.post(
        "/api/v1/texture", files=upload, data={"seed": "4242"}
    ).json()
    assert payload["parameters"]["seed"] == 4242
    assert fake_paint.calls[0]["seed"] == 4242


def test_the_seed_falls_back_to_the_setting(client, upload, fake_paint):
    """Omitted means ``H3D_DEFAULT_SEED``, not "whatever the last job used".

    The expected value is read from ``/info`` rather than hard-coded: the
    setting has an environment variable, and the suite must not assume the
    checkout's ``.env`` leaves it at the code default.
    """
    default = client.get("/api/v1/info").json()["defaults"]["seed"]
    payload = client.post("/api/v1/texture", files=upload).json()
    assert payload["parameters"]["seed"] == default
    assert fake_paint.calls[0]["seed"] == default


def test_seed_out_of_range_is_422(client, upload):
    """Only -1 is special ("draw a fresh one"); the rest must fit torch."""
    response = client.post("/api/v1/texture", files=upload, data={"seed": "-2"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_parameter"
    assert "seed" in response.json()["error"]["message"]


def test_texture_size_out_of_range_is_422(client, upload):
    response = client.post("/api/v1/texture", files=upload, data={"texture_size": "99"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_parameter"
    assert "texture_size" in response.json()["error"]["message"]


def test_decimation_budget_is_applied(client, fake_paint):
    """A target below the input face count must actually shrink the mesh."""
    payload = client.post(
        "/api/v1/texture",
        files=stubs.upload_payload(
            stubs.glb_bytes(), stubs.png_bytes(), mesh_filename="big.glb"
        ),
        data={"target_faces": "6"},
    ).json()
    # A 12-face box decimated to ~6, and the atlas is built on the result.
    assert payload["uv"]["faces_after"] < 12
    assert fake_paint.calls[0]["faces"] == payload["uv"]["faces_after"]


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("bad.txt", "text/plain"),
        ("archive.zip", "application/zip"),
        ("noextension", "application/octet-stream"),
    ],
)
def test_unsupported_mesh_type_is_415(client, image_bytes, filename, content_type):
    response = client.post(
        "/api/v1/texture",
        files={
            "mesh": (filename, b"nope", content_type),
            "image": ("ref.png", image_bytes, "image/png"),
        },
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_mesh"


def test_unsupported_image_type_is_415(client, mesh_bytes):
    response = client.post(
        "/api/v1/texture",
        files={
            "mesh": ("cube.stl", mesh_bytes, "model/stl"),
            "image": ("ref.tiff", b"nope", "image/tiff"),
        },
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_image"


def test_an_unreadable_mesh_is_422(client, image_bytes):
    """Right extension, garbage content: a 422, not a 500."""
    response = client.post(
        "/api/v1/texture",
        files={
            "mesh": ("broken.stl", b"not really a mesh", "model/stl"),
            "image": ("ref.png", image_bytes, "image/png"),
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "mesh_unreadable"


def test_oversized_mesh_is_413(client, image_bytes):
    """``H3D_MAX_MESH_UPLOAD_MB`` is 4 in the test environment."""
    response = client.post(
        "/api/v1/texture",
        files={
            "mesh": ("huge.stl", b"0" * (5 * 1024 * 1024), "model/stl"),
            "image": ("ref.png", image_bytes, "image/png"),
        },
    )
    assert response.status_code == 413
    assert response.json()["error"]["details"]["role"] == "mesh"


def test_reuse_cached_skips_the_model(client, upload, fake_paint):
    first = client.post("/api/v1/texture", files=upload).json()
    calls = len(fake_paint.calls)
    second = client.post("/api/v1/texture", files=upload).json()
    assert second["reused_cache"] is True
    assert len(fake_paint.calls) == calls
    assert second["textured_mesh"]["sha256"] == first["textured_mesh"]["sha256"]


def test_force_regeneration_ignores_the_cache(client, upload, fake_paint):
    client.post("/api/v1/texture", files=upload)
    calls = len(fake_paint.calls)
    payload = client.post(
        "/api/v1/texture", files=upload, data={"reuse_cached": "false"}
    ).json()
    assert payload["reused_cache"] is False
    assert len(fake_paint.calls) == calls + 1


def test_a_different_image_is_a_different_job(client, mesh_bytes, image_bytes, fake_paint):
    """The whole point of keying on both names: a second photo must repaint."""
    first = client.post(
        "/api/v1/texture",
        files=stubs.upload_payload(mesh_bytes, image_bytes, image_filename="red.png"),
    ).json()
    second = client.post(
        "/api/v1/texture",
        files=stubs.upload_payload(mesh_bytes, stubs.png_bytes(color=(255, 0, 0, 255)),
                                   image_filename="blue.png"),
    ).json()
    assert first["job_key"] != second["job_key"]
    assert second["reused_cache"] is False
    assert len(fake_paint.calls) == 2


def test_a_paint_failure_is_an_error_not_a_silent_success(client, upload, fake_paint):
    """Unlike the shape service, there is no useful partial result here."""

    def boom(mesh, image, *, texture_size=None, seed=None):
        raise RuntimeError("no VRAM left")

    fake_paint.paint = boom
    response = client.post("/api/v1/texture", files=upload)
    assert response.status_code == 500
    payload = response.json()["error"]
    assert payload["code"] == "texture_failed"
    assert "no VRAM left" in payload["message"]


def test_jobs_list_and_get(client, upload):
    created = client.post("/api/v1/texture", files=upload).json()
    listing = client.get("/api/v1/texture/jobs").json()
    assert listing["total"] >= 1
    entry = next(job for job in listing["jobs"] if job["job_key"] == created["job_key"])
    assert entry["textured_mesh_url"]

    fetched = client.get(f"/api/v1/texture/jobs/{created['job_key']}").json()
    assert fetched["job_key"] == created["job_key"]
    # URLs are rebuilt for the caller, not replayed from disk.
    assert fetched["textured_mesh"]["url"].startswith("http://testserver")


def test_unknown_job_is_404(client):
    assert client.get("/api/v1/texture/jobs/nope").status_code == 404


def test_artifact_path_escape_is_404(client, upload):
    created = client.post("/api/v1/texture", files=upload).json()
    response = client.get(
        f"/api/v1/texture/jobs/{created['job_key']}/files/..%2Fjob.json"
    )
    assert response.status_code == 404


def test_delete_job(client, upload):
    created = client.post("/api/v1/texture", files=upload).json()
    assert client.delete(f"/api/v1/texture/jobs/{created['job_key']}").status_code == 204
    assert client.get(f"/api/v1/texture/jobs/{created['job_key']}").status_code == 404
    assert client.delete(f"/api/v1/texture/jobs/{created['job_key']}").status_code == 404
