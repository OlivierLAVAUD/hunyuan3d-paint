"""Mesh preparation: cleaning, smoothing, decimation, UV unwrap."""
from __future__ import annotations

import pytest

from app.exceptions import MeshLoadError, PreprocessError
from app.services.mesh import (
    clean_mesh,
    has_uvs,
    load_mesh,
    prepare_mesh,
    unwrap_uvs,
)


def _box(with_uvs: bool = False):
    import numpy as np
    import trimesh

    box = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    if with_uvs:
        box.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(box.vertices), 2), dtype=np.float64)
        )
    return box


def _write(mesh, tmp_path, name: str):
    path = tmp_path / name
    mesh.export(str(path))
    return path


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #


def test_load_mesh_reads_an_stl(tmp_path):
    path = _write(_box(), tmp_path, "box.stl")
    mesh = load_mesh(path)
    assert len(mesh.faces) == 12
    assert len(mesh.vertices) == 8


def test_load_mesh_flattens_a_scene(tmp_path):
    """A multi-object GLB concatenates rather than silently dropping nodes."""
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(_box(), node_name="a")
    second = _box()
    second.apply_translation((5.0, 0.0, 0.0))
    scene.add_geometry(second, node_name="b")
    path = tmp_path / "two.glb"
    scene.export(str(path))

    mesh = load_mesh(path)
    assert len(mesh.faces) == 24
    # The second box really was moved: the combined bounds are wider than one.
    assert mesh.extents[0] > 5.0


def test_load_mesh_rejects_a_point_cloud(tmp_path):
    import numpy as np
    import trimesh

    cloud = trimesh.PointCloud(np.random.default_rng(0).random((10, 3)))
    path = tmp_path / "cloud.ply"
    cloud.export(str(path))
    with pytest.raises(MeshLoadError):
        load_mesh(path)


def test_load_mesh_rejects_garbage(tmp_path):
    path = tmp_path / "broken.stl"
    path.write_bytes(b"this is not a mesh at all")
    with pytest.raises(MeshLoadError):
        load_mesh(path)


def test_load_mesh_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.stl"
    path.write_bytes(b"")
    with pytest.raises(MeshLoadError):
        load_mesh(path)


# --------------------------------------------------------------------------- #
# UVs
# --------------------------------------------------------------------------- #


def test_has_uvs_detects_an_atlas():
    assert has_uvs(_box(with_uvs=True)) is True
    assert has_uvs(_box(with_uvs=False)) is False


def test_unwrap_adds_an_atlas_to_an_stl():
    """The whole reason this module exists: an STL has no UVs and paint needs them."""
    mesh = _box()
    assert has_uvs(mesh) is False
    unwrapped = unwrap_uvs(mesh)
    assert has_uvs(unwrapped) is True
    # xatlas cuts seams, so it may split vertices; it never loses faces.
    assert len(unwrapped.faces) == len(mesh.faces)


def test_unwrap_uvs_lie_in_the_unit_square():
    unwrapped = unwrap_uvs(_box())
    uv = unwrapped.visual.uv
    assert uv.min() >= -1e-6
    assert uv.max() <= 1.0 + 1e-6


def test_clean_mesh_removes_degenerate_faces():
    import numpy as np
    import trimesh

    box = _box()
    degenerate = trimesh.Trimesh(
        vertices=np.vstack([box.vertices, box.vertices[:2]]),
        faces=np.vstack([box.faces, [[8, 9, 8]]]),
        process=False,
    )
    cleaned = clean_mesh(degenerate)
    assert len(cleaned.faces) == 12


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #


def test_prepare_mesh_unwraps_and_reports_what_it_did(tmp_path):
    source = _write(_box(), tmp_path, "in.stl")
    destination = tmp_path / "out.glb"
    mesh, info = prepare_mesh(source, destination, target_faces=0, uv_unwrap=True)

    assert destination.is_file()
    assert info.unwrapped is True
    assert info.had_uvs is False
    assert info.faces_before == 12
    assert info.faces_after == 12
    assert has_uvs(mesh) is True


def test_prepare_mesh_keeps_an_existing_atlas(tmp_path):
    source = _write(_box(with_uvs=True), tmp_path, "uv.glb")
    destination = tmp_path / "out.glb"
    _mesh, info = prepare_mesh(source, destination, uv_unwrap=True)
    assert info.had_uvs is True
    assert info.unwrapped is False


def test_prepare_mesh_decimates_before_unwrapping(tmp_path):
    """Order matters: the atlas is built on the final geometry, not the input."""
    import trimesh

    dense = trimesh.creation.icosphere(subdivisions=3)
    source = _write(dense, tmp_path, "dense.stl")
    destination = tmp_path / "out.glb"
    mesh, info = prepare_mesh(source, destination, target_faces=500, uv_unwrap=True)

    assert info.faces_before > 500
    assert info.faces_after <= 500
    assert len(mesh.faces) == info.faces_after
    # And the mesh that came out is the one that carries the atlas.
    assert has_uvs(mesh) is True


def test_prepare_mesh_smooths_when_asked(tmp_path):
    source = _write(_box(), tmp_path, "in.stl")
    destination = tmp_path / "out.glb"
    _mesh, info = prepare_mesh(
        source, destination, taubin_steps=5, taubin_lambda=0.5, taubin_mu=-0.53
    )
    assert info.smoothed is True


def test_prepare_mesh_reports_no_smoothing_when_disabled(tmp_path):
    source = _write(_box(), tmp_path, "in.stl")
    destination = tmp_path / "out.glb"
    _mesh, info = prepare_mesh(source, destination, taubin_steps=0)
    assert info.smoothed is False


def test_taubin_does_not_shrink_the_mesh(tmp_path):
    """The sign of `mu` is the difference between smoothing and collapsing.

    Taubin's published parameter is the signed ``mu = -0.53``, but trimesh's
    ``filter_taubin`` takes the positive magnitude and applies the sign itself.
    Passing ``-0.53`` through makes *both* passes shrink; on a real mesh that is
    a steady ~8% volume loss per call, and on a low-poly box it collapses the
    geometry to a point. This is asserted on a subdivided sphere because a
    12-face box has no interior vertices and shrinks legitimately.
    """
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=4)
    source = tmp_path / "sphere.stl"
    sphere.export(str(source))

    _mesh, info = prepare_mesh(
        source,
        tmp_path / "sphere.glb",
        taubin_steps=20,
        taubin_lambda=0.5,
        taubin_mu=-0.53,
        uv_unwrap=False,
    )
    assert info.smoothed is True

    smoothed = trimesh.load(str(tmp_path / "sphere.glb"), force="mesh")
    # Taubin is volume-preserving by construction: a few percent at most.
    assert smoothed.volume == pytest.approx(sphere.volume, rel=0.05), (
        f"volume went from {sphere.volume:.4f} to {smoothed.volume:.4f}"
    )


def test_prepare_mesh_refuses_to_write_uvs_to_an_stl(tmp_path):
    """STL has no concept of UVs: writing one would silently drop the atlas."""
    source = _write(_box(), tmp_path, "in.stl")
    with pytest.raises(PreprocessError):
        prepare_mesh(source, tmp_path / "out.stl", uv_unwrap=True, export_uvs=True)


def test_prepare_mesh_allows_an_stl_without_uvs(tmp_path):
    """With the unwrap off, an STL destination is fine — there is no atlas to lose."""
    source = _write(_box(), tmp_path, "in.stl")
    destination = tmp_path / "out.stl"
    prepare_mesh(source, destination, uv_unwrap=False, export_uvs=True)
    assert destination.is_file()


def test_prepare_mesh_applies_the_uv_face_ceiling(tmp_path):
    """xatlas on a huge mesh is minutes of work for nothing: the ceiling holds."""
    import trimesh

    from app.services.mesh import UV_UNWRAP_FACE_CEILING

    dense = trimesh.creation.icosphere(subdivisions=5)  # 20480 faces
    assert len(dense.faces) < UV_UNWRAP_FACE_CEILING
    source = _write(dense, tmp_path, "dense.stl")
    mesh, info = prepare_mesh(
        source, tmp_path / "out.glb", target_faces=0, uv_unwrap=True
    )
    # Under the ceiling, so nothing was decimated away.
    assert len(mesh.faces) == len(dense.faces)
    assert info.decimated_to is None
