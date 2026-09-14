"""Front-end viewer helpers: the GLB re-export behind the ``gr.Model3D`` windows.

``ui.viewers`` is the only module of the front-end that can be imported from
this suite — it needs nothing but trimesh, where ``ui.app`` imports gradio, which
the API image does not ship. What is pinned down here is the re-export itself:
the prepared mesh of a run is read **once** for both its viewer GLB and the
journal line, and the uploaded mesh is deliberately not part of it — its window
is the picker, an interactive ``gr.Model3D`` that Gradio fills on its own.
"""
from __future__ import annotations

import trimesh

from ui.viewers import clear_views, load_mesh, render_mesh, view_name


def _write(mesh, directory, name: str):
    path = directory / name
    mesh.export(str(path))
    return path


def test_view_name_follows_the_uploaded_file(tmp_path):
    assert view_name("brick_mesh_processed.stl") == "brick_mesh_processed.glb"


def test_render_mesh_exports_a_tinted_glb(tmp_path):
    """The viewer shows the mesh in the colour the page asked for."""
    source = _write(trimesh.creation.box(), tmp_path, "cube.stl")

    export, _ = render_mesh(
        source,
        color="#d32f2f",
        wireframe=False,
        output_dir=tmp_path / "view",
        filename=view_name(source),
        label="prepared",
    )

    assert export is not None
    assert export.endswith("cube.glb")
    mesh = load_mesh(export)
    assert len(mesh.faces) == 12
    assert tuple(mesh.visual.vertex_colors[0][:3]) == (211, 47, 47)


def test_render_mesh_can_add_the_wireframe(tmp_path):
    """One geometry means the mesh alone, two mean the mesh and its edges."""
    source = _write(trimesh.creation.icosphere(subdivisions=1), tmp_path, "ball.stl")

    plain, _ = render_mesh(
        source, color="#9e9e9e", output_dir=tmp_path / "plain", label="prepared"
    )
    wired, _ = render_mesh(
        source,
        color="#9e9e9e",
        wireframe=True,
        output_dir=tmp_path / "wired",
        label="prepared",
    )

    assert len(trimesh.load(plain).geometry) == 1
    assert len(trimesh.load(wired).geometry) == 2


def test_render_mesh_describes_what_it_exported(tmp_path):
    """The journal line comes out of the same read as the export — or not at all."""
    source = _write(trimesh.creation.box(), tmp_path, "cube.stl")

    _, line = render_mesh(
        source, color="#9e9e9e", output_dir=tmp_path / "view", label="prepared"
    )
    _, silent = render_mesh(
        source,
        color="#9e9e9e",
        output_dir=tmp_path / "view",
        label="prepared",
        describe=False,
    )

    assert line.startswith("prepared — 12 faces")
    assert "watertight" in line
    # A re-render that only restyles: a mesh does not change when its colour does.
    assert silent == ""


def test_render_mesh_reports_a_mesh_it_cannot_read(tmp_path):
    """An unreadable mesh yields no viewer, and a line that says so."""
    unreadable = tmp_path / "broken.stl"
    unreadable.write_bytes(b"this is not a mesh at all")

    export, line = render_mesh(
        unreadable, color="#9e9e9e", output_dir=tmp_path / "view", label="prepared"
    )

    assert export is None
    assert line == "prepared — ❌ unreadable"
    assert not list((tmp_path / "view").glob("*.glb"))


def test_clear_views_keeps_the_current_export(tmp_path):
    """A re-render that *reuses* a GLB must not delete it on the way in."""
    directory = tmp_path / "view"
    directory.mkdir()
    kept = _write(trimesh.creation.box(), directory, "kept.glb")
    _ = _write(trimesh.creation.box(), directory, "stale.glb")

    clear_views(directory, keep=[kept])

    assert [path.name for path in directory.iterdir()] == ["kept.glb"]
