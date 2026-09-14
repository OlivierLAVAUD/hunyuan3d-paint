"""Mesh helpers behind the ``gr.Model3D`` viewers.

``gr.Model3D`` renders what three.js can load, but a bare STL comes out as an
untextured grey blob and a raw GLB is often too dense to read. Both are
therefore re-exported through trimesh: the surface is tinted with the chosen
colour and the edges can be overlaid as a wireframe — the same treatment the
CADFit tab applied, kept here because it is front-end-only work (the API has no
opinion on how its meshes are displayed).

Nothing here may *require* an optional trimesh dependency: the front-end image
ships only numpy/trimesh/Pillow (see ``requirements-ui.txt``), so a code path
that silently needs more only fails once deployed. ``_tint`` is written around
exactly that trap - see its docstring.

An export depends on two things only: the mesh it shows and the style it was
asked for. The front-end keeps that in mind (see ``ui.app``'s viewer cache) so a
second click on "Refresh" does not redo the work, and ``render_mesh`` hands
back the GLB *and* the journal line from a single read of the file — loading a
dense mesh costs half a second before any of the work even starts.

The uploaded mesh is not re-exported at all: the mesh picker is itself an
interactive ``gr.Model3D``, so Gradio shows what it received in the window that
received it and there is nothing to hand it (see ``ui.app``).
"""
from __future__ import annotations

import logging
import tempfile
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

logger = logging.getLogger("ui.viewers")

# A 500k-face mesh has ~750k unique edges; drawing them all makes a GLB the
# browser cannot rotate. The wireframe is sub-sampled above this.
MAX_WIREFRAME_EDGES = 60_000
DEFAULT_COLOR = "#9e9e9e"

# Every viewer export lives in per-job ``view/`` directory, so clearing that
# pattern can never touch an artifact of the API.
_GLOB = "*.glb"


def view_name(mesh_filename: str | Path) -> str:
    """GLB name of a viewer export, taken from the file it displays.

    ``box_mesh_processed.stl`` -> ``box_mesh_processed.glb``. Its download
    button hands the browser this name, so the file the user gets is named after
    the image they submitted rather than after an internal id.
    """
    return f"{Path(mesh_filename).stem}.glb"


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh | None:
    """Load any mesh/GLB as a single mesh, or return ``None`` if unreadable."""
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
    except Exception:  # trimesh raises everything it can: IO, formats, parsing
        return None
    if mesh is None or len(getattr(mesh, "vertices", ())) == 0:
        return None
    return mesh


def clear_views(output_dir: str | Path, keep: Iterable[str | Path] = ()) -> None:
    """Drop the previous viewer GLBs of a job so they do not pile up.

    ``keep`` lists the GLBs of the current render: a re-render that *reuses* an
    export instead of writing it again would otherwise delete it on the way in.
    """
    directory = Path(output_dir)
    if not directory.is_dir():
        return
    kept = {str(Path(path)) for path in keep}
    for stale in directory.glob(_GLOB):
        if str(stale) not in kept:
            stale.unlink(missing_ok=True)


def _tint(mesh: trimesh.Trimesh, color: str) -> None:
    """Paint the whole surface with one flat colour.

    The colour goes to the *vertex* colours on purpose. Assigning face colours
    instead lets trimesh convert them to vertex colours at export time through
    ``mesh.faces_sparse``, which imports ``scipy.sparse`` — absent from the
    front-end image by design. That conversion raised, the exception was caught
    below, and both viewers stayed empty in the browser while every other part
    of the run looked successful.
    """
    rgba = np.asarray(trimesh.visual.color.hex_to_rgba(color), dtype=np.uint8)
    mesh.visual.vertex_colors = np.tile(rgba, (len(mesh.vertices), 1))


def export_glb(
    mesh: trimesh.Trimesh,
    *,
    color: str = DEFAULT_COLOR,
    wireframe: bool = False,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> str | None:
    """Write an already-loaded mesh as a viewer-ready GLB, or ``None`` on failure.

    ``filename`` is the name of the exported GLB — what the viewer's ⤓ button
    hands the browser (see ``view_name``); a random one is used when omitted.

    ``None`` is a legitimate answer (nothing to show), so failures are logged:
    a silent ``None`` is indistinguishable from an empty viewer in the browser.
    """
    directory = Path(output_dir) if output_dir else Path(tempfile.gettempdir())
    destination = directory / (filename or f"view_{uuid.uuid4().hex[:8]}.glb")
    try:
        directory.mkdir(parents=True, exist_ok=True)
        _tint(mesh, color)
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="mesh")
        if wireframe:
            edges = _edge_path(mesh)
            if edges is not None:
                scene.add_geometry(edges, node_name="edges")
        scene.export(str(destination), file_type="glb")
    except Exception as exc:  # a viewer failure must never fail the run
        logger.warning("could not export a viewer GLB for %s: %s", destination.name, exc)
        return None
    return str(destination)


def viewer_glb(
    mesh_path: str | Path | None,
    *,
    color: str = DEFAULT_COLOR,
    wireframe: bool = False,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> str | None:
    """Export ``mesh_path`` as a viewer-ready GLB, or ``None`` on failure.

    Reads the mesh, exports it and forgets it — the front-end uses
    ``render_mesh`` to get the GLB *and* the mesh's characteristics out of a
    single read, which is half the work of asking this twice.
    """
    if not mesh_path:
        return None
    mesh = load_mesh(mesh_path)
    if mesh is None:
        logger.warning("unreadable mesh, no viewer generated: %s", mesh_path)
        return None
    return export_glb(
        mesh,
        color=color,
        wireframe=wireframe,
        output_dir=output_dir,
        filename=filename,
    )


def render_mesh(
    mesh_path: str | Path,
    *,
    color: str = DEFAULT_COLOR,
    wireframe: bool = False,
    output_dir: str | Path | None = None,
    filename: str | None = None,
    label: str | None = None,
    describe: bool = True,
) -> tuple[str | None, str]:
    """Export the viewer GLB *and* describe the mesh, reading the file once.

    Reading is the expensive half — half a second for a dense GLB, and seconds
    more for its watertightness and volume — and the two answers are wanted at
    the same moment, so they come out of one read instead of two.

    Returns ``(GLB path or None, journal line)``. The line is empty when
    ``describe`` is false, which is what a re-render asks for: a mesh does not
    change when its colour does.
    """
    mesh = load_mesh(mesh_path)
    if mesh is None:
        logger.warning("unreadable mesh, no viewer generated: %s", mesh_path)
        return None, f"{label or Path(mesh_path).name} — ❌ unreadable"
    export = export_glb(
        mesh,
        color=color,
        wireframe=wireframe,
        output_dir=output_dir,
        filename=filename,
    )
    line = "" if not describe else mesh_line(mesh_path, label=label, mesh=mesh)
    return export, line


def _edge_path(mesh: trimesh.Trimesh) -> trimesh.path.Path3D | None:
    """The mesh edges, sub-sampled, as a line-only path for the GLB."""
    edges = mesh.edges_unique
    if len(edges) == 0:
        return None
    if len(edges) > MAX_WIREFRAME_EDGES:
        keep = np.linspace(0, len(edges) - 1, MAX_WIREFRAME_EDGES).astype(int)
        edges = edges[keep]
    vertices = np.asarray(mesh.vertices)[edges].reshape(-1, 3)
    pairs = np.arange(len(edges) * 2).reshape(-1, 2)
    entities = [trimesh.path.entities.Line(points=pair) for pair in pairs]
    try:
        return trimesh.path.Path3D(entities=entities, vertices=vertices)
    except Exception:
        return None


def face_count(mesh_path: str | Path) -> int | None:
    """Number of faces of a mesh file, or ``None`` if unreadable."""
    mesh = load_mesh(mesh_path)
    return None if mesh is None else len(mesh.faces)


def mesh_stats(
    mesh_path: str | Path, *, mesh: trimesh.Trimesh | None = None
) -> dict[str, Any] | None:
    """Faces, vertices, bounds, watertightness and volume of a mesh file.

    ``None`` when the file cannot be read as a mesh — the caller decides how to
    report that (the journal logs it, a viewer export skips it). ``mesh`` lets a
    caller that has the mesh in hand (``render_mesh``) hand it over rather than
    pay for a second read of the same file.
    """
    loaded = mesh if mesh is not None else load_mesh(mesh_path)
    if loaded is None:
        return None
    watertight = bool(loaded.is_watertight)
    try:
        volume: float | None = float(loaded.volume) if watertight else None
    except Exception:  # a broken mesh can complain about its own volume
        volume = None
    return {
        "name": Path(mesh_path).name,
        "faces": len(loaded.faces),
        "vertices": len(loaded.vertices),
        "extents": tuple(float(value) for value in loaded.extents),
        "watertight": watertight,
        "volume": volume,
    }


def mesh_line(
    mesh_path: str | Path,
    label: str | None = None,
    *,
    mesh: trimesh.Trimesh | None = None,
) -> str:
    """Characteristics of a mesh file, on one line — what the journal logs.

    Plain text on purpose: it goes to a text box, where the ``**`` of markdown
    would show up literally. ``label`` names the mesh the way the page does
    ("raw", "preprocessed"), so the line reads without the file name. ``mesh``
    hands over an already-loaded mesh (see ``mesh_stats``).
    """
    stats = mesh_stats(mesh_path, mesh=mesh)
    name = label or Path(mesh_path).name
    if stats is None:
        return f"{name} — ❌ unreadable"
    west, north, height = stats["extents"]
    parts = [
        f"{stats['faces']:,} faces",
        f"{stats['vertices']:,} vertices",
        f"{west:.3f} × {north:.3f} × {height:.3f}",
        "watertight" if stats["watertight"] else "non watertight",
    ]
    if stats["volume"] is not None:
        parts.append(f"volume {stats['volume']:.4f}")
    return f"{name} — " + " · ".join(parts)
