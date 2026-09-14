"""Mesh preparation: read the upload, clean it, UV-unwrap it, hand it to paint.

The paint model needs three things the upload may not have:

1. **UVs.** Its whole output is a baked texture atlas, so a mesh with no UV
   layer cannot be textured at all. This is the common case, not the edge one:
   the shape service's raw export is a marching-cubes mesh (``_mesh_raw.glb``)
   and a bare ``.stl`` carries no UVs either. ``xatlas`` generates the atlas.
2. **A sane face count.** A raw marching-cubes mesh is 200k-500k faces; both
   the rasterizer's memory and the bake time scale with it, and the texture is
   baked at a fixed resolution anyway, so the extra geometry buys nothing.
   Taubin smoothing then quadric decimation, exactly as the shape service does
   it, so both halves of the pipeline agree on what "processed" means.
3. **Clean topology.** Duplicate/degenerate faces make xatlas produce a
   swiss-cheese atlas and the rasterizer produce holes in the render.

Everything here is CPU work on trimesh + xatlas. pymeshlab is used for the
smoothing/decimation pass *when it is installed and works*, because it cleans
topology better, but trimesh is a complete fallback: this service must not lose
a paint run over a missing C++ binding.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..exceptions import MeshLoadError, PreprocessError
from ..schemas import MeshStats, UvInfo

logger = logging.getLogger(__name__)

# A UV atlas needs a little padding or the bake bleeds between islands.
DEFAULT_UV_PADDING = 2
# xatlas on a 500k-face mesh is slow (tens of seconds) and pointless: the bake
# resolution caps what the atlas can hold. Decimation runs first for that
# reason, and this is the ceiling it aims for when the caller asked for more.
UV_UNWRAP_FACE_CEILING = 120_000


def load_mesh(path: str | Path) -> Any:
    """Read any supported 3D file as a single ``trimesh.Trimesh``.

    A ``trimesh.Scene`` (what a multi-object GLB loads as) is concatenated:
    the paint model takes one mesh, and a scene's per-node transforms would
    otherwise be dropped silently by ``force="mesh"``.
    """
    import trimesh

    try:
        loaded = trimesh.load(str(path), force=None)
    except Exception as exc:
        raise MeshLoadError(f"could not read the mesh: {exc}") from exc
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise MeshLoadError("the uploaded file contains no geometry")
        try:
            mesh = trimesh.util.concatenate(geometries)
        except Exception as exc:
            raise MeshLoadError(
                f"could not merge the objects of this scene: {exc}"
            ) from exc
    else:
        mesh = loaded
    if mesh is None or len(getattr(mesh, "vertices", ())) == 0:
        raise MeshLoadError("the uploaded file contains no geometry")
    if len(getattr(mesh, "faces", ())) == 0:
        raise MeshLoadError("the uploaded file contains no faces (a point cloud?)")
    return mesh


def has_uvs(mesh: Any) -> bool:
    """True when the mesh carries a usable UV layer.

    ``visual.uv`` exists but is empty on most non-UV exports, so the length is
    checked rather than the attribute: an empty array is the same as none for
    the rasterizer, but a truthiness test on the ndarray would be ambiguous.
    """
    uv = getattr(getattr(mesh, "visual", None), "uv", None)
    if uv is None:
        return False
    try:
        return len(uv) == len(mesh.vertices)
    except TypeError:  # pragma: no cover - an unexpected type
        return False


def clean_mesh(mesh: Any) -> Any:
    """Drop degenerate and duplicate faces, then fix the winding.

    The same cleanup the shape service's trimesh fallback runs, and for the same
    reason: a marching-cubes mesh comes out with slivers and duplicated faces,
    and both break the UV atlas (islands that overlap or have no area).
    """
    mesh = mesh.copy()
    try:
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    except Exception as exc:  # pragma: no cover - a very broken mesh
        logger.warning("could not clean the mesh topology: %s", exc)
        return mesh
    try:
        mesh.fix_normals()
    except Exception as exc:  # pragma: no cover - non-manifold input
        # Not fatal: the rasterizer mostly cares about the winding it can see,
        # and xatlas does not need consistent normals at all.
        logger.debug("could not fix the normals: %s", exc)
    return mesh


def decimate(mesh: Any, target_faces: int) -> Any:
    """Quadric-decimate down to ``target_faces`` (0 or more-than-now = no-op)."""
    if target_faces <= 0 or len(mesh.faces) <= target_faces:
        return mesh
    try:
        # face_count is keyword-only in trimesh 4.x: passing it positionally
        # hits the ``percent`` parameter instead.
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except (ImportError, ValueError) as exc:
        # trimesh delegates to the optional fast_simplification package.
        # Without it the mesh is still usable, just heavier than asked for.
        logger.warning(
            "quadric decimation unavailable (%s); keeping %d faces",
            exc,
            len(mesh.faces),
        )
        return mesh


def smooth(mesh: Any, *, steps: int, lamb: float, nu: float) -> tuple[Any, bool]:
    """Taubin-smooth the mesh (volume-preserving, unlike plain Laplacian).

    Returns ``(mesh, applied)``. ``applied`` is False when smoothing was not
    asked for *or* when trimesh refused - which happens for real: its
    ``smoothing`` module needs ``scipy.sparse``, so a missing scipy turns the
    whole step into a logged warning. Reporting the request instead of the
    outcome would tell the client a mesh was smoothed when it was exported raw.

    ``nu`` is the *signed* Taubin parameter (the usual ``mu = -0.53``), but
    trimesh wants the positive magnitude - ``filter_taubin`` applies the sign
    itself (``vertices -= nu * dot``). Passing ``-0.53`` straight through makes
    both passes shrink the mesh, and a few dozen iterations collapse it to a
    point. ``abs`` here is what keeps that convention from becoming a bug.
    """
    if steps <= 0:
        return mesh, False
    from trimesh.smoothing import filter_taubin

    try:
        smoothed = filter_taubin(mesh, lamb=lamb, nu=abs(nu), iterations=steps)
    except Exception as exc:  # pragma: no cover - a very broken mesh
        logger.warning("Taubin smoothing failed (%s); keeping the mesh as is", exc)
        return mesh, False
    return smoothed, True


def unwrap_uvs(mesh: Any, *, padding: int = DEFAULT_UV_PADDING) -> Any:
    """Generate a UV atlas with xatlas and write it onto the mesh.

    xatlas wants plain vertices/faces; anything else on the mesh (a texture from
    a previous run, vertex colours) is irrelevant to it and is left alone.
    """
    try:
        import xatlas
    except ImportError as exc:
        raise PreprocessError(
            "xatlas is not installed, so a UV atlas cannot be generated; the "
            "uploaded mesh has no UVs and cannot be textured without one"
        ) from exc

    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    try:
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        # `padding` is a *pack* option in xatlas >= 0.0.9; passing it to
        # generate() as a keyword raises "incompatible function arguments".
        pack_options = xatlas.PackOptions()
        pack_options.padding = padding
        atlas.generate(pack_options=pack_options)
        vmapping, indices, uvs = atlas[0]
    except Exception as exc:
        raise PreprocessError(f"the UV unwrap failed: {exc}") from exc

    import trimesh

    # xatlas may split vertices (the seam has to be cut), so the geometry is
    # rebuilt from its mapping rather than written onto the old arrays.
    unwrapped = trimesh.Trimesh(
        vertices=vertices[vmapping],
        faces=indices,
        process=False,
    )
    unwrapped.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uvs, dtype=np.float64))
    # Carry the original normals over when they survived the rebuild; a mesh
    # without them gets them recomputed on export.
    return unwrapped


def prepare_mesh(
    path: str | Path,
    out_path: str | Path,
    *,
    target_faces: int = 0,
    taubin_steps: int = 0,
    taubin_lambda: float = 0.5,
    taubin_mu: float = -0.53,
    uv_unwrap: bool = True,
    export_uvs: bool = True,
) -> tuple[Any, UvInfo]:
    """Read, clean, smooth, decimate and (if needed) UV-unwrap a mesh file.

    Writes the prepared mesh to ``out_path`` and returns ``(mesh, info)``. The
    written file is what the caller downloads as ``<key>_mesh_prepared.stl``:
    seeing the geometry the texture was actually baked onto is the difference
    between "the colour is wrong" and "the mesh I sent is wrong".

    The order matters — clean, smooth, decimate, unwrap:

    * smoothing a decimated mesh pushes the collapse artefacts around;
    * decimating after unwrapping invalidates the atlas (every collapse moves
      the UV seams);
    * so the atlas is built last, on the final geometry.

    ``export_uvs`` writes a mesh format that can carry UVs (``.glb``/``.obj``)
    instead of ``.stl``, which cannot. Callers that keep the prepared mesh for
    their own use only want the in-memory object; the on-disk copy is for the
    operator.
    """
    mesh = load_mesh(path)
    faces_before = len(mesh.faces)
    vertices_before = len(mesh.vertices)
    had_uvs = has_uvs(mesh)

    mesh = clean_mesh(mesh)
    mesh, smoothed = smooth(mesh, steps=taubin_steps, lamb=taubin_lambda, nu=taubin_mu)

    # xatlas' cost is in the face count, so the ceiling is enforced even when
    # the caller asked for more: an atlas on a 400k-face mesh takes minutes and
    # buys nothing at bake resolution.
    budget = target_faces
    if uv_unwrap and not had_uvs and (budget <= 0 or budget > UV_UNWRAP_FACE_CEILING):
        budget = UV_UNWRAP_FACE_CEILING
    mesh = decimate(mesh, budget)

    unwrapped = False
    if uv_unwrap and not had_uvs:
        mesh = unwrap_uvs(mesh)
        unwrapped = True

    faces_after = len(mesh.faces)
    vertices_after = len(mesh.vertices)
    info = UvInfo(
        unwrapped=unwrapped,
        had_uvs=had_uvs,
        smoothed=smoothed,
        decimated_to=budget if faces_after < faces_before else None,
        faces_before=faces_before,
        faces_after=faces_after,
        vertices_before=vertices_before,
        vertices_after=vertices_after,
    )

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _export(mesh, destination, keep_uvs=export_uvs)
    logger.info(
        "prepared %s -> %s (%d -> %d faces, uvs=%s)",
        Path(path).name,
        destination.name,
        faces_before,
        faces_after,
        "generated" if unwrapped else ("kept" if had_uvs else "none"),
    )
    return mesh, info


def _export(mesh: Any, destination: Path, *, keep_uvs: bool) -> None:
    """Write the prepared mesh, refusing a format that would drop the atlas."""
    if keep_uvs and has_uvs(mesh) and destination.suffix.lower() == ".stl":
        # STL has no concept of UVs: writing one here would silently throw away
        # the atlas the unwrap just paid for.
        raise PreprocessError(
            f"cannot write a UV-mapped mesh to {destination.suffix}: use .glb or .obj"
        )
    try:
        mesh.export(str(destination))
    except Exception as exc:
        raise PreprocessError(f"could not write the prepared mesh: {exc}") from exc


# --------------------------------------------------------------------------- #
# Statistics (cheap, trimesh only - safe to run in the API process)
# --------------------------------------------------------------------------- #

def mesh_stats(path: str | Path) -> MeshStats:
    """Faces / vertices / bounds / watertightness of a mesh file."""
    import trimesh

    try:
        loaded = trimesh.load(str(path), force=None)
    except Exception as exc:
        raise PreprocessError(f"could not read mesh {Path(path).name}: {exc}") from exc
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(list(loaded.geometry.values()))
    return MeshStats(
        faces=len(loaded.faces),
        vertices=len(loaded.vertices),
        extents=[float(value) for value in loaded.extents],
        watertight=bool(loaded.is_watertight),
    )
