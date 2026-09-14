"""(mesh + image) -> textured mesh orchestration.

The exact sequence, minus the UI bookkeeping:

    1. keep the uploaded 3D file in the job directory (``<key>_mesh_src.<ext>``);
    2. keep the reference image likewise (``<key>_image_src.<ext>``) and save the
       RGBA version the paint model is actually given
       (``<key>_image_processed.png``);
    3. prepare the geometry — clean, Taubin-smooth, decimate, and generate a UV
       atlas when the upload has none (``<key>_mesh_prepared.glb``);
    4. paint it (``<key>_mesh_textured.glb``);
    5. describe every artifact with a size, a sha256 and a download URL.

Steps 3-4 are skipped when ``reuse_cached`` is set and the artifacts of the same
(mesh, image) pair are already on disk.

Unlike the shape service, a paint failure is **not** swallowed: there is no
"the rest of the job is still useful" here. The prepared mesh is an intermediate
the caller never asked for, so returning it with an error attached would be a
200 with nothing in it — every failure raises, with the envelope the API
documents.
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..config import Settings
from ..exceptions import (
    ApiError,
    GenerationError,
    MeshLoadError,
    PreprocessError,
    TextureError,
)
from ..schemas import (
    Artifact,
    MeshArtifact,
    PaintParameters,
    TextureResponse,
    UvInfo,
)
from ..storage import JobStore, utcnow
from .mesh import mesh_stats, prepare_mesh
from .paint import PaintingEngine

logger = logging.getLogger(__name__)

# The prepared geometry is written as GLB, not STL: the UV atlas the unwrap
# generates has to survive the round-trip, and STL cannot carry one.
PREPARED_SUFFIX = "_mesh_prepared.glb"
TEXTURED_SUFFIX = "_mesh_textured.glb"
MESH_SRC_SUFFIX = "_mesh_src"
IMAGE_SRC_SUFFIX = "_image_src"
PROCESSED_SUFFIX = "_image_processed.png"
METADATA_KEY = "response"


def prepared_mesh_name(key: str) -> str:
    """Output name of the geometry that was painted."""
    return f"{key}{PREPARED_SUFFIX}"


def textured_mesh_name(key: str) -> str:
    """Output name of the coloured result."""
    return f"{key}{TEXTURED_SUFFIX}"


def mesh_source_name(key: str, filename: str | None) -> str:
    """Stored copy of the uploaded 3D file, keeping its format."""
    suffix = Path(filename or "").suffix.lower() or ".glb"
    return f"{key}{MESH_SRC_SUFFIX}{suffix}"


def image_source_name(key: str, filename: str | None) -> str:
    """Stored copy of the uploaded reference image, keeping its format."""
    suffix = Path(filename or "").suffix.lower() or ".png"
    return f"{key}{IMAGE_SRC_SUFFIX}{suffix}"


def file_url(base_url: str, key: str, filename: str) -> str:
    """Download URL of one artifact, shared by the pipeline and the routes."""
    return (
        f"{base_url.rstrip('/')}/api/v1/texture/jobs/"
        f"{quote(key, safe='')}/files/{quote(filename, safe='')}"
    )


class TexturePipeline:
    """Runs one paint job: prepare the geometry, then texture it."""

    def __init__(
        self,
        settings: Settings,
        engine: PaintingEngine,
        store: JobStore,
    ):
        self._settings = settings
        self._engine = engine
        self._store = store

    # -- public entry point ---------------------------------------------

    def run(
        self,
        *,
        job_key: str,
        mesh_path: Path,
        mesh_filename: str,
        image_path: Path,
        image_filename: str,
        params: PaintParameters,
        base_url: str,
    ) -> TextureResponse:
        job_dir = self._store.prepare_job_dir(job_key)
        mesh_src_name = mesh_source_name(job_key, mesh_filename)
        image_src_name = image_source_name(job_key, image_filename)
        prepared_name = prepared_mesh_name(job_key)
        textured_name = textured_mesh_name(job_key)
        processed_name = f"{job_key}{PROCESSED_SUFFIX}"

        prepared_path = job_dir / prepared_name
        textured_path = job_dir / textured_name
        processed_path = job_dir / processed_name

        # Keep both uploads inside the job directory: downloads then all live
        # under one URL space, and the job survives an upload-cache wipe.
        _adopt(mesh_path, job_dir / mesh_src_name)
        _adopt(image_path, job_dir / image_src_name)

        started = time.perf_counter()
        cache_complete = prepared_path.is_file() and textured_path.is_file()
        if params.reuse_cached and cache_complete:
            logger.info("[%s] reusing the cached artifacts", job_key)
            response = self._response_from_cache(
                job_key=job_key,
                mesh_src_name=mesh_src_name,
                image_src_name=image_src_name,
                prepared_name=prepared_name,
                textured_name=textured_name,
                processed_name=processed_name,
                params=params,
                base_url=base_url,
                duration_s=time.perf_counter() - started,
            )
        else:
            response = self._run_generation(
                job_key=job_key,
                mesh_src_name=mesh_src_name,
                image_src_name=image_src_name,
                prepared_path=prepared_path,
                prepared_name=prepared_name,
                textured_path=textured_path,
                processed_path=processed_path,
                params=params,
                base_url=base_url,
                started=started,
            )

        self._store.write_metadata(job_key, {METADATA_KEY: response.model_dump(mode="json")})
        return response

    # -- fresh run -------------------------------------------------------

    def _run_generation(
        self,
        *,
        job_key: str,
        mesh_src_name: str,
        image_src_name: str,
        prepared_path: Path,
        prepared_name: str,
        textured_path: Path,
        processed_path: Path,
        params: PaintParameters,
        base_url: str,
        started: float,
    ) -> TextureResponse:
        job_dir = self._store.job_dir(job_key)
        rgba = self._prepare_image(
            job_key=job_key,
            image_path=job_dir / image_src_name,
            processed_path=processed_path,
        )
        try:
            logger.info(
                "[%s] Taubin(%d) + decimation to %d faces + unwrap=%s",
                job_key,
                params.taubin_steps,
                params.target_faces,
                params.uv_unwrap,
            )
            try:
                mesh, uv = prepare_mesh(
                    job_dir / mesh_src_name,
                    prepared_path,
                    target_faces=params.target_faces,
                    taubin_steps=params.taubin_steps,
                    taubin_lambda=params.taubin_lambda,
                    taubin_mu=params.taubin_mu,
                    uv_unwrap=params.uv_unwrap,
                    # GLB so the atlas survives; see PREPARED_SUFFIX.
                    export_uvs=True,
                )
            except ApiError:
                raise
            except Exception as exc:
                raise GenerationError(f"could not prepare the mesh: {exc}") from exc

            logger.info(
                "[%s] texturing with %s at %dpx (seed %d)",
                job_key,
                self._settings.subfolder,
                params.texture_size,
                params.seed,
            )
            textured_path.unlink(missing_ok=True)
            try:
                textured = self._engine.paint(
                    mesh,
                    rgba,
                    texture_size=params.texture_size,
                    seed=params.seed,
                )
            except ApiError:
                # A `TextureError` (or a `VramError`) already carries the right
                # status and code - let it through instead of flattening it into
                # a generic generation failure.
                raise
            except Exception as exc:
                raise TextureError(f"Hunyuan3D paint failed: {exc}") from exc
            try:
                textured.export(str(textured_path))
            except Exception as exc:
                raise TextureError(
                    f"could not export the textured mesh: {exc}"
                ) from exc
        finally:
            rgba.close()

        return TextureResponse(
            job_id=job_key,
            job_key=job_key,
            created_at=utcnow(),
            duration_s=time.perf_counter() - started,
            device=self._engine.device or "unknown",
            reused_cache=False,
            input_mesh=self._artifact("input_mesh", job_key, mesh_src_name, base_url),
            mesh=self._mesh_artifact("mesh", job_key, prepared_name, base_url),
            source_image=self._artifact(
                "source_image", job_key, image_src_name, base_url
            ),
            processed_image=self._artifact(
                "processed_image", job_key, processed_path.name, base_url
            ),
            textured_mesh=self._mesh_artifact(
                "textured_mesh", job_key, textured_path.name, base_url
            ),
            uv=uv,
            parameters=params,
            files=self._files(
                job_key,
                [mesh_src_name, prepared_name, image_src_name, processed_path.name,
                 textured_path.name],
                base_url,
            ),
        )

    def _prepare_image(self, *, job_key: str, image_path: Path, processed_path: Path) -> Any:
        """Return the RGBA image the paint model is given, and keep a copy.

        No rembg here, unlike the shape service: the paint model de-lights the
        reference image itself and background removal is the shape stage's job.
        A reference image that arrives with a background, however, still paints
        that background onto the mesh — which is why the ``_processed.png`` is
        kept on disk and downloadable, so the operator can see exactly what was
        given to the model.
        """
        from PIL import Image, UnidentifiedImageError

        try:
            source = Image.open(image_path)
            source.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise PreprocessError(
                f"the uploaded file is not a readable image: {exc}"
            ) from exc
        rgba = source.convert("RGBA")
        source.close()
        rgba.save(processed_path)
        logger.info("[%s] reference image %dx%d", job_key, rgba.width, rgba.height)
        return rgba

    # -- cached run ------------------------------------------------------

    def _response_from_cache(
        self,
        *,
        job_key: str,
        mesh_src_name: str,
        image_src_name: str,
        prepared_name: str,
        textured_name: str,
        processed_name: str,
        params: PaintParameters,
        base_url: str,
        duration_s: float,
    ) -> TextureResponse:
        stored = (self._store.read_metadata(job_key) or {}).get(METADATA_KEY) or {}
        uv = stored.get("uv")
        uv_info = UvInfo.model_validate(uv) if uv else None
        if uv_info is None:
            # An older payload with no UV record: recompute it from the file
            # rather than inventing one. The geometry is on disk either way.
            uv_info = _fallback_uv_info(self._store.job_dir(job_key) / prepared_name)

        present = [
            name
            for name in (
                mesh_src_name,
                prepared_name,
                image_src_name,
                processed_name,
                textured_name,
            )
            if (self._store.job_dir(job_key) / name).is_file()
        ]
        return TextureResponse(
            job_id=job_key,
            job_key=job_key,
            created_at=utcnow(),
            duration_s=duration_s,
            device=self._engine.device or stored.get("device", "unknown"),
            reused_cache=True,
            input_mesh=self._artifact("input_mesh", job_key, mesh_src_name, base_url),
            mesh=self._mesh_artifact("mesh", job_key, prepared_name, base_url),
            source_image=self._artifact(
                "source_image", job_key, image_src_name, base_url
            ),
            processed_image=self._artifact(
                "processed_image", job_key, processed_name, base_url
            ),
            textured_mesh=self._mesh_artifact(
                "textured_mesh", job_key, textured_name, base_url
            ),
            uv=uv_info,
            parameters=params,
            files=self._files(job_key, present, base_url),
        )

    # -- artifact helpers ------------------------------------------------

    @staticmethod
    def _file_url(base_url: str, key: str, filename: str) -> str:
        return file_url(base_url, key, filename)

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _artifact(self, name: str, key: str, filename: str, base_url: str) -> Artifact:
        path = self._store.artifact_path(key, filename)
        return Artifact(
            name=name,
            filename=filename,
            url=self._file_url(base_url, key, filename),
            size_bytes=path.stat().st_size,
            sha256=self._sha256(path),
        )

    def _mesh_artifact(
        self, name: str, key: str, filename: str, base_url: str
    ) -> MeshArtifact:
        artifact = self._artifact(name, key, filename, base_url)
        stats = mesh_stats(self._store.artifact_path(key, filename))
        return MeshArtifact(**artifact.model_dump(), stats=stats)

    def _files(self, key: str, names: list[str], base_url: str) -> dict[str, str]:
        return {name: self._file_url(base_url, key, name) for name in names}


def _adopt(source: Path, destination: Path) -> None:
    """Copy an upload into the job directory, unless it is already there.

    The route streams uploads to the upload directory first (so the size limit
    is enforced while copying), then the pipeline adopts them: one layout for
    everything a job needs, and no dependency on the upload cache surviving.
    """
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    try:
        shutil.copyfile(source, destination)
    except OSError as exc:
        raise MeshLoadError(f"could not store the upload: {exc}") from exc


def _fallback_uv_info(prepared_path: Path) -> UvInfo:
    """Best-effort ``UvInfo`` for a payload that predates the field.

    Reports what can be read off the prepared file — its face count and whether
    it carries UVs — and nothing it cannot know (whether a *smoothed* pass ran).
    """
    from .mesh import has_uvs, load_mesh

    try:
        mesh = load_mesh(prepared_path)
    except Exception:  # pragma: no cover - the file is corrupt or gone
        return UvInfo(
            unwrapped=False,
            had_uvs=False,
            smoothed=False,
            faces_before=0,
            faces_after=0,
            vertices_before=0,
            vertices_after=0,
        )
    faces, vertices = len(mesh.faces), len(mesh.vertices)
    carries_uvs = has_uvs(mesh)
    return UvInfo(
        unwrapped=False,
        had_uvs=carries_uvs,
        smoothed=False,
        faces_before=faces,
        faces_after=faces,
        vertices_before=vertices,
        vertices_after=vertices,
    )
