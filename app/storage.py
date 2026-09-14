"""Job storage.

Layout (one directory per (mesh, image) pair):

    <output_dir>/<job_key>/
        <job_key>_mesh_src<ext>        # the uploaded 3D file, as received
        <job_key>_image_src<ext>       # the uploaded reference image, as received
        <job_key>_image_processed.png  # the RGBA image the paint model saw
        <job_key>_mesh_prepared.glb    # cleaned + smoothed + decimated + UV-mapped
        <job_key>_mesh_textured.glb    # the coloured result
        job.json                       # last response payload, for GET /jobs/{id}

Keeping the artifacts under a key derived from the *two* upload names (rather
than a random job id) is what makes ``reuse_cached`` work: resubmitting the same
pair reuses the files already on disk instead of re-running a multi-minute GPU
job.

The two size limits differ on purpose (see ``Settings.max_upload_mb`` and
``max_mesh_upload_mb``): a reference image is a photo, a mesh is geometry, and
the same ceiling for both would either refuse legitimate meshes or allow a
photo large enough to blow up the process reading it.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from .config import Settings
from .exceptions import ArtifactNotFound, JobNotFound, UploadTooLarge

METADATA_FILENAME = "job.json"

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_ID_LENGTH = 48
_COPY_CHUNK = 1024 * 1024


def sanitize_id(filename: str | None, fallback: str = "input") -> str:
    """Turn an uploaded file name into a safe, stable identifier fragment."""
    stem = Path(filename or "").stem or fallback
    stem = _UNSAFE_CHARS.sub("_", stem).strip("._-")
    return stem[:_MAX_ID_LENGTH] or fallback


def job_key_for(mesh_filename: str | None, image_filename: str | None) -> str:
    """The job directory name of a (mesh, image) pair.

    Both names are in it, so a second image painted onto the same mesh is a
    different job — which is the common way this service is used, and the
    reason a mesh-only key would silently serve the wrong texture from cache.
    """
    mesh_part = sanitize_id(mesh_filename, "mesh")
    image_part = sanitize_id(image_filename, "image")
    return f"{mesh_part}_{image_part}"[:_MAX_ID_LENGTH * 2]


class JobStore:
    """Owns every read/write on the output and upload directories."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self.output_dir = Path(settings.output_dir)
        self.upload_dir = Path(settings.upload_dir)

    # -- setup -----------------------------------------------------------

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # -- paths -----------------------------------------------------------

    def job_key_for(self, mesh_filename: str | None, image_filename: str | None) -> str:
        return job_key_for(mesh_filename, image_filename)

    def job_dir(self, key: str) -> Path:
        return self.output_dir / key

    def prepare_job_dir(self, key: str) -> Path:
        job_dir = self.job_dir(key)
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def job_exists(self, key: str) -> bool:
        return self.job_dir(key).is_dir()

    def metadata_path(self, key: str) -> Path:
        return self.job_dir(key) / METADATA_FILENAME

    # -- uploads ---------------------------------------------------------

    def save_upload(
        self,
        filename: str | None,
        stream: BinaryIO,
        *,
        role: str = "mesh",
    ) -> tuple[str, Path]:
        """Stream an upload to disk, enforcing the matching size limit.

        Returns ``(stem, path)``. The size check happens while copying, so an
        oversized body is rejected without ever being held in memory. ``role``
        picks the limit: ``"mesh"`` for geometry, anything else for an image.
        """
        self.ensure_dirs()
        stem = sanitize_id(filename, role)
        suffix = Path(filename or "").suffix.lower() or (".glb" if role == "mesh" else ".png")
        destination = self.upload_dir / f"{stem}{suffix}"

        is_mesh = role == "mesh"
        limit = (
            self._settings.max_mesh_upload_bytes if is_mesh else self._settings.max_upload_bytes
        )
        limit_mb = (
            self._settings.max_mesh_upload_mb if is_mesh else self._settings.max_upload_mb
        )
        label = "mesh" if is_mesh else "image"

        written = 0
        try:
            with destination.open("wb") as handle:
                while True:
                    chunk = stream.read(_COPY_CHUNK)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise UploadTooLarge(
                            f"{label} exceeds the {limit_mb} MB limit",
                            details={
                                "max_upload_mb": limit_mb,
                                "received_bytes": written,
                                "role": label,
                            },
                        )
                    handle.write(chunk)
        except UploadTooLarge:
            destination.unlink(missing_ok=True)
            raise
        except OSError as exc:
            destination.unlink(missing_ok=True)
            raise UploadTooLarge(f"could not store the upload: {exc}") from exc

        if written == 0:
            destination.unlink(missing_ok=True)
            raise UploadTooLarge(f"the uploaded {label} is empty")

        return stem, destination

    # -- metadata --------------------------------------------------------

    def write_metadata(self, key: str, payload: dict[str, Any]) -> None:
        job_dir = self.prepare_job_dir(key)
        tmp = job_dir / f".{METADATA_FILENAME}.tmp"
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.metadata_path(key))

    def read_metadata(self, key: str) -> dict[str, Any] | None:
        path = self.metadata_path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def require_metadata(self, key: str) -> dict[str, Any]:
        metadata = self.read_metadata(key)
        if metadata is None:
            raise JobNotFound(f"unknown job {key!r}")
        return metadata

    def list_jobs(self, limit: int = 50, offset: int = 0) -> tuple[int, list[dict[str, Any]]]:
        """Return ``(total, page)`` of jobs, most recent first."""
        if not self.output_dir.is_dir():
            return 0, []
        entries: list[tuple[float, dict[str, Any]]] = []
        for metadata_path in self.output_dir.glob(f"*/{METADATA_FILENAME}"):
            try:
                mtime = metadata_path.stat().st_mtime
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            entries.append((mtime, payload))
        entries.sort(key=lambda item: item[0], reverse=True)
        page = [payload for _, payload in entries[offset : offset + limit]]
        return len(entries), page

    # -- artifacts -------------------------------------------------------

    def artifact_path(self, key: str, filename: str) -> Path:
        """Resolve ``filename`` inside the job directory, rejecting escapes."""
        job_dir = self.job_dir(key).resolve()
        candidate = (job_dir / filename).resolve()
        if candidate.parent != job_dir or not candidate.is_file():
            raise ArtifactNotFound(f"{filename!r} is not part of job {key!r}")
        return candidate

    def delete_job(self, key: str) -> bool:
        job_dir = self.job_dir(key)
        if not job_dir.is_dir():
            return False
        shutil.rmtree(job_dir)
        return True


def utcnow() -> datetime:
    return datetime.now(UTC)
