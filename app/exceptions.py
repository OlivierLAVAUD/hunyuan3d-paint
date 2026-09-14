"""Domain exceptions.

Services raise these; ``app.api.errors`` translates them into the JSON error
envelope described in the README. Same envelope as the shape service, so a
client that learned to read one already reads the other.
"""
from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for every error the API reports to clients."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return {"error": payload}


class InvalidParameter(ApiError):
    status_code = 422
    code = "invalid_parameter"


class UnsupportedImage(ApiError):
    status_code = 415
    code = "unsupported_image"


class UnsupportedMesh(ApiError):
    """The uploaded 3D file is not one we can paint.

    Separate from ``UnsupportedImage`` so the caller can tell which of the two
    uploads the service refused, and separate from ``InvalidParameter`` because
    a 415 is what the front-end's hint table keys on.
    """

    status_code = 415
    code = "unsupported_mesh"


class UploadTooLarge(ApiError):
    status_code = 413
    code = "upload_too_large"


class JobNotFound(ApiError):
    status_code = 404
    code = "job_not_found"


class ArtifactNotFound(ApiError):
    status_code = 404
    code = "artifact_not_found"


class EngineUnavailable(ApiError):
    status_code = 503
    code = "engine_unavailable"


class EngineLoadError(ApiError):
    status_code = 500
    code = "engine_load_failed"


class GenerationError(ApiError):
    status_code = 500
    code = "generation_failed"


class HostMemoryError(GenerationError):
    """The process ran out of *system* RAM, not of VRAM.

    Reported separately from a plain ``GenerationError`` because the remedy is
    different and the message has to say which one it is: a CUDA OOM is fixed by
    lowering the texture size, a host OOM by giving the container more memory.
    """

    code = "host_out_of_memory"


class VramError(GenerationError):
    """The GPU is too small for the paint model.

    Its own code because this is the failure a laptop hits first and it is not
    fixable from inside the container: a 6 GB card cannot hold the ~8 GB
    resident set,
    and "lower the texture size" does not help either. The message names the
    card's total VRAM and what is needed, so the answer is obvious.
    """

    status_code = 503
    code = "insufficient_vram"


class PreprocessError(ApiError):
    status_code = 500
    code = "preprocess_failed"


class TextureError(ApiError):
    status_code = 500
    code = "texture_failed"


class MeshLoadError(ApiError):
    """The uploaded mesh could not be read as a mesh at all."""

    status_code = 422
    code = "mesh_unreadable"
