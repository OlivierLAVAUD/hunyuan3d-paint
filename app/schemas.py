"""Public API contract: every request/response body the service exchanges.

The models here are the OpenAPI schema too — ``/docs`` renders them directly,
so field descriptions double as API documentation.

The shape is deliberately the mirror image of the shape service's: where that
one takes ``image`` and returns ``raw_mesh`` + ``mesh``, this one takes ``mesh``
**plus** ``image`` and returns ``textured_mesh``. Everything downstream of the
paint model (a job id, a directory of artifacts, absolute download URLs, mesh
statistics) is modelled the same way, so the two front-ends are the same code
with a different form.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

JobStatus = Literal["succeeded"]


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error identifier.")
    message: str = Field(description="Human-readable explanation.")
    details: dict[str, Any] | None = Field(
        default=None, description="Optional extra context (offending value, log tail...)."
    )


class ErrorResponse(BaseModel):
    """Envelope returned for every 4xx/5xx response."""

    error: ErrorDetail


class Artifact(BaseModel):
    """A file produced by a job, downloadable through ``url``."""

    name: str = Field(description="Logical name, e.g. 'textured_mesh' or 'mesh'.")
    filename: str = Field(description="File name inside the job directory.")
    url: str = Field(description="Absolute URL of the download endpoint.")
    size_bytes: int
    sha256: str


class MeshStats(BaseModel):
    faces: int
    vertices: int
    extents: list[float] = Field(description="Bounding-box size on x, y, z.")
    watertight: bool


class MeshArtifact(Artifact):
    stats: MeshStats


class UvInfo(BaseModel):
    """What the geometry preparation did to the mesh before painting."""

    unwrapped: bool = Field(
        description="True when a UV atlas was generated (the input had none)."
    )
    had_uvs: bool = Field(description="True when the upload already carried UVs.")
    smoothed: bool = Field(description="True when Taubin smoothing was applied.")
    decimated_to: int | None = Field(
        default=None,
        description="Face budget the mesh was decimated to, or None when it was "
        "already below it.",
    )
    faces_before: int
    faces_after: int
    vertices_before: int
    vertices_after: int


class PaintParameters(BaseModel):
    """Painting parameters, echoed back for reproducibility."""

    texture_size: int = Field(
        description="Resolution of the baked UV atlas. The main VRAM knob."
    )
    target_faces: int = Field(
        description="Decimation budget applied before painting (0 keeps the "
        "input face count). Lowering it is the other way to fit a small GPU."
    )
    taubin_steps: int
    taubin_lambda: float
    taubin_mu: float
    uv_unwrap: bool = Field(
        description="Generate a UV atlas when the uploaded mesh has none. A "
        "marching-cubes mesh (the shape service's raw GLB, a bare STL) has no "
        "UVs and the paint model cannot texture it without them."
    )
    seed: int = Field(
        description="Multiview diffusion seed. -1 draws a fresh one per job "
        "(the run is then not reproducible)."
    )
    reuse_cached: bool = Field(
        description="Serve artifacts already on disk for this mesh+image pair."
    )

    @field_validator("target_faces", "taubin_steps")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @field_validator("texture_size")
    @classmethod
    def _positive_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("texture_size must be >= 1")
        return value


class TextureResponse(BaseModel):
    """Result of one (mesh + image) -> textured mesh job."""

    job_id: str
    job_key: str = Field(
        description="Stable id derived from the mesh and image file names; the "
        "directory the artifacts live in."
    )
    status: JobStatus = "succeeded"
    created_at: datetime
    duration_s: float
    device: str = Field(description="Torch device the paint model ran on.")
    reused_cache: bool = Field(
        description="True when previously produced artifacts were served as-is."
    )
    input_mesh: Artifact = Field(description="The uploaded 3D file, as received.")
    mesh: MeshArtifact = Field(
        description="The geometry that was actually painted "
        "(<key>_mesh_prepared.stl) - smoothed, decimated and UV-unwrapped. "
        "Download it when you want to see what the texture was baked onto."
    )
    source_image: Artifact = Field(
        description="The uploaded reference image, as received."
    )
    processed_image: Artifact = Field(
        description="Exactly the RGBA image the paint model was given."
    )
    textured_mesh: MeshArtifact = Field(
        description="The coloured result (<key>_mesh_textured.glb), with the "
        "baked UV texture travelling inside the GLB."
    )
    uv: UvInfo
    parameters: PaintParameters
    files: dict[str, str] = Field(
        description="Logical name -> download URL, for clients that just want links."
    )


class JobSummary(BaseModel):
    job_id: str
    job_key: str
    status: str
    created_at: datetime
    duration_s: float | None = None
    faces: int | None = None
    mesh_url: str | None = None
    textured_mesh_url: str | None = None


class JobListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    jobs: list[JobSummary]


class ModelInfo(BaseModel):
    model_id: str
    subfolder: str
    use_safetensors: bool
    device: str | None
    state: str
    loaded: bool
    error: str | None = None
    # Whether the weights are streamed from system RAM instead of being held
    # on the card, and the VRAM budget that mode enforces. Both are worth
    # reporting: an operator wondering why texturing is slow (or why a load
    # was refused) should not have to read the container's environment.
    low_vram: bool = False
    vram_required_gb: float | None = None


class Limits(BaseModel):
    max_upload_mb: int
    max_mesh_upload_mb: int
    min_texture_size: int
    max_texture_size: int
    taubin_steps_max: int
    max_concurrent_jobs: int


class InfoResponse(BaseModel):
    """Service metadata: what the API accepts, defaults, and current model state."""

    name: str
    version: str
    api_prefix: str
    allowed_image_suffixes: list[str]
    allowed_mesh_suffixes: list[str]
    model: ModelInfo = Field(description="The paint (texture) model.")
    defaults: PaintParameters
    limits: Limits


class HealthResponse(BaseModel):
    """Liveness: the process is up and serving HTTP."""

    status: Literal["ok"] = "ok"
    name: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness: the paint model can serve requests.

    ``cuda`` is reported separately from ``model_loaded`` because the paint
    model has no CPU path: a container running on a host without a visible GPU
    is not "still loading", it will never be ready, and saying so up front beats
    a 503 that never clears.
    """

    status: Literal["ready", "loading", "error"]
    model_loaded: bool
    device: str | None = None
    cuda: bool = Field(description="True when torch sees a CUDA device.")
    error: str | None = None
