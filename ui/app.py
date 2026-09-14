"""Gradio front-end for the Hunyuan3D texture (paint) API.

Where the shape service's page is *one image in, two meshes out*, this one is
**a mesh and an image in, a coloured mesh out** — and the viewers follow.

The visualisation windows work on the same principle as the sister app's, which
is the point of the split: the API's artifacts are downloaded into
``H3D_UI_DATA_DIR`` and handed to ``gr.Model3D`` through trimesh re-exports
(``ui.viewers``), so the display does not depend on the browser being able to
reach the API. Three meshes are shown:

* **input mesh** — the upload window *is* its viewer. The picker is an
  interactive ``gr.Model3D``, which Gradio renders as a drop zone while it is
  empty and as the model itself once it holds one: the file that enters the page
  appears in the very window that received it, as soon as it lands — no
  re-export, and no waiting for a run that takes minutes to find out what was
  sent. That window stays the run's *input*, so nothing writes to it afterwards:
  handing it a viewer GLB would swap the mesh the next run sends for the file
  that GLB was exported from;
* **prepared mesh** — the geometry that was actually painted, after cleaning,
  Taubin smoothing, decimation and the UV unwrap (``<key>_mesh_prepared.glb``),
  re-exported so the colour picker and the wireframe apply. Next to the result,
  seeing it is the difference between "the colour is wrong" and "the mesh I
  sent is wrong";
* **textured mesh** — the result, shown *as the API exported it*. It is already a
  GLB carrying its own baked texture, so it is deliberately not re-exported:
  tinting it with the surface colour would throw the texture away. The colour
  picker and the wireframe therefore apply to the prepared mesh only.

Two consequences of the API being synchronous shape the code, exactly as in the
sister app:

* the request runs in a worker thread while the handler keeps yielding, which is
  what makes the elapsed-time banner tick during a multi-minute run;
* the payload a generator *ends* on is delivered twice by Gradio, and a
  ``gr.Model3D`` gets its effect torn down by the second delivery — so the run
  only *produces* the artifacts (progress, status, state) and ``build_ui`` hands
  them to the viewers from a separate single-shot event chained with ``.then``.
  Exporting the viewers is the run's own work, done just before its last
  message, so the chained event only has to hand files over.

A repeat run of the same (mesh, image) pair gets one more trick: its artifacts
are already on disk, so ``preload_views`` exports their viewers in a second
worker thread *while the API paints*, and the render at the end of the run finds
them ready — ``promote_preloads`` keeps only those whose sha256 the API
confirms, so a mesh that was really repainted is exported again.

Because the mesh and the image are two separate uploads, the job key is derived
from both names (``<mesh>_<image>``): painting a second photo onto the same mesh
is a different job, and a mesh-only key would serve the first texture from cache
forever.

Run it with ``python -m ui.app`` (the container does exactly that).
"""
from __future__ import annotations

import hashlib
import logging
import sys
import time
import zipfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr

from app.storage import job_key_for
from ui.client import APIError, TextureClient
from ui.config import UISettings, get_settings
from ui.viewers import clear_views, mesh_line, render_mesh, view_name

logger = logging.getLogger("ui")

# How often the run handler pushes a refreshed elapsed-time banner.
TICK_S = 0.5
PROBE_TIMEOUT_S = 10.0

LOG_MAX_LINES = 500

# Sub-details of a journal line (one artifact, one mesh) are indented under it.
DETAIL_INDENT = "    · "

# One height for the four windows of the page — the mesh picker (which is also
# the viewer of the mesh it received), the source image and the two result
# viewers — so they line up from top to bottom.
VIEW_HEIGHT = 420

# Surface colour of a freshly loaded page, and what a run exports without being
# told otherwise (``ui.viewers`` leaves a mesh grey for the same reason).
DEFAULT_VIEW_COLOR = "#9e9e9e"

# Used when the API is unreachable at startup: the page still has to render,
# since it is precisely the page that reports the API as unreachable.
FALLBACK_DEFAULTS: dict[str, Any] = {
    "texture_size": 768,
    "target_faces": 50_000,
    "taubin_steps": 20,
    "taubin_lambda": 0.5,
    "taubin_mu": -0.53,
    "uv_unwrap": True,
    "seed": 12345,
    "reuse_cached": True,
}
FALLBACK_LIMITS: dict[str, Any] = {
    "min_texture_size": 256,
    "max_texture_size": 2048,
    "taubin_steps_max": 200,
    "max_upload_mb": 32,
    "max_mesh_upload_mb": 256,
}
FALLBACK_MESH_SUFFIXES = (".glb", ".gltf", ".obj", ".ply", ".stl")

_MISSING_MESH = "invalid_response"
_MISSING_MESH_MSG = "incomplete API response (no textured mesh)"

# The "download everything" entry of the picker; a real artifact name looks
# nothing like it, so it can never collide with one.
DOWNLOAD_ALL = "📦 All files (.zip)"


@dataclass
class ViewExport:
    """One mesh of a render: what it is, where its GLB went, how it was made."""

    key: str
    label: str
    artifact: str
    glb: str | None = None
    line: str = ""
    reused: bool = False
    # Set on the exports made before the run finished (``preload_views``): the
    # journal says where the waiting time went, and the cache entry remembers it
    # so the event that displays them can say it too.
    preloaded: bool = False
    # Digest of the artifact the export was made from; the run compares it with
    # the sha256 the API publishes before reusing a preload.
    digest: str = ""


@dataclass
class FormDefaults:
    """Everything the form needs, resolved from the service."""

    defaults: dict[str, Any] = field(default_factory=lambda: dict(FALLBACK_DEFAULTS))
    limits: dict[str, Any] = field(default_factory=lambda: dict(FALLBACK_LIMITS))
    mesh_suffixes: tuple[str, ...] = FALLBACK_MESH_SUFFIXES
    badge: str = ""

    @classmethod
    def fallback(cls, badge: str) -> FormDefaults:
        return cls(badge=badge)


# ---------------------------------------------------------------------------
# API probes (the page must render even when the API is down)
# ---------------------------------------------------------------------------


def api_badge(settings: UISettings) -> str:
    """One-line status of the service, shown at the top of the page."""
    try:
        with TextureClient(settings.api_url, timeout_s=PROBE_TIMEOUT_S) as client:
            info = client.info()
            ready = client.ready()
    except APIError as exc:
        return f"🔴 **API** `{settings.api_url}` — {exc.message}"

    model = info.get("model") or {}
    state = ready.get("status") or model.get("state") or "?"
    icon = {"ready": "🟢", "loading": "🟡", "error": "🔴"}.get(state, "⚪")
    device = ready.get("device") or model.get("device") or "?"
    line = (
        f"{icon} **API** `{settings.api_url}` — state `{state}`, device `{device}`, "
        f"model `{model.get('model_id')}`/`{model.get('subfolder')}` "
        f"(v{info.get('version')})"
    )
    # The paint model has no CPU path, so a host without a visible GPU is a
    # permanent 503: saying which of the two it is saves a long look at /ready.
    if not ready.get("cuda", True):
        line += " · ⚠️ **no CUDA GPU visible** — the paint model cannot run"
    if ready.get("error"):
        line += f" · ❌ {ready['error']}"
    return line


def fetch_form(settings: UISettings) -> FormDefaults:
    """Read ``/info`` so the form matches the service, with a fallback."""
    try:
        with TextureClient(settings.api_url, timeout_s=PROBE_TIMEOUT_S) as client:
            info = client.info()
    except APIError as exc:
        logger.warning("API unreachable at startup: %s", exc)
        return FormDefaults.fallback(api_badge(settings))

    return FormDefaults(
        defaults={**FALLBACK_DEFAULTS, **(info.get("defaults") or {})},
        limits={**FALLBACK_LIMITS, **(info.get("limits") or {})},
        mesh_suffixes=tuple(info.get("allowed_mesh_suffixes") or FALLBACK_MESH_SUFFIXES),
        badge=api_badge(settings),
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes} min {rest:02d} s"


def format_size(num_bytes: float | int | None) -> str:
    """Human-readable size, for the journal (``318 KB``, ``1.20 MB``)."""
    value = float(num_bytes or 0)
    if value < 1024:
        return f"{value:.0f} B"
    for unit in ("KB", "MB", "GB"):
        value /= 1024
        if value < 1024 or unit == "GB":
            return f"{value:.2f} {unit}"
    return f"{value:.2f} GB"


def stamped(message: str, *, elapsed: float | None = None) -> str:
    """One journal line: wall clock, plus the run's elapsed time when known."""
    prefix = f"[{time.strftime('%H:%M:%S')}]"
    if elapsed is not None:
        prefix += f" [{format_elapsed(elapsed)}]"
    return f"{prefix} {message}"


def log_text(lines: list[str] | None) -> str:
    """The journal as the log window shows it."""
    return "\n".join(lines or [])


def logged(state: dict[str, Any] | None, *entries: str) -> tuple[dict[str, Any], str]:
    """Append ``entries`` to the journal carried by ``state``.

    Keeping the lines in the state rather than in a closure is what lets the
    events chained *after* the run (the viewer render, for one) keep writing to
    the same window.
    """
    updated = {**(state or {})}
    lines = [*(updated.get("log") or []), *entries][-LOG_MAX_LINES:]
    updated["log"] = lines
    return updated, log_text(lines)


# state -> (border, background, text colour, icon, label)
_BANNERS = {
    "done": ("#4caf50", "#f1f8e9", "#2e7d32", "✅", "Done"),
    "failed": ("#d32f2f", "#ffebee", "#c62828", "❌", "Failed"),
}


def progress_html(elapsed: float, task: str, state: str = "running") -> str:
    """Spinner + live timer + current step, as a self-contained HTML snippet.

    Same widget as the sister app. ``state`` is ``running`` (animated spinner),
    ``done`` (green) or ``failed`` (red): a single boolean used to render the
    green "Done" banner on a failed run, which is a lie in the most visible
    place of the page.
    """
    banner = _BANNERS.get(state)
    if banner is not None:
        border, background, colour, icon, label = banner
        return (
            '<div style="display:flex;align-items:center;gap:10px;font-size:15px;'
            f'padding:8px 12px;border:1px solid {border};border-radius:8px;'
            f'background:{background};color:{colour};">'
            f"{icon} <b>{label}</b>"
            '<span style="color:#888;">|</span> ⏱️ '
            f"{format_elapsed(elapsed)}"
            '<span style="color:#888;">|</span> '
            f"<span>{task or ''}</span>"
            "</div>"
        )
    return (
        '<div style="display:flex;align-items:center;gap:10px;font-size:15px;'
        'padding:8px 12px;border:1px solid #ddd;border-radius:8px;'
        'background:#fafafa;">'
        '<span style="width:16px;height:16px;border:3px solid #cfd8dc;'
        'border-top-color:#d32f2f;border-radius:50%;display:inline-block;'
        'animation:h3d-ui-spin .8s linear infinite;"></span>'
        f"<b>⏱️ {format_elapsed(elapsed)}</b>"
        '<span style="color:#888;">|</span> '
        f"<span>{task or 'Starting…'}</span>"
        "</div>"
        "<style>@keyframes h3d-ui-spin{to{transform:rotate(360deg)}}</style>"
    )


def result_outputs(
    *,
    progress: str,
    status: str,
    state: dict[str, Any] | None = None,
    log: str = "",
) -> tuple[Any, ...]:
    """The run outputs, in the order declared by ``build_ui``.

    progress, status, state, log — the viewers are filled by the event chained
    after the run, so they are not part of this payload (see the module
    docstring).
    """
    return (progress, status, state, log)


def artifact_files(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every downloadable artifact of a job, the textured mesh first.

    Read from the response rather than from a hard-coded list of names, so an
    artifact the API starts producing shows up in the picker on its own.
    """
    artifacts: list[dict[str, Any]] = []
    for key in (
        "textured_mesh",
        "mesh",
        "input_mesh",
        "processed_image",
        "source_image",
    ):
        artifact = payload.get(key)
        if isinstance(artifact, dict):
            artifacts.append(artifact)
    return [
        artifact
        for artifact in artifacts
        if artifact.get("url") and artifact.get("filename")
    ]


def download_artifacts(
    client: TextureClient, payload: dict[str, Any], job_dir: Path
) -> dict[str, Path]:
    """Fetch every artifact into ``job_dir``; returns ``{filename: local path}``.

    The textured mesh is what makes the response usable: without it the payload
    is incomplete and the viewer would have nothing to show.
    """
    artifacts = artifact_files(payload)
    names = {artifact.get("name") for artifact in artifacts}
    if "textured_mesh" not in names:
        raise APIError(None, _MISSING_MESH, _MISSING_MESH_MSG)
    job_dir.mkdir(parents=True, exist_ok=True)
    local: dict[str, Path] = {}
    for artifact in artifacts:
        filename = str(artifact["filename"])
        local[filename] = client.download(str(artifact["url"]), job_dir / filename)
    return local


def viewer_style(color: str, wireframe: bool) -> str:
    """How a viewer was styled — shown in the journal *and* kept as a cache key.

    An export is tinted and optionally wired, so these two settings are what a
    cached GLB was made with: changing either one invalidates it, while clicking
    "Refresh" a second time does not.
    """
    return f"colour `{color}`" + (", wireframe" if wireframe else "")


def artifact_identity(
    state: dict[str, Any] | None, artifact: str | Path
) -> list[Any] | None:
    """What the viewer cache keys on for one artifact.

    The sha256 the API published with the artifact is enough on its own: it says
    the bytes are the same, so the GLB exported from them still is too — even
    after a run re-downloaded the file and moved its mtime. Without it (an older
    payload, a state built by hand), the file's size and mtime stand in.
    """
    path = Path(artifact)
    digest = ((state or {}).get("digests") or {}).get(path.name)
    if digest:
        return [str(path), str(digest)]
    try:
        stat = path.stat()
    except OSError:
        return None
    return [str(path), str(stat.st_size), str(stat.st_mtime_ns)]


def cached_view(
    state: dict[str, Any] | None,
    key: str,
    artifact: str | Path,
    style: str,
) -> dict[str, Any] | None:
    """The cache entry of an earlier render of these exact bytes, if it still holds.

    ``None`` when nothing was rendered for this mesh yet, when the style changed,
    or when the GLB has been cleaned up since. The whole entry comes back — not
    just the path — because the entry is also what says the export was made
    while the API was painting (see ``preload_views``).
    """
    entry = ((state or {}).get("views") or {}).get(key) or {}
    if entry.get("style") != style:
        return None
    if entry.get("identity") != artifact_identity(state, artifact):
        return None
    path = Path(str(entry.get("glb") or ""))
    return entry if path.is_file() else None


def export_views(
    state: dict[str, Any] | None,
    color: str,
    wireframe: bool,
    views_dir: Path,
    *,
    describe: bool = False,
) -> list[ViewExport]:
    """Export the viewer GLB of every mesh of the job — or reuse the one at hand.

    The prepared mesh is the artifact worth re-exporting: a plain mesh whose
    surface colour and wireframe are the page's to choose. The uploaded mesh is
    not here — its window is the picker, which Gradio fills on its own (see the
    module docstring), and the textured result is deliberately absent because it
    is already a GLB with its own baked texture, which tinting would destroy;
    ``render_views`` hands that file over as is.

    This is where the waiting time went: re-exporting reads a mesh with trimesh
    and writes a new GLB, about a second for a dense mesh and four with the
    wireframe on, for bytes that are byte-for-byte the export already on disk as
    long as neither the mesh nor the style changed.

    ``describe`` asks for the mesh's characteristics as well, out of the same
    read (see ``ui.viewers.render_mesh``); a re-render that only restyles does
    not want them, since a mesh does not change when its colour does.
    """
    style = viewer_style(color, wireframe)
    views: list[ViewExport] = []
    for key, label in (("prepared", "prepared"),):
        artifact = (state or {}).get(key)
        if not artifact or not Path(artifact).is_file():
            continue
        view = ViewExport(key=key, label=label, artifact=str(artifact))
        identity = artifact_identity(state, artifact)
        previous = ((state or {}).get("views") or {}).get(key) or {}
        if previous.get("identity") == identity:
            # Same bytes as last time: what the journal said about them still
            # holds, whether or not the style changed.
            view.line = str(previous.get("line") or "")
        cached = cached_view(state, key, artifact, style)
        if cached:
            view.glb, view.reused = str(cached["glb"]), True
            view.preloaded = bool(cached.get("preloaded"))
            if not view.line:
                view.line = str(cached.get("line") or "")
        else:
            view.glb, view.line = render_mesh(
                artifact,
                color=color,
                wireframe=wireframe,
                output_dir=views_dir,
                filename=view_name(artifact),
                label=label,
                describe=describe,
            )
        if describe and not view.line:
            view.line = mesh_line(artifact, label=label)
        views.append(view)
    return views


def view_cache(
    state: dict[str, Any] | None,
    exports: Iterable[ViewExport],
    style: str,
) -> dict[str, dict[str, Any]]:
    """Cache entries of a render: what a later one can reuse without reading.

    The characteristics come from the mesh alone, so they survive a restyle and
    can be logged again without reading the file.
    """
    return {
        view.key: {
            "glb": view.glb,
            "style": style,
            "identity": artifact_identity(state, view.artifact),
            "line": view.line,
            "preloaded": view.preloaded,
        }
        for view in exports
    }


def viewer_header(exports: Iterable[ViewExport], style: str) -> str:
    """Opening line of the journal block describing what the viewers show."""
    views = list(exports)
    if views and all(view.reused for view in views):
        if any(view.preloaded for view in views):
            return f"🎨 viewers preloaded while the API painted ({style})"
        return f"🎨 viewers already exported ({style})"
    return f"🎨 viewers exported ({style})"


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Digest of a local file, in the same format the API publishes for its own."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_meshes(job_dir: str | Path) -> list[tuple[str, str, Path]]:
    """The re-exportable meshes a run of this job leaves on disk.

    ``(key, label, path)``, discovered rather than built: the names are only
    known to the front-end once the API has sent them, and the case that matters
    here — a repeat run of a pair its owner already painted — is exactly the one
    where the previous ones are still there. The ``view/`` directory holds the
    exported GLBs and is not searched, so a viewer can never be mistaken for an
    artifact.

    Neither the textured result nor the uploaded mesh is in the list: the first
    is not re-exported, the second is shown by the picker (see ``export_views``).
    """
    directory = Path(job_dir)
    if not directory.is_dir():
        return []
    found: list[tuple[str, str, Path]] = []
    for key, label, pattern in (("prepared", "prepared", "*_mesh_prepared.glb"),):
        matches = sorted(path for path in directory.glob(pattern) if path.is_file())
        if matches:
            found.append((key, label, matches[0]))
    return found


def preload_views(
    meshes: Iterable[tuple[str, str, Path]],
    views_dir: Path,
    color: str,
    wireframe: bool,
) -> list[ViewExport]:
    """Export the viewers of meshes already on disk, while the API paints.

    A paint run takes minutes during which this process has nothing to do but
    tick the banner, and a repeat run of the same pair already has its artifacts
    on disk: the expensive half of the display — trimesh reading a dense mesh,
    then writing a GLB, seconds in all — fits in that idle time.

    The digest of each artifact travels back with the export: the run compares it
    with the sha256 the API publishes, so a mesh the API really rewrote (other
    bytes) is exported again instead of being shown stale.
    """
    views: list[ViewExport] = []
    for key, label, artifact in meshes:
        view = ViewExport(key=key, label=label, artifact=str(artifact))
        try:
            view.digest = sha256_file(artifact)
        except OSError as exc:
            logger.warning("could not read %s: %s", artifact, exc)
            continue
        view.glb, view.line = render_mesh(
            artifact,
            color=color,
            wireframe=wireframe,
            output_dir=views_dir,
            filename=view_name(artifact),
            label=label,
        )
        views.append(view)
    return views


def promote_preloads(
    state: dict[str, Any] | None,
    preloaded: Iterable[ViewExport],
    style: str,
) -> dict[str, dict[str, Any]]:
    """Preloads turned into cache entries, kept only if the API confirms the bytes.

    A preload was exported from the artifacts of an earlier run; the run that
    just finished says what the artifacts are *now* (``state["digests"]``, the
    sha256 the API published). Same digest means the same mesh, so the GLB still
    stands — the render at the end of the run reuses it instead of reading the
    mesh again. A different digest means the API rewrote it, and the preload is
    dropped so that the render redoes the work.
    """
    digests = (state or {}).get("digests") or {}
    promoted: dict[str, dict[str, Any]] = {}
    for view in preloaded:
        if not view.glb or not view.digest:
            continue
        if digests.get(Path(view.artifact).name) != view.digest:
            continue
        promoted[view.key] = {
            "glb": view.glb,
            "style": style,
            "identity": artifact_identity(state, view.artifact),
            "line": view.line,
            "preloaded": True,
        }
    return promoted


def artifact_choices(state: dict[str, Any] | None) -> list[str]:
    """Picker options: every file the last run produced, plus the zip of them."""
    files = (state or {}).get("files") or {}
    return [DOWNLOAD_ALL, *files] if files else []


def prepare_download(
    state: dict[str, Any] | None, choice: str | None
) -> tuple[str | None, str]:
    """Resolve a picker choice to a local file, and describe it.

    "All" is zipped: one artifact per link would make "all" mean N downloads,
    which is not what it says. No API call here — the artifacts are already on
    disk, downloaded by the run.
    """
    files: dict[str, str] = (state or {}).get("files") or {}
    if not files:
        return None, "❌ Run the job first to generate the files."

    if choice in (None, "", DOWNLOAD_ALL):
        job_id = str((state or {}).get("job_id") or "job")
        archive_path = Path(next(iter(files.values()))).parent / f"{job_id}_outputs.zip"
        try:
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
                for filename, path in files.items():
                    if Path(path).is_file():
                        archive.write(path, arcname=filename)
        except OSError as exc:
            logger.warning("could not build %s: %s", archive_path, exc)
            return None, f"❌ Could not create the archive: `{exc}`"
        return str(archive_path), _describe(archive_path, len(files))

    path = files.get(str(choice))
    if not path or not Path(path).is_file():
        return None, f"❌ File `{choice}` not found — run the job again."
    return str(path), _describe(Path(path), 1)


def populate_download(
    state: dict[str, Any] | None, choice: str | None
) -> tuple[Any, str]:
    """Arm the download button with the picked artifact, and describe it.

    ``gr.DownloadButton`` downloads its *value* when clicked, so the file has to
    be handed over before the click: this runs whenever the picker changes (and
    right after a run) and stores the choice in the button. Nothing is
    transferred until the user actually clicks.
    """
    path, message = prepare_download(state, choice)
    return gr.update(value=path, interactive=bool(path)), message


def _describe(path: Path, count: int) -> str:
    """One-line summary of what the download button just built."""
    size_mb = path.stat().st_size / (1024 * 1024)
    icon = "📦" if path.suffix == ".zip" else "📄"
    plural = "files" if count > 1 else "file"
    return f"{icon} `{path.name}` — {count} {plural}, {size_mb:.2f} MB"


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def thousands(value: Any) -> str:
    """``50000`` -> ``50,000``: numbers stay readable in the journal."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}" if isinstance(value, int) else str(value)


def params_line(params: dict[str, Any]) -> str:
    """The paint parameters, as one journal line."""
    return (
        "⚙️ texture {}px · budget {} faces · taubin {} (λ {}, µ {}) · uv {} · seed {}"
    ).format(
        params.get("texture_size"),
        thousands(params.get("target_faces")),
        params.get("taubin_steps"),
        params.get("taubin_lambda"),
        params.get("taubin_mu"),
        "auto" if params.get("uv_unwrap") else "off",
        params.get("seed"),
    )


def payload_notes(payload: dict[str, Any]) -> list[str]:
    """What the API answered: geometry preparation, then the paint. One line each.

    The device and the cache hit belong on the same line: "painted on cuda"
    followed by "taken from cache" two lines down contradicts itself, and a
    cached run reports a near-zero duration that means nothing on its own.
    """
    device = payload.get("device") or "?"
    duration = payload.get("duration_s")
    seconds = f"{float(duration):.1f} s" if duration is not None else "unknown duration"
    ready = f"✅ texture ready in {seconds} on `{device}`"
    if payload.get("reused_cache"):
        ready += " · ♻️ taken from cache, paint not re-run"

    lines = [ready]

    uv = payload.get("uv") or {}
    before, after = uv.get("faces_before"), uv.get("faces_after")
    if uv.get("unwrapped"):
        lines.append("🧭 UV atlas generated (the mesh had none)")
    elif uv.get("had_uvs"):
        lines.append("🧭 UV atlas of the mesh kept")
    else:
        lines.append("🧭 no UV atlas (unwrap disabled)")
    if before is not None and after is not None and before != after:
        parts = [f"📐 {thousands(before)} faces → {thousands(after)}"]
        if uv.get("smoothed"):
            parts.append("Taubin applied")
        lines.append(" · ".join(parts))
    elif after is not None:
        lines.append(f"📐 {thousands(after)} faces (unchanged)")

    return lines


def run_texture(
    *,
    settings: UISettings,
    limits: dict[str, Any],
    mesh_path: str | None,
    mesh_name: str | None,
    image_path: str | None,
    image_name: str | None,
    params: dict[str, Any],
    state: dict[str, Any] | None,
    color: str = DEFAULT_VIEW_COLOR,
    wireframe: bool = False,
) -> Iterator[tuple[Any, ...]]:
    """Submit one (mesh, image) pair and stream progress until the result is local.

    A generator: the API only answers when the job is finished, so the POST runs
    in a worker thread while each tick yields a refreshed banner. Every failure
    (unreachable API, rejected parameter, insufficient VRAM, failed download) is
    rendered in the status field instead of raising, so the page stays usable —
    and each one is written to the journal too, which is where the full error
    detail lives.

    The viewers are exported before the last message, so that the event chained
    after the run only has to hand files over and the first display is immediate.
    The meshes are still *not* given to the viewers from here: Gradio cancels a
    ``gr.Model3D`` that gets its value in the last message of a generator.

    A second worker thread exports the viewers of whatever meshes the job
    already has on disk while this one waits for the API — a repeat run of the
    same pair, where the whole display is ready before the GPU returns.
    """
    journal: list[str] = []

    def trim() -> str:
        del journal[:-LOG_MAX_LINES]
        return log_text(journal)

    def note(message: str, *, elapsed: float | None = None) -> str:
        """Add a dated line to the journal; returns the text the window shows."""
        journal.append(stamped(message, elapsed=elapsed))
        return trim()

    def detail(message: str) -> str:
        """Add an indented continuation under the last dated line.

        Sub-details (one artifact, one mesh) belong to the line they explain:
        giving each its own timestamp would make the journal look like a stream
        of independent events.
        """
        journal.append(f"{DETAIL_INDENT}{message}")
        return trim()

    def failed(status: str, message: str, *, elapsed: float = 0.0) -> tuple[Any, ...]:
        """A failure payload: red banner, status detail, and one journal line.

        The state is handed back untouched: a failed run must not wipe the
        previous job's artifacts from the viewers and the download picker.
        """
        return result_outputs(
            progress=progress_html(
                elapsed, "Failed" if elapsed else "", state="failed"
            ),
            status=status,
            state=state,
            log=note(f"❌ {message}", elapsed=elapsed or None),
        )

    if not mesh_path:
        yield failed("❌ Provide a 3D file.", "no 3D file provided")
        return
    if not image_path:
        yield failed("❌ Provide a source image.", "no source image provided")
        return

    mesh_source = Path(mesh_path)
    image_source = Path(image_path)
    if not mesh_source.is_file():
        yield failed(
            f"❌ Mesh not found: `{mesh_path}`",
            f"mesh not found: `{mesh_path}`",
        )
        return
    if not image_source.is_file():
        yield failed(
            f"❌ Image not found: `{image_path}`",
            f"image not found: `{image_path}`",
        )
        return

    mesh_size = mesh_source.stat().st_size
    image_size = image_source.stat().st_size
    max_mesh_mb = float(limits.get("max_mesh_upload_mb") or 0)
    max_image_mb = float(limits.get("max_upload_mb") or 0)
    mesh_mb = mesh_size / (1024 * 1024)
    image_mb = image_size / (1024 * 1024)
    if max_mesh_mb and mesh_mb > max_mesh_mb:
        yield failed(
            f"❌ Mesh too large ({mesh_mb:.1f} MB) — API limit: "
            f"{max_mesh_mb:.0f} MB.",
            f"mesh too large: {format_size(mesh_size)} > limit of "
            f"{max_mesh_mb:.0f} MB",
        )
        return
    if max_image_mb and image_mb > max_image_mb:
        yield failed(
            f"❌ Image too large ({image_mb:.1f} MB) — API limit: "
            f"{max_image_mb:.0f} MB.",
            (
                f"image too large: {format_size(image_size)} > limit of "
                f"{max_image_mb:.0f} MB"
            ),
        )
        return

    job_id = job_key_for(mesh_name or mesh_source.name, image_name or image_source.name)
    job_dir = Path(settings.data_dir) / "artifacts" / job_id
    views_dir = job_dir / "view"
    style = viewer_style(color, wireframe)
    # The meshes an earlier run of this pair left behind — minus the ones this
    # session already has a viewer for: those fit in the minutes the API spends
    # painting (see ``preload_views``).
    pending = [
        mesh
        for mesh in job_meshes(job_dir)
        if not cached_view(state, mesh[0], mesh[2], style)
    ]
    note(
        f"▶️ `{mesh_name or mesh_source.name}` ({format_size(mesh_size)}) + "
        f"`{image_name or image_source.name}` ({format_size(image_size)}) → job `{job_id}`"
    )
    # What was sent, in numbers. The picker shows the mesh but says nothing about
    # it, and the run used to get this line from the input viewer's export — a
    # read of the file, without the GLB that no window displays any more.
    detail(mesh_line(mesh_source, label="input"))
    note(params_line(params))
    note(f"🔌 `{settings.api_url}`")
    started = time.monotonic()
    client = TextureClient(settings.api_url, timeout_s=settings.timeout_s)
    preloaded: list[ViewExport] = []
    payload: dict[str, Any] | None = None
    try:
        # Two workers: the one the API answers is busy for minutes, the other
        # exports viewers out of the idle time that creates.
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(client.texture, mesh_source, image_source, params)
            preload = pool.submit(preload_views, pending, views_dir, color, wireframe)
            note("📤 files sent: preparing the geometry, then painting…")
            if pending:
                note(
                    "🎨 preloading the viewers of the job already on disk, "
                    "while the API paints…"
                )
            # The journal stays as it is while the API paints — a "still
            # running" line every few seconds only buried the lines that say
            # what actually happened. The banner is what ticks, and it carries
            # the elapsed time (see ``progress_html``).
            while not future.done():
                elapsed = time.monotonic() - started
                yield result_outputs(
                    progress=progress_html(
                        elapsed,
                        "Preparing the mesh + Hunyuan3D painting…",
                    ),
                    status="",
                    state=state,
                    log=log_text(journal),
                )
                time.sleep(TICK_S)
            try:
                payload = future.result()
            except APIError as exc:
                yield failed(
                    exc.as_markdown(),
                    f"{exc.message} ({exc.code or 'API error'})",
                    elapsed=time.monotonic() - started,
                )
                return
            except Exception as exc:  # a bug in the front-end must stay readable
                logger.exception("texturing failed")
                yield failed(
                    f"❌ **Unexpected error:** `{exc}`",
                    f"unexpected error: `{exc}`",
                    elapsed=time.monotonic() - started,
                )
                return

            # The first payload note says what the API answered and how long it
            # took, so no separate "done" line: the journal states it once. The
            # notes that follow describe that same instant, so only the first
            # carries the elapsed time.
            answered = time.monotonic() - started
            for index, line in enumerate(payload_notes(payload)):
                note(line, elapsed=answered if index == 0 else None)

            yield result_outputs(
                progress=progress_html(
                    time.monotonic() - started, "Downloading the artifacts…"
                ),
                status="",
                state=state,
                log=log_text(journal),
            )
            try:
                local = download_artifacts(client, payload, job_dir)
            except APIError as exc:
                yield failed(
                    exc.as_markdown(),
                    f"{exc.message} ({exc.code or 'API error'})",
                    elapsed=time.monotonic() - started,
                )
                return
            total = sum(path.stat().st_size for path in local.values())
            note(f"⬇️ {len(local)} files ({format_size(total)}):")
            for filename, path in sorted(local.items()):
                detail(f"`{filename}` — {format_size(path.stat().st_size)}")
            # The exports made during the painting, if that job had meshes on
            # disk: the render below reuses them as soon as the API confirms
            # they are the bytes it just produced.
            preloaded = preload.result()
    finally:
        client.close()

    if payload is None:  # pragma: no cover - the early returns cover this
        yield failed("❌ Empty response from the API.", "empty response from the API")
        return

    prepared_name = str((payload.get("mesh") or {}).get("filename") or "")
    textured_name = str((payload.get("textured_mesh") or {}).get("filename") or "")
    note(f"🏁 output directory: `{job_dir}`")
    fresh: dict[str, Any] = {
        # The previous job (if any) is kept on disk and in the picker; only
        # what a successful run produced is updated here.
        **(state or {}),
        "job_id": job_id,
        "prepared": str(local[prepared_name]) if prepared_name in local else None,
        # The coloured model, shown untouched (its texture *is* the point), so
        # it is not part of the re-export cache.
        "textured": str(local[textured_name]) if textured_name in local else None,
        # Every artifact of the job, in the order the API listed them: this is
        # what the download picker offers.
        "files": {name: str(path) for name, path in local.items()},
        # The sha256 the API published with each artifact: what the viewer
        # cache keys on, so re-running a cached job reuses the GLBs it already
        # has instead of reading the meshes again (see export_views).
        "digests": {
            str(artifact["filename"]): str(artifact.get("sha256") or "")
            for artifact in artifact_files(payload)
        },
        # The journal travels with the state so the chained events (viewer
        # render, download arming) can keep appending to the same window.
        "log": journal,
        # A fresh run describes its meshes again, whatever the last render
        # logged (see render_views).
        "described": False,
    }
    fresh["views"] = {
        **(fresh.get("views") or {}),
        **promote_preloads(fresh, preloaded, style),
    }
    yield result_outputs(
        progress=progress_html(time.monotonic() - started, "Preparing the viewers…"),
        status="",
        state=fresh,
        log=log_text(journal),
    )
    # Exporting the viewers belongs to the run, not to the event chained after
    # it: seconds of trimesh and GLB writing that would otherwise happen with
    # the run already reported as finished, next to two empty windows.
    rendered = render_views(fresh, color, wireframe)
    yield result_outputs(
        progress=progress_html(
            time.monotonic() - started, "Textured mesh", state="done"
        ),
        # Nothing above the viewers: a successful run needs no commentary, the
        # banner reports what happened and the journal holds the detail.
        status="",
        state=rendered.state,
        log=rendered.log,
    )


def begin_refresh(
    state: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None, str]:
    """Show that a click on "🔄 Refresh the display" is working.

    Re-exporting the viewers loads the meshes with trimesh and writes GLBs —
    seconds on a dense mesh, during which the page used to sit perfectly still:
    from the outside, a button that does nothing.

    The banner takes the shape of a run's, and ``refresh_views`` closes it once
    the meshes are re-exported. The start time travels in
    ``state["refreshing"]``, so the banner ends on the real duration.
    """
    updated, log = logged(state, stamped("🎨 re-exporting the viewers…"))
    updated["refreshing"] = time.monotonic()
    return progress_html(0.0, "Re-exporting the viewers…"), updated, log


def _refresh_banner(state: dict[str, Any] | None, task: str, *, ok: bool = True) -> Any:
    """Close the banner ``begin_refresh`` opened, or leave the page's own alone.

    Chained after a run, there is no refresh in flight and the banner belongs to
    the run, whose duration would be lost by overwriting it: ``gr.skip()`` says
    "not mine to touch".
    """
    started = (state or {}).get("refreshing")
    if started is None:
        return gr.skip()
    return progress_html(
        time.monotonic() - float(started),
        task,
        state="done" if ok else "failed",
    )


@dataclass
class RenderedViews:
    """What one render produced, in the order ``build_ui`` declares its outputs."""

    prepared_mesh: str | None = None
    textured_mesh: str | None = None
    choices: list[str] = field(default_factory=list)
    selection: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    log: str = ""
    empty: bool = False

    def outputs(self, banner: Any = None) -> tuple[Any, ...]:
        """The payload Gradio expects from a render: viewers, picker, state, log."""
        # The mesh picker is not in here on purpose: it holds the run's *input*,
        # and a render that wrote to it would replace the mesh the next run sends
        # with the GLB it exported.
        return (
            self.prepared_mesh,
            self.textured_mesh,
            gr.update(choices=self.choices, value=self.selection),
            self.state,
            self.log,
            banner,
        )


def render_views(
    state: dict[str, Any] | None,
    color: str,
    wireframe: bool,
    choice: str | None = None,
    *,
    quiet: bool = False,
) -> RenderedViews:
    """Render the viewers from the local files, without touching the API.

    Called at the end of every run — where the exports are written, so that the
    event which follows only has to hand them over — and on "🔄 Refresh the
    display", which is how a job already on disk is re-styled without
    repainting it.

    The picker is refreshed here too, so it lists exactly the files of the run
    that just finished. A selection that no longer exists falls back to "all".

    The journal gets the details once: the characteristics of the re-exported
    mesh and the name and size of what its ⤓ button will hand over.
    Re-styling the same meshes only appends one line saying so — the
    characteristics of a mesh do not change when its colour does, and repeating
    the whole block on every click buried the run's own lines. ``quiet`` says the
    block is already there: it is what the run asks for, since the run wrote it
    on its own timeline.
    """
    if not state or not state.get("textured"):
        updated, log = logged(
            state, stamped("❌ nothing to show: run the job first")
        )
        updated.pop("refreshing", None)
        return RenderedViews(state=updated, log=log, empty=True)

    style = viewer_style(color, wireframe)
    described = bool(state.get("described"))
    views_dir = Path(state["textured"]).parent / "view"
    exports = export_views(state, color, wireframe, views_dir, describe=not described)
    clear_views(views_dir, keep=[view.glb for view in exports if view.glb])
    choices = artifact_choices(state)
    selection = choice if choice in choices else DOWNLOAD_ALL

    entries: list[str] = []
    if not quiet:
        if described:
            if exports and all(view.reused for view in exports):
                entries = [stamped(f"🎨 viewers unchanged ({style}) · meshes not re-read")]
            else:
                entries = [
                    stamped(
                        f"🎨 viewers re-exported ({style}) · characteristics unchanged"
                    )
                ]
        else:
            entries = [stamped(f"{viewer_header(exports, style)} :")]
            for view in exports:
                line = f"{DETAIL_INDENT}{view.line or mesh_line(view.artifact, label=view.label)}"
                if view.glb:
                    line += (
                        f" · ⤓ `{Path(view.glb).name}`"
                        f" ({format_size(Path(view.glb).stat().st_size)})"
                    )
                else:
                    line += " · ⚠️ viewer not exported (see the front-end logs)"
                entries.append(line)
            textured = state.get("textured")
            if textured and Path(textured).is_file():
                path = Path(textured)
                entries.append(
                    f"{DETAIL_INDENT}textured mesh — shown as is "
                    f"(`{path.name}`, {format_size(path.stat().st_size)})"
                )

    updated, log = logged(
        {
            **state,
            "color": color,
            "wireframe": bool(wireframe),
            "described": True,
            "views": view_cache(state, exports, style),
        },
        *entries,
    )
    # The textured mesh is shown exactly as the API exported it: it is already a
    # GLB carrying its own texture, so there is nothing to tint or re-export.
    textured = state.get("textured")
    textured = str(textured) if textured and Path(textured).is_file() else None

    updated.pop("refreshing", None)
    return RenderedViews(
        prepared_mesh=next((view.glb for view in exports if view.key == "prepared"), None),
        textured_mesh=textured,
        choices=choices,
        selection=selection,
        state=updated,
        log=log,
    )


def refresh_views(
    state: dict[str, Any] | None,
    color: str,
    wireframe: bool,
    choice: str | None = None,
) -> tuple[Any, ...]:
    """The "🔄 Refresh the display" button: render, then close its own banner."""
    rendered = render_views(state, color, wireframe, choice)
    task = "nothing to show" if rendered.empty else "Display refreshed"
    return rendered.outputs(_refresh_banner(state, task, ok=not rendered.empty))


def finish_views(
    state: dict[str, Any] | None,
    color: str,
    wireframe: bool,
    choice: str | None = None,
) -> tuple[Any, ...]:
    """Chained after every run: hand the viewers the GLBs the run already wrote.

    The journal stays as the run left it — the run wrote that block itself — and
    so does the banner, which belongs to the run and would lose its duration
    here (``_refresh_banner`` skips it when no refresh is in flight).
    """
    rendered = render_views(state, color, wireframe, choice, quiet=True)
    return rendered.outputs(_refresh_banner(state, "Display refreshed"))


# ---------------------------------------------------------------------------
# The UI
# ---------------------------------------------------------------------------


def build_ui(settings: UISettings | None = None) -> gr.Blocks:
    """Build the Gradio app.

    Split from ``main`` so tests can construct it without launching a server.
    """
    settings = settings or UISettings.from_env()
    form = fetch_form(settings)
    defaults, limits = form.defaults, form.limits

    with gr.Blocks(title=settings.title, analytics_enabled=False) as demo:
        gr.Markdown("# Texture a mesh with Hunyuan3D")
        badge = gr.Markdown(form.badge)

        # The two inputs side by side: the mesh to colour and the photo whose
        # colours it should take. One row rather than two so the page reads as
        # a single gesture ("this mesh + this photo").
        with gr.Row():
            # An *interactive* Model3D is the picker and the viewer in one
            # window: Gradio draws it as a drop zone while it is empty and as the
            # model itself as soon as it holds one — the mesh appears where it
            # was dropped, without waiting for a run. It is also the run's input,
            # and no handler writes to its value afterwards (see the module
            # docstring). Its formats are Gradio's own list, so the service's
            # ``/info`` suffixes go in the label, which is what the user reads
            # before sending anything.
            mesh_input = gr.Model3D(
                label=f"Input mesh ({', '.join(form.mesh_suffixes)})",
                height=VIEW_HEIGHT,
                interactive=True,
            )
            image_input = gr.Image(
                type="filepath", label="Source image (colours)", height=VIEW_HEIGHT
            )

        with gr.Accordion("Advanced options", open=False):
            with gr.Row():
                texture_size = gr.Slider(
                    minimum=int(limits["min_texture_size"]),
                    maximum=int(limits["max_texture_size"]),
                    value=defaults["texture_size"],
                    step=64,
                    label="Texture size (px)",
                    info="The main VRAM knob; 768 by default",
                )
                target_faces = gr.Number(
                    value=defaults["target_faces"],
                    precision=0,
                    label="Face budget (decimation)",
                    info="0 keeps every face; lowering it helps on a small GPU",
                )
                taubin_steps = gr.Slider(
                    minimum=0,
                    maximum=int(limits["taubin_steps_max"]),
                    value=defaults["taubin_steps"],
                    step=1,
                    label="Taubin smoothing iterations",
                    info="0 disables the smoothing",
                )
            with gr.Row():
                taubin_lambda = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=defaults["taubin_lambda"],
                    step=0.01,
                    label="Taubin λ",
                )
                taubin_mu = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=defaults["taubin_mu"],
                    step=0.01,
                    label="Taubin µ",
                )
                seed = gr.Number(value=defaults["seed"], precision=0, label="Seed")
            with gr.Row():
                uv_unwrap = gr.Checkbox(
                    value=bool(defaults["uv_unwrap"]),
                    label="Generate UVs when missing",
                    info="Required for a marching-cubes mesh: without UVs the "
                    "paint model has nothing to texture",
                )
                reuse_cached = gr.Checkbox(
                    value=bool(defaults["reuse_cached"]),
                    label="Reuse the cache",
                    info="Uncheck to repaint a mesh that is already textured",
                )

        run = gr.Button("🚀 Texture the mesh", variant="primary")

        with gr.Row():
            with gr.Column():
                progress = gr.HTML(label="Progress")
                status = gr.Markdown()
            with gr.Column():
                color = gr.ColorPicker(value=DEFAULT_VIEW_COLOR, label="Surface colour")
                wireframe = gr.Checkbox(value=False, label="Wireframe (edges)")
                refresh = gr.Button("🔄 Refresh the display")

        # The two windows of the result, same principle as the sister app: the
        # prepared mesh is re-exported through trimesh (so the colour picker and
        # the wireframe apply), the textured one is the API's own GLB, shown
        # untouched. The input mesh is not repeated here — the picker shows it.
        with gr.Row():
            prepared_model = gr.Model3D(
                label="Prepared mesh (cleaned + UV)", height=VIEW_HEIGHT
            )
            textured_model = gr.Model3D(
                label="Textured mesh (Hunyuan3D paint colour)", height=VIEW_HEIGHT
            )

        with gr.Row():
            download_choice = gr.Dropdown(
                choices=[],
                value=None,
                label="📥 Output file to download",
                info="Filled at the end of the run: every artifact produced, or "
                "all of them in a .zip archive",
                scale=3,
            )
            # A DownloadButton, not a Button + gr.File: gr.File only shows a
            # link to click again, this one downloads its value on the click.
            download_button = gr.DownloadButton(
                "📥 Download", value=None, interactive=False, scale=1
            )
        download_status = gr.Markdown()

        with gr.Accordion("📜 Journal", open=False):
            journal = gr.Textbox(
                value="",
                lines=12,
                max_lines=12,
                autoscroll=True,
                interactive=False,
                show_label=False,
                placeholder=(
                    "The run's detail shows up here: files sent and their size, "
                    "parameters, progress, files produced with their size, mesh "
                    "characteristics, full errors."
                ),
            )

        state = gr.State(value=None)

        def handle_run(
            mesh_path: str | None,
            image_path: str | None,
            texture_size: int,
            target_faces: int,
            taubin_steps: int,
            taubin_lambda: float,
            taubin_mu: float,
            uv_unwrap: bool,
            seed: int,
            reuse_cached: bool,
            color: str,
            wireframe: bool,
            state: dict[str, Any] | None,
        ) -> Iterator[tuple[Any, ...]]:
            params = {
                "texture_size": texture_size,
                "target_faces": target_faces,
                "taubin_steps": taubin_steps,
                "taubin_lambda": taubin_lambda,
                "taubin_mu": taubin_mu,
                "uv_unwrap": uv_unwrap,
                "seed": seed,
                "reuse_cached": reuse_cached,
            }
            mesh_name = Path(mesh_path).name if mesh_path else None
            image_name = Path(image_path).name if image_path else None
            # The viewer style travels with the run: it exports the viewers
            # before its last message, and exports depend on the style.
            yield from run_texture(
                settings=settings,
                limits=limits,
                mesh_path=mesh_path,
                mesh_name=mesh_name,
                image_path=image_path,
                image_name=image_name,
                params=params,
                state=state,
                color=color,
                wireframe=bool(wireframe),
            )

        run.click(
            handle_run,
            inputs=[
                mesh_input,
                image_input,
                texture_size,
                target_faces,
                taubin_steps,
                taubin_lambda,
                taubin_mu,
                uv_unwrap,
                seed,
                reuse_cached,
                color,
                wireframe,
                state,
            ],
            outputs=[progress, status, state, journal],
            # One GPU behind the API: queue the runs instead of stacking them.
            concurrency_limit=1,
            concurrency_id="texture",
            show_progress="hidden",
        ).then(
            # Chained, single-shot: this is what makes the meshes show up without
            # the user having to ask for a redraw. The run already exported the
            # viewers, so this only hands them over — and stays silent in the
            # journal, which the run filled itself (see the module docstring).
            finish_views,
            inputs=[state, color, wireframe, download_choice],
            outputs=[
                prepared_model,
                textured_model,
                download_choice,
                state,
                journal,
                progress,
            ],
            show_progress="hidden",
        ).then(
            # Arm the download button on the choice the render just selected, so
            # one click on it is enough as soon as the run is over.
            populate_download,
            inputs=[state, download_choice],
            outputs=[download_button, download_status],
            show_progress="hidden",
        )
        refresh.click(
            # First, immediate feedback: re-exporting the meshes takes seconds,
            # and a page that does not move reads as a click that did nothing.
            # It only writes the banner and the journal, so the meshes keep
            # arriving from a single-shot event (see the module docstring).
            begin_refresh,
            inputs=[state],
            outputs=[progress, state, journal],
            show_progress="hidden",
        ).then(
            refresh_views,
            inputs=[state, color, wireframe, download_choice],
            outputs=[
                prepared_model,
                textured_model,
                download_choice,
                state,
                journal,
                progress,
            ],
            show_progress="hidden",
        ).then(
            populate_download,
            inputs=[state, download_choice],
            outputs=[download_button, download_status],
            show_progress="hidden",
        )
        # Picking another file re-arms the button; the download itself happens
        # when the button is clicked. Local files only: the run already
        # downloaded every artifact, so this works with the API down.
        download_choice.change(
            populate_download,
            inputs=[state, download_choice],
            outputs=[download_button, download_status],
            show_progress="hidden",
        )
        # The mesh picker needs no event: it is the viewer. Gradio shows the
        # file it received in the same window, on its own (see the module
        # docstring), so there is nothing to export when a file lands.
        demo.load(
            lambda: (
                api_badge(settings),
                stamped(
                    "🔌 page loaded — the journal fills up on every run; "
                    "the meshes are written under the viewers."
                ),
            ),
            outputs=[badge, journal],
        )

    demo.queue(default_concurrency_limit=1)
    return demo


def main() -> int:
    """Entry point of the front-end container (``python -m ui.app``)."""
    settings = get_settings()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    logger.info(
        "front-end starting - api=%s data=%s http://%s:%s",
        settings.api_url,
        settings.data_dir,
        settings.host,
        settings.port,
    )
    demo = build_ui(settings)
    demo.launch(
        server_name=settings.host,
        server_port=settings.port,
        share=settings.share,
        show_error=True,
        allowed_paths=[str(settings.data_dir)],
        quiet=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
