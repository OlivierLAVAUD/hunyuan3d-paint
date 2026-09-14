"""Hunyuan3D paint (texture) generation, wrapped as a process-wide singleton.

This is the texture half of Tencent's pipeline, on its own: where the shape
service turns an image into geometry, this one paints geometry somebody else
produced with the colours of a reference image.

The model is ``hunyuan3d-paint-v2-0-turbo`` — the multiview texture model of
``tencent/Hunyuan3D-2``, living next to ``hunyuan3d-delight-v2-0``. *turbo* is
the low-step LCM variant, which is what keeps a texture pass affordable on a
consumer GPU.

Paint **requires CUDA**: its custom rasterizer and differentiable renderer are
CUDA extensions (no CPU fallback upstream), so a ``cpu`` device reports the
engine as unavailable instead of failing a job.

The checkpoints are also *far* heavier than the shape ones — ~7.5 GB of
fp16 weights in RAM (multiview pipeline ~5.3 GB, delight ~2.2 GB), and a ~14 GB
download because the repo ships each file both as an fp32 ``.bin`` and as fp16
safetensors — which is why the load path is the fragile part on a small host,
not the inference. Two guards stand in front of it, because it fails in two
different ways and neither message from the kernel is usable:

* ``_check_host_ram`` — the UNet is read into *system* RAM before moving to the
  GPU, so a container with less than ``min_host_ram_gb`` of reachable memory is
  OOM-killed mid-load and the client only sees a dropped socket;
* ``_check_vram`` — the weights then have to fit on the card, and the ~8 GB
  resident set on a 6 GB laptop GPU fails with a CUDA driver error buried in a
  7-minute log.

``H3D_LOW_VRAM=1`` changes the second constraint rather than working around it:
the pipelines are built *CPU-first* and their weights are streamed onto the
card one sub-module at a time (diffusers' *sequential* CPU offload), so what
has to fit is the largest block instead of the whole checkpoint. The guards
still run — the low-VRAM budget is a different number, not "no number" —
because the offload is not free: the renderer, the CUDA context and the
activations stay on the card.

Torch and ``hy3dgen`` are imported lazily, in the methods, so ``import
app.main`` stays cheap and the API can boot on a machine without the ML stack.
"""
from __future__ import annotations

import logging
import os
import random
import threading
from typing import TYPE_CHECKING, Any

from ..config import Settings
from ..exceptions import (
    EngineLoadError,
    EngineUnavailable,
    HostMemoryError,
    TextureError,
    VramError,
)
from .cuda import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_LOADING,
    STATE_READY,
    _is_host_oom,
    _release_cuda_memory,
    cuda_ready,
    vram_info,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of imports
    import trimesh
    from PIL import Image

logger = logging.getLogger(__name__)


def _meminfo_kb() -> dict[str, int] | None:
    """The ``/proc/meminfo`` fields this module cares about, in kB."""
    wanted = {"MemTotal": 0, "MemAvailable": 0, "SwapTotal": 0, "SwapFree": 0}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                key = line.split(":", 1)[0]
                if key in wanted:
                    wanted[key] = int(line.split()[1])
    except (OSError, ValueError, IndexError):  # pragma: no cover - non-Linux
        return None
    return wanted


def _read_int(path: str) -> int | None:
    """Read a single integer from a cgroup file, or ``None`` if unreadable.

    ``memory.max``/``memory.swap.max`` hold either a number of bytes or the
    literal ``max`` (no limit). Both spellings are normal here.
    """
    try:
        with open(path, encoding="ascii") as handle:
            raw = handle.read().strip()
    except OSError:  # pragma: no cover - no cgroup v2, or a restricted view
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:  # pragma: no cover - unexpected content
        return None


def _swap_reachable_by_container() -> bool:
    """True when this container is allowed to swap at all.

    This is the difference between "the host has 16 GB of swap" and "the paint
    load may *use* it", and getting it wrong is how a guard green-lights a load
    the kernel is about to kill. Docker sets ``memory.swap.max`` to 0 whenever
    ``memswap_limit`` equals ``mem_limit`` (the compose default), and with
    ``memory.swap.max=0`` the cgroup OOM killer fires at the RAM ceiling with
    the swap file sitting full of free space, untouched.

    An unreadable cgroup means "assume yes": on a host without a cgroup swap
    limit the process can genuinely swap, and refusing a load that would work
    is the worse of the two errors.
    """
    for path in (
        "/sys/fs/cgroup/memory.swap.max",  # cgroup v2
        "/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes",  # cgroup v1
    ):
        limit = _read_int(path)
        if limit is not None:
            # v1 exposes the *combined* limit; v2 the swap-only allowance.
            if path.endswith("memsw.limit_in_bytes"):  # pragma: no cover - v1
                mem_max = _read_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
                if mem_max is not None:
                    return limit > mem_max
            return limit > 0
    return True


def _host_mem_available_gb() -> float | None:
    """Headroom the paint load may use, counting only memory it can reach.

    Starts from ``MemAvailable`` and adds the free swap **only when the cgroup
    actually lets this container swap** (see ``_swap_reachable_by_container``).
    Adding swap unconditionally is not a conservative error: it is exactly the
    bug that made this guard report 28.6 GB on a container whose ``memory.swap.max``
    was 0, wave the load through, and let the kernel kill uvicorn at 14 GB with
    the swap file untouched.

    Also caps the answer by the container's own ``memory.max`` plus reachable
    swap, so a limit smaller than the host RAM is respected rather than papered
    over.

    Returns ``None`` when ``/proc/meminfo`` cannot be read (non-Linux, a
    restricted ``/proc``), which callers treat as "skip the check".
    """
    meminfo = _meminfo_kb()
    if meminfo is None or meminfo["MemAvailable"] <= 0:
        return None

    budget_kb = meminfo["MemAvailable"]
    if _swap_reachable_by_container():
        budget_kb += meminfo["SwapFree"]

    mem_max = _read_int("/sys/fs/cgroup/memory.max") or _read_int(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes"
    )
    if mem_max is not None:
        swap_max = _read_int("/sys/fs/cgroup/memory.swap.max") or 0
        cgroup_kb = (mem_max + swap_max) / 1024
        budget_kb = min(budget_kb, cgroup_kb)

    return budget_kb / 1024 / 1024


class PaintingEngine:
    """Loads the Hunyuan3D paint pipeline once and textures meshes on demand."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._state = STATE_IDLE
        self._error: str | None = None
        self._device: str | None = None
        self._pipeline: Any = None
        self._unloaded = False
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # -- introspection ---------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def device(self) -> str | None:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._state == STATE_READY

    @property
    def low_vram(self) -> bool:
        """Whether this engine is streaming its weights from system RAM."""
        return self._settings.low_vram

    def is_available(self) -> bool:
        """Readiness: could a texture request be served?

        An idle, on-demand model counts as available, and so does a released
        one. The hard requirement on top is CUDA: paint has no CPU path, so a
        host without a visible GPU is not "loading", it is unavailable.
        """
        if self._state == STATE_READY:
            return True
        if self._state == STATE_ERROR:
            return False
        return (self._unloaded or not self._settings.preload_model) and cuda_ready()

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "loaded": self.is_loaded,
            "device": self._device,
            "error": self._error,
            "low_vram": self._settings.low_vram,
            "vram_required_gb": self._settings.vram_required_gb,
        }

    # -- lifecycle -------------------------------------------------------

    def load(self) -> None:
        """Load the paint weights. Idempotent and safe to call from any thread."""
        with self._load_lock:
            if self._state == STATE_READY:
                return
            self._state = STATE_LOADING
            self._error = None
            self._unloaded = False
            try:
                self._pipeline = self._build_pipeline()
            except (HostMemoryError, VramError):
                # Neither is a statement about the weights being loadable: free
                # RAM or a free GPU later may well allow it. Leaving the state
                # untouched is what lets the next request retry, instead of
                # pinning the worker into a permanent STATE_ERROR.
                raise
            except Exception as exc:
                self._state = STATE_ERROR
                self._error = f"{type(exc).__name__}: {exc}"
                logger.exception("failed to load the Hunyuan3D paint pipeline")
                if _is_host_oom(exc):
                    raise HostMemoryError(
                        "not enough system memory to load the Hunyuan3D paint "
                        "model; give the container more RAM or lower the "
                        "texture size",
                        details={"reason": self._error, "stage": "load"},
                    ) from exc
                raise EngineLoadError(
                    "could not load the Hunyuan3D paint model",
                    details={"reason": self._error},
                ) from exc
            self._state = STATE_READY
            logger.info("Hunyuan3D paint pipeline ready on %s", self._device)

    def ensure_loaded(self) -> None:
        if self._state == STATE_READY:
            return
        if self._state == STATE_ERROR:
            raise EngineUnavailable(
                "the paint model failed to load and the worker is degraded",
                details={"reason": self._error},
            )
        self.load()

    def unload(self) -> None:
        """Release the checkpoint, handing the GPU back."""
        with self._load_lock:
            if self._pipeline is None:
                return
            self._pipeline = None
            self._state = STATE_IDLE
            self._unloaded = True
        logger.info("released the Hunyuan3D paint model (GPU handed back)")
        _release_cuda_memory(self._device)

    def _build_pipeline(self) -> Any:
        self._settings.apply_runtime_env()

        if not cuda_ready():
            raise EngineLoadError(
                "the Hunyuan3D paint model needs CUDA "
                "(its rasterizer has no CPU implementation)"
            )
        self._device = "cuda"
        # Both guards run *before* the ~7.5 GB read starts, so the caller gets a
        # sentence it can act on instead of a dead socket (host OOM) or a CUDA
        # driver error 7 minutes in (VRAM).
        self._check_host_ram()
        self._check_vram()
        logger.info(
            "loading paint %s/%s on %s%s",
            self._settings.model_id,
            self._settings.subfolder,
            self._device,
            " (low VRAM: weights streamed from system RAM)"
            if self._settings.low_vram
            else "",
        )
        if self._settings.low_vram:
            # CPU-first, and it has to happen *before* from_pretrained: the
            # vendored constructors end with pipeline.to(self.device), where
            # the device comes from Hunyuan3DTexGenConfig - hardcoded 'cuda'
            # upstream, overridable through this env var in our fork. Without
            # it, construction itself moves the whole ~8 GB onto the card and
            # a 6 GB GPU dies before the offload hook below ever runs.
            os.environ["HY3DGEN_TEXGEN_DEVICE"] = "cpu"

        from hy3dgen.texgen import Hunyuan3DPaintPipeline

        pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            self._settings.model_id,
            subfolder=self._settings.subfolder,
        )
        if self._settings.low_vram:
            self._enable_low_vram(pipeline)
        return pipeline

    def _enable_low_vram(self, pipeline: Any) -> None:
        """Stream the weights sub-module by sub-module instead of holding them.

        Two things had to be true *before* this method runs, and both are what
        separates it from upstream's ``--low_vram_mode``:

        * the pipelines were built **CPU-first** (``_build_pipeline`` sets
          ``HY3DGEN_TEXGEN_DEVICE=cpu`` first) — otherwise the vendored
          constructors' own ``pipeline.to("cuda")`` moves the whole ~8 GB onto
          the card during construction and a 6 GB GPU dies before any hook
          can help;
        * the offload is **sequential**, not model-level: upstream's
          ``enable_model_cpu_offload`` hands Accelerate each *whole* model,
          so the 5.3 GB multiview pipeline still has to fit on the card in
          one piece when its turn comes. ``enable_sequential_cpu_offload``
          (the method our vendored fork forwards to both sub-pipelines)
          moves one sub-module at a time, the only granularity that fits a
          6 GB card.

        A pipeline without the method is a hard error rather than a silent
        fallback: the operator asked for a memory mode this build cannot
        provide, and quietly loading 8 GB onto a 6 GB card would fail minutes
        later with a message that says nothing about the real cause.
        """
        enable = getattr(pipeline, "enable_sequential_cpu_offload", None)
        if not callable(enable):
            raise EngineLoadError(
                "H3D_LOW_VRAM is on but this build of the Hunyuan3D paint "
                "pipeline has no enable_sequential_cpu_offload(); the vendored "
                "hy3dgen is too old to stream its weights sub-module by "
                "sub-module",
                details={"model_id": self._settings.model_id},
            )
        try:
            enable()
        except Exception as exc:
            raise EngineLoadError(
                f"could not enable the low-VRAM CPU offload: {exc}",
                details={"model_id": self._settings.model_id},
            ) from exc
        logger.info(
            "paint weights streamed from system RAM (H3D_LOW_VRAM=1): only "
            "one sub-module is resident on the GPU, at the cost of speed"
        )

    def _check_host_ram(self) -> None:
        """Refuse the load when the host clearly cannot hold the checkpoints.

        The checkpoints are ~7.5 GB of fp16 weights, read into host memory
        before anything reaches the GPU, so a host without the RAM for it is
        OOM-killed
        *during* the load — the client sees a dead socket and the front-end
        reports "Server disconnected without sending a response", which says
        nothing about the real problem. Answering up front with a readable
        error, and a number in it, is worth far more than a container restart.

        The budget is ``MemAvailable`` plus free swap, but only when this
        container is allowed to swap (see ``_host_mem_available_gb`` and
        ``_swap_reachable_by_container``): with ``memory.swap.max=0`` the kernel
        kills the process at the RAM ceiling however much host swap is idle, so
        counting it would refuse nothing and merely delay the crash.

        Best-effort: when ``/proc/meminfo`` cannot be read (non-Linux, a
        restricted ``/proc``) the check is skipped rather than guessed.
        """
        required_gb = self._settings.min_host_ram_gb
        if required_gb <= 0:
            return
        available = _host_mem_available_gb()
        if available is None:
            return
        if available < required_gb:
            swap_reachable = _swap_reachable_by_container()
            swap_note = (
                ""
                if swap_reachable
                else " Swap is not available to this container "
                "(memory.swap.max=0: set memswap_limit above mem_limit, or the "
                "container cannot use the host swap)."
            )
            raise HostMemoryError(
                f"not enough system memory to load the Hunyuan3D paint model: "
                f"{available:.1f} GB reachable, about "
                f"{required_gb:.0f} GB needed.{swap_note} Give the container more "
                f"RAM, raise the WSL memory/swap limits, or paint with a smaller "
                f"texture_size.",
                details={
                    "available_gb": round(available, 1),
                    "required_gb": required_gb,
                    "stage": "precheck",
                    "swap_reachable": swap_reachable,
                },
            )

    def _check_vram(self) -> None:
        """Refuse the load when the GPU cannot hold what has to be resident.

        The harder of the two constraints on a laptop, and the one no container
        setting can fix: the full-resident set is ~8 GB of fp16 weights
        (multiview ~5.3 + delight ~2.2 + renderer ~0.7), so a 6 GB card fails
        however much system RAM the host has and however small the render
        gets — the weights themselves have to fit.

        With ``low_vram`` on, what has to fit is a single sub-module rather
        than the whole checkpoint (see ``_enable_low_vram``), and the budget
        comes from ``min_vram_low_gb`` instead. It is deliberately not zero:
        the offload moves the *weights*, not the renderer, the CUDA context or
        the activations, and those are what a 2 GB card would still run out of.

        Reading the card in use first (rather than a driver lookup inside torch)
        keeps the failure honest: "5936 MiB of 6144 MiB already in use" is a
        different problem from "6144 MiB total is not enough", and on a shared
        box the first one is the one that happens.
        """
        required_gb = self._settings.vram_required_gb
        if required_gb <= 0:
            return
        info = vram_info()
        if info is None:
            # No nvidia-smi / no driver: the load itself will say so, and a
            # guess here would refuse working hosts.
            return
        total_gb = info["total_mb"] / 1024
        free_gb = info["free_mb"] / 1024
        # What is resident depends on the mode, so the explanation has to as
        # well: in low-VRAM mode the weights are *not* on the card, and
        # saying they are would send the operator looking for the wrong
        # problem.
        if self._settings.low_vram:
            weights_note = (
                "H3D_LOW_VRAM is on, so the weights stream from system RAM and "
                "only one sub-module is resident at a time; what still has to "
                "fit is the renderer, the CUDA context, the activations and "
                "the largest block."
            )
            remedy = (
                "Free the card, or lower texture_size / target_faces to shrink "
                "the activations."
            )
        else:
            weights_note = (
                "the full-resident set is ~8 GB of fp16 weights "
                "(multiview ~5.3 GB + delight ~2.2 GB + renderer)"
            )
            remedy = (
                "Run this service on a machine with at least "
                f"{required_gb:.0f} GB of VRAM, or set H3D_LOW_VRAM=1 to stream "
                "the weights from system RAM instead; lowering texture_size does "
                "not help, the weights themselves have to fit."
            )

        if total_gb < required_gb:
            raise VramError(
                f"the GPU is too small for the Hunyuan3D paint model: "
                f"{info['name']} has {info['total_mb']} MiB of VRAM, about "
                f"{required_gb:.0f} GB needed ({weights_note}). {remedy}",
                details={
                    "gpu": info["name"],
                    "total_mb": info["total_mb"],
                    "free_mb": info["free_mb"],
                    "required_gb": required_gb,
                    "low_vram": self._settings.low_vram,
                    "stage": "precheck",
                },
            )
        if free_gb < required_gb:
            raise VramError(
                f"not enough free VRAM for the Hunyuan3D paint model: "
                f"{info['free_mb']} MiB free of {info['total_mb']} MiB on "
                f"{info['name']}, about {required_gb:.0f} GB needed ({weights_note}). "
                f"Another process is holding the GPU.",
                details={
                    "gpu": info["name"],
                    "total_mb": info["total_mb"],
                    "free_mb": info["free_mb"],
                    "required_gb": required_gb,
                    "low_vram": self._settings.low_vram,
                    "stage": "precheck",
                },
            )

    # -- inference -------------------------------------------------------

    def _apply_texture_size(self, texture_size: int) -> None:
        """Point the vendored pipeline at ``texture_size`` for the next run.

        Upstream's ``Hunyuan3DPaintPipeline.__call__`` takes no resolution
        argument - both sizes are construction-time constants of
        ``Hunyuan3DTexGenConfig`` (render and bake, 2048 each) - so passing one
        as a keyword is a ``TypeError`` raised *after* the ~7.5 GB load, which
        is exactly how a documented VRAM knob turned into a 13-minute run
        ending in HTTP 500. Both sizes move instead:

        * ``render.texture_size`` - the baked atlas, its cos map and its trust
          map are all allocated at that size;
        * ``render.default_resolution`` and ``config.render_size`` - the six
          normal/position maps, their rasterisation and the multiview images
          the bake back-projects, which are the larger allocation of the two.

        The two are kept equal on purpose: that is upstream's own 1:1 ratio, so
        lowering the knob shrinks the whole pass instead of half of it.

        A build without the setters is a hard error rather than a silent
        fallback, for the same reason as ``_enable_low_vram``: baking at the
        config default instead of the requested size would OOM on the small
        cards this knob exists for.
        """
        render = getattr(self._pipeline, "render", None)
        set_texture = getattr(render, "set_default_texture_resolution", None)
        set_render = getattr(render, "set_default_render_resolution", None)
        if not callable(set_texture) or not callable(set_render):
            raise EngineLoadError(
                "this build of the Hunyuan3D paint pipeline exposes no "
                "MeshRender.set_default_texture_resolution() / "
                "set_default_render_resolution(); the vendored hy3dgen is too "
                "old to make the resolution a per-job knob",
                details={"model_id": self._settings.model_id},
            )
        # Kept in step with the renderer: the multiview images are resized to
        # ``config.render_size`` and ``set_texture`` resizes to
        # ``render.texture_size``, so the config and the two setters have to
        # agree on the same number.
        config = getattr(self._pipeline, "config", None)
        if config is not None:
            config.texture_size = texture_size
            config.render_size = texture_size
        set_texture(texture_size)
        set_render(texture_size)
        # Info, not debug: this is the value the vendored renderer actually got,
        # which is what an operator has to compare against the size they asked
        # for when a texture comes back coarser than expected.
        logger.info(
            "bake and render resolution set to %dpx (atlas, normal/position "
            "renders and the multiview bake all follow it)",
            texture_size,
        )

    @staticmethod
    def _resolve_seed(seed: int) -> int:
        """Turn the API's "pick one" (``-1``) into a seed the model can use.

        ``-1`` is documented as "not reproducible": a fresh seed is drawn per
        job, so asking for it twice gives two different textures. Any other
        value passes through unchanged, which is what makes a run repeatable.
        """
        if seed == -1:
            return random.randrange(2**31)
        return seed

    def paint(
        self,
        mesh: trimesh.Trimesh,
        image: Image.Image,
        *,
        texture_size: int | None = None,
        seed: int | None = None,
    ) -> trimesh.Trimesh:
        """Paint ``mesh`` with the colours of ``image``, returning the textured mesh.

        The pipeline de-lights the reference image, renders the mesh from its
        six canonical viewpoints and bakes the generated views back into a UV
        texture on ``mesh``. ``texture_size`` is the resolution of that whole
        pass - it is applied to the renderer just before the call (see
        ``_apply_texture_size``), not passed to it. ``seed`` is the multiview
        diffusion seed (``-1`` draws a fresh one, see ``_resolve_seed``); it is
        logged once resolved, since it is what makes a run reproducible.
        """
        self.ensure_loaded()
        import torch

        with self._infer_lock:
            kwargs: dict[str, Any] = {}
            if texture_size is not None:
                # Inside the lock: the resolution is state of the shared
                # pipeline, and two jobs asking for different sizes would
                # otherwise race on it.
                self._apply_texture_size(texture_size)
            if seed is not None:
                kwargs["seed"] = self._resolve_seed(seed)
                logger.info("multiview diffusion seed %d", kwargs["seed"])
            try:
                textured = self._pipeline(mesh, image, **kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise TextureError(
                    "CUDA ran out of memory while texturing; try a smaller "
                    "texture_size or target_faces",
                    details={"device": self._device},
                ) from exc
            except Exception as exc:
                if _is_host_oom(exc):
                    raise HostMemoryError(
                        "ran out of system memory while texturing",
                        details={"device": self._device, "stage": "paint"},
                    ) from exc
                # With the traceback: the message alone ("META device type not
                # an accelerator", say) says nothing about where a run that
                # took twenty minutes actually broke, and the client only gets
                # the one-line envelope.
                logger.exception("Hunyuan3D paint call failed")
                raise TextureError(f"Hunyuan3D paint failed: {exc}") from exc
            finally:
                if self._device and self._device.startswith("cuda"):
                    torch.cuda.empty_cache()

        if textured is None or len(getattr(textured, "vertices", [])) == 0:
            raise TextureError("the paint model returned an empty mesh")
        return textured
