"""CUDA / host-memory helpers shared by the paint engine and its guards.

Split out of ``services/engine`` so the paint repo has no shape-generation code
to import: this module is only the *hardware* half of what that one carried —
the state constants, the OOM classifier, the VRAM probe and the memory-release
helper. Nothing here imports torch at module level, so ``/info`` and ``/ready``
still answer on a machine where the ML stack is missing or broken.
"""
from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_ERROR = "error"

# Signatures of a *host* allocation failure rather than a CUDA one. glibc
# reports the first two when it cannot grow its arena, torch the last two when
# it cannot allocate the staging buffer of a (mostly CPU-first) load. They are
# checked before the CUDA ones because a failed host allocation during
# `from_pretrained` is the failure mode on a small-RAM host, and saying "lower
# texture_size" would be advice that cannot help.
_HOST_OOM_MARKERS = (
    "cannot allocate memory",
    "unable to allocate",
    "std::bad_alloc",
    "memoryerror",
    "defaultcpuallocator",
    "not enough memory",
)

_SMI_TIMEOUT_S = 10.0


def _is_host_oom(exc: BaseException) -> bool:
    """True when ``exc`` looks like a system-RAM failure rather than a VRAM one.

    The CUDA check comes first on purpose: ``torch.cuda.OutOfMemoryError``
    subclasses ``MemoryError``, so an ``isinstance`` test alone would classify
    every VRAM failure as a host one and bury the advice that actually helps
    (lower ``texture_size``) under advice that does not.
    """
    message = f"{type(exc).__name__}: {exc}".lower()
    if "outofmemoryerror" in message or "cuda out of memory" in message:
        return False
    if isinstance(exc, MemoryError):
        return True
    return any(marker in message for marker in _HOST_OOM_MARKERS)


def cuda_ready() -> bool:
    """True when torch is importable and sees a CUDA device.

    Used for readiness instead of importing torch at module level, so ``/info``
    still answers on a machine where the ML stack is absent.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - a missing/broken torch is not fatal
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - broken driver
        return False


def vram_info() -> dict[str, object] | None:
    """Total and free VRAM of GPU 0, straight from ``nvidia-smi``.

    Deliberately not ``torch.cuda.mem_get_info()``: that reports what *torch*
    can see, which on a driver mismatch is nothing at all, and it cannot be
    called before torch is imported. ``nvidia-smi`` answers in the one case
    that matters — telling the caller its card is too small *before* paying for
    a ~7.5 GB read.

    Returns ``None`` when nvidia-smi is absent, times out, or answers something
    unexpected; callers treat that as "cannot tell" and let the load decide.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            capture_output=True,
            text=True,
            timeout=_SMI_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        logger.debug("nvidia-smi unavailable: %s", exc)
        return None
    if completed.returncode != 0:
        logger.debug("nvidia-smi exited %d: %s", completed.returncode, completed.stderr)
        return None
    line = (completed.stdout or "").strip().splitlines()
    if not line:
        return None
    parts = [part.strip() for part in line[0].split(",")]
    if len(parts) < 3:
        return None
    try:
        return {
            "name": parts[0],
            "total_mb": int(parts[1]),
            "free_mb": int(parts[2]),
        }
    except ValueError:  # pragma: no cover - an unexpected nvidia-smi format
        logger.debug("could not parse nvidia-smi output: %r", line[0])
        return None


def _release_cuda_memory(device: str | None) -> None:
    """Drop the Python cycles a released pipeline may hold, then free the cache."""
    import gc

    gc.collect()
    if device and device.startswith("cuda"):
        try:
            import torch
        except ImportError:  # pragma: no cover - the ML stack is optional here
            torch = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
    # Return the freed heap to the OS. glibc keeps the arena otherwise: after
    # `unload()` the weights are unreferenced but still counted in the process
    # RSS, which is exactly the memory the next load peak needs on a small host
    # (the kernel OOM-kills the server mid-load otherwise).
    #
    # Done unconditionally, including on a CPU device: the host heap is what
    # runs out on a small-RAM box, and `malloc_trim` is the only thing that
    # gives it back before the next `from_pretrained`.
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:  # pragma: no cover - non-glibc platforms
        pass
