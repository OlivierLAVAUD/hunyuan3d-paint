"""Host-RAM and VRAM guards: "out of memory" must not always mean VRAM.

Three failures produce messages that all read "out of memory" and have three
different remedies:

* a host OOM is fixed by giving the container more RAM (or by letting it swap);
* a CUDA OOM by lowering ``texture_size``/``target_faces``;
* too little VRAM at all by running on a bigger card — no knob helps.

``_is_host_oom`` tells the first two apart, the two ``_check_*`` guards catch the
third kind before the load starts, and both are pinned here on the exact strings
the real failures produce.
"""
from __future__ import annotations

import io
import os
import sys
import types
from pathlib import Path

import pytest

from app.exceptions import (
    EngineLoadError,
    GenerationError,
    HostMemoryError,
    VramError,
)
from app.services.cuda import _is_host_oom


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError(),
        RuntimeError("std::bad_alloc"),
        RuntimeError("terminate called after throwing an instance of 'std::bad_alloc'"),
        RuntimeError("cannot allocate memory"),
        RuntimeError("Cannot allocate memory"),
        RuntimeError("unable to allocate 3.10 GiB for an array"),
        RuntimeError(
            "DefaultCPUAllocator: not enough memory: you tried to allocate "
            "4194304 bytes. Buy new RAM!"
        ),
    ],
)
def test_host_failures_are_recognised(exc):
    assert _is_host_oom(exc) is True


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("torch.cuda.OutOfMemoryError: CUDA out of memory"),
        RuntimeError("CUDA driver error: device not ready"),
        OSError("No such file or directory: '/data/models'"),
    ],
)
def test_vram_and_unrelated_failures_are_not_flagged(exc):
    """A CUDA OOM must keep its own advice: lowering texture_size is what helps."""
    assert _is_host_oom(exc) is False


def test_host_memory_error_reports_its_own_code():
    """The client has to be able to branch on the code, not parse the message."""
    error = HostMemoryError("not enough RAM", details={"stage": "load"})
    assert error.status_code == 500
    assert error.code == "host_out_of_memory"
    assert error.to_payload() == {
        "error": {
            "code": "host_out_of_memory",
            "message": "not enough RAM",
            "details": {"stage": "load"},
        }
    }


def test_host_memory_error_stays_a_generation_error():
    """Existing handlers that catch ``GenerationError`` must keep working."""
    assert issubclass(HostMemoryError, GenerationError)
    assert HostMemoryError("x").status_code == GenerationError.status_code


def test_vram_error_is_a_503_with_its_own_code():
    """Not fixable from inside the container, so it must not read as a bug."""
    error = VramError("GPU too small", details={"total_mb": 6144})
    assert error.status_code == 503
    assert error.code == "insufficient_vram"
    # Still a GenerationError, so existing handlers keep catching it.
    assert isinstance(error, GenerationError)


class _FakeSettings:
    """The fields the engine reads, plus the derived budget it uses.

    ``vram_required_gb`` is duplicated here rather than imported because the
    guards must be exercised *without* building a real ``Settings`` (which
    would read the process environment). It mirrors ``Settings.vram_required_gb``
    exactly; if the two ever disagree, ``test_vram_budget_matches_the_setting``
    below is what fails.
    """

    def __init__(
        self,
        *,
        min_host_ram_gb: float = 0.0,
        min_vram_gb: float = 0.0,
        low_vram: bool = False,
        min_vram_low_gb: float = 0.0,
        model_id: str = "tencent/Hunyuan3D-2",
        subfolder: str = "hunyuan3d-paint-v2-0-turbo",
    ) -> None:
        self.model_id = model_id
        self.subfolder = subfolder
        self.min_host_ram_gb = min_host_ram_gb
        self.min_vram_gb = min_vram_gb
        self.low_vram = low_vram
        self.min_vram_low_gb = min_vram_low_gb

    def apply_runtime_env(self) -> None:
        # The real Settings exports env vars other code reads back (HF_HOME
        # and friends); the fake has nothing to export, and _build_pipeline
        # only calls this for its side effects.
        return None

    @property
    def vram_required_gb(self) -> float:
        return self.min_vram_low_gb if self.low_vram else self.min_vram_gb


def _engine(
    *,
    min_host_ram_gb: float = 0.0,
    min_vram_gb: float = 0.0,
    low_vram: bool = False,
    min_vram_low_gb: float = 0.0,
):
    from app.services.paint import PaintingEngine

    return PaintingEngine(
        _FakeSettings(
            min_host_ram_gb=min_host_ram_gb,
            min_vram_gb=min_vram_gb,
            low_vram=low_vram,
            min_vram_low_gb=min_vram_low_gb,
        )
    )


# --------------------------------------------------------------------------- #
# Host RAM
# --------------------------------------------------------------------------- #


def test_paint_refuses_to_load_when_ram_is_too_low(monkeypatch):
    """The whole point of the guard: a readable error, not an OOM-killed run."""
    from app.services import paint

    monkeypatch.setattr(paint, "_host_mem_available_gb", lambda: 6.0)
    engine = _engine(min_host_ram_gb=14.0)
    with pytest.raises(HostMemoryError) as caught:
        engine._check_host_ram()
    assert caught.value.code == "host_out_of_memory"
    assert caught.value.details["available_gb"] == 6.0
    assert caught.value.details["required_gb"] == 14.0


def test_paint_loads_when_ram_is_sufficient(monkeypatch):
    from app.services import paint

    monkeypatch.setattr(paint, "_host_mem_available_gb", lambda: 32.0)
    _engine(min_host_ram_gb=14.0)._check_host_ram()  # must not raise


_MEMINFO_WITH_SWAP = (
    "MemTotal:       15360000 kB\n"
    "MemAvailable:   13107200 kB\n"  # 12.5 GiB of RAM
    "SwapTotal:      16777216 kB\n"
    "SwapFree:       16777216 kB\n"  # 16 GiB of swap
)


def _patch_meminfo(
    monkeypatch,
    text: str,
    *,
    mem_max: int | None = None,
    swap_max: int | None = None,
) -> None:
    """Serve ``/proc/meminfo`` and the two cgroup limit files from one ``open``.

    A single patch has to cover both sources: the module reads meminfo and the
    cgroup limits through the same builtin, so patching them separately would
    have the second call shadow the first.
    """
    from app.services import paint

    cgroup = {
        "/sys/fs/cgroup/memory.max": mem_max,
        "/sys/fs/cgroup/memory.swap.max": swap_max,
    }

    def fake_open(path, *args, **kwargs):
        path = str(path)
        if path.endswith("meminfo"):
            return io.StringIO(text)
        if path in cgroup:
            value = cgroup[path]
            if value is None:
                raise FileNotFoundError(path)
            return io.StringIO(str(value) + "\n")
        raise FileNotFoundError(path)

    monkeypatch.setattr(paint, "open", fake_open, raising=False)


def _patch_cgroup_for_reachability(monkeypatch, swap_max: int | None) -> None:
    """Pin only ``memory.swap.max``; meminfo is irrelevant to the helper."""
    _patch_meminfo(monkeypatch, "", mem_max=15 * 1024**3, swap_max=swap_max)


def test_budget_counts_free_swap_when_the_container_may_swap(monkeypatch):
    """Swap is headroom, but only if this container is allowed to reach it.

    With ``memory.swap.max`` above zero the kernel really does spill the load's
    overflow to swap, so free swap is genuine headroom and belongs in the sum.
    """
    from app.services import paint

    _patch_meminfo(monkeypatch, _MEMINFO_WITH_SWAP)  # unreadable cgroup => assume yes
    # 12.5 + 16 = 28.5 GiB, comfortably over the 14 GiB requirement.
    assert paint._host_mem_available_gb() > 14.0
    _engine(min_host_ram_gb=14.0)._check_host_ram()  # must not raise


def test_budget_excludes_swap_the_container_cannot_use(monkeypatch):
    """``memory.swap.max=0`` must not be counted, however much host swap is free.

    This is the failure that killed the first textured run of the sister
    project: the host had 16 GB of *unused* swap, Docker had disabled swapping
    for the container, and the old helper summed them anyway — reporting
    28.6 GB reachable, waving the load through, and letting the cgroup OOM killer
    take uvicorn at ~14 GB with the swap file untouched. Counting only reachable
    memory is the fix, and it is carried over here verbatim.
    """
    from app.services import paint

    _patch_meminfo(monkeypatch, _MEMINFO_WITH_SWAP, mem_max=None, swap_max=0)
    # Only the 12.5 GiB of free RAM is reachable, so the guard must refuse.
    assert paint._host_mem_available_gb() == pytest.approx(12.5, abs=0.01)
    with pytest.raises(HostMemoryError) as caught:
        _engine(min_host_ram_gb=14.0)._check_host_ram()
    assert caught.value.details["swap_reachable"] is False
    assert "Swap is not available to this container" in caught.value.message


def test_budget_is_capped_by_the_cgroup_ram_limit(monkeypatch):
    """A container limit below the host RAM is honoured, not papered over.

    WSL reports ~15 GiB while ``mem_limit: 8g`` caps the container at 8: the
    guard has to answer 8, or it green-lights a load that dies at the ceiling.
    """
    from app.services import paint

    meminfo = "MemAvailable:   14680064 kB\nSwapFree:       16777216 kB\n"  # 14 GiB + 16 GiB
    _patch_meminfo(monkeypatch, meminfo, mem_max=8 * 1024**3, swap_max=2 * 1024**3)
    # reachable = min(RAM + swap, mem_max + swap_max) = min(30, 10) = 10 GiB
    assert paint._host_mem_available_gb() == pytest.approx(10.0, abs=0.01)


def test_budget_excludes_used_swap(monkeypatch):
    """Only *free* swap counts: swap already holding pages adds no headroom."""
    from app.services import paint

    meminfo = (
        "MemAvailable:   4194304 kB\n"  # 4 GiB free RAM
        "SwapFree:       2097152 kB\n"  # 2 GiB free swap only
    )
    _patch_meminfo(monkeypatch, meminfo, mem_max=None, swap_max=None)  # swapping allowed
    assert paint._host_mem_available_gb() == pytest.approx(6.0, abs=0.01)


def test_guard_answers_none_without_memavailable(monkeypatch):
    """A ``/proc/meminfo`` with no ``MemAvailable`` must not be read as zero."""
    from app.services import paint

    _patch_meminfo(monkeypatch, "SwapFree: 100 kB\n")
    assert paint._host_mem_available_gb() is None


def test_ram_guard_is_opt_out(monkeypatch):
    """``H3D_MIN_HOST_RAM_GB=0`` disables the check entirely."""
    from app.services import paint

    monkeypatch.setattr(paint, "_host_mem_available_gb", lambda: 1.0)
    _engine(min_host_ram_gb=0.0)._check_host_ram()  # must not raise


def test_ram_guard_is_skipped_when_meminfo_is_unreadable(monkeypatch):
    """Non-Linux / restricted ``/proc`` must not turn into a spurious refusal."""
    from app.services import paint

    monkeypatch.setattr(paint, "_host_mem_available_gb", lambda: None)
    _engine(min_host_ram_gb=14.0)._check_host_ram()  # must not raise


@pytest.mark.parametrize(
    ("swap_max", "expected"),
    [
        (0, False),  # `memswap_limit == mem_limit` -> Docker writes 0
        (1, True),  # any positive allowance lets the load spill
        (16 * 1024**3, True),
        (None, True),  # no limit file => cannot tell => assume swapping is on
    ],
)
def test_swap_reachability_reads_the_cgroup(monkeypatch, swap_max, expected):
    """The helper has to distinguish "host has swap" from "container may use it"."""
    from app.services import paint

    _patch_cgroup_for_reachability(monkeypatch, swap_max)
    assert paint._swap_reachable_by_container() is expected


def test_swap_reachability_defaults_to_true_without_cgroup(monkeypatch):
    """A host with no cgroup swap limit can genuinely swap: do not refuse it."""
    from app.services import paint

    monkeypatch.setattr(paint, "_read_int", lambda path: None)
    assert paint._swap_reachable_by_container() is True


# --------------------------------------------------------------------------- #
# VRAM
# --------------------------------------------------------------------------- #

_RTX_3060_LAPTOP = {"name": "NVIDIA GeForce RTX 3060 Laptop GPU", "total_mb": 6144, "free_mb": 59}
_A100_40G = {"name": "NVIDIA A100-SXM4-40GB", "total_mb": 40960, "free_mb": 39000}
# A 6 GB card that is actually *idle*. `_RTX_3060_LAPTOP` above has only
# 59 MiB free (it is a snapshot of this machine under load), so it fails the
# free-memory check whatever the budget - which is correct, but not what the
# low-VRAM tests are about.
_IDLE_6GB = {"name": "NVIDIA GeForce RTX 3060 Laptop GPU", "total_mb": 6144, "free_mb": 5800}


def test_a_card_smaller_than_the_weights_is_refused(monkeypatch):
    """The blocker on this very machine: 6 GB of VRAM, a ~8 GB resident set.

    The message has to name the card and say that lowering ``texture_size``
    cannot help, because that is the advice the *other* OOM message gives and it
    is wrong here.
    """
    from app.services import paint

    monkeypatch.setattr(paint, "vram_info", lambda: dict(_RTX_3060_LAPTOP))
    engine = _engine(min_vram_gb=9.0)
    with pytest.raises(VramError) as caught:
        engine._check_vram()
    assert caught.value.code == "insufficient_vram"
    assert caught.value.status_code == 503
    assert caught.value.details["total_mb"] == 6144
    assert caught.value.details["gpu"] == "NVIDIA GeForce RTX 3060 Laptop GPU"
    assert "6144 MiB" in caught.value.message
    # The refusal has to offer the way out that actually exists.
    assert "H3D_LOW_VRAM=1" in caught.value.message
    assert caught.value.details["low_vram"] is False


def test_a_big_card_passes(monkeypatch):
    from app.services import paint

    monkeypatch.setattr(paint, "vram_info", lambda: dict(_A100_40G))
    _engine(min_vram_gb=9.0)._check_vram()  # must not raise


def test_a_busy_card_is_refused_with_its_own_message(monkeypatch):
    """Full card, small card: same refusal, different remedy — say which."""
    from app.services import paint

    busy = {"name": "NVIDIA A100-SXM4-40GB", "total_mb": 40960, "free_mb": 2048}
    monkeypatch.setattr(paint, "vram_info", lambda: busy)
    with pytest.raises(VramError) as caught:
        _engine(min_vram_gb=9.0)._check_vram()
    assert "Another process is holding the GPU" in caught.value.message
    assert caught.value.details["free_mb"] == 2048


def test_the_vram_guard_is_opt_out(monkeypatch):
    """``H3D_MIN_VRAM_GB=0`` disables it (the test suite relies on this)."""
    from app.services import paint

    monkeypatch.setattr(paint, "vram_info", lambda: dict(_RTX_3060_LAPTOP))
    _engine(min_vram_gb=0.0)._check_vram()  # must not raise


def test_the_vram_guard_is_skipped_when_nvidia_smi_is_unavailable(monkeypatch):
    """No driver / no nvidia-smi must not become a spurious refusal."""
    from app.services import paint

    monkeypatch.setattr(paint, "vram_info", lambda: None)
    _engine(min_vram_gb=9.0)._check_vram()  # must not raise


@pytest.mark.parametrize(
    "line",
    [
        "NVIDIA GeForce RTX 3060 Laptop GPU, 6144, 5936",
        "NVIDIA A100-SXM4-40GB, 40960, 39000",
    ],
)
def test_vram_info_parses_nvidia_smi_output(monkeypatch, line):
    """The parser is the only thing between the guard and a silent no-op."""
    from app.services import cuda

    class _Completed:
        returncode = 0
        stdout = line + "\n"
        stderr = ""

    monkeypatch.setattr(cuda.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(cuda.subprocess, "run", lambda *a, **k: _Completed())
    info = cuda.vram_info()
    assert info is not None
    assert info["total_mb"] == int(line.split(",")[1])
    assert info["free_mb"] == int(line.split(",")[2])


def test_vram_info_is_none_without_nvidia_smi(monkeypatch):
    from app.services import cuda

    monkeypatch.setattr(cuda.shutil, "which", lambda name: None)
    assert cuda.vram_info() is None


def test_vram_info_is_none_on_a_failed_command(monkeypatch):
    from app.services import cuda

    class _Completed:
        returncode = 9
        stdout = ""
        stderr = "NVML: Driver/library version mismatch"

    monkeypatch.setattr(cuda.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(cuda.subprocess, "run", lambda *a, **k: _Completed())
    assert cuda.vram_info() is None


# --------------------------------------------------------------------------- #
# Low VRAM (sequential CPU offload)
# --------------------------------------------------------------------------- #


def test_vram_budget_matches_the_setting():
    """The stub's derived budget must not drift from the real one."""
    from app.config import Settings

    on = Settings(low_vram=True, min_vram_gb=9.0, min_vram_low_gb=3.0)
    off = Settings(low_vram=False, min_vram_gb=9.0, min_vram_low_gb=3.0)
    assert on.vram_required_gb == 3.0
    assert off.vram_required_gb == 9.0


def test_defaults_pin_the_real_footprint():
    """The budgets are measured, not folklore.

    ~8 GB resident in full mode (9 with margin), ~2-2.5 GB at the
    sequential-offload peak (3 with margin). If these move, the checkpoint or
    the offload granularity changed - and the README and .env.example quotes
    must move with them.
    """
    from app.config import Settings

    defaults = Settings()
    assert defaults.min_vram_gb == 9.0
    assert defaults.min_vram_low_gb == 3.0


def test_low_vram_mode_uses_its_own_budget(monkeypatch):
    """The 6 GB card that fails outright passes once the weights are streamed.

    This is the whole point of the mode, and the reason the two budgets are
    separate settings: offloading changes what is resident, so it is not
    "min_vram_gb with a fudge factor".
    """
    from app.services import paint

    monkeypatch.setattr(paint, "vram_info", lambda: dict(_IDLE_6GB))
    _engine(low_vram=True, min_vram_gb=9.0, min_vram_low_gb=3.0)._check_vram()


def test_low_vram_budget_is_still_enforced(monkeypatch):
    """Streaming the weights is not a licence to ignore VRAM.

    The renderer, the CUDA context, the activations and the largest single
    block are not offloaded, so a card that cannot hold them must still be
    refused - and the message must not claim the whole checkpoint has to fit,
    which is exactly what it no longer does.
    """
    from app.services import paint

    tiny = {"name": "NVIDIA GeForce MX150", "total_mb": 2048, "free_mb": 1900}
    monkeypatch.setattr(paint, "vram_info", lambda: tiny)
    with pytest.raises(VramError) as caught:
        _engine(low_vram=True, min_vram_gb=9.0, min_vram_low_gb=3.0)._check_vram()
    assert caught.value.details["low_vram"] is True
    assert caught.value.details["required_gb"] == 3.0
    assert "stream from system RAM" in caught.value.message
    assert "~8 GB" not in caught.value.message


def test_low_vram_budget_is_opt_out(monkeypatch):
    """``H3D_MIN_VRAM_LOW_GB=0`` disables the check in that mode too."""
    from app.services import paint

    monkeypatch.setattr(paint, "vram_info", lambda: dict(_RTX_3060_LAPTOP))
    _engine(low_vram=True, min_vram_low_gb=0.0)._check_vram()  # must not raise


def _fake_hy3dgen(monkeypatch, pipeline_cls):
    """Make ``from hy3dgen.texgen import Hunyuan3DPaintPipeline`` importable.

    A parent package with an empty ``__path__`` plus a pre-registered child
    module is enough for the import system; ``_build_pipeline`` then runs its
    real body - guards, env var, offload call - against the recording class,
    with no torch, no CUDA and none of the ~14 GB.
    """
    texgen = types.ModuleType("hy3dgen.texgen")
    texgen.Hunyuan3DPaintPipeline = pipeline_cls
    hy3dgen = types.ModuleType("hy3dgen")
    hy3dgen.__path__ = []
    hy3dgen.texgen = texgen
    monkeypatch.setitem(sys.modules, "hy3dgen", hy3dgen)
    monkeypatch.setitem(sys.modules, "hy3dgen.texgen", texgen)


@pytest.fixture
def _clean_device_env(monkeypatch):
    """Start with the fork's env var unset, and unset it again afterwards.

    ``_build_pipeline`` writes ``HY3DGEN_TEXGEN_DEVICE`` through ``os.environ``
    directly, so no monkeypatch undo will remove it; the manual pop keeps one
    test's mode from leaking into the next.
    """
    monkeypatch.delenv("HY3DGEN_TEXGEN_DEVICE", raising=False)
    yield
    os.environ.pop("HY3DGEN_TEXGEN_DEVICE", None)


class _RecordingPipeline:
    """Records the vendored device env *at from_pretrained time* and the hooks.

    The whole order bug was that the env var was set too late: the recording
    has to happen inside ``from_pretrained``, not after ``_build_pipeline``
    returns, or this test would pass against exactly the broken build.
    """

    def __init__(self, device_env):
        self.device_env = device_env
        self.sequential_offloads = 0
        self.model_offloads = 0

    @classmethod
    def from_pretrained(cls, model_id, subfolder=None):
        return cls(os.environ.get("HY3DGEN_TEXGEN_DEVICE"))

    def enable_sequential_cpu_offload(self) -> None:
        self.sequential_offloads += 1

    def enable_model_cpu_offload(self) -> None:
        self.model_offloads += 1


def test_low_vram_builds_cpu_first(monkeypatch, _clean_device_env):
    """The env var has to be set *before* from_pretrained, not after.

    The vendored constructors end with ``pipeline.to(self.device)`` where the
    device comes from that env var: setting it after the import-and-build
    would still move the whole ~8 GB onto the card during construction, which
    is the order bug that shipped in the first version of this mode.
    """
    from app.services import paint

    monkeypatch.setattr(paint, "cuda_ready", lambda: True)
    _fake_hy3dgen(monkeypatch, _RecordingPipeline)

    pipeline = _engine(low_vram=True)._build_pipeline()

    assert pipeline.device_env == "cpu"
    assert pipeline.sequential_offloads == 1


def test_full_mode_leaves_the_vendored_device_env_alone(monkeypatch, _clean_device_env):
    """Full mode must not poison the process env for whoever comes next.

    A stale ``HY3DGEN_TEXGEN_DEVICE=cpu`` would silently flip a later
    full-resident load into building on the CPU and then streaming everything
    onto the card one sub-module at a time - the slow mode, chosen by nobody.
    """
    from app.services import paint

    monkeypatch.setattr(paint, "cuda_ready", lambda: True)
    _fake_hy3dgen(monkeypatch, _RecordingPipeline)

    pipeline = _engine(low_vram=False)._build_pipeline()

    assert pipeline.device_env is None
    assert pipeline.sequential_offloads == 0
    assert "HY3DGEN_TEXGEN_DEVICE" not in os.environ


def test_the_sequential_offload_is_what_runs():
    """Model-level offload exists on the same object and must stay unused.

    Upstream's ``enable_model_cpu_offload`` moves each *whole* model onto the
    card when its turn comes; the 5.3 GB multiview pipeline does not fit a
    6 GB card in one piece, so calling it would be the second half of the
    original bug - right order, wrong granularity.
    """
    from app.services.paint import PaintingEngine

    engine = PaintingEngine(_FakeSettings(low_vram=True))
    pipeline = _RecordingPipeline(None)
    engine._enable_low_vram(pipeline)
    assert pipeline.sequential_offloads == 1
    assert pipeline.model_offloads == 0


def test_offload_absent_from_the_build_is_a_hard_error():
    """A build without the hook must not silently load ~8 GB onto a 6 GB card.

    Falling back to "load it anyway" would move the failure minutes later and
    strip it of its cause, which is the failure mode this whole module exists
    to prevent. A build that only has upstream's model-level hook counts as
    absent: the wrong granularity is no better than no hook at all.
    """
    from app.services.paint import PaintingEngine

    class _UpstreamOnly:
        def enable_model_cpu_offload(self) -> None:  # not good enough
            pass

    engine = PaintingEngine(_FakeSettings(low_vram=True))
    with pytest.raises(EngineLoadError) as caught:
        engine._enable_low_vram(_UpstreamOnly())
    assert "enable_sequential_cpu_offload" in caught.value.message


def test_offload_failure_is_reported_not_swallowed():
    """Accelerate raising must surface as a load error, not a later OOM."""
    from app.services.paint import PaintingEngine

    class _Broken:
        def enable_sequential_cpu_offload(self) -> None:
            raise RuntimeError("accelerate: no device map")

    engine = PaintingEngine(_FakeSettings(low_vram=True))
    with pytest.raises(EngineLoadError) as caught:
        engine._enable_low_vram(_Broken())
    assert "no device map" in caught.value.message


def test_status_reports_the_mode():
    """An operator should not have to read the container env to know the mode."""
    engine = _engine(low_vram=True, min_vram_low_gb=3.0)
    status = engine.status()
    assert status["low_vram"] is True
    assert status["vram_required_gb"] == 3.0


def test_low_vram_is_off_by_default():
    """It costs speed and host RAM, so it must never be opt-out."""
    from app.config import Settings

    assert Settings().low_vram is False


# --------------------------------------------------------------------------- #
# Bake resolution (texture_size) and seed
# --------------------------------------------------------------------------- #


class _RecordingRenderer:
    """Stands in for ``MeshRender``: records the resolutions it is set to."""

    def __init__(self) -> None:
        self.texture_resolutions: list[int] = []
        self.render_resolutions: list[int] = []

    def set_default_texture_resolution(self, resolution) -> None:
        self.texture_resolutions.append(resolution)

    def set_default_render_resolution(self, resolution) -> None:
        self.render_resolutions.append(resolution)


class _VendoredPaintPipeline:
    """The vendored pipeline's call shape: ``(mesh, image, seed=None)``.

    Upstream's ``Hunyuan3DPaintPipeline.__call__`` takes neither a resolution
    nor a seed - both are constants of ``Hunyuan3DTexGenConfig``, set at
    construction, and the multiview pass hardcodes its own RNG - so a keyword
    the engine invents raises a ``TypeError`` minutes into a run, *after* the
    ~7.5 GB load. Modelling the exact signature here is what turns that
    regression into a test failure instead of a 500. ``seed`` is the fork's
    addition (see ``test_the_vendored_pipeline_still_takes_a_seed``).
    """

    def __init__(self) -> None:
        self.config = types.SimpleNamespace(texture_size=2048, render_size=2048)
        self.render = _RecordingRenderer()
        self.calls: list = []

    def __call__(self, mesh, image, seed=None):
        self.calls.append((mesh, image, seed))
        return types.SimpleNamespace(vertices=[0.0, 0.0, 0.0])


@pytest.fixture
def _fake_torch(monkeypatch):
    """``paint()`` imports torch to recognise a CUDA OOM; the suite has no torch.

    The failure handling is not what is under test here - the call the engine
    makes is - so the module only carries the two attributes it touches.
    """
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}),
        empty_cache=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


@pytest.mark.usefixtures("_fake_torch")
def test_paint_applies_the_bake_resolution_instead_of_passing_it():
    """The bug that shipped: ``texture_size`` was a keyword ``__call__`` never had.

    It surfaced as ``Hunyuan3DPaintPipeline.call() got an unexpected keyword
    argument 'texture_size'`` 13 minutes in, because the load comes first; the
    resolution has to be pushed onto the renderer, and the call itself has to
    stay ``(mesh, image)``.
    """
    from app.services.cuda import STATE_READY

    engine = _engine()
    engine._pipeline = _VendoredPaintPipeline()
    engine._state = STATE_READY
    engine._device = "cuda"
    mesh, image = object(), object()

    textured = engine.paint(mesh, image, texture_size=768)

    assert textured.vertices
    # Nothing was invented for the call itself; the seed is left at the
    # vendored default when the caller does not ask for one.
    assert engine._pipeline.calls == [(mesh, image, None)]
    assert engine._pipeline.render.texture_resolutions == [768]
    assert engine._pipeline.render.render_resolutions == [768]
    assert engine._pipeline.config.texture_size == 768
    assert engine._pipeline.config.render_size == 768


@pytest.mark.usefixtures("_fake_torch")
def test_omitting_the_bake_resolution_leaves_the_renderer_alone():
    """``None`` means "what the pipeline was built with", not "reset to 2048"."""
    from app.services.cuda import STATE_READY

    engine = _engine()
    engine._pipeline = _VendoredPaintPipeline()
    engine._state = STATE_READY
    engine._device = "cuda"

    engine.paint(object(), object(), texture_size=None)

    assert engine._pipeline.render.texture_resolutions == []
    assert engine._pipeline.render.render_resolutions == []
    assert engine._pipeline.config.texture_size == 2048
    assert engine._pipeline.config.render_size == 2048


def test_a_build_without_the_resolution_setter_is_a_hard_error():
    """A renderer too old to resize must not silently bake at 2048.

    Falling back would paint at five times the requested resolution - the exact
    failure the knob exists to avoid on a small card - and the message has to
    say so rather than surfacing as a later CUDA OOM.
    """
    engine = _engine()
    engine._pipeline = types.SimpleNamespace(config=None, render=types.SimpleNamespace())

    with pytest.raises(EngineLoadError) as caught:
        engine._apply_texture_size(768)
    assert "set_default_texture_resolution" in caught.value.message


def test_a_build_with_only_half_the_setters_is_a_hard_error():
    """Shrinking only the atlas would leave the larger render pass at 2048.

    Half the knob is not the knob: the render maps are the bigger allocation,
    so a build that can only resize the texture must be refused rather than
    accepted as "close enough".
    """
    from app.services.paint import PaintingEngine

    class _TextureOnly:
        def set_default_texture_resolution(self, resolution) -> None:
            pass

    engine = PaintingEngine(_FakeSettings())
    engine._pipeline = types.SimpleNamespace(config=None, render=_TextureOnly())
    with pytest.raises(EngineLoadError) as caught:
        engine._apply_texture_size(768)
    assert "set_default_render_resolution" in caught.value.message


@pytest.mark.usefixtures("_fake_torch")
def test_paint_forwards_the_seed_to_the_vendored_pipeline():
    """Upstream seeds the multiview pass with 0, so the knob needs the fork."""
    from app.services.cuda import STATE_READY

    engine = _engine()
    engine._pipeline = _VendoredPaintPipeline()
    engine._state = STATE_READY
    engine._device = "cuda"
    mesh, image = object(), object()

    engine.paint(mesh, image, seed=4242)

    assert engine._pipeline.calls == [(mesh, image, 4242)]


@pytest.mark.usefixtures("_fake_torch")
def test_a_negative_seed_draws_a_fresh_one_per_job():
    """``-1`` is documented as "not reproducible", and has to mean it.

    A fixed fallback would make two ``-1`` jobs return the same texture, which
    is the opposite of what the API promises; the value also has to land in
    torch's accepted range, since it is fed to ``manual_seed``.
    """
    from app.services.cuda import STATE_READY

    engine = _engine()
    engine._pipeline = _VendoredPaintPipeline()
    engine._state = STATE_READY
    engine._device = "cuda"

    engine.paint(object(), object(), seed=-1)
    engine.paint(object(), object(), seed=-1)

    drawn = [call[2] for call in engine._pipeline.calls]
    assert all(0 <= seed < 2**31 for seed in drawn)
    assert drawn[0] != drawn[1]


@pytest.mark.usefixtures("_fake_torch")
def test_zero_is_a_real_seed():
    """``0`` is reproducible and must be passed through, not treated as absent."""
    from app.services.cuda import STATE_READY

    engine = _engine()
    engine._pipeline = _VendoredPaintPipeline()
    engine._state = STATE_READY
    engine._device = "cuda"

    engine.paint(object(), object(), seed=0)

    assert engine._pipeline.calls[0][2] == 0


def _vendored_call_args(relative_path: str, class_name: str) -> list[str]:
    """Argument names of one vendored ``__call__``, read without importing it.

    The fork's additions cannot be exercised here - importing ``hy3dgen.texgen``
    pulls in torch and the compiled CUDA rasterizer - and a re-vendored copy
    that dropped them would only fail once a real job reaches the model,
    minutes in. Parsing the source pins the contract before that.
    """
    import ast

    root = Path(__file__).resolve().parents[1] / "third_party" / "Hunyuan3D-2" / "hy3dgen"
    tree = ast.parse((root / relative_path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__call__":
                    return [arg.arg for arg in child.args.args]
    raise AssertionError(f"{class_name}.__call__ not found in {relative_path}")


def test_the_vendored_pipeline_still_takes_a_seed():
    """The engine's ``seed`` keyword must exist all the way down the stack.

    Both halves are fork additions: the paint pipeline has to accept ``seed``
    and the multiview model has to consume it (upstream's are hardcoded to 0),
    so losing either one silently reverts every run to seed 0 - or worse,
    raises ``TypeError`` once the model is loaded.
    """
    texgen = _vendored_call_args(
        "texgen/pipelines.py", "Hunyuan3DPaintPipeline"
    )
    assert "seed" in texgen

    multiview = _vendored_call_args(
        "texgen/utils/multiview_utils.py", "Multiview_Diffusion_Net"
    )
    assert "seed" in multiview


def test_the_vendored_generator_is_not_built_on_a_meta_device():
    """``pipeline.device`` is 'meta' behind sequential CPU offload.

    accelerate stores the offloaded weights as meta tensors, so that property -
    "the device of the first parameter" - answers 'meta', and
    ``torch.Generator(device='meta')`` raises "META device type not an
    accelerator". It cost a twenty-minute low-VRAM run to discover, and no unit
    test can import the real pipeline (torch + the CUDA rasterizer), so the
    source is what pins it.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "Hunyuan3D-2"
        / "hy3dgen"
        / "texgen"
        / "utils"
        / "multiview_utils.py"
    ).read_text(encoding="utf-8")
    assert "_execution_device" in source
    assert "torch.Generator(device=self.pipeline.device)" not in source


def test_the_low_vram_offload_pins_the_weights_read_outside_forward():
    """A parameter read outside ``forward`` is dataless once it is offloaded.

    ``hunyuanpaint/pipeline.py`` builds ``prompt_embeds`` from
    ``unet.learned_text_clip_gen`` before the UNet is entered, and accelerate's
    sequential CPU offload leaves every *parameter* on the 'meta' device
    between forwards - so that read returns a tensor with no storage and the
    ``.to(dtype, device)`` after it raises ``Cannot copy out of meta tensor; no
    data!``, twenty minutes into a low-VRAM run and in no other mode. The two
    embeddings have to be buffers *before* the hooks are installed.

    Pinned at the source for the same reason as the seed and generator forks:
    the vendored ``hy3dgen`` is not importable here (torch, diffusers and the
    compiled rasterizer), so a re-vendor that drops this would only fail once a
    real job reaches the multiview pipeline minutes in.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "Hunyuan3D-2"
        / "hy3dgen"
        / "texgen"
        / "pipelines.py"
    ).read_text(encoding="utf-8")

    assert "register_buffer" in source
    for name in ("learned_text_clip_gen", "learned_text_clip_ref"):
        assert f'"{name}"' in source
    # A DiffusionPipeline is not an nn.Module (it holds its components as plain
    # attributes), so the walk starts from `vars(pipeline)` - guarding on
    # isinstance() instead finds nothing and skips the fix without a word.
    assert "vars(pipeline)" in source
    for model in ("delight_model", "multiview_model"):
        pinned = source.index(f"_pin_offload_unsafe_weights(self.models['{model}']")
        offloaded = source.index(
            f"self.models['{model}'].pipeline.enable_sequential_cpu_offload"
        )
        assert pinned < offloaded


def test_the_vendored_resolution_setters_still_exist():
    """The engine resizes the renderer through these two methods."""
    import ast

    root = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "Hunyuan3D-2"
        / "hy3dgen"
        / "texgen"
        / "differentiable_renderer"
        / "mesh_render.py"
    )
    methods = {
        node.name
        for node in ast.walk(ast.parse(root.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    assert {"set_default_texture_resolution", "set_default_render_resolution"} <= methods
