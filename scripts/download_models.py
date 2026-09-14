#!/usr/bin/env python3
"""Pre-download the Hunyuan3D paint checkpoints into the Hugging Face cache.

Run by the container entrypoint (``H3D_DOWNLOAD_MODELS=1``) so the first HTTP
request does not pay for a multi-gigabyte download while holding a client
connection open. Also usable locally::

    python scripts/download_models.py
    python scripts/download_models.py --check-only    # everything cached?

One repository is involved — ``tencent/Hunyuan3D-2`` — but two subfolders, so
the *multiview texture* model and its *de-lighting* companion are fetched from
it. ``Hunyuan3DPaintPipeline.from_pretrained`` loads the second one next to the
first, and a cache missing it fails deep inside the load.

``--check-only`` exits 0 when everything requested is cached, 1 otherwise, so CI
and orchestration can gate on it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# The de-lighting model the paint pipeline loads next to the multiview one.
DELIGHT_SUBFOLDER = "hunyuan3d-delight-v2-0"


def is_cached(model_id: str, subfolders: list[str]) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot_download(
            repo_id=model_id,
            allow_patterns=[f"{subfolder}/*" for subfolder in subfolders],
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, FileNotFoundError, OSError):
        return False
    return True


def targets(args: argparse.Namespace, settings) -> list[tuple[str, list[str]]]:
    """The ``(repo_id, subfolders)`` pairs this run has to have on disk."""
    model_id = args.model_id or settings.model_id
    multiview = args.subfolder or settings.subfolder
    subfolders = [multiview]
    if not args.no_delight:
        subfolders.append(DELIGHT_SUBFOLDER)
    return [(model_id, subfolders)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=None, help="Override H3D_MODEL_ID.")
    parser.add_argument("--subfolder", default=None, help="Override H3D_MODEL_SUBFOLDER.")
    parser.add_argument(
        "--no-delight",
        action="store_true",
        help="Skip hunyuan3d-delight-v2-0 (the paint pipeline loads it too, so "
        "only useful when pre-fetching by hand).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only report whether the checkpoints are cached (exit 0 = cached).",
    )
    args = parser.parse_args(argv)

    from app.config import get_settings

    settings = get_settings()
    # HF_HOME / HF_HUB_CACHE must point at the mounted volume before the hub
    # client is imported, otherwise the snapshot lands in ~/.cache.
    settings.apply_runtime_env()

    wanted = targets(args, settings)

    if args.check_only:
        missing = False
        for model_id, subfolders in wanted:
            cached = is_cached(model_id, subfolders)
            missing = missing or not cached
            print(f"{model_id}/{','.join(subfolders)}: {'cached' if cached else 'missing'}")
        return 1 if missing else 0

    from huggingface_hub import snapshot_download

    exit_code = 0
    for model_id, subfolders in wanted:
        if is_cached(model_id, subfolders):
            print(f"[models] {model_id}/{','.join(subfolders)} already cached - nothing to do")
            continue
        print(f"[models] downloading {model_id}/{','.join(subfolders)} ...", flush=True)
        started = time.time()
        try:
            path = snapshot_download(
                repo_id=model_id,
                allow_patterns=[f"{subfolder}/*" for subfolder in subfolders],
            )
        except Exception as exc:
            print(f"[models] download failed: {exc}", file=sys.stderr, flush=True)
            exit_code = 1
            continue
        print(f"[models] ready in {time.time() - started:.1f}s -> {path}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
