#!/usr/bin/env python3
"""Launcher: ``python main.py`` is a friendlier ``uvicorn app.main:app``.

Inside the container the entrypoint calls uvicorn directly (so signals reach the
server); this script is for local development, where ``--reload`` and the
printed URL are handy.
"""
from __future__ import annotations

import argparse
import os

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=settings.app_name)
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes.")
    parser.add_argument(
        "--log-level", default=settings.log_level.lower(), help="uvicorn log level."
    )
    args = parser.parse_args()

    # GPU work runs in threads off the event loop; the API itself is not
    # CPU-bound, so one worker process is the right shape - and the ~7.5 GB paint
    # model must live in exactly one process anyway.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print(f"-> http://{args.host}:{args.port}/docs  (OpenAPI at /openapi.json)")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
