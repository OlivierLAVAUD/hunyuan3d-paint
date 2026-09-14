"""Aggregated ``/api/v1`` router."""
from __future__ import annotations

from fastapi import APIRouter

from . import info, texture

router = APIRouter()
router.include_router(info.router)
router.include_router(texture.router)
