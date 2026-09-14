"""Settings of the Gradio front-end, read from ``H3D_UI_*`` environment variables.

Only the API location and the server binding really matter: the *form* (default
values, ranges, allowed formats) is read from ``GET /api/v1/info``, so the UI can
never drift from the service it talks to.

The ``H3D_*`` parsing helpers and the dotenv reader come from ``app.config``
instead of being re-implemented: one env convention for the whole project — and
the same one the sister image-to-mesh repo uses, so an operator moving between
the two sees identical names.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# The underscore-prefixed readers are the API's own H3D_* parsing helpers; this
# front-end deliberately reuses them instead of growing a second convention.
from app.config import _bool, _float, _int, _str, load_dotenv


@dataclass(frozen=True)
class UISettings:
    """Immutable snapshot of the front-end configuration."""

    # Where the API lives. In compose this is the service name, not localhost.
    api_url: str = "http://127.0.0.1:8082"
    host: str = "0.0.0.0"
    # 7861, not 7860: the shape service's front-end already holds 7860, and the
    # two are meant to be run side by side.
    port: int = 7861
    # A paint run takes a couple of minutes; the HTTP client must outlive it or
    # the UI would report a timeout the API never had.
    timeout_s: float = 1800.0
    # Downloaded artifacts, so the viewers and the download links work whatever
    # route the browser has to the API.
    data_dir: Path = Path("/data/ui")
    title: str = "Hunyuan3D — Mesh + Image → Texture"
    share: bool = False

    @classmethod
    def from_env(cls, dotenv_path: Path | str | None = None) -> UISettings:
        if dotenv_path is not None:
            load_dotenv(Path(dotenv_path))
        return cls(
            api_url=_str("H3D_UI_API_URL", cls.api_url),
            host=_str("H3D_UI_HOST", cls.host),
            port=_int("H3D_UI_PORT", cls.port),
            timeout_s=_float("H3D_UI_TIMEOUT_S", cls.timeout_s),
            data_dir=Path(_str("H3D_UI_DATA_DIR", str(cls.data_dir))),
            title=_str("H3D_UI_TITLE", cls.title),
            share=_bool("H3D_UI_SHARE", cls.share),
        )


@lru_cache(maxsize=1)
def get_settings() -> UISettings:
    """Return the process-wide front-end settings (built once)."""
    return UISettings.from_env(Path(".env"))
