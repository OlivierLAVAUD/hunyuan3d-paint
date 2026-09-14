"""HTTP client for the texture (paint) API.

Three things make this more than a thin ``httpx`` wrapper:

* **The error envelope is decoded.** Every 4xx/5xx carries
  ``{"error": {"code", "message", "details"}}``; it is turned into an
  ``APIError`` whose markdown rendering is what the UI displays.
* **Artifact URLs are re-anchored.** Responses embed absolute URLs built from
  the *API's* host (``http://hunyuan3d-paint-api:8082`` inside compose, which no
  browser can reach); only the path is kept and re-prefixed with our own base.
* **Timeouts are honest.** A paint run takes minutes, so the read timeout is
  minutes too — a front-end timeout would be a bug in the front-end, not in the
  service.
"""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx

# What the documented status codes mean for the person clicking "run".
_HINTS = {
    404: "unknown job or artifact for the API",
    413: "file too large: see H3D_MAX_MESH_UPLOAD_MB on the API side",
    415: "unsupported format (mesh: glb, gltf, obj, ply, stl — image: png, jpg, webp, bmp)",
    422: "invalid parameter or unreadable mesh — the detail the API returns says which",
    500: "the painting or the mesh preparation failed on the API side (see its logs)",
    503: "the paint model is unavailable (not enough VRAM?): watch /ready",
}


class APIError(RuntimeError):
    """A failed API call: a non-2xx response, or no response at all.

    ``status_code`` is ``None`` when the API could not be reached or did not
    answer in time, which is a different problem from a rejected request.
    """

    def __init__(
        self,
        status_code: int | None,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details

    @property
    def hint(self) -> str | None:
        return _HINTS.get(self.status_code or 0)

    def as_markdown(self) -> str:
        """Render the failure as the Markdown block the UI shows."""
        status = "API unreachable" if self.status_code is None else f"HTTP {self.status_code}"
        lines = [f"❌ **{self.message}**", f"`{self.code}` · {status}"]
        if self.hint:
            lines.append(f"💡 {self.hint}")
        if self.details:
            payload = json.dumps(self.details, indent=2, ensure_ascii=False, default=str)
            lines.append(f"```json\n{payload}\n```")
        return "  \n".join(lines)


def _form_value(value: Any) -> Any:
    """Multipart fields are text: booleans have to go out as ``true``/``false``."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


class TextureClient:
    """The texture endpoints, from the front-end's point of view.

    ``http_client`` exists so the tests can hand over a client bound to the ASGI
    app (no socket, no GPU); when omitted the wrapper owns its client.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8082",
        *,
        timeout_s: float = 1800.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(timeout_s, connect=15.0),
            follow_redirects=True,
        )

    # -- plumbing --------------------------------------------------------

    @property
    def base_url(self) -> str:
        """The API root, taken from the underlying HTTP client."""
        return str(self._client.base_url).rstrip("/")

    def resolve_url(self, url: str) -> str:
        """Re-anchor an artifact URL on this client's base URL.

        Only the path and query are kept: the host in the response is the one
        the API saw in the request, which is not necessarily a host the browser
        (or this container) can reach.
        """
        parts = urlsplit(url)
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        return f"{self.base_url}{path}"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TextureClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            timeout = self._client.timeout.read
            raise APIError(
                None,
                "timeout",
                f"no response from {self.base_url} within the timeout ({timeout} s)",
            ) from exc
        except httpx.HTTPError as exc:
            raise APIError(
                None, "unreachable", f"API unreachable at {self.base_url} — {exc}"
            ) from exc

    @staticmethod
    def _decode(response: httpx.Response, *, accepted: tuple[int, ...] = ()) -> dict[str, Any]:
        """Return the JSON body, or raise the API's own error envelope."""
        if response.status_code in accepted or response.is_success:
            try:
                return response.json()
            except ValueError as exc:
                raise APIError(
                    response.status_code,
                    "invalid_response",
                    "the API returned something other than JSON",
                    details={"body": response.text[:500]},
                ) from exc
        code, message, details = "http_error", response.text[:500], None
        try:
            error = (response.json() or {}).get("error") or {}
            code = error.get("code", code)
            message = error.get("message", message)
            details = error.get("details")
        except ValueError:
            pass  # not the documented envelope: keep the raw body as the message
        raise APIError(response.status_code, code, message, details)

    # -- endpoints -------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Liveness of the API process."""
        return self._decode(self._request("GET", "/health"))

    def ready(self) -> dict[str, Any]:
        """Readiness of the paint model.

        A 503 is a normal answer here — it means the weights are still loading,
        or the GPU is too small to hold them — so it is decoded instead of
        raised.
        """
        return self._decode(self._request("GET", "/ready"), accepted=(503,))

    def info(self) -> dict[str, Any]:
        """Defaults, limits and model state: what the form is built from."""
        return self._decode(self._request("GET", "/api/v1/info"))

    def texture(
        self,
        mesh_path: str | Path,
        image_path: str | Path,
        params: dict[str, Any] | None = None,
        *,
        mesh_filename: str | None = None,
        image_filename: str | None = None,
    ) -> dict[str, Any]:
        """Run one (mesh + image) -> texture job and return the result payload.

        Blocks until the API has the coloured GLB on disk, which is why callers
        run it off the UI thread.

        Both files are streamed as one multipart body: the reference image is a
        photo and the mesh can be tens of megabytes, so neither is read into
        memory here — ``httpx`` streams the handles.
        """
        mesh = Path(mesh_path)
        image = Path(image_path)
        mesh_type = mimetypes.guess_type(mesh.name)[0] or "application/octet-stream"
        image_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
        form = {
            key: _form_value(value) for key, value in (params or {}).items() if value is not None
        }
        with mesh.open("rb") as mesh_handle, image.open("rb") as image_handle:
            response = self._request(
                "POST",
                "/api/v1/texture",
                files={
                    "mesh": (mesh_filename or mesh.name, mesh_handle, mesh_type),
                    "image": (image_filename or image.name, image_handle, image_type),
                },
                data=form,
            )
        return self._decode(response)

    def download(
        self, url: str, destination: str | Path, *, chunk_size: int = 1024 * 1024
    ) -> Path:
        """Stream an artifact to ``destination`` and return the local path."""
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._client.stream("GET", self.resolve_url(url)) as response:
                if response.is_error:
                    response.read()  # materialise the body so _decode can read it
                    self._decode(response)
                with target.open("wb") as handle:
                    for chunk in response.iter_bytes(chunk_size):
                        handle.write(chunk)
        except httpx.HTTPError as exc:
            raise APIError(None, "unreachable", f"download failed — {exc}") from exc
        if target.stat().st_size == 0:
            target.unlink(missing_ok=True)
            raise APIError(None, "empty_artifact", f"artifact {target.name!r} is empty")
        return target
