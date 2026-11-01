"""
Krusch Agentic Proxy — Unified Waterfall Router.

Consolidates waterfall routing into a single, pooled class with consistent
error handling, timeouts, and ENV: API key resolution.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from src.exceptions import ConfigurationError, RouteConfigurationError, resolve_api_key

_TOOLKIT_LIB_PY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "lib-py")
)
if os.path.isdir(_TOOLKIT_LIB_PY) and _TOOLKIT_LIB_PY not in sys.path:
    sys.path.insert(0, _TOOLKIT_LIB_PY)

try:
    from krusch_toolkit.fastapi_app import sse_event_response
except ImportError:
    import inspect

    async def _close_fallback_source(source: Any) -> None:
        aclose_fn = getattr(source, "aclose", None)
        if callable(aclose_fn):
            res = aclose_fn()
            if inspect.isawaitable(res):
                await res

    async def _is_disconnected(req: Any) -> bool:
        if req is None:
            return False
        fn = getattr(req, "is_disconnected", None)
        if not callable(fn):
            return False
        res = fn()
        return bool(await res if inspect.isawaitable(res) else res)

    def sse_event_response(
        event_generator: Any,
        request: Any = None,
        *,
        default_event: str = "token",
        heartbeat_comment: Optional[str] = "ping",
        passthrough: bool = False,
    ) -> StreamingResponse:
        async def _fallback_stream():
            aiter = None
            try:
                if await _is_disconnected(request):
                    return
                if heartbeat_comment:
                    yield f": {heartbeat_comment}\n\n"
                aiter = event_generator.__aiter__()
                while True:
                    if await _is_disconnected(request):
                        break
                    try:
                        chunk = await aiter.__anext__()
                    except StopAsyncIteration:
                        break
                    yield chunk
            finally:
                if aiter is not None and aiter is not event_generator:
                    await _close_fallback_source(aiter)
                await _close_fallback_source(event_generator)

        return StreamingResponse(
            _fallback_stream(),
            media_type="/".join(("text", "event-stream")),
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

logger = logging.getLogger(__name__)


class _UpstreamByteStream:
    """AsyncIterable wrapper ensuring deterministic resp/client cleanup even before first iteration."""

    def __init__(self, resp: httpx.Response, client: httpx.AsyncClient) -> None:
        self._resp = resp
        self._client = client
        self._closed = False

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._resp.aclose()
        finally:
            await self._client.aclose()

    async def __aiter__(self):
        try:
            async for chunk in self._resp.aiter_bytes():
                yield chunk
        finally:
            await self.aclose()


def validate_api_key(raw_key: str) -> str:
    """Resolve and strictly validate a non-empty API key."""
    return resolve_api_key(raw_key, strict=True)


class WaterfallRouter:
    """
    Iterates through configured routes until one succeeds.
    Supports both streaming and non-streaming responses with connection pooling.
    """

    def __init__(
        self,
        routes: List[Dict[str, Any]],
        timeout: float = 120.0,
        client: Optional[httpx.AsyncClient] = None,
        strict_routes: bool = False,
    ):
        self.routes = routes or []
        self.timeout = timeout
        self._client = client
        self.strict_routes = strict_routes

    async def route_chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Route a simple chat completion through the waterfall and return the content string.
        Used by the internal KruschEngine client for LLM calls.
        Raises ConfigurationError / RuntimeError if all routes fail.
        """
        if not self.routes:
            raise ConfigurationError("No waterfall_routes configured for auto-cloud routing.")

        last_error = ""
        attempted_any = False

        for route in self.routes:
            route_name = route.get("name", "Unknown")
            api_url = route.get("api_url")
            if not api_url:
                msg = f"Route {route_name} missing required field 'api_url'"
                logger.warning(f"[Waterfall] {msg}")
                if self.strict_routes:
                    raise RouteConfigurationError(msg)
                last_error = msg
                continue

            raw_key = route.get("api_key", "")
            api_key = resolve_api_key(raw_key)
            if "api_key" in route and not api_key:
                msg = f"Route {route_name} has empty or unresolved api_key"
                logger.error(f"[Waterfall] {msg}")
                if self.strict_routes:
                    raise ConfigurationError(msg)
                last_error = msg
                continue

            attempted_any = True
            logger.info(f"[Waterfall] Trying route: {route_name}")

            payload: Dict[str, Any] = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            }

            if "models" in route:
                payload["models"] = route["models"]
            elif "model" in route:
                payload["model"] = route["model"]

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "http://localhost:5440",
                "X-Title": "Krusch Agentic Waterfall Router",
            }

            try:
                if self._client is not None and getattr(self._client, "is_closed", False) is not True:
                    response = await self._client.post(
                        api_url, json=payload, headers=headers, timeout=self.timeout
                    )
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(api_url, json=payload, headers=headers)

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                last_error = f"HTTP {response.status_code}: {response.text}"
                logger.warning(f"[Waterfall] Route {route_name} failed: {last_error}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"[Waterfall] Route {route_name} exception: {last_error}")

        if not attempted_any and "missing required field 'api_url'" in last_error:
            raise RouteConfigurationError(f"All waterfall routes failed. Last error: {last_error}")
        raise RuntimeError(f"All waterfall routes failed. Last error: {last_error}")

    async def route_proxy(
        self,
        data: dict,
        is_stream: bool = False,
        request: Any = None,
    ) -> JSONResponse | StreamingResponse:
        """
        Route a full OpenAI-compatible request through the waterfall.
        Returns a FastAPI response (JSON or Streaming).
        Used by the API gateway for external HTTP requests.
        """
        if not self.routes:
            return JSONResponse(
                {"error": "No waterfall_routes configured for auto-cloud routing."},
                status_code=500,
            )

        last_error = "No routes attempted."

        for route in self.routes:
            api_url = route.get("api_url")
            route_name = route.get("name", "Unknown Route")
            if not api_url:
                last_error = f"Route {route_name} missing required field 'api_url'"
                logger.warning(f"[Waterfall] {last_error}")
                continue
            logger.info(f"[Waterfall] Attempting route: {route_name} ({api_url})")

            payload = {k: v for k, v in data.items() if k != "model"}
            if "models" in route:
                payload["models"] = route["models"]
            elif "model" in route:
                payload["model"] = route["model"]

            raw_key = route.get("api_key", "")
            api_key = resolve_api_key(raw_key)
            if "api_key" in route and not api_key:
                last_error = f"Route {route_name} has empty or unresolved api_key"
                logger.error(f"[Waterfall] {last_error}")
                continue

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5440",
                "X-Title": "Krusch Agentic Waterfall Router",
            }

            try:
                if is_stream:
                    response = await self._try_stream(api_url, payload, headers, request=request)
                    if response is not None:
                        return response
                    last_error = f"Stream connection failed for route {route_name}"
                else:
                    if self._client is not None and getattr(self._client, "is_closed", False) is not True:
                        resp = await self._client.post(
                            api_url, json=payload, headers=headers, timeout=self.timeout
                        )
                    else:
                        async with httpx.AsyncClient(timeout=self.timeout) as client:
                            resp = await client.post(api_url, json=payload, headers=headers)

                    if resp.status_code == 200:
                        return JSONResponse(resp.json())
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    logger.warning(f"[Waterfall] Route {route_name} failed: {last_error}")
            except Exception as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"[Waterfall] Route {route_name} exception: {last_error}")

        return JSONResponse(
            {"error": f"All waterfall routes failed. Last error: {last_error}"},
            status_code=502,
        )

    async def _try_stream(
        self,
        api_url: str,
        payload: dict,
        headers: dict,
        request: Any = None,
    ) -> Optional[StreamingResponse]:
        """
        Attempt a streaming connection. Returns a StreamingResponse on success,
        or None on failure. Ensures httpx client and response are properly cleaned up.
        """
        client = httpx.AsyncClient(timeout=self.timeout)
        resp: Optional[httpx.Response] = None
        handed_off = False
        try:
            req = client.build_request("POST", api_url, json=payload, headers=headers)
            resp = await client.send(req, stream=True)

            if resp.status_code != 200:
                error_body = await resp.aread()
                logger.warning(
                    f"[Waterfall] Stream route failed: HTTP {resp.status_code}: "
                    f"{error_body.decode('utf-8', errors='ignore')}"
                )
                return None

            byte_stream = _UpstreamByteStream(resp, client)
            response = sse_event_response(
                byte_stream,
                request=request,
                passthrough=True,
                heartbeat_comment="",
            )
            handed_off = True
            return response

        except Exception as e:
            logger.warning(f"[Waterfall] Stream connection exception: {e}")
            return None
        finally:
            if not handed_off:
                try:
                    if resp is not None:
                        await resp.aclose()
                finally:
                    await client.aclose()
