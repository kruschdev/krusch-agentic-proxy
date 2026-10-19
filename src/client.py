"""
Krusch Agentic Proxy — LLM Chat Client.

Provides the low-level async chat function used by KruschEngine.
Supports both single-endpoint pooled HTTP execution and waterfall multi-provider routing.
"""

import logging
import os
import re
from typing import Any, Dict, Optional

import httpx

from src.router import WaterfallRouter, resolve_api_key

logger = logging.getLogger(__name__)

_THINK_TAG_RE = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)
DEFAULT_LOCAL_OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"

# Shared connection pool for low-latency persistent keep-alive connections
_shared_client: Optional[httpx.AsyncClient] = None


def get_http_client(timeout: float = 300.0) -> httpx.AsyncClient:
    """Return a shared, connection-pooled httpx.AsyncClient instance."""
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
    return _shared_client


async def close_http_client() -> None:
    """Gracefully close the shared httpx.AsyncClient connection pool."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
    _shared_client = None


async def chat(system_prompt: str, user_message: str, config: Dict[str, Any]) -> str:
    """
    Standard OpenAI-compatible async chat client using pooled httpx connections.
    Routes through the WaterfallRouter when model is 'auto-cloud' or USE_AUTO_CLOUD=1,
    and supports automatic fleet node failover when a remote node is unreachable.
    """
    api_url = config.get("api_url", DEFAULT_LOCAL_OLLAMA_URL)
    model = config.get("model", "qwen2.5-coder:7b")
    temperature = config.get("temperature", 0.1)
    max_tokens = config.get("max_tokens", 2048)

    # Waterfall routing for cloud fallback
    if model == "auto-cloud" or os.environ.get("USE_AUTO_CLOUD") == "1":
        routes = config.get("waterfall_routes", [])
        router = WaterfallRouter(routes, timeout=120.0, client=get_http_client(120.0))
        raw_content = await router.route_chat(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _THINK_TAG_RE.sub("", raw_content or "").strip()

    # Standard single-endpoint execution with fleet node failover
    api_key = config.get("api_key", "sk-none")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {resolve_api_key(api_key)}",
    }

    base_payload: Dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    # Only include Ollama options/num_ctx when explicitly requested to prevent
    # VRAM eviction of models pinned with keep_alive: -1 at 4096 or 6144 ctx (INV-1)
    if "options" in config and isinstance(config["options"], dict):
        base_payload["options"] = config["options"]
    elif "num_ctx" in config and config["num_ctx"]:
        base_payload["options"] = {"num_ctx": int(config["num_ctx"])}

    if "response_format" in config:
        base_payload["response_format"] = config["response_format"]

    fallback_model = config.get(
        "fallback_model", os.environ.get("KRUSCH_FALLBACK_MODEL", "qwen2.5-coder:7b")
    )
    local_fallback_url = os.environ.get("KRUSCH_FALLBACK_OLLAMA_URL", DEFAULT_LOCAL_OLLAMA_URL)

    candidates = [(api_url, model)]
    for extra_url in config.get("fallback_urls", []) or []:
        if extra_url and all(extra_url != u for u, _ in candidates):
            candidates.append((extra_url, fallback_model))

    if (
        os.environ.get("KRUSCH_FLEET_FAILOVER", "1") != "0"
        and local_fallback_url
        and all(local_fallback_url != u for u, _ in candidates)
    ):
        candidates.append((local_fallback_url, fallback_model))

    client = get_http_client(timeout=float(config.get("timeout", 300.0)))
    last_exc: Optional[Exception] = None

    for idx, (target_url, target_model) in enumerate(candidates):
        payload = dict(base_payload)
        payload["model"] = target_model
        try:
            response = await client.post(target_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            raw_content = data["choices"][0]["message"].get("content") or ""
            return _THINK_TAG_RE.sub("", raw_content).strip()
        except Exception as e:
            last_exc = e
            if idx + 1 < len(candidates):
                logger.warning(
                    f"[Client] Fleet node {target_url} ({target_model}) failed ({e}); "
                    f"failing over to {candidates[idx + 1][0]}"
                )

    waterfall_routes = config.get("waterfall_routes", [])
    if waterfall_routes:
        try:
            logger.warning("[Client] All local fleet nodes failed; attempting WaterfallRouter failover.")
            router = WaterfallRouter(waterfall_routes, timeout=120.0, client=get_http_client(120.0))
            raw_content = await router.route_chat(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return _THINK_TAG_RE.sub("", raw_content or "").strip()
        except Exception as wf_exc:
            last_exc = wf_exc

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM chat request failed with no available endpoints.")
