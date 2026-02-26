"""
Lab Agent proxy endpoint.

Registers an APIRouter at the /lab prefix (auto-discovered by ownquesta_agents).
All requests to /lab/<path> are forwarded transparently to the lab-agent service
running on LAB_AGENT_URL (default: http://localhost:8020).

Streaming endpoints (SSE) are handled with httpx streaming so that server-sent
events flow through without buffering.
"""

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, Response
import httpx
import logging

from .config import LAB_AGENT_URL

logger = logging.getLogger("ownquesta_agents.lab_agent")

router = APIRouter()

# Paths that return Server-Sent Events (text/event-stream).
# These must be streamed rather than buffered.
_SSE_PATHS = {
    "/v2/analyze-stream",
    "/v2/build-pipeline-stream",
}


def _forward_headers(request: Request) -> dict:
    """Strip hop-by-hop headers that must not be forwarded."""
    skip = {"host", "content-length", "transfer-encoding", "connection"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_to_lab_agent(request: Request, path: str):
    """
    Transparent proxy: forwards every request to the lab-agent service and
    streams the response back.  SSE endpoints are handled with chunked streaming
    so events reach the browser in real-time.
    """
    target_url = f"{LAB_AGENT_URL}/{path}"
    params     = dict(request.query_params)
    headers    = _forward_headers(request)
    body       = await request.body()
    is_sse     = f"/{path}" in _SSE_PATHS

    logger.debug("Proxying %s %s → %s (sse=%s)", request.method, path, target_url, is_sse)

    if is_sse:
        async def _event_stream():
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                    async with client.stream(
                        request.method, target_url,
                        params=params, content=body, headers=headers,
                    ) as resp:
                        async for chunk in resp.aiter_bytes():
                            yield chunk
            except httpx.ConnectError:
                yield b'data: {"type":"error","text":"lab-agent is offline (http://localhost:8020)"}\n\n'
            except Exception as exc:
                err = str(exc).replace('"', "'")
                yield f'data: {{"type":"error","text":"{err}"}}\n\n'.encode()

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Non-streaming request ──────────────────────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.request(
                request.method, target_url,
                params=params, content=body, headers=headers,
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )
    except httpx.ConnectError:
        return Response(
            content=b'{"detail":"lab-agent is offline. Start it with: uvicorn main:app --port 8020"}',
            status_code=503,
            media_type="application/json",
        )
    except Exception as exc:
        logger.exception("Proxy error for %s", target_url)
        return Response(
            content=f'{{"detail":"{exc}"}}'.encode(),
            status_code=502,
            media_type="application/json",
        )
