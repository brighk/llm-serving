"""HTTP API. The only component reachable from outside the cluster.

Request flow for POST /v1/chat/completions:
  1. Authenticate the bearer token against Postgres.
  2. Refuse (429) and kill the user's pod if the token quota is used up.
  3. Load the conversation (owned by this user) and prepend stored history.
  4. Get the user's worker pod, creating it if needed (503 if the GPU is full).
  5. Proxy the request; meter tokens from the worker's response.
  6. Save the new messages and usage; kill the pod if the quota is now exhausted.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, select, update

from .auth import authenticate, parse_bearer
from .config import Settings
from .db import (
    Conversation,
    Message,
    UsageEvent,
    User,
    init_db,
    make_engine,
    make_sessionmaker,
    utcnow,
)
from .orchestrator import CapacityError, Orchestrator, WorkerStartError
from .pods import FakePodBackend, K8sPodBackend, PodBackend

log = logging.getLogger("gateway")


# ---- request / response schemas -------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=32_000)


class ChatRequest(BaseModel):
    # With server-side history, send only the NEW messages for this turn.
    messages: list[ChatMessage] = Field(min_length=1, max_length=50)
    conversation_id: str | None = None
    max_tokens: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False
    model: str | None = None  # accepted for OpenAI-client compatibility; ignored


# ---- app factory ----------------------------------------------------------


def make_backend(settings: Settings) -> PodBackend:
    if settings.pod_backend == "fake":
        return FakePodBackend()
    return K8sPodBackend(settings)


def create_app(
    settings: Settings | None = None,
    backend: PodBackend | None = None,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    start_reaper: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = make_engine(settings.database_url)
        await init_db(engine)
        sm = make_sessionmaker(engine)
        orch = Orchestrator(settings, backend or make_backend(settings), sm)
        http = httpx.AsyncClient(
            transport=upstream_transport,
            timeout=httpx.Timeout(settings.upstream_timeout_s, connect=5.0),
        )
        app.state.settings = settings
        app.state.sessionmaker = sm
        app.state.orchestrator = orch
        app.state.http = http
        # One request per user at a time: matches --parallel 1 in the worker and
        # keeps the quota check and the usage update from racing each other.
        app.state.user_request_locks = defaultdict(asyncio.Lock)
        reaper = asyncio.create_task(orch.run_reaper()) if start_reaper else None
        try:
            yield
        finally:
            if reaper:
                reaper.cancel()
            await http.aclose()
            await engine.dispose()

    # No /docs, /redoc or /openapi.json: they would be public, before any auth.
    app = FastAPI(title="secure-llm-serving gateway", version="0.1.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    async def current_user(request: Request, authorization: str | None = Header(default=None)) -> User:
        async with request.app.state.sessionmaker() as session:
            user = await authenticate(session, parse_bearer(authorization))
        if user is None:
            raise HTTPException(401, "invalid or missing token", headers={"WWW-Authenticate": "Bearer"})
        return user

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/v1/usage")
    async def usage(user: User = Depends(current_user)) -> dict:
        return {
            "tokens_used": user.tokens_used,
            "token_limit": user.token_limit,
            "tokens_remaining": max(0, user.token_limit - user.tokens_used),
        }

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest, request: Request, response: Response, user: User = Depends(current_user)) -> dict:
        if body.stream:
            raise HTTPException(400, "streaming is not supported in v0.1")

        st = request.app.state
        orch: Orchestrator = st.orchestrator

        async with st.user_request_locks[user.id]:
            async with st.sessionmaker() as session:
                fresh = await session.get(User, user.id)
                remaining = fresh.token_limit - fresh.tokens_used
            if remaining <= 0:
                await orch.release(user.id, "quota")
                raise HTTPException(429, "token quota exhausted")

            max_tokens = min(body.max_tokens, settings.max_output_tokens, remaining)

            # -- conversation and history (stored in Postgres, never in the pod)
            async with st.sessionmaker() as session:
                if body.conversation_id:
                    conv = await session.get(Conversation, body.conversation_id)
                    # 404 (not 403) so other users' conversation IDs are not confirmed.
                    if conv is None or conv.user_id != user.id:
                        raise HTTPException(404, "conversation not found")
                else:
                    conv = Conversation(user_id=user.id, title=body.messages[-1].content[:80])
                    session.add(conv)
                    await session.commit()
                history = (
                    await session.execute(
                        select(Message)
                        .where(Message.conversation_id == conv.id)
                        .order_by(Message.id.desc())
                        .limit(settings.max_history_messages)
                    )
                ).scalars().all()

            prompt = [{"role": m.role, "content": m.content} for m in reversed(history)]
            prompt += [m.model_dump() for m in body.messages]

            # -- get the user's worker and proxy the request
            try:
                async with orch.lease(user.id) as base_url:
                    upstream = await st.http.post(
                        f"{base_url}/v1/chat/completions",
                        json={
                            "messages": prompt,
                            "max_tokens": max_tokens,
                            "temperature": body.temperature,
                            "stream": False,
                        },
                    )
            except CapacityError:
                raise HTTPException(503, "all GPU slots are busy, retry shortly", headers={"Retry-After": "10"})
            except WorkerStartError:
                raise HTTPException(503, "model worker failed to start", headers={"Retry-After": "30"})
            except httpx.HTTPError as exc:
                log.warning("upstream error user=%s err=%s", user.id, type(exc).__name__)
                raise HTTPException(502, "model worker unavailable")

            if upstream.status_code != 200:
                log.warning("upstream status user=%s code=%s", user.id, upstream.status_code)
                raise HTTPException(502, "model worker returned an error")

            data = upstream.json()
            reply = data["choices"][0]["message"]["content"]

            # -- meter in the gateway. If the worker reports no usage, fail closed:
            # charge the full output budget plus a rough prompt estimate.
            reported = data.get("usage") or {}
            fallback = "prompt_tokens" not in reported or "completion_tokens" not in reported
            if fallback:
                prompt_tokens = sum(len(m["content"]) for m in prompt) // 3
                completion_tokens = max_tokens
            else:
                prompt_tokens = int(reported["prompt_tokens"])
                completion_tokens = int(reported["completion_tokens"])
            charged = prompt_tokens + completion_tokens

            async with st.sessionmaker() as session:
                for m in body.messages:
                    session.add(Message(conversation_id=conv.id, role=m.role, content=m.content))
                session.add(Message(conversation_id=conv.id, role="assistant", content=reply))
                session.add(
                    UsageEvent(
                        user_id=user.id,
                        conversation_id=conv.id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        metered_by_fallback=fallback,
                    )
                )
                await session.execute(
                    update(User).where(User.id == user.id).values(tokens_used=User.tokens_used + charged)
                )
                await session.execute(
                    update(Conversation).where(Conversation.id == conv.id).values(updated_at=utcnow())
                )
                await session.commit()
                fresh = await session.get(User, user.id, populate_existing=True)
                remaining_after = max(0, fresh.token_limit - fresh.tokens_used)

            if remaining_after == 0:
                await orch.release(user.id, "quota")  # the limit kills the pod

        response.headers["X-Quota-Remaining"] = str(remaining_after)
        return {
            "id": data.get("id"),
            "object": "chat.completion",
            "conversation_id": conv.id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                         "finish_reason": data["choices"][0].get("finish_reason")}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": charged},
        }

    @app.get("/v1/conversations")
    async def list_conversations(request: Request, user: User = Depends(current_user)) -> list[dict]:
        async with request.app.state.sessionmaker() as session:
            rows = (
                await session.execute(
                    select(Conversation)
                    .where(Conversation.user_id == user.id)
                    .order_by(Conversation.updated_at.desc())
                    .limit(100)
                )
            ).scalars().all()
        return [{"id": c.id, "title": c.title, "updated_at": c.updated_at.isoformat()} for c in rows]

    @app.get("/v1/conversations/{conversation_id}/messages")
    async def get_messages(conversation_id: str, request: Request, user: User = Depends(current_user)) -> list[dict]:
        async with request.app.state.sessionmaker() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user.id:
                raise HTTPException(404, "conversation not found")
            rows = (
                await session.execute(
                    select(Message).where(Message.conversation_id == conv.id).order_by(Message.id)
                )
            ).scalars().all()
        return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in rows]

    @app.delete("/v1/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str, request: Request, user: User = Depends(current_user)) -> Response:
        async with request.app.state.sessionmaker() as session:
            conv = await session.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user.id:
                raise HTTPException(404, "conversation not found")
            # Explicit delete of messages: SQLite ignores ON DELETE CASCADE by default.
            await session.execute(delete(Message).where(Message.conversation_id == conv.id))
            await session.delete(conv)
            await session.commit()
        return Response(status_code=204)

    return app


def asgi() -> FastAPI:
    """Entrypoint for uvicorn: `uvicorn app.main:asgi --factory`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
