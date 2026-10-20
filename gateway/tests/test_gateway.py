import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.admin import create_user, issue_token, revoke_token
from app.config import Settings
from app.db import UsageEvent, User, init_db, make_engine, make_sessionmaker
from app.main import create_app
from app.pods import FakePodBackend, build_pod_manifest


class FakeWorker:
    """Stands in for llama.cpp's OpenAI-compatible server."""

    def __init__(self, usage=(10, 5), reply="hello from the model"):
        self.usage = usage
        self.reply = reply
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        payload = {
            "id": "cmpl-1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.reply},
                         "finish_reason": "stop"}],
        }
        if self.usage is not None:
            p, c = self.usage
            payload["usage"] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}
        return httpx.Response(200, json=payload)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def env(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/test.db"

    async def setup():
        engine = make_engine(url)
        await init_db(engine)
        sm = make_sessionmaker(engine)
        async with sm() as s:
            await create_user(s, "alice", 1_000)
            await create_user(s, "bob", 1_000)
            await create_user(s, "carol", 1_000)
            tokens = {n: await issue_token(s, n) for n in ("alice", "bob", "carol")}
        await engine.dispose()
        return tokens

    tokens = run(setup())
    settings = Settings(database_url=url, pod_backend="fake", pod_poll_interval_s=0.01,
                        pod_ready_timeout_s=2, max_active_workers=2)
    backend = FakePodBackend()
    worker = FakeWorker()
    app = create_app(settings, backend=backend,
                     upstream_transport=httpx.MockTransport(worker.handler), start_reaper=False)
    with TestClient(app) as client:
        yield {"client": client, "app": app, "backend": backend, "worker": worker,
               "tokens": tokens, "url": url}


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def chat(client, token, text="hi", **extra):
    return client.post("/v1/chat/completions", headers=auth(token),
                       json={"messages": [{"role": "user", "content": text}], **extra})


# ---- authentication ---------------------------------------------------------

def test_rejects_missing_and_bad_tokens(env):
    c = env["client"]
    assert c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 401
    assert chat(c, "sk-llm-not-a-real-token").status_code == 401
    assert chat(c, "Basic abc").status_code == 401
    assert env["backend"].created == []  # no pod for unauthenticated traffic


def test_api_docs_are_not_served(env):
    c = env["client"]
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404


def test_revoked_token_is_rejected(env):
    token = env["tokens"]["alice"]
    assert chat(env["client"], token).status_code == 200

    async def revoke():
        engine = make_engine(env["url"])
        async with make_sessionmaker(engine)() as s:
            await revoke_token(s, token[:11])
        await engine.dispose()

    run(revoke())
    assert chat(env["client"], token).status_code == 401


# ---- pods, metering and history ---------------------------------------------

def test_first_request_creates_pod_and_meters_usage(env):
    r = chat(env["client"], env["tokens"]["alice"])
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello from the model"
    assert body["usage"]["total_tokens"] == 15
    assert r.headers["X-Quota-Remaining"] == "985"
    assert len(env["backend"].created) == 1

    # Second request reuses the same pod.
    chat(env["client"], env["tokens"]["alice"])
    assert len(env["backend"].created) == 1
    assert env["client"].get("/v1/usage", headers=auth(env["tokens"]["alice"])).json()["tokens_used"] == 30


def test_history_is_stored_outside_the_pod_and_replayed(env):
    c, token = env["client"], env["tokens"]["alice"]
    conv_id = chat(c, token, "my name is Alice").json()["conversation_id"]

    # Kill the pod: history must survive because it lives in the database.
    env["client"].portal.call(env["app"].state.orchestrator.release, _user_id(env, "alice"), "test")
    assert chat(c, token, "what is my name?", conversation_id=conv_id).status_code == 200

    sent = env["worker"].requests[-1]["messages"]
    assert [m["content"] for m in sent] == ["my name is Alice", "hello from the model", "what is my name?"]
    msgs = c.get(f"/v1/conversations/{conv_id}/messages", headers=auth(token)).json()
    assert len(msgs) == 4


def test_users_cannot_read_each_others_conversations(env):
    c = env["client"]
    conv_id = chat(c, env["tokens"]["alice"]).json()["conversation_id"]
    bob = env["tokens"]["bob"]
    assert c.get(f"/v1/conversations/{conv_id}/messages", headers=auth(bob)).status_code == 404
    assert chat(c, bob, conversation_id=conv_id).status_code == 404
    assert c.delete(f"/v1/conversations/{conv_id}", headers=auth(bob)).status_code == 404
    assert c.get("/v1/conversations", headers=auth(bob)).json() == []


def test_each_user_gets_their_own_pod(env):
    chat(env["client"], env["tokens"]["alice"])
    chat(env["client"], env["tokens"]["bob"])
    users = {p["user_id"] for p in env["backend"].pods.values()}
    assert users == {_user_id(env, "alice"), _user_id(env, "bob")}


# ---- quota enforcement --------------------------------------------------------

def test_reaching_the_limit_kills_the_pod_and_blocks_further_requests(env):
    c, token = env["client"], env["tokens"]["alice"]
    env["worker"].usage = (400, 600)  # 1000 tokens = the whole quota
    r = chat(c, token)
    assert r.status_code == 200
    assert r.headers["X-Quota-Remaining"] == "0"
    assert env["backend"].pods == {}  # pod deleted
    assert len(env["backend"].deleted) == 1

    assert chat(c, token).status_code == 429
    assert len(env["backend"].created) == 1  # no new pod after the limit


def test_max_tokens_is_clamped_to_remaining_quota(env):
    env["worker"].usage = (100, 800)
    chat(env["client"], env["tokens"]["alice"])  # 100 remaining
    chat(env["client"], env["tokens"]["alice"], max_tokens=4000)
    assert env["worker"].requests[-1]["max_tokens"] == 100


def test_missing_usage_is_charged_conservatively(env):
    env["worker"].usage = None
    chat(env["client"], env["tokens"]["alice"], max_tokens=200)

    async def events():
        engine = make_engine(env["url"])
        async with make_sessionmaker(engine)() as s:
            rows = (await s.execute(select(UsageEvent))).scalars().all()
        await engine.dispose()
        return rows

    (event,) = run(events())
    assert event.metered_by_fallback and event.completion_tokens == 200


# ---- GPU admission control ----------------------------------------------------

def test_cap_evicts_least_recently_used_idle_pod(env):
    c, t = env["client"], env["tokens"]
    chat(c, t["alice"])
    chat(c, t["bob"])
    chat(c, t["carol"])  # cap is 2: alice (LRU) must be evicted
    users = {p["user_id"] for p in env["backend"].pods.values()}
    assert users == {_user_id(env, "bob"), _user_id(env, "carol")}


def test_cap_returns_503_when_every_pod_is_busy(env):
    orch = env["app"].state.orchestrator
    alice, bob = _user_id(env, "alice"), _user_id(env, "bob")
    chat(env["client"], env["tokens"]["alice"])
    chat(env["client"], env["tokens"]["bob"])
    orch._in_flight.update([alice, bob])  # simulate two requests in progress
    r = chat(env["client"], env["tokens"]["carol"])
    assert r.status_code == 503 and r.headers["Retry-After"] == "10"


# ---- reaper -------------------------------------------------------------------

def test_reaper_kills_idle_and_disabled_users_pods(env):
    app = env["app"]
    chat(env["client"], env["tokens"]["alice"])
    chat(env["client"], env["tokens"]["bob"])

    async def disable_bob():
        engine = make_engine(env["url"])
        async with make_sessionmaker(engine)() as s:
            user = (await s.execute(select(User).where(User.name == "bob"))).scalar_one()
            user.disabled = True
            await s.commit()
        await engine.dispose()

    run(disable_bob())
    env["client"].portal.call(app.state.orchestrator.reap)
    assert {p["user_id"] for p in env["backend"].pods.values()} == {_user_id(env, "alice")}

    object.__setattr__(app.state.settings, "idle_timeout_s", 0)
    env["client"].portal.call(app.state.orchestrator.reap)
    assert env["backend"].pods == {}


def test_reaper_deletes_orphaned_pods(env):
    backend = env["backend"]
    env["client"].portal.call(backend.create, "ghost-user-id")
    env["client"].portal.call(env["app"].state.orchestrator.reap)
    assert backend.pods == {}


# ---- pod spec -------------------------------------------------------------------

def test_worker_pod_spec_is_locked_down():
    s = Settings(model_sha256="ab" * 32)
    pod = build_pod_manifest(s, "user-1", "llm-user-1-abc")
    spec = pod["spec"]
    ctr = spec["containers"][0]
    assert spec["automountServiceAccountToken"] is False
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
    assert ctr["securityContext"] == {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                                      "capabilities": {"drop": ["ALL"]}}
    assert ctr["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert all(m["readOnly"] for m in ctr["volumeMounts"] if m["name"] == "models")
    assert "sha256sum -c" in spec["initContainers"][0]["args"][0]
    assert pod["metadata"]["labels"]["llm-gateway/user-id"] == "user-1"


def _user_id(env, name):
    async def q():
        engine = make_engine(env["url"])
        async with make_sessionmaker(engine)() as s:
            uid = (await s.execute(select(User.id).where(User.name == name))).scalar_one()
        await engine.dispose()
        return uid
    return run(q())
