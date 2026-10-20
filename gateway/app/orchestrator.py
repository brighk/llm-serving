"""Worker lifecycle: one pod per user, created on demand, killed on limit or idle.

GPU sharing policy (one RTX 3090, time-sliced):
  * VRAM is not shared between pods, so at most `max_active_workers` exist.
  * When a new user needs a pod and the cap is reached, the least recently
    used idle pod is evicted.
  * If every pod is busy serving a request, the caller gets CapacityError
    (HTTP 503 + Retry-After) instead of oversubscribing the GPU.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import AsyncIterator

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .config import Settings
from .db import User, WorkerPod, as_utc, utcnow
from .pods import PodBackend, PodInfo

log = logging.getLogger("gateway.orchestrator")


class CapacityError(Exception):
    """All GPU slots are busy; try again later."""


class WorkerStartError(Exception):
    """The worker pod did not become ready."""


class Orchestrator:
    def __init__(self, settings: Settings, backend: PodBackend, sessionmaker: async_sessionmaker):
        self._settings = settings
        self._backend = backend
        self._sm = sessionmaker
        self._user_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # Serialises "count pods, maybe evict, create" so the cap is never exceeded.
        self._capacity_lock = asyncio.Lock()
        # Users with a request in progress. Their pods are never evicted or reaped.
        self._in_flight: Counter[str] = Counter()

    # ---- public API -------------------------------------------------------

    @asynccontextmanager
    async def lease(self, user_id: str) -> AsyncIterator[str]:
        """Yield the base URL of the user's worker, keeping it busy meanwhile."""
        self._in_flight[user_id] += 1  # mark busy *before* any eviction decision
        try:
            base_url = await self.ensure_worker(user_id)
            yield base_url
        finally:
            self._in_flight[user_id] -= 1
            if self._in_flight[user_id] <= 0:
                del self._in_flight[user_id]
            await self._touch(user_id)

    async def ensure_worker(self, user_id: str) -> str:
        async with self._user_locks[user_id]:
            name: str | None = None
            async with self._sm() as session:
                worker = await session.get(WorkerPod, user_id)
                if worker is not None:
                    info = await self._backend.get(worker.pod_name)
                    if info is not None and not info.dead:
                        if info.ready and info.ip:
                            worker.status, worker.pod_ip = "ready", info.ip
                            worker.last_used_at = utcnow()
                            await session.commit()
                            return self._url(info.ip)
                        name = worker.pod_name  # still starting; wait below
                    else:
                        # Pod vanished or crashed: forget it and start a fresh one.
                        if info is not None:
                            await self._backend.delete(worker.pod_name)
                        await self._delete_row(session, user_id, worker.pod_name)

                if name is None:
                    name = await self._start_pod(session, user_id)

            info = await self._wait_ready(name)
            if info is None:
                await self._forget(user_id, name)
                raise WorkerStartError(f"worker {name} did not become ready")

            async with self._sm() as session:
                await session.execute(
                    update(WorkerPod)
                    .where(WorkerPod.user_id == user_id)
                    .values(status="ready", pod_ip=info.ip, last_used_at=utcnow())
                )
                await session.commit()
            return self._url(info.ip)

    async def release(self, user_id: str, reason: str) -> bool:
        """Delete the user's worker pod, if any."""
        async with self._user_locks[user_id]:
            async with self._sm() as session:
                worker = await session.get(WorkerPod, user_id)
                if worker is None:
                    return False
                await self._backend.delete(worker.pod_name)
                await self._delete_row(session, user_id, worker.pod_name)
        log.info("worker released user=%s pod=%s reason=%s", user_id, worker.pod_name, reason)
        return True

    async def reap(self) -> None:
        """Kill idle workers, workers of blocked users, and orphaned pods."""
        cutoff = utcnow() - timedelta(seconds=self._settings.idle_timeout_s)
        async with self._sm() as session:
            workers = (await session.execute(select(WorkerPod))).scalars().all()
            blocked = set(
                (
                    await session.execute(
                        select(User.id).where(
                            or_(User.disabled.is_(True), User.tokens_used >= User.token_limit)
                        )
                    )
                ).scalars()
            )

        for w in workers:
            if w.user_id in blocked:
                await self.release(w.user_id, "blocked")
            elif w.user_id not in self._in_flight and as_utc(w.last_used_at) <= cutoff:
                await self.release(w.user_id, "idle")

        # Orphans: pods in the cluster with no DB row, e.g. after a gateway crash.
        async with self._capacity_lock:
            async with self._sm() as session:
                known = set((await session.execute(select(WorkerPod.pod_name))).scalars())
            for name in await self._backend.list_managed():
                if name not in known:
                    log.warning("deleting orphaned worker pod=%s", name)
                    await self._backend.delete(name)

    async def run_reaper(self) -> None:
        while True:
            try:
                await self.reap()
            except Exception:  # keep the loop alive; the next pass will retry
                log.exception("reaper pass failed")
            await asyncio.sleep(self._settings.reaper_interval_s)

    async def active_workers(self) -> int:
        async with self._sm() as session:
            return (await session.execute(select(func.count()).select_from(WorkerPod))).scalar_one()

    # ---- internals --------------------------------------------------------

    def _url(self, ip: str) -> str:
        return f"http://{ip}:{self._settings.worker_port}"

    async def _start_pod(self, session: AsyncSession, user_id: str) -> str:
        async with self._capacity_lock:
            workers = (await session.execute(select(WorkerPod))).scalars().all()
            if len(workers) >= self._settings.max_active_workers:
                idle = [w for w in workers if w.user_id not in self._in_flight]
                if not idle:
                    raise CapacityError("all GPU slots are busy")
                victim = min(idle, key=lambda w: as_utc(w.last_used_at))
                await self._backend.delete(victim.pod_name)
                await self._delete_row(session, victim.user_id, victim.pod_name)
                log.info("evicted LRU worker user=%s pod=%s", victim.user_id, victim.pod_name)

            name = await self._backend.create(user_id)
            session.add(WorkerPod(user_id=user_id, pod_name=name, status="pending"))
            await session.commit()
            log.info("worker created user=%s pod=%s", user_id, name)
            return name

    async def _wait_ready(self, name: str) -> PodInfo | None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.pod_ready_timeout_s
        while loop.time() < deadline:
            info = await self._backend.get(name)
            if info is None or info.dead:
                return None
            if info.ready and info.ip:
                return info
            await asyncio.sleep(self._settings.pod_poll_interval_s)
        return None

    async def _forget(self, user_id: str, name: str) -> None:
        await self._backend.delete(name)
        async with self._sm() as session:
            await self._delete_row(session, user_id, name)

    @staticmethod
    async def _delete_row(session: AsyncSession, user_id: str, pod_name: str) -> None:
        # A DELETE statement (not session.delete) so a concurrent release of the
        # same row is a no-op instead of a stale-data error.
        await session.execute(
            delete(WorkerPod).where(WorkerPod.user_id == user_id, WorkerPod.pod_name == pod_name)
        )
        await session.commit()
        session.expunge_all()

    async def _touch(self, user_id: str) -> None:
        async with self._sm() as session:
            await session.execute(
                update(WorkerPod).where(WorkerPod.user_id == user_id).values(last_used_at=utcnow())
            )
            await session.commit()
