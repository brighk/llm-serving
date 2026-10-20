"""Gateway settings, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite+aiosqlite:///./gateway.db"

    # Which pod backend to use: "k8s" in the cluster, "fake" for local dev/tests.
    pod_backend: str = "k8s"
    worker_namespace: str = "llm-workers"
    worker_image: str = "ghcr.io/ggml-org/llama.cpp:server-cuda"
    worker_port: int = 8080
    runtime_class: str = "nvidia"
    models_pvc: str = "models"
    model_file: str = "model.gguf"
    # SHA-256 of the model file. When set, an init container refuses to start
    # the worker if the file on disk does not match.
    model_sha256: str = ""
    context_size: int = 8192
    worker_memory_limit: str = "8Gi"

    # GPU admission control. With time-slicing, VRAM is NOT shared between pods:
    # each worker loads its own copy of the weights. Set this to the number of
    # workers that fit in VRAM (a 7-8B Q4 model on a 24 GB 3090: about 3).
    max_active_workers: int = 3

    pod_ready_timeout_s: float = 180.0
    pod_poll_interval_s: float = 1.0
    idle_timeout_s: float = 600.0
    reaper_interval_s: float = 30.0
    upstream_timeout_s: float = 300.0

    max_history_messages: int = 20
    max_output_tokens: int = 2048

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            database_url=_env("DATABASE_URL", cls.database_url),
            pod_backend=_env("POD_BACKEND", cls.pod_backend),
            worker_namespace=_env("WORKER_NAMESPACE", cls.worker_namespace),
            worker_image=_env("WORKER_IMAGE", cls.worker_image),
            worker_port=int(_env("WORKER_PORT", str(cls.worker_port))),
            runtime_class=_env("RUNTIME_CLASS", cls.runtime_class),
            models_pvc=_env("MODELS_PVC", cls.models_pvc),
            model_file=_env("MODEL_FILE", cls.model_file),
            model_sha256=_env("MODEL_SHA256", cls.model_sha256).strip().lower(),
            context_size=int(_env("CONTEXT_SIZE", str(cls.context_size))),
            worker_memory_limit=_env("WORKER_MEMORY_LIMIT", cls.worker_memory_limit),
            max_active_workers=int(_env("MAX_ACTIVE_WORKERS", str(cls.max_active_workers))),
            pod_ready_timeout_s=float(_env("POD_READY_TIMEOUT_S", str(cls.pod_ready_timeout_s))),
            idle_timeout_s=float(_env("IDLE_TIMEOUT_S", str(cls.idle_timeout_s))),
            reaper_interval_s=float(_env("REAPER_INTERVAL_S", str(cls.reaper_interval_s))),
            upstream_timeout_s=float(_env("UPSTREAM_TIMEOUT_S", str(cls.upstream_timeout_s))),
            max_history_messages=int(_env("MAX_HISTORY_MESSAGES", str(cls.max_history_messages))),
            max_output_tokens=int(_env("MAX_OUTPUT_TOKENS", str(cls.max_output_tokens))),
        )
