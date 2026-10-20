"""Pod backends: how worker pods are created, inspected and deleted.

K8sPodBackend talks to the Kubernetes API. FakePodBackend keeps pods in memory
for local development and tests. The orchestrator only sees the PodBackend
interface, so a different backend (for example one shared vLLM server) can be
swapped in later for benchmarking.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from typing import Protocol

from .config import Settings

MANAGED_LABEL = "app.kubernetes.io/managed-by"
MANAGED_VALUE = "llm-gateway"
USER_LABEL = "llm-gateway/user-id"


@dataclass
class PodInfo:
    name: str
    phase: str  # Pending | Running | Succeeded | Failed | Unknown
    ready: bool
    ip: str | None

    @property
    def dead(self) -> bool:
        return self.phase in ("Succeeded", "Failed")


class PodBackend(Protocol):
    async def create(self, user_id: str) -> str: ...
    async def get(self, name: str) -> PodInfo | None: ...
    async def delete(self, name: str) -> None: ...
    async def list_managed(self) -> list[str]: ...


def pod_name_for(user_id: str) -> str:
    # Random suffix so a recreated pod never reuses an old name (or old IP cache).
    return f"llm-{user_id[:8]}-{secrets.token_hex(3)}"


def build_pod_manifest(settings: Settings, user_id: str, name: str) -> dict:
    """Worker pod spec. Passes the Pod Security 'restricted' profile."""
    port = settings.worker_port
    model_path = f"/models/{settings.model_file}"

    container_security = {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
        "capabilities": {"drop": ["ALL"]},
    }

    spec: dict = {
        "automountServiceAccountToken": False,  # workers never talk to the K8s API
        "enableServiceLinks": False,
        "restartPolicy": "Never",  # a crashed worker is replaced, not restarted
        "securityContext": {
            "runAsNonRoot": True,
            "runAsUser": 65532,
            "runAsGroup": 65532,
            "seccompProfile": {"type": "RuntimeDefault"},
        },
        "containers": [
            {
                "name": "llama",
                "image": settings.worker_image,
                "args": [
                    "-m", model_path,
                    "--host", "0.0.0.0",
                    "--port", str(port),
                    "-ngl", "99",  # offload all layers to the GPU
                    "-c", str(settings.context_size),
                    "--parallel", "1",  # one user per pod; one request at a time
                ],
                "ports": [{"containerPort": port, "name": "http"}],
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": port},
                    "periodSeconds": 2,
                    "failureThreshold": 3,
                },
                "resources": {
                    "requests": {"cpu": "1", "memory": "2Gi"},
                    "limits": {
                        "nvidia.com/gpu": "1",  # one time-sliced GPU slot
                        "memory": settings.worker_memory_limit,
                    },
                },
                "securityContext": container_security,
                "volumeMounts": [
                    {"name": "models", "mountPath": "/models", "readOnly": True},
                    {"name": "tmp", "mountPath": "/tmp"},
                ],
            }
        ],
        "volumes": [
            {
                "name": "models",
                "persistentVolumeClaim": {"claimName": settings.models_pvc, "readOnly": True},
            },
            {"name": "tmp", "emptyDir": {"sizeLimit": "256Mi"}},
        ],
    }

    if settings.runtime_class:
        spec["runtimeClassName"] = settings.runtime_class

    if settings.model_sha256:
        # Refuse to serve a model file that does not match the pinned hash.
        spec["initContainers"] = [
            {
                "name": "verify-model",
                "image": settings.worker_image,
                "command": ["sh", "-c"],
                "args": [
                    f'echo "{settings.model_sha256}  {model_path}" | sha256sum -c -'
                ],
                "resources": {"limits": {"memory": "256Mi"}},
                "securityContext": container_security,
                "volumeMounts": [{"name": "models", "mountPath": "/models", "readOnly": True}],
            }
        ]

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": settings.worker_namespace,
            "labels": {
                "app.kubernetes.io/name": "llm-worker",
                MANAGED_LABEL: MANAGED_VALUE,
                USER_LABEL: user_id,
            },
        },
        "spec": spec,
    }


class K8sPodBackend:
    """Creates worker pods through the Kubernetes API.

    The gateway's ServiceAccount is bound to a Role that only allows pod
    create/get/list/delete in the worker namespace (see k8s/20-rbac.yaml).
    """

    def __init__(self, settings: Settings):
        from kubernetes import client, config

        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()  # local development against a cluster
        self._api = client.CoreV1Api()
        self._settings = settings
        self._ApiException = client.exceptions.ApiException

    async def create(self, user_id: str) -> str:
        name = pod_name_for(user_id)
        body = build_pod_manifest(self._settings, user_id, name)
        await asyncio.to_thread(
            self._api.create_namespaced_pod, self._settings.worker_namespace, body
        )
        return name

    async def get(self, name: str) -> PodInfo | None:
        try:
            pod = await asyncio.to_thread(
                self._api.read_namespaced_pod, name, self._settings.worker_namespace
            )
        except self._ApiException as exc:
            if exc.status == 404:
                return None
            raise
        conditions = pod.status.conditions or []
        ready = any(c.type == "Ready" and c.status == "True" for c in conditions)
        return PodInfo(name=name, phase=pod.status.phase or "Unknown", ready=ready, ip=pod.status.pod_ip)

    async def delete(self, name: str) -> None:
        try:
            await asyncio.to_thread(
                self._api.delete_namespaced_pod,
                name,
                self._settings.worker_namespace,
                grace_period_seconds=5,
            )
        except self._ApiException as exc:
            if exc.status != 404:
                raise

    async def list_managed(self) -> list[str]:
        pods = await asyncio.to_thread(
            self._api.list_namespaced_pod,
            self._settings.worker_namespace,
            label_selector=f"{MANAGED_LABEL}={MANAGED_VALUE}",
        )
        return [p.metadata.name for p in pods.items]


@dataclass
class FakePodBackend:
    """In-memory backend for tests and local runs without a cluster."""

    ip: str = "127.0.0.1"
    ready_after_polls: int = 0
    pods: dict[str, dict] = field(default_factory=dict)
    created: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    async def create(self, user_id: str) -> str:
        name = pod_name_for(user_id)
        self.pods[name] = {"user_id": user_id, "polls": 0}
        self.created.append(name)
        return name

    async def get(self, name: str) -> PodInfo | None:
        pod = self.pods.get(name)
        if pod is None:
            return None
        pod["polls"] += 1
        ready = pod["polls"] > self.ready_after_polls
        return PodInfo(
            name=name,
            phase="Running" if ready else "Pending",
            ready=ready,
            ip=self.ip if ready else None,
        )

    async def delete(self, name: str) -> None:
        if self.pods.pop(name, None) is not None:
            self.deleted.append(name)

    async def list_managed(self) -> list[str]:
        return list(self.pods)
