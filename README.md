# secure-llm-serving

Multi-tenant LLM serving on k3s with one isolated model pod per user, token-based quotas, and chat history kept outside the pods.

```
client ──Bearer token──▶ Traefik ──▶ gateway (FastAPI) ──▶ worker pod (user A, llama.cpp + GPU slot)
                                        │    │         └─▶ worker pod (user B)
                                        │    └─ K8s API: create / delete pods in llm-workers only
                                        ▼
                                     Postgres (host disk): users · tokens · usage · conversations
```

## How it works

1. A request arrives with `Authorization: Bearer sk-llm-…`. The gateway hashes the token and looks it up in Postgres.
2. If the user's token quota is used up, the request gets `429` and the user's pod is deleted.
3. The gateway loads the conversation (only if this user owns it) and prepends the stored history.
4. It finds the user's worker pod, or creates one and waits for it to become ready.
5. It proxies the request to the pod, reads the token counts from the response and records them.
6. If that request used up the quota, the pod is deleted immediately.

A background reaper also deletes pods that are idle, belong to disabled or over-quota users, or have no database record.

### Sharing one GPU

The RTX 3090 is shared with NVIDIA time-slicing. VRAM is **not** shared between pods: each pod loads its own copy of the weights. The gateway therefore enforces a hard cap (`MAX_ACTIVE_WORKERS`):

| Situation | Behaviour |
| --- | --- |
| Below the cap | New user gets a pod |
| At the cap, some pod idle | Least recently used idle pod is evicted |
| At the cap, every pod busy | `503` with `Retry-After: 10` |

With a 7–8B model at Q4 (about 5–6 GB per pod with its KV cache), 24 GB fits about 3 pods.

## Security controls

| Threat | Control | Where |
| --- | --- | --- |
| Stolen database leaks usable tokens | Only SHA-256 hashes of 256-bit tokens are stored; raw token shown once | `app/auth.py` |
| One user reads another's chats | Every conversation lookup checks ownership; returns `404`, not `403` | `app/main.py` |
| One user reaches another's model | Pod per user; gateway routes only by authenticated user ID | `app/orchestrator.py` |
| Compromised model server | No egress, no service-account token, non-root, read-only root FS, all capabilities dropped, seccomp | `k8s/50-…`, `app/pods.py` |
| Tampered model weights | Init container checks the pinned SHA-256 before the server starts | `app/pods.py` |
| Gateway compromise escalates | ServiceAccount can only manage pods in `llm-workers` | `k8s/20-rbac.yaml` |
| Quota bypass or under-metering | Metering in the gateway; `max_tokens` clamped to remaining quota; missing usage charged at the maximum | `app/main.py` |
| GPU exhaustion | Admission cap, LRU eviction, back-pressure | `app/orchestrator.py` |
| Admin API abuse | No admin HTTP API; admin actions need `kubectl exec` | `app/admin.py` |

## Deploy on k3s

Prerequisites: NVIDIA driver and `nvidia-container-toolkit` installed **before** k3s starts, so k3s registers the `nvidia` RuntimeClass (`kubectl get runtimeclass`).

```bash
# 1. Model on the host
sudo mkdir -p /srv/models && sudo cp your-model.gguf /srv/models/model.gguf
sha256sum /srv/models/model.gguf          # paste into MODEL_SHA256 in k8s/40-gateway.yaml

# 2. GPU sharing
kubectl apply -f k8s/60-gpu-time-slicing.yaml
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
helm upgrade -i nvdp nvdp/nvidia-device-plugin -n kube-system \
  --set runtimeClassName=nvidia --set config.name=nvidia-device-plugin-config

# 3. Everything else
kubectl apply -f k8s/00-namespaces.yaml
kubectl -n llm-system create secret generic postgres-auth \
  --from-literal=password="$(openssl rand -base64 32)"
docker build -t llm-gateway:0.1.0 gateway/
docker save llm-gateway:0.1.0 | sudo k3s ctr images import -
kubectl apply -f k8s/

# 4. Users and tokens
kubectl -n llm-system exec deploy/llm-gateway -- python -m app.admin create-user alice --limit 200000
kubectl -n llm-system exec deploy/llm-gateway -- python -m app.admin issue-token alice
```

## API

OpenAI-compatible chat, plus server-side history. Send only the **new** messages each turn and pass `conversation_id` to continue.

```bash
curl -s http://llm.home.arpa/v1/chat/completions \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/chat/completions` | Chat; returns `conversation_id`, header `X-Quota-Remaining` |
| `GET /v1/usage` | Tokens used and remaining |
| `GET /v1/conversations` | Your conversations |
| `GET /v1/conversations/{id}/messages` | Full history |
| `DELETE /v1/conversations/{id}` | Delete a conversation and its messages |

## Develop and test

```bash
cd gateway
pip install -e ".[dev]"
pytest -q
```

Tests use SQLite and an in-memory pod backend, so no cluster or GPU is needed.

## Known limits of v0.1 and next steps

- **Single gateway replica.** Locks and the GPU cap live in process memory. Scaling out needs Postgres row locks (`SELECT … FOR UPDATE`) or a controller.
- **No streaming.** `stream: true` returns `400`.
- **Lifetime quota.** Usage does not reset monthly; add a usage window.
- **Chat history is stored in plain text.** Next: encryption at rest with per-user keys, and a retention period.
- **Gateway egress is open.** Next: restrict it to Postgres, the API server and worker pods.
- **Metrics.** Next: Prometheus metrics (time to first token, tokens/s, active pods, evictions, 503s) scraped by VictoriaMetrics, with vmalert rules.
- **Benchmark.** Compare per-user pods against one shared vLLM server with continuous batching.
