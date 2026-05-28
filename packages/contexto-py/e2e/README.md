# LocalBackend E2E

Two ways to exercise the new `LocalBackend` against a real provider.

## Prereqs

Set the provider key in `e2e/.env` (gitignored):

```
OPENROUTER_API_KEY=sk-or-v1-...
# or
OPENAI_API_KEY=sk-...
```

## 1. Slim Python container (fast — recommended)

Builds a minimal Python image, installs `contexto-hermes`, runs an ingest +
search round-trip directly against the configured provider.

```bash
cd contexto/packages/contexto-py
docker build -f e2e/Dockerfile -t contexto-local-e2e .
docker run --rm --env-file e2e/.env contexto-local-e2e
```

Expected (last line):

```
... INFO  e2e | E2E PASSED
```

What it verifies:

- Provider/key resolution (spec §7).
- `extract_episode_text` reads Hermes' flat `data.messages` shape.
- Real HTTP calls to `{base}/chat/completions` and `{base}/embeddings`.
- scipy AGNES clustering + atomic JSON write.
- `version: 1` schema written to disk.
- Spec §6 metadata (`source`, `status`, `confidence`, `evidence_refs`,
  `open_questions`, `trace_ref`, `sessionKey`, `episode.extracted_text`).
- Beam search retrieves semantically relevant items (Kubernetes-shaped
  episodes outrank an unrelated Italian-restaurant one).
- Persisted state reloads from disk on a fresh instance.

## 2. Full Hermes container (integration with the agent)

Runs the real `hermes-agent` gateway with `CONTEXTO_BACKEND=local`. The bundled
`plugins/context_engine/contexto/` is symlinked to this source tree in the
hermes-agent repo, so a fresh image build picks up the changes automatically.

```bash
# One-time: build the hermes-agent base image (slow — Playwright + npm).
cd ../../../hermes-agent
docker build -t hermes-agent .

# Run the gateway against the local backend.
cd ../contexto/packages/contexto-py
export HERMES_UID=$(id -u) HERMES_GID=$(id -g)
docker compose -f e2e/docker-compose.hermes-local.yml --env-file e2e/.env up
```

After driving a chat session that triggers `compress()`, the mindmap lands at:

```
~/.hermes/data/contexto/mindmap.json
```

(Inside the container that resolves to `/opt/data/data/contexto/mindmap.json`
via `$HERMES_HOME`.)

Quick check:

```bash
jq '.version, .stats' ~/.hermes/data/contexto/mindmap.json
```

## Files in this directory

| File | Purpose |
|---|---|
| `.env` | Provider secrets — **gitignored**. |
| `Dockerfile` | Slim Python container running `run_local_e2e.py`. |
| `run_local_e2e.py` | The actual end-to-end test script. |
| `docker-compose.hermes-local.yml` | Compose override for running the full hermes-agent against the local backend. |
| `README.md` | This file. |

## Cost

A single slim-container run makes ~3 chat completions + ~4 embeddings against
OpenRouter using `openai/gpt-4o-mini` + `openai/text-embedding-3-small`.
Approximate cost per run: < $0.01.
