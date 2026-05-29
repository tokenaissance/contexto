# Contexto-Hermes Quickstart

Install Contexto as a context engine in [hermes-agent](https://hermes-agent.nousresearch.com) in a few minutes. Two backends to choose from — they install the same way and differ only in env vars.

| | **Remote** *(default)* | **Local** *(v0.2.0+)* |
|---|---|---|
| Storage | Contexto cloud | On-disk JSON (`$HERMES_HOME/data/contexto/mindmap.json`) |
| Embeddings + summarization | Contexto-hosted | Your OpenAI **or** OpenRouter key |
| Required key | `CONTEXTO_API_KEY` from [getcontexto.com](https://getcontexto.com) | `OPENAI_API_KEY` *or* `OPENROUTER_API_KEY` |
| Extra deps | none beyond `httpx` | `numpy`, `scipy` (installed automatically) |
| Best for | Zero local setup, managed retrieval | Air-gapped / BYOK / "no third-party context store" |

## 0. Prereqs

- Python 3.10 or newer.
- `hermes-agent` installed and importable (i.e. `python -c "import plugins.context_engine"` resolves). If you don't have it yet, follow the hermes-agent install instructions first, then come back.

## 1. Install the plugin (both backends)

```bash
pip install contexto-hermes
contexto-hermes-install        # symlinks the plugin into hermes-agent's tree
```

`contexto-hermes-install` finds hermes via the same import path Hermes uses. To point at a non-default checkout: `HERMES_AGENT_ROOT=/path/to/hermes-agent contexto-hermes-install`.

Then enable the engine in `~/.hermes/config.yaml`:

```yaml
context:
  engine: contexto
```

That's it for installation. The remaining step is one or two env vars to pick a backend.

## 2a. Remote backend (default)

```bash
export CONTEXTO_API_KEY=ckai_xxx          # from https://getcontexto.com
hermes chat                                # or `hermes gateway run`
```

Nothing else to configure. `CONTEXTO_BACKEND` defaults to `remote`.

## 2b. Local backend

```bash
export CONTEXTO_BACKEND=local
export OPENROUTER_API_KEY=sk-or-xxx        # or: export OPENAI_API_KEY=sk-xxx
hermes chat                                # or `hermes gateway run`
```

Provider is auto-detected from the key you set. If both keys are exported, OpenRouter wins; pin explicitly with `CONTEXTO_LOCAL_PROVIDER=openai|openrouter`. The mindmap lands at `~/.hermes/data/contexto/mindmap.json` after the first `compress()`.

## 3. Verify it's wired up

After enough chat turns to trigger compaction:

```bash
# Hermes logs — confirm no registration error.
# If anything is wrong you'll see one of these and Hermes falls back silently:
#   "Contexto plugin not registered: CONTEXTO_API_KEY is not set"
#   "Contexto plugin (local) not registered: local config invalid"
grep -i "contexto plugin" ~/.hermes/logs/hermes.log

# Local backend only — inspect the on-disk mindmap.
jq '.version, .stats' ~/.hermes/data/contexto/mindmap.json
```

A healthy local backend prints `1` and `{"total_items": N, "total_clusters": M, ...}`.

## Running inside Docker

The hermes-agent base image needs two adjustments when using the local backend:

1. **Plugin source.** The bundled `plugins/context_engine/contexto/` is an absolute symlink that `docker build` resolves to a host path. Bind-mount the source at runtime instead:
   ```yaml
   volumes:
     - /path/to/contexto/packages/contexto-py/src/contexto_hermes:/opt/hermes/plugins/context_engine/contexto:ro
   ```
2. **Runtime deps.** Install `numpy` + `scipy` into the image's venv before launching:
   ```yaml
   command:
     - sh
     - -c
     - |
       uv pip install --python /opt/hermes/.venv/bin/python numpy scipy \
         && exec hermes gateway run
   ```

A ready-to-use compose file is at [`packages/contexto-py/e2e/docker-compose.hermes-local.yml`](../packages/contexto-py/e2e/docker-compose.hermes-local.yml).

The remote backend has no extra Docker requirements — just `CONTEXTO_API_KEY` in the environment.

## Full config reference

The most common knobs are above. For everything else — `CONTEXTO_MAX_RESULTS`, `CONTEXTO_MIN_SCORE`, `CONTEXTO_LOCAL_SIMILARITY_THRESHOLD`, model overrides, timeouts, status fields — see the package README at [`packages/contexto-py/README.md`](../packages/contexto-py/README.md).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Contexto plugin not registered: CONTEXTO_API_KEY is not set` | Remote backend selected (default) but no API key in env. |
| `Contexto plugin (local) not registered: local config invalid` | Local backend selected but neither `OPENAI_API_KEY` nor `OPENROUTER_API_KEY` is set, or `CONTEXTO_LOCAL_PROVIDER` is unknown. Check the line above this in the log for the specific reason. |
| `ModuleNotFoundError: No module named 'numpy'` (Docker) | The image doesn't ship numpy/scipy — install them into the venv as shown above. |
| `Could not locate hermes-agent's plugins/context_engine directory` from `contexto-hermes-install` | hermes-agent isn't installed in the same Python environment. Activate the venv first, or set `HERMES_AGENT_ROOT`. |
