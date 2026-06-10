# Slim E2E test container — exercises LocalBackend against a real provider.
#
# Build context: contexto-py root.
#   cd contexto/packages/contexto-py
#   docker build -f e2e/Dockerfile -t contexto-local-e2e .
#
# Run (loads e2e/.env automatically):
#   docker run --rm --env-file e2e/.env contexto-local-e2e

FROM python:3.12-slim-bookworm

# Build-essential is required for scipy wheels to fall back to source on some platforms;
# trim runtime image with --no-install-recommends.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the package, install with deps.
COPY pyproject.toml ./
COPY src/ ./src/
COPY README.md ./

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir .

# Copy the e2e harness.
COPY e2e/run_local_e2e.py ./e2e/run_local_e2e.py

# Default: run the E2E script. Provider creds must come from --env or --env-file.
CMD ["python", "/app/e2e/run_local_e2e.py"]
