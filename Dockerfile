# ER-MAP OpenEnv server image
# ---------------------------------------------------------------------------
# Builds the TriageEnv FastAPI server for HF Spaces / local Docker.
# Excludes training-only deps (torch / unsloth / trl) so the server image
# stays small. To train, use a separate environment with requirements.txt
# fully installed.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only the env-server runtime deps. Training deps are excluded.
RUN pip install --upgrade pip && pip install \
    "gymnasium>=0.29.0" \
    "groq>=0.4.0" \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.27.0" \
    "pydantic>=2.0.0"

COPY ER_MAP /app/ER_MAP
COPY README.md /app/README.md

# HF Spaces convention: app listens on $PORT (default 7860).
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn ER_MAP.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
