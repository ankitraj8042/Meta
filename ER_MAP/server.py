"""
ER_MAP/server.py
================
FastAPI wrapper around TriageEnv for OpenEnv-style deployment
(HF Space / Docker / local Uvicorn).

Endpoints:
    POST /reset         -> {observation, info}
    POST /step          -> {observation, reward, done, truncated, info}
    GET  /state         -> full internal env state
    GET  /health        -> {"status": "ok", "version": ...}
    GET  /              -> human-readable landing page

Usage (local):
    uvicorn ER_MAP.server:app --host 0.0.0.0 --port 7860 --reload

Usage (HF Space):
    The HF Space spec auto-launches uvicorn against this app.

Environment variables:
    GROQ_API_KEY         shared key for both nurse and patient
    GROQ_NURSE_API_KEY   per-role override (optional)
    GROQ_PATIENT_API_KEY per-role override (optional)
    ERMAP_MODEL          Groq model id (default: llama-3.3-70b-versatile)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ER_MAP.envs.triage_env import TriageEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ER_MAP.server")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(None, description="Seed for reproducible scenarios")
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description='Reset options. Supports {"phase": 1|2|3, "difficulty": "easy|medium|hard"}.',
    )


class StepRequest(BaseModel):
    action: str = Field(
        ..., description="Doctor's JSON action string (e.g., '{\"tool\":\"read_soap\"}')"
    )


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]


class ResetResponse(BaseModel):
    observation: str
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# App + singleton env
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ER-MAP Triage Environment",
    version="1.0.0",
    description=(
        "OpenEnv-compatible FastAPI wrapper for the ER-MAP multi-agent "
        "medical triage environment. The Doctor RL agent communicates "
        "with this server via /reset and /step."
    ),
)

_ENV: Optional[TriageEnv] = None


def _get_env() -> TriageEnv:
    """Lazily construct a single shared TriageEnv per server process."""
    global _ENV
    if _ENV is None:
        _ENV = TriageEnv(
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
            nurse_api_key=os.environ.get("GROQ_NURSE_API_KEY", ""),
            patient_api_key=os.environ.get("GROQ_PATIENT_API_KEY", ""),
            model=os.environ.get("ERMAP_MODEL", "llama-3.3-70b-versatile"),
            render_mode=None,
        )
        logger.info("TriageEnv initialized.")
    return _ENV


def _sanitize(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip ground-truth disease from info before returning to the agent.
    The Doctor must NEVER see the true disease in observations or info,
    only the verifier output at terminal_discharge time.
    """
    safe = dict(info)
    safe.pop("ground_truth_disease", None)
    return safe


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "version": app.version,
        "env": "ER-MAP-Triage",
    }


@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest) -> ResetResponse:
    env = _get_env()
    try:
        obs, info = env.reset(seed=req.seed, options=req.options or {})
    except Exception as e:
        logger.exception("reset failed")
        raise HTTPException(status_code=500, detail=f"reset failed: {e}")
    return ResetResponse(observation=obs, info=_sanitize(info))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    env = _get_env()
    try:
        obs, reward, done, truncated, info = env.step(req.action)
    except Exception as e:
        logger.exception("step failed")
        raise HTTPException(status_code=500, detail=f"step failed: {e}")
    return StepResponse(
        observation=obs,
        reward=float(reward),
        done=bool(done),
        truncated=bool(truncated),
        info=_sanitize(info),
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    """
    Return the full internal env state. Includes ground truth, so this
    endpoint is for debugging / dashboards, NOT for the RL agent.
    """
    env = _get_env()
    return env.state()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return f"""
    <html>
      <head><title>ER-MAP Triage Env</title></head>
      <body style="font-family: ui-sans-serif, system-ui; max-width: 720px; margin: 2rem auto;">
        <h1>ER-MAP Triage Environment</h1>
        <p>OpenEnv-compatible FastAPI wrapper. Version <code>{app.version}</code>.</p>
        <h3>Endpoints</h3>
        <ul>
          <li><code>POST /reset</code> &mdash; start a new episode</li>
          <li><code>POST /step</code>  &mdash; submit a Doctor action (JSON string)</li>
          <li><code>GET  /state</code> &mdash; full env state (debug, exposes ground truth)</li>
          <li><code>GET  /health</code> &mdash; liveness check</li>
          <li><code>GET  /docs</code>  &mdash; interactive Swagger UI</li>
        </ul>
        <h3>Quick start</h3>
<pre>curl -X POST $URL/reset -H 'content-type: application/json' \\
     -d '{{"options": {{"phase": 1, "difficulty": "easy"}}}}'
curl -X POST $URL/step -H 'content-type: application/json' \\
     -d '{{"action": "{{\\"tool\\": \\"read_soap\\"}}"}}'</pre>
      </body>
    </html>
    """
