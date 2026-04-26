---
title: ER-MAP Triage Environment
emoji: 🏥
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Multi-agent ER triage env for LLM RL training
tags:
  - openenv
  - reinforcement-learning
  - multi-agent
  - medical
  - llm
  - grpo
  - curriculum-learning
---

# ER-MAP: Emergency Room Multi-Agent Protocol

> A multi-agent RL environment for training medical triage LLMs with curriculum learning, empathy-aware rewards, and consent-based treatment mechanics.

Built for the **Meta × PyTorch OpenEnv Hackathon (April 2026)**.

## What this Space hosts

This is the **OpenEnv-compatible FastAPI server** for the ER-MAP environment. The Doctor RL agent (trained externally with GRPO + Unsloth) talks to it via:

- `POST /reset` — start a new triage case (50-disease library, 3 difficulty phases)
- `POST /step` — submit a Doctor JSON action (`speak_to`, `order_lab`, `read_soap`, `update_soap`, `terminal_discharge`)
- `GET /state` — full internal env state (debug; exposes ground-truth disease)
- `GET /health` — liveness check
- `GET /docs` — interactive Swagger UI

## Quickstart

```python
import requests, json
URL = "https://YOUR-USERNAME-er-map-triage.hf.space"

obs = requests.post(f"{URL}/reset", json={"options": {"phase": 1, "difficulty": "easy"}}).json()
print(obs["observation"][:300])

step = requests.post(f"{URL}/step", json={"action": json.dumps({"tool": "read_soap"})}).json()
print(f"reward = {step['reward']:+.3f}")
```

## Architecture (TL;DR)

```
Doctor (RL policy, Qwen3/Llama-3.1-8B + LoRA via Unsloth)
   ⇅ JSON actions
TriageEnv (Gymnasium)
   ├─ Nurse LLM    (Llama-3.1-8B-instant via Groq)
   ├─ Patient LLM  (Llama-3.1-8B-instant via Groq, 4 personas × 5 traits)
   ├─ Empathy Judge (Llama-3.3-70B-versatile, intent grading)
   ├─ Medical Judge (Llama-3.3-70B-versatile, treatment grading)
   ├─ SOAP EMR (Subjective/Objective/Assessment/Plan)
   └─ Reward = process (60%) + terminal (40%), 11 components, capped per-episode
```

## Required Space Secrets

Set these under **Settings → Variables and secrets** (Secret type, NOT public):

| Name | Required | Used for |
|---|---|---|
| `GROQ_NURSE_API_KEY` | yes | Nurse actor inside the env |
| `GROQ_PATIENT_API_KEY` | yes | Patient actor inside the env |
| `GROQ_EMPATHY_JUDGE_API_KEY` | recommended | LLM judge for empathy intent |
| `GROQ_MEDICAL_JUDGE_API_KEY` | recommended | LLM judge for treatment correctness |
| `GROQ_API_KEY` | optional | Shared fallback if any role-specific key is missing |

All keys can come from a single Groq account (free tier works), but separate keys give you 5× the daily token quota.

## Links

- **GitHub repo:** https://github.com/VaibhavDeopa/Multi-Agents-for-Clinical-Decision-Making
- **Trained LoRA adapter:** https://huggingface.co/ankitraj86/ermap-doctor-lora *(updated after training run completes)*
- **Mini-blog & demo video:** see GitHub README

## License

MIT.
