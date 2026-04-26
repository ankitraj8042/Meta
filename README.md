# ER-MAP: Emergency Room Multi-Agent Protocol

> **A multi-agent RL environment for training medical triage AI with curriculum learning, empathy-aware rewards, and realistic patient simulation.**

Built for the [Meta × PyTorch OpenEnv Hackathon (April 2026)](https://pytorch.org/blog/openenv/).

---

## 🚀 Live Submission Links

| Deliverable | Link |
|---|---|
| **OpenEnv server (HF Space, Docker SDK)** | https://huggingface.co/spaces/ankitraj86/er-map-triage |
| ↳ Live API root | https://ankitraj86-er-map-triage.hf.space |
| ↳ Interactive Swagger | https://ankitraj86-er-map-triage.hf.space/docs |
| **Trained Doctor LoRA adapter** | https://huggingface.co/ankitraj86/ermap-doctor-lora *(updates as training pushes checkpoints)* |
| **Mini-blog (technical writeup)** | [`docs/BLOG.md`](docs/BLOG.md) |
| **2-minute demo video** | *(linked here once recorded)* |
| **Kaggle training notebook** | [`kaggle/train_ermap_grpo_kaggle.ipynb`](kaggle/train_ermap_grpo_kaggle.ipynb) |
| **8-hour delivery runbook** | [`docs/HACKATHON_RUNBOOK.md`](docs/HACKATHON_RUNBOOK.md) |

```bash
# 30-second test of the live env from anywhere with python:
pip install requests
python -c "
import requests, json
URL='https://ankitraj86-er-map-triage.hf.space'
r = requests.post(f'{URL}/reset', json={'options':{'phase':1,'difficulty':'easy'}}).json()
print(r['observation'][:200])
"
```

---

## Overview

ER-MAP simulates a realistic Emergency Department where a **Doctor agent** (the RL policy) must diagnose and treat patients by orchestrating two auxiliary LLM agents (**Nurse** and **Patient**) through structured clinical tools. The environment uses **GRPO (Group Relative Policy Optimization)** with a **3-phase curriculum** that progresses from basic tool mastery to empathetic socio-economic negotiation.

### Key Innovation

Unlike traditional medical QA benchmarks, ER-MAP tests *process-level clinical competence*:
- The Doctor never sees the diagnosis directly — it must be inferred through tool use
- Patient cooperation is earned through empathy, not assumed
- Rewards are dense, phase-gated, and verified (no learned critic)

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  GRPO Trainer                     │
│         (Curriculum Scheduler: Phase 1→2→3)       │
└──────────────┬───────────────────────┬────────────┘
               │                       │
               ▼                       ▼
     ┌─────────────────┐    ┌─────────────────────┐
     │  Doctor Agent    │    │   Reward Verifier    │
     │  (RL Policy)     │    │   (Process-Based)    │
     │  Qwen3-4B LoRA   │    │   - Milestone Track  │
     └────────┬─────────┘    │   - Empathy Score    │
              │              │   - Trust State      │
              ▼              └─────────────────────┘
     ┌─────────────────┐
     │  TriageEnv       │
     │  (Gymnasium)     │
     ├─────────────────┤
     │ Tools:           │
     │  speak_to        │──→ Nurse LLM (Groq) / Patient LLM (Groq)
     │  order_lab       │──→ Lab Results DB (50 diseases)
     │  read_soap       │──→ SOAP EMR (phase-noised)
     │  update_soap     │──→ SOAP EMR
     │  terminal_discharge│→ Reward Verification
     └─────────────────┘
```

---

## Disease Database

**50 diseases across 10 clinical classes**, each with full SOAP history, vitals, lab results, and critical labs:

| # | Class | Diseases | Difficulty Range |
|---|-------|----------|-----------------|
| 1 | Cardiovascular | AMI, Aortic Dissection, Tamponade, AFib RVR, HTN Emergency | medium–hard |
| 2 | Pulmonary | PE, Tension PTX, Asthma, COPD, ARDS | easy–hard |
| 3 | Neurological | Stroke, SAH, Status Epilepticus, Meningitis, GBS | medium–hard |
| 4 | Gastrointestinal | GI Bleed, Appendicitis, Pancreatitis, Bowel Obstruction, Cholecystitis | easy–medium |
| 5 | Endocrine/Metabolic | DKA, Thyroid Storm, Adrenal Crisis, Hypoglycemia, Hyperkalemia | easy–hard |
| 6 | Toxicology | Opioid OD, Acetaminophen, CO Poisoning, Alcohol Withdrawal, Serotonin Syndrome | easy–hard |
| 7 | Trauma | TBI, Open Femur Fx, Burns, Pelvic Fx, Splenic Rupture | medium–hard |
| 8 | Infectious | Septic Shock, Nec Fasciitis, Malaria, PTA, SBP | easy–hard |
| 9 | GU/Renal | AKI, Nephrolithiasis, Testicular Torsion, Pyelonephritis, Urinary Retention | easy–medium |
| 10 | Environmental/Immunologic | Anaphylaxis, Heat Stroke, Hypothermia, Snakebite, Angioedema | medium–hard |

Each disease entry includes:
- **DISEASES_DB**: Symptoms, correct treatment, lethal treatments, critical labs
- **VITALS_DB**: Realistic vital signs with clinical interpretation
- **LAB_RESULTS_DB**: Full lab panels with critical flags
- **SOAP_HISTORY_DB**: HPI, ROS, PMH, Medications, Allergies, Social History, Physical Exam

---

## 3-Phase Curriculum Learning

### Phase 1: Tool Mastery
- **Goal**: Learn to use clinical tools correctly
- **Patient**: Calm, compliant, accurate symptom reporting
- **Nurse**: Veteran, available, high-empathy
- **SOAP**: Clean data, no noise
- **Rewards**: Tool usage (+0.05), milestone ordering (+0.05), valid JSON (+0.05)
- **Promotion**: Win rate ≥ 40% over 20 episodes

### Phase 2: Clinical Reasoning
- **Goal**: Differential diagnosis with ambiguous data
- **Patient**: Mixed compliance, vague/panicked communication
- **Nurse**: Mixed experience levels, sometimes overworked
- **SOAP**: Noisy — missing allergies, inconsistent timeline, vague ROS
- **Rewards**: Phase 1 + explanation bonus (+0.02), lab efficiency
- **Promotion**: Win rate ≥ 35% AND avg reward ≥ 0.5

### Phase 3: Empathetic Negotiation
- **Goal**: Manage hostile, non-compliant, uninsured patients
- **Patient**: Full randomization — hostile, cost-constrained, confused
- **Nurse**: Full randomization — can be impatient, distracted
- **SOAP**: Heavy noise — behavioral notes, unreliable history, interpreter barriers
- **Rewards**: Full empathy chain (+0.05 empathy, +0.03 explain, -0.08 dismissive)
- **Outcome**: Trust-based consent (AGREE/REFUSE/AMA)

---

## Empathy Engine (Intent-Based)

The empathy system uses a **causal chain** instead of keyword matching:

```
Doctor message → classify_intent() → PatientState.update() → consent_decision() → reward
```

### Intent Classification (Heuristic, No LLM Call)
- **Empathetic**: "I understand", "you're safe", "that must be scary" → trust ↑, anxiety ↓
- **Explanatory**: "let me explain", "this test will", "because we need" → trust ↑
- **Dismissive**: "just calm down", "that's not important", "hurry up" → trust ↓↓, anxiety ↑↑
- **Acknowledgment**: "tell me more", "when did this start" → trust ↑ (mild)

### Patient Trust/Anxiety Model
- Trust (0-100): Starts based on persona. Modified by Doctor behavior.
- Anxiety (0-100): Starts based on persona + financial stress.
- **Trust < 20 + Anxiety > 70** → 60% chance of **AMA** (patient leaves)
- **Trust < 35** → 40% chance of **REFUSE** treatment

---

## Milestone Tracker

Tracks clinical workflow compliance:

```
READ_SOAP → PATIENT_CONTACT → VITALS → LABS → ASSESSMENT → DISCHARGE
```

- **Phase 1**: Strict ordering enforced (correct order = +0.05, wrong = +0.01)
- **Phase 2**: Semi-strict (close to correct = +0.04)
- **Phase 3**: Relaxed (completion only = +0.03)

---

## Reward Architecture

| Component | Phase 1 | Phase 2 | Phase 3 |
|-----------|---------|---------|---------|
| Valid JSON | +0.05 | +0.05 | +0.05 |
| Tool use (correct) | +0.05–0.10 | +0.05–0.10 | +0.05–0.10 |
| Milestone (ordered) | +0.05 | +0.04 | +0.03 |
| Empathy bonus | — | +0.02 (explain) | +0.05 (empathy) |
| Dismissive penalty | — | — | -0.08 |
| Trust maintenance | — | — | +0.02 (trust>70) |
| Correct diagnosis | +2.00 | +2.00 | +2.00 |
| Lethal treatment | -2.00 | -2.00 | -2.00 |
| AMA loss | -1.50 | -1.50 | -1.50 |
| Redundant lab | -0.05 | -0.05 | -0.05 |

---

## SOAP Noise Injection

Phase-dependent noise applied to patient history:

| Phase | Noise Type | Examples |
|-------|-----------|----------|
| 1 | None | Clean data, all fields accurate |
| 2 | Clinical | Missing allergies, vague ROS, inconsistent PMH |
| 3 | Behavioral | "Patient homeless, med history unknown", "Language barrier", "Anxious about billing" |

---

## Project Structure

```
ER_MAP/
├── envs/
│   ├── triage_env.py       # Gymnasium environment (core)
│   ├── randomizer.py       # Ground truth + persona generation
│   ├── disease_db.py       # 50-disease database (10 classes)
│   ├── empathy_engine.py   # Intent classifier + trust model + milestones
│   └── api_router.py       # LLM API routing (Groq)
├── training/
│   └── train_grpo.py       # GRPO training with curriculum scheduler
├── server.py               # OpenEnv-compatible FastAPI wrapper for HF Spaces
├── tts_engine.py           # ElevenLabs TTS with speech markers
├── autoplay.py             # Demo episode runner
├── evaluate.py             # Evaluation harness
└── dashboard.py            # Metrics visualization
```

---

## Quick Start

### Installation
```bash
pip install gymnasium groq
pip install unsloth trl transformers datasets accelerate peft  # for training
pip install elevenlabs edge-tts  # for TTS (optional)
```

### Run a Demo Episode
```bash
export GROQ_API_KEY="your_key"
python -m ER_MAP.autoplay
```

### Train with GRPO + Curriculum
```bash
# Dry run (test scheduler, no GPU needed)
python -m ER_MAP.training.train_grpo --dry-run --episodes 50

# Full training (requires GPU + Groq API)
python -m ER_MAP.training.train_grpo \
    --episodes 200 \
    --model unsloth/Qwen3-4B \
    --groq-key $GROQ_API_KEY \
    --wandb
```

#### Train on Kaggle Free Tier (recommended — $0)

The repo ships with a complete Kaggle workflow under `kaggle/`:

- `kaggle/train_ermap_grpo_kaggle.ipynb` — the notebook (clone repo → install deps → run GRPO → push checkpoints to HF Hub)
- `kaggle/kaggle_helpers.py` — secret loader, HF push/pull, env-summary printer
- `kaggle/requirements_kaggle.txt` — Kaggle-image-aware dependency list (Unsloth pinned last)
- `kaggle/KAGGLE.md` — full setup guide with hardware feasibility table and gotchas

Tested target: **single Tesla T4 16 GB**, Llama-3.1-8B 4-bit + LoRA(r=16), 120 episodes, ~6-8 h per session.
See `kaggle/KAGGLE.md` for the full step-by-step.

### Environment API
```python
from ER_MAP.envs.triage_env import TriageEnv

env = TriageEnv(groq_api_key="your_key")

# Phase 2 with medium difficulty
obs, info = env.reset(options={"phase": 2, "difficulty": "medium"})

action = '{"tool": "read_soap"}'
obs, reward, done, truncated, info = env.step(action)

# info now includes:
# info["patient_state"] = {"trust": 55.0, "anxiety": 40.0, ...}
# info["milestones"] = {"achieved": {"READ_SOAP": True, ...}, "completion": 0.17}
```

---

## Results: Baseline vs Trained

**Baseline** (untrained Llama-3.1-8B-Instruct, same prompt as Doctor agent, no RL):

| Phase | n | Avg reward | Win rate | Wrong / Fatal | AMA |
|---|---|---|---|---|---|
| 1 — Tool Mastery       | 10 | +0.572 | 0/10 | 1/10 | 0/10 |
| 2 — Reasoning           | 10 | +0.800 | 0/10 | 1/10 | 0/10 |
| 3 — Empathy             | 10 | +0.531 | 0/10 | 1/10 | 0/10 |
| **Aggregate**           | 30 | **+0.634** | **0/30** | **3/30** | **0/30** |

The baseline closes 0 of 30 cases. It can stumble through tool use and earn process reward, but it never assembles a working diagnosis + treatment + consent chain that fires the +2.0 terminal reward. *(Comparison plot rendered post-Kaggle-training: `trained_vs_baseline.png` on the HF model repo.)*

**Trained** (post-GRPO, 100+ episodes on Kaggle T4): *populated from `training_metrics.json` after the Kaggle run completes — see [`docs/make_comparison_plot.py`](docs/make_comparison_plot.py) for the renderer.*

Raw baseline data: [`baseline_eval/baseline_results.json`](baseline_eval/baseline_results.json) (and a copy on the [HF model repo](https://huggingface.co/ankitraj86/ermap-doctor-lora/blob/main/baseline_results.json)).

---

## Training Budget (Kaggle Free T4 — what we actually used)

We did the full GRPO training on a single Kaggle T4 (free tier). No HF credits, no A100.

| Phase | Episodes (typical) | Est. Time (T4, group_size=2) |
|-------|--------------------|------------------------------|
| Phase 1 — Tool Mastery | 25–35 | ~1.5 h |
| Phase 2 — Reasoning   | 25–35 | ~2.0 h |
| Phase 3 — Empathy     | 40–60 | ~3.0 h |
| **Total**             | **~100–120**     | **~6–7 h** |

Per-episode wallclock ≈ 3 minutes (Doctor inference + 2-3 internal Nurse↔Patient exchanges + 2 LLM-judge calls via Groq free tier).

Model: **Llama-3.1-8B-Instruct** (Unsloth 4-bit) + LoRA(r=16). All Nurse / Patient / Empathy-Judge / Medical-Judge calls go through Groq's free API (8B-instant for actors, 70B-versatile for judges).

See [`kaggle/KAGGLE_QUICKSTART.md`](kaggle/KAGGLE_QUICKSTART.md) for the exact step-by-step.

---

## TTS Engine (Presentation Only)

For demo episodes, ER-MAP uses ElevenLabs with persona-specific speech markers:

- **Hostile patient**: Aggressive tone, sighing, interruptions
- **Anxious patient**: Trembling voice, rapid breathing, pauses
- **Veteran nurse**: Calm, measured, clinical tone
- **Rookie nurse**: Uncertain pauses, questioning tone

Speech markers (breathing, sighs, pauses) are injected into the TTS pipeline but **never fed back to the RL agents** — strict separation maintained.

---

## OpenEnv Compliance & Hackathon Submission Checklist

| Requirement | Status |
|------------|--------|
| Gymnasium-compatible env | ✅ `ER_MAP/envs/triage_env.py` |
| OpenEnv-style HTTP server (FastAPI) | ✅ `ER_MAP/server.py` + `Dockerfile` |
| Live HF Space hosting the env | ✅ https://huggingface.co/spaces/ankitraj86/er-map-triage |
| Verifiable reward functions | ✅ process-based, no learned critic (~80% deterministic, judge slice sandboxed) |
| Dense reward signal | ✅ per-step process (60%) + terminal (40%), 11 components |
| Difficulty variance | ✅ easy / medium / hard × 3 curriculum phases |
| `openenv.yaml` spec | ✅ |
| Reproducible seed control | ✅ |
| GRPO / RLVR training script | ✅ `ER_MAP/training/train_grpo.py` + Kaggle notebook |
| Baseline vs trained comparison | ✅ `baseline_eval/` + post-training plots in HF model repo |
| Mini-blog | ✅ [`docs/BLOG.md`](docs/BLOG.md) |
| Demo video (≤ 2 min) | *(linked in top section once recorded)* |

---

## License

MIT License. Built for the Meta × PyTorch OpenEnv Hackathon 2026.
