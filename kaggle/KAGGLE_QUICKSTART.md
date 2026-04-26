# Kaggle Quickstart — ER-MAP GRPO Training (v3 stable)

The Kaggle notebook is in `kaggle/train_ermap_grpo_kaggle.ipynb`. This file
is the cheat sheet for running it end-to-end without the dependency hell
that bit us in earlier attempts.

## 0. Prerequisites (one-time)

1. **GitHub fork** of this repo. The notebook clones from a public fork at
   cell 6 — edit `GIT_URL`. Alternatively, upload the repo as a Kaggle
   Dataset named `ermap-source` (Add Data → Upload).
2. **Hugging Face write token** (`HF_TOKEN`) for pushing the trained
   adapter. Create at https://huggingface.co/settings/tokens (fine-grained,
   write access on a single model repo is enough).
3. **Five Groq keys** (one each for Nurse / Patient / Empathy Judge /
   Medical Judge / shared fallback). Free-tier accounts are fine; the
   per-account limits multiply across keys.

## 1. Create the Kaggle notebook

1. Sign in to https://www.kaggle.com/code → **New Notebook**.
2. Right sidebar:
   - Accelerator: **GPU T4 ×2** (or P100)
   - Internet: **On**
   - Persistence: Files only
3. **File → Upload Notebook** → choose `kaggle/train_ermap_grpo_kaggle.ipynb`
   from this repo.

## 2. Add Kaggle Secrets

Add-ons → Secrets → Add a new secret. Required labels (exactly):

| Label | Value |
|---|---|
| `GROQ_NURSE_API_KEY` | your nurse Groq key |
| `GROQ_PATIENT_API_KEY` | your patient Groq key |
| `GROQ_EMPATHY_JUDGE_API_KEY` | your empathy-judge Groq key |
| `GROQ_MEDICAL_JUDGE_API_KEY` | your medical-judge Groq key |
| `HF_TOKEN` | your HF write token |
| `WANDB_API_KEY` *(optional)* | your W&B key (skip — disabled by default) |

The notebook reads them via `kaggle_helpers.load_kaggle_secrets()` and
exports them as env vars.

## 3. Edit two placeholders in the notebook

- **Cell 6:** `GIT_URL = "https://github.com/VaibhavDeopa/Multi-Agents-for-Clinical-Decision-Making.git"` (or your own fork)
- **Cell 8:** `HF_PUSH_REPO = "ankitraj86/ermap-doctor-lora"` (your HF username)

If you uploaded the repo as a Kaggle Dataset instead, leave `GIT_URL` as the
placeholder — cell 6 will detect `/kaggle/input/ermap-source` and copy from
there.

## 4. Run order (the only sequence that works)

| Cell | What it does | Expected output |
|---|---|---|
| 2 | GPU + disk + python + internet sanity check | GPU listed, disk free > 8 GB |
| 3 | **REPAIR** — pin torch 2.10 cu128, reinstall bitsandbytes, upgrade unsloth | `REPAIR OK` (or `RESTART REQUIRED`) |
| **(restart)** | If cell 3 said RESTART REQUIRED → Run → Restart kernel | — |
| 5 | Post-restart import verify | All `OK`, GPUs listed |
| 6 | Clone / mount the repo | `OK. Repo at /kaggle/working/Multi-Agents-for-Clinical-Decision-Making` |
| 7 | Wire Kaggle Secrets → env vars | `OK — at least one Groq key is wired` |
| 8 | HF Hub config | `Starting fresh — no resume.` |
| 9 | Hyperparameters (P1=+1.2, P2=+1.1, P3=+1.0) | thresholds printed |
| 10 | **Pre-flight** — Groq routing + 4× PING | 4× `[PASS]`, then `OK` |
| 11 | Dry-run smoke test (no GPU) | `Dry-run OK` |
| 12 | Wire HF push hook | `Hub-push hook installed.` |
| 13 | **REAL TRAINING** (4–6 h) | per-group rolling stats, eventual `EARLY STOP` |
| 14 | Final push to HF | `Final checkpoints pushed: https://huggingface.co/...` |
| 15 | Per-phase plots | 5 PNGs displayed inline |
| 16 | Push plots to HF | `Plots pushed: ...` |
| 17 | Inference smoke-test (optional) | 3 sample Doctor actions printed |

## 5. Common failures & fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `numpy was upgraded mid-session` | numpy import poisoned by a previous cell | Restart kernel, re-run from cell 3 |
| `Pillow incompatible with torchvision` | Pillow ABI mismatch | Restart kernel, re-run from cell 3 |
| `PyTorch and torchvision compiled with different CUDA major` | torch upgraded to cu13 by a transient resolve | Re-run cell 3 (it pins cu128) and restart |
| `cannot import name 'create_gradient_checkpointing_buffer'` | unsloth ↔ unsloth_zoo version drift | Re-run cell 3 (upgrades both in lockstep) |
| `libnvJitLink.so.13 missing` | bitsandbytes built against different CUDA | Re-run cell 3 (force-reinstalls bitsandbytes after torch pin) |
| Disk usage > quota | Kaggle's 20 GB working partition fills up | First line of cell 3 cleans `/tmp` and pip cache |
| Pre-flight `[FAIL]` for a role | Groq key dead / quota exceeded | Generate a new key in console.groq.com → update Kaggle Secret → re-run cell 7+10 |
| `[FAIL]` says `routing=WRONG` | env var not set when `AgentRouter()` was constructed | Re-run cell 9 BEFORE cell 10 |
| Training freezes at episode 1 for >10 min | Doctor.generate hung; Unsloth import broke silently | Check cell 5 output for `unsloth` line; restart kernel and re-run cell 3 if missing |

## 6. What the trained model gives you

After cell 13 finishes (or hits the 12 h Kaggle session cap), you have:

- `OUTPUT_DIR/final_lora/` — LoRA adapter weights (~50 MB), pushed to
  `HF_PUSH_REPO`
- `OUTPUT_DIR/final_merged_fp16/` — full Llama-3.1-8B fp16 merge with the
  adapter applied (~16 GB), pushed to `HF_PUSH_REPO-merged`
- `OUTPUT_DIR/training_metrics.json` — per-episode rewards, outcomes,
  rolling stats — input for the per-phase plots
- `OUTPUT_DIR/plots/*.png` — 5 dashboards (one per phase + cross-phase
  overview + comparison bar)

Use the LoRA adapter for the demo (quick to load, runs on a 4050 6 GB at
~30 tok/s); use the merged fp16 if you need to host on a Vercel/HF Space
without `peft`.
