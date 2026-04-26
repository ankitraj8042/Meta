"""
kaggle/build_notebook.py
========================
Programmatically (re)builds `train_ermap_grpo_kaggle.ipynb` from scratch.

Why a builder script?
--------------------
The hand-edited notebook drifted into a fragile state across many sessions:
mixed early-stop / fixed-budget params, stale install snippets, dead pre-flight
checks, etc. This script is the single source of truth — run it once and the
notebook is regenerated as a clean, deterministic v3 layout.

Run:
    python kaggle/build_notebook.py

Output:
    kaggle/train_ermap_grpo_kaggle.ipynb     (overwritten)
    kaggle/KAGGLE_QUICKSTART.md              (overwritten)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split_keep_newlines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split_keep_newlines(text),
    }


def _split_keep_newlines(text: str) -> list[str]:
    """Notebook 'source' fields expect each line to terminate with '\n'
    except the last one. Splitting like this keeps `git diff` clean when
    the notebook is regenerated."""
    text = textwrap.dedent(text).lstrip("\n")
    if not text.endswith("\n"):
        text = text + "\n"
    lines = text.splitlines(keepends=True)
    if lines:
        # The last line should NOT have a trailing newline (Jupyter convention).
        if lines[-1].endswith("\n"):
            lines[-1] = lines[-1].rstrip("\n")
    return lines


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

CELL_01_TITLE = """\
# ER-MAP — Doctor Agent GRPO Training (Kaggle Free-Tier · v3 stable)

Trains the **Doctor LLM** (Llama-3.1-8B-Instruct, 4-bit + LoRA r=16) via GRPO
with a 3-phase curriculum on Kaggle's free GPU. Designed to survive Kaggle's
pre-baked image quirks (numpy / Pillow ABI mismatches, torch + torchvision
CUDA-major mismatches, transient `unsloth_zoo` upgrades).

## TL;DR — How to run this notebook

1. **Notebook settings (right sidebar):**
   - Accelerator: **GPU T4 ×2** (or P100)
   - Internet: **On**
   - Persistence: Files only
2. **Kaggle Secrets** (Add-ons → Secrets):
   - **Required:** `GROQ_NURSE_API_KEY`, `GROQ_PATIENT_API_KEY`,
     `GROQ_EMPATHY_JUDGE_API_KEY`, `GROQ_MEDICAL_JUDGE_API_KEY`, `HF_TOKEN`
   - **Optional:** `WANDB_API_KEY`
3. **Run cells 2 → 3 (sanity + REPAIR).** When cell 3 prints
   `RESTART REQUIRED`, click **Run → Restart kernel**, then resume from cell 5.
4. **Run cells 5 → 11 (verify + configure + dry-run + pre-flight).** Each cell
   should print an `OK` line before moving on.
5. **Run cell 13 (the long training cell, 4–6 hours).**
6. **Run cells 14 → 17 (final push + plots + inference smoke-test).**

## Curriculum + reward thresholds (this run)

Constant per-phase rolling-avg-reward bars; sustained for **3 consecutive
GRPO groups** triggers either a phase promotion or end-of-training.

| Phase | Reward target (sustained ×3 groups) | Action when met |
|---|---|---|
| 1 — Tool Mastery | `+1.2` | force-promote to Phase 2 |
| 2 — Clinical Reasoning | `+1.1` | force-promote to Phase 3 |
| 3 — Empathetic Negotiation | `+1.0` | END TRAINING |

Why these numbers? The un-trained 8B Doctor's baseline on the same env is
`P1=+0.76, P2=+0.59, P3=+0.39`. Targets of `+1.2 / +1.1 / +1.0` correspond
to roughly `1.6× / 1.9× / 2.6×` improvement over baseline — a meaningful
signal but reachable inside Kaggle's 12 h session limit.
"""

CELL_02_SANITY = """\
# === CELL 2 — Sanity check (GPU + disk + python + internet) ===
# Run this FIRST. If any check fails, fix it before running the REPAIR cell.

import os, shutil, subprocess, sys, socket

print("--- GPU ---")
try:
    print(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv"],
        timeout=10,
    ).decode())
except Exception as e:
    print(f"nvidia-smi failed: {e}")
    print("-> Set Accelerator to 'GPU T4 x2' in the right sidebar.")

print("--- Disk (/kaggle/working) ---")
total, used, free = shutil.disk_usage("/kaggle/working")
print(f"  total={total/1e9:5.1f} GB | used={used/1e9:5.1f} GB | free={free/1e9:5.1f} GB")
if free < 8 * 1e9:
    print("  WARNING: free disk < 8 GB — repair cell may fail. "
          "Consider 'Run > Restart and clear cell outputs' to reset /tmp.")

print("--- Python ---")
print(f"  python={sys.version.split()[0]} | exe={sys.executable}")

print("--- Internet (api.groq.com:443) ---")
try:
    socket.create_connection(("api.groq.com", 443), timeout=5).close()
    print("  reachable")
except Exception as e:
    print(f"  UNREACHABLE: {e}")
    print("  -> Settings (right sidebar) -> Internet -> ON")
"""

CELL_03_REPAIR = """\
# === CELL 3 — REPAIR CELL (idempotent full environment rebuild) ===
# Single source of truth for ER-MAP's GPU stack. Safe to re-run. After it
# finishes you'll see one of two final lines:
#
#   RESTART REQUIRED  -> Run -> Restart kernel, then resume from cell 5
#   REPAIR OK         -> proceed directly to cell 5
#
# Note: this cell only runs shell commands and one isolated subprocess.
# It deliberately does NOT `import torch / numpy / Pillow / unsloth` in the
# kernel, so re-running it after a botched install does not poison further
# attempts.

print("=" * 72); print("  CELL 3 — REPAIR"); print("=" * 72)

# 1. Clean caches (Kaggle's /kaggle/working is only 20 GB — installs
#    routinely fill it after a few re-runs).
print("[1/6] Cleaning pip + tmp + HF dataset caches...")
get_ipython().system('pip cache purge -q || true')
get_ipython().system('rm -rf /tmp/* /root/.cache/pip /root/.cache/huggingface/datasets 2>/dev/null || true')

# 2. Pin torch + torchvision to the cu128 wheel (matches Kaggle's CUDA 12.8
#    base image). DON'T let pip pull a generic CUDA-13 build — that breaks
#    bitsandbytes (libnvJitLink.so.13 missing) and torchvision (CUDA-major
#    mismatch RuntimeError at import time).
print("[2/6] Installing torch==2.10.0 + torchvision==0.25.0 (cu128)...")
get_ipython().system('pip install -q --no-cache-dir --force-reinstall '
                     'torch==2.10.0 torchvision==0.25.0 '
                     '--index-url https://download.pytorch.org/whl/cu128')

# 3. Reinstall bitsandbytes against the now-pinned torch.
print("[3/6] Reinstalling bitsandbytes...")
get_ipython().system('pip install -q --no-cache-dir --force-reinstall bitsandbytes')

# 4. Upgrade unsloth + unsloth_zoo + trl in lockstep. unsloth and
#    unsloth_zoo are released as a matched pair; if pip pulls a fresh
#    unsloth_zoo against an old unsloth you get
#       ImportError: cannot import name 'create_gradient_checkpointing_buffer'
print("[4/6] Upgrading unsloth + unsloth_zoo + trl...")
get_ipython().system('pip install -q --upgrade --no-cache-dir '
                     'unsloth unsloth_zoo "trl>=0.18.2"')

# 5. ER-MAP runtime deps that aren't pre-installed on Kaggle.
print("[5/6] Installing ER-MAP runtime deps...")
get_ipython().system('pip install -q --no-cache-dir '
                     '"groq>=0.18.0" "huggingface_hub>=0.25.0" '
                     '"gymnasium>=0.29.0" "openenv-core>=0.1.0"')

# 6. Verify in a SUBPROCESS (so the parent kernel never imports any of these
#    while pip is mid-flight, which is what causes the
#       'numpy was upgraded mid-session (loaded: X, installed: Y)' RuntimeError
#    we kept hitting before).
print("[6/6] Verifying via subprocess...")
import subprocess, sys, json

verify_script = r'''
import json, sys
out = {"ok": True, "details": {}, "errors": []}
try:
    import importlib.metadata as md
    for pkg in ("torch", "torchvision", "bitsandbytes", "unsloth", "unsloth_zoo",
                "trl", "transformers", "peft", "accelerate", "groq",
                "huggingface_hub", "gymnasium", "numpy", "Pillow"):
        try:
            out["details"][pkg + "_installed"] = md.version(pkg)
        except md.PackageNotFoundError:
            out["details"][pkg + "_installed"] = None

    import torch, torchvision, numpy as np, PIL, unsloth, unsloth_zoo, bitsandbytes, trl
    out["details"]["torch_loaded"]        = torch.__version__
    out["details"]["torch_cuda"]          = torch.version.cuda
    out["details"]["cuda_available"]      = bool(torch.cuda.is_available())
    out["details"]["gpu_count"]           = int(torch.cuda.device_count())
    out["details"]["torchvision_loaded"]  = torchvision.__version__
    out["details"]["numpy_loaded"]        = np.__version__
    out["details"]["pillow_loaded"]       = PIL.__version__
    out["details"]["unsloth_loaded"]      = unsloth.__version__
    out["details"]["unsloth_zoo_loaded"]  = unsloth_zoo.__version__
    out["details"]["bitsandbytes_loaded"] = bitsandbytes.__version__
    out["details"]["trl_loaded"]          = trl.__version__

    # Cross-check loaded-vs-installed for the C-extension libs that bit us
    # on every previous run.
    for pkg, loaded_key, installed_key in [
        ("numpy",  "numpy_loaded",  "numpy_installed"),
        ("Pillow", "pillow_loaded", "Pillow_installed"),
        ("torch",  "torch_loaded",  "torch_installed"),
    ]:
        loaded = out["details"].get(loaded_key)
        installed = out["details"].get(installed_key)
        if loaded and installed and loaded != installed:
            # Strip any local-version suffix (e.g. '+cu128') before compare.
            if loaded.split("+")[0] != installed.split("+")[0]:
                out["errors"].append(
                    f"{pkg} mismatch: loaded={loaded} installed={installed}"
                )
except Exception as e:
    out["ok"] = False
    out["errors"].append(f"{type(e).__name__}: {e}")
print(json.dumps(out, default=str))
'''.lstrip()

res = subprocess.run([sys.executable, "-c", verify_script],
                     capture_output=True, text=True, timeout=180)
print(res.stdout if res.stdout else "<no stdout>")
if res.stderr:
    print("---- subprocess stderr ----"); print(res.stderr)

# Parse the LAST line of stdout (others are prints from package init).
try:
    last = res.stdout.strip().splitlines()[-1]
    parsed = json.loads(last)
except Exception:
    parsed = {"ok": False, "errors": ["could not parse verification output"]}

ok = parsed.get("ok") and not parsed.get("errors")
d = parsed.get("details", {})

print("\n" + "=" * 72)
if ok:
    print("  REPAIR OK")
    print(f"    torch       : {d.get('torch_loaded')}  (CUDA {d.get('torch_cuda')})")
    print(f"    torchvision : {d.get('torchvision_loaded')}")
    print(f"    bitsandbytes: {d.get('bitsandbytes_loaded')}")
    print(f"    unsloth     : {d.get('unsloth_loaded')} | unsloth_zoo: {d.get('unsloth_zoo_loaded')}")
    print(f"    trl         : {d.get('trl_loaded')}")
    print(f"    numpy       : {d.get('numpy_loaded')} | Pillow: {d.get('pillow_loaded')}")
    print(f"    GPUs        : {d.get('gpu_count')}  (cuda_available={d.get('cuda_available')})")
    print()
    print("  -> If this kernel previously imported torch/numpy/Pillow/unsloth,")
    print("     RESTART NOW (Run -> Restart kernel) before continuing to cell 5.")
    print("     If this is a fresh kernel, you can proceed directly.")
else:
    print("  RESTART REQUIRED — issues detected:")
    for e in parsed.get("errors", []):
        print(f"    - {e}")
    print()
    print("  Action: Run -> Restart kernel, then re-run from cell 2.")
print("=" * 72)
"""

CELL_04_RESTART = """\
## ⚠ Restart kernel here if cell 3 said `RESTART REQUIRED`

Click **Run → Restart kernel** (or **Run → Restart & clear cell outputs**),
then resume from **cell 5**. Skipping the restart will produce ABI mismatch
errors at the first GPU op.

If cell 3 said `REPAIR OK` AND this is a fresh kernel that hasn't imported
torch/numpy/Pillow/unsloth yet, you can proceed to cell 5 directly.
"""

CELL_05_VERIFY = """\
# === CELL 5 — Post-restart verify (this kernel can import everything) ===
import importlib.metadata as md

print("--- Loaded versions in this kernel ---")
import torch, numpy, PIL, torchvision, unsloth, unsloth_zoo, bitsandbytes, trl, transformers, peft

versions = {
    "torch":          torch.__version__,
    "torchvision":    torchvision.__version__,
    "numpy":          numpy.__version__,
    "Pillow":         PIL.__version__,
    "unsloth":        unsloth.__version__,
    "unsloth_zoo":    unsloth_zoo.__version__,
    "bitsandbytes":   bitsandbytes.__version__,
    "trl":            trl.__version__,
    "transformers":   transformers.__version__,
    "peft":           peft.__version__,
}
all_ok = True
for k, v in versions.items():
    try:
        inst = md.version(k)
    except md.PackageNotFoundError:
        inst = "(not installed)"
    # Tolerate local version suffixes like '+cu128'
    flag = "OK" if inst.split("+")[0] == v.split("+")[0] else f"MISMATCH (installed={inst})"
    if "MISMATCH" in flag:
        all_ok = False
    print(f"  {k:14s}: loaded={v:20s} [{flag}]")

print()
print(f"  CUDA available : {torch.cuda.is_available()}")
print(f"  GPU count      : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}          : {p.name} ({p.total_memory/1e9:.1f} GB)")

print()
print("OK" if all_ok else "NOT OK — re-run cell 3 and restart kernel.")
"""

CELL_06_REPO = """\
# === CELL 6 — Mount the ER-MAP repo into /kaggle/working ===
import os, subprocess, sys

# OPTION A: clone a public GitHub fork (preferred). Edit GIT_URL.
GIT_URL    = "https://github.com/ankitraj8042/Meta.git"
BRANCH     = "main"
REPO_ROOT  = "/kaggle/working/Meta"

# OPTION B: Kaggle Dataset upload — set this if you uploaded the repo
# as a Kaggle Dataset named "ermap-source" (Add Data -> Upload).
DATASET_DIR = "/kaggle/input/ermap-source"

if not os.path.isdir(f"{REPO_ROOT}/ER_MAP"):
    if "<your-fork>" not in GIT_URL:
        print(f"Cloning {GIT_URL}@{BRANCH} -> {REPO_ROOT}...")
        out = subprocess.run(
            ["git", "clone", "--depth", "1", "-b", BRANCH, GIT_URL, REPO_ROOT],
            capture_output=True, text=True,
        )
        print(out.stdout); print(out.stderr)
    elif os.path.isdir(DATASET_DIR):
        print(f"Copying {DATASET_DIR} -> {REPO_ROOT}...")
        import shutil
        shutil.copytree(DATASET_DIR, REPO_ROOT, dirs_exist_ok=True)

assert os.path.isdir(f"{REPO_ROOT}/ER_MAP"), (
    "Repo not found.\\n"
    " - Edit GIT_URL above to your GitHub fork, OR\\n"
    " - Upload the repo as a Kaggle Dataset named 'ermap-source' (Add Data -> Upload)."
)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, f"{REPO_ROOT}/kaggle")
print(f"OK. Repo at {REPO_ROOT}")
"""

CELL_07_SECRETS = """\
# === CELL 7 — Wire Kaggle Secrets into env vars ===
import os
from kaggle_helpers import load_kaggle_secrets, kaggle_env_summary

load_kaggle_secrets()
kaggle_env_summary()

# Hard fail if no Groq key — training would silently use mock LLMs.
assert any(os.environ.get(k) for k in (
    "GROQ_NURSE_API_KEY", "GROQ_PATIENT_API_KEY",
    "GROQ_EMPATHY_JUDGE_API_KEY", "GROQ_MEDICAL_JUDGE_API_KEY",
    "GROQ_API_KEY",
)), ("No Groq key found in Kaggle Secrets. "
     "Add at least GROQ_NURSE_API_KEY in Add-ons -> Secrets.")
print("OK — at least one Groq key is wired.")
"""

CELL_08_HF = """\
# === CELL 8 — Hugging Face Hub config (for checkpoint backup) ===
import os
from kaggle_helpers import push_checkpoint_to_hub, download_checkpoint_from_hub

# EDIT the line below to your HF model id (e.g. "udayd/ermap-doctor-lora").
HF_PUSH_REPO   = "ankitraj86/ermap-doctor-lora"
# To resume from a previous run, paste the same repo id here. Empty = fresh.
HF_RESUME_REPO = ""

RESUME_DIR = "/kaggle/working/checkpoints/resume"
if HF_RESUME_REPO:
    download_checkpoint_from_hub(HF_RESUME_REPO, RESUME_DIR)
    contents = os.listdir(RESUME_DIR) if os.path.isdir(RESUME_DIR) else []
    print(f"Resume dir: {contents or '(empty)'}")
else:
    print("Starting fresh — no resume.")

if "<your-username>" in HF_PUSH_REPO:
    print("\\nWARNING: HF_PUSH_REPO still has <your-username> placeholder.")
    print("         Checkpoints will NOT be pushed to HF Hub.")
    print("         Edit the cell above and re-run before training if you want backups.")
"""

CELL_09_HYPERPARAMS = """\
# === CELL 9 — GRPO hyperparameters ===
import os

# --- CUDA memory: expandable_segments avoids fragmentation-OOM on T4 -------
# Without this, PyTorch's CUDA caching allocator can fail to find a contiguous
# block for a new allocation even when enough total free memory exists.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_NAME       = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
GROUP_SIZE       = 2
LEARNING_RATE    = 5e-6
KL_BETA          = 0.04
OUTPUT_DIR       = "/kaggle/working/er_map_grpo_checkpoints"
PUSH_EVERY_EPS   = 20
USE_WANDB        = False  # WANDB conflicts with protobuf 7 on Kaggle base image
NUM_EPISODES     = 200    # hard cap; early-stop usually finishes first

# --- Per-phase reward thresholds (constant for this run) -------------------
# After every GRPO update we look at the last CONVERGENCE_WINDOW groups; if
# ALL of them belong to the same current phase AND each has
# rolling_avg_reward >= PHASE_REWARD_TARGETS[current_phase] AND
# rolling_win_rate >= PHASE_MIN_WIN_RATE, we either:
#   - force-promote to the next phase (Phase 1 / Phase 2), OR
#   - terminate training (Phase 3).
EARLY_STOP_ENABLED   = True
PHASE_REWARD_TARGETS = {1: 1.2, 2: 1.1, 3: 1.0}
PHASE_MIN_WIN_RATE   = 0.20
CONVERGENCE_WINDOW   = 3

# --- Per-episode budget controls (read by triage_env) ----------------------
os.environ["ERMAP_MAX_EPISODE_STEPS"]      = "20"
os.environ["ERMAP_MAX_INTERNAL_EXCHANGES"] = "5"

# --- Groq traffic-shaping (8B for actors, 70B for judges) ------------------
# High-volume conversational roles (Nurse + Patient) on the 8B-instant pool
# (500K TPD, 14,400 RPD). Empathy judge fires on EVERY speak_to call
# (~5-10x/episode) so it uses 8B-instant (500K TPD) to avoid rate limits.
# Medical judge fires once per episode at terminal_discharge — stays on 70B
# for treatment-correctness grading quality.
os.environ["ERMAP_NURSE_MODEL"]            = "llama-3.1-8b-instant"
os.environ["ERMAP_PATIENT_MODEL"]          = "llama-3.1-8b-instant"
os.environ["ERMAP_EMPATHY_JUDGE_MODEL"]    = "llama-3.1-8b-instant"
os.environ["ERMAP_MEDICAL_JUDGE_MODEL"]    = "llama-3.3-70b-versatile"

print("Hyperparameters set:")
print(f"  NUM_EPISODES         = {NUM_EPISODES}")
print(f"  GROUP_SIZE           = {GROUP_SIZE}")
print(f"  PHASE_REWARD_TARGETS = {PHASE_REWARD_TARGETS}")
print(f"  PHASE_MIN_WIN_RATE   = {PHASE_MIN_WIN_RATE}")
print(f"  CONVERGENCE_WINDOW   = {CONVERGENCE_WINDOW}")
print(f"  Nurse / Patient      = llama-3.1-8b-instant (actors, high-volume)")
print(f"  Empathy judge        = llama-3.1-8b-instant (high-volume, 500K TPD)")
print(f"  Medical judge        = llama-3.3-70b-versatile (once/episode, quality)")
print(f"  PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True (OOM guard)")
"""

CELL_10_PREFLIGHT = """\
# === CELL 10 — Pre-flight: Groq routing + key liveness ===
# Verifies that:
#  - each role is routed to the model you set in cell 9, and
#  - each role's Groq key actually answers a 1-token "PING" prompt.

import os
from ER_MAP.envs.api_router import AgentRouter

router = AgentRouter()
expected = {
    "nurse":         "llama-3.1-8b-instant",
    "patient":       "llama-3.1-8b-instant",
    "empathy_judge": "llama-3.1-8b-instant",
    "medical_judge": "llama-3.3-70b-versatile",
}

print("=" * 60); print("  PRE-FLIGHT — Groq routing + smoke test"); print("=" * 60)
all_pass = True
for role, exp in expected.items():
    actual = router._models.get(role, "?")
    routing_ok = (actual == exp)
    client = router._clients.get(role)

    if client is None:
        print(f"  [SKIP] {role:14s} -> no Groq client (key missing)")
        all_pass = False
        continue

    try:
        resp = client.chat.completions.create(
            model=exp,
            messages=[{"role": "user", "content": "Reply with exactly: PING"}],
            max_tokens=4, temperature=0,
        )
        api_ok = "PING" in (resp.choices[0].message.content or "").upper()
        err = ""
    except Exception as e:
        api_ok = False
        err = f" ({type(e).__name__}: {str(e)[:80]})"

    flag = "PASS" if (routing_ok and api_ok) else "FAIL"
    if flag == "FAIL":
        all_pass = False
    print(f"  [{flag}] {role:14s} -> {actual:30s} "
          f"routing={'ok' if routing_ok else 'WRONG'}, "
          f"api={'ok' if api_ok else 'fail'}{err}")

print("=" * 60)
print("OK" if all_pass else "NOT OK — fix routing/keys before training.")
print("=" * 60)
assert all_pass, "Pre-flight failed; do not proceed to training."
"""

CELL_11_DRYRUN = """\
# === CELL 11 — Dry-run smoke test (no GPU, no model load) ===
# Verifies the curriculum scheduler + reward verifier + per-phase early-stop
# wiring before we burn GPU minutes on the real run.

from ER_MAP.training.train_grpo import train

_ = train(
    num_episodes=8,
    group_size=2,
    model_name=MODEL_NAME,
    learning_rate=LEARNING_RATE,
    kl_beta=KL_BETA,
    output_dir="/kaggle/working/_dryrun",
    dry_run=True,
    phase_reward_targets=PHASE_REWARD_TARGETS,
    phase_min_win_rate=PHASE_MIN_WIN_RATE,
    convergence_window=CONVERGENCE_WINDOW,
    early_stop=EARLY_STOP_ENABLED,
)
print("\\nDry-run OK — scheduler + verifier + per-phase early-stop wiring is healthy.")
"""

CELL_12_HOOK = """\
# === CELL 12 — Wire periodic HF Hub push into training ===
# We monkey-patch save_lora_adapters so every checkpoint dump also pushes
# the LoRA adapter to HF Hub. Failures are non-fatal — training keeps
# running even if a push fails (e.g. transient HF 502).

from ER_MAP.training import train_grpo as _tg
_original_save = _tg.save_lora_adapters

def save_lora_adapters_with_push(model, tokenizer, output_dir):
    _original_save(model, tokenizer, output_dir)
    if HF_PUSH_REPO and "<your-username>" not in HF_PUSH_REPO:
        try:
            push_checkpoint_to_hub(
                output_dir, HF_PUSH_REPO,
                commit_message=f"checkpoint @ {os.path.basename(output_dir)}",
            )
        except Exception as e:
            print(f"  [hub-push] non-fatal failure: {e}")

_tg.save_lora_adapters = save_lora_adapters_with_push
print("Hub-push hook installed.")
"""

CELL_13_TRAIN_MD = """\
## 13 · Run real training (the 4–6 hour cell)

**Estimated wall-clock on Kaggle T4 ×2:**

- ~3–5 min per episode (6–14 env steps × Doctor.generate + 4–8 × Groq calls)
- ~1–2 min amortized per GRPO update (G=2 trajectories × response-token log-probs)
- **Per-group ≈ 8–12 min** (2 episodes + 1 update)

| Phase | Typical episodes to reach target | Wall-clock |
|---|---|---|
| 1 (target `+1.2` × 3) | 12 – 24 episodes (6 – 12 groups) | ~1.0 – 2.0 h |
| 2 (target `+1.1` × 3) | 16 – 32 episodes (8 – 16 groups) | ~1.5 – 2.5 h |
| 3 (target `+1.0` × 3) | 20 – 50 episodes (10 – 25 groups) | ~2.0 – 4.0 h |
| **Total** | 50 – 100 episodes | **~4.5 – 8.5 h** |

If `NUM_EPISODES=200` is exhausted before Phase 3 converges, training
stops at the cap and the latest LoRA checkpoint is on HF Hub already
(we push every 20 episodes), so resume in a fresh session via
`HF_RESUME_REPO` in cell 8.
"""

CELL_13_TRAIN = """\
# === CELL 13 — REAL TRAINING (4-6 h cell) ===
metrics = train(
    num_episodes=NUM_EPISODES,
    group_size=GROUP_SIZE,
    model_name=MODEL_NAME,
    groq_api_key=os.environ.get("GROQ_NURSE_API_KEY", "")
                  or os.environ.get("GROQ_API_KEY", ""),
    learning_rate=LEARNING_RATE,
    kl_beta=KL_BETA,
    use_wandb=USE_WANDB,
    output_dir=OUTPUT_DIR,
    dry_run=False,
    phase_reward_targets=PHASE_REWARD_TARGETS,
    phase_min_win_rate=PHASE_MIN_WIN_RATE,
    convergence_window=CONVERGENCE_WINDOW,
    early_stop=EARLY_STOP_ENABLED,
)
print(f"\\nTraining returned {len(metrics)} metric records.")
"""

CELL_14_FINAL_PUSH = """\
# === CELL 14 — Final push: adapters + merged fp16 ===
FINAL_LORA_DIR   = f"{OUTPUT_DIR}/final_lora"
FINAL_MERGED_DIR = f"{OUTPUT_DIR}/final_merged_fp16"

if HF_PUSH_REPO and "<your-username>" not in HF_PUSH_REPO:
    push_checkpoint_to_hub(FINAL_LORA_DIR, HF_PUSH_REPO,
                           commit_message="final LoRA adapter")
    if os.path.isdir(FINAL_MERGED_DIR):
        push_checkpoint_to_hub(FINAL_MERGED_DIR, f"{HF_PUSH_REPO}-merged",
                               commit_message="final merged fp16")
    print(f"Final checkpoints pushed: https://huggingface.co/{HF_PUSH_REPO}")
else:
    print("HF_PUSH_REPO not configured — skipping final push.")
"""

CELL_15_PLOTS_MD = """\
## 15 · Per-phase training graphs (one dashboard per curriculum phase)

We render a 6-panel dashboard for **every phase that contains episodes**,
plus a cross-phase overview and a phase-comparison bar chart. All PNGs are
written to `er_map_grpo_checkpoints/plots/` and uploaded to HF Hub in the
next cell so they survive Kaggle session expiry.

Each per-phase dashboard contains:

1. **Reward growth** — raw scatter + rolling mean (w=10) + verified rolling mean
2. **Rolling win rate** — w=20 win-rate evolution within the phase
3. **Outcome distribution over time** — stacked bars (WIN/PARTIAL/INCORRECT/AMA_LOSS/FATAL_LOSS)
4. **Reward components** — mean of each component (process / treatment / empathy / labs / etc.)
5. **GRPO update stats** — loss + KL divergence per group update
6. **Episode length distribution** — histogram of step counts
"""

CELL_15_PLOTS = """\
# === CELL 15 — Per-phase training dashboards ===
from ER_MAP.plotting import plot_per_phase_dashboards
from IPython.display import Image, display, Markdown

PLOTS_DIR = f"{OUTPUT_DIR}/plots"
written = plot_per_phase_dashboards(
    metrics_path=f"{OUTPUT_DIR}/training_metrics.json",
    output_dir=PLOTS_DIR,
)

print(f"Saved {len(written)} chart(s) to {PLOTS_DIR}:")
for name, path in written.items():
    size_kb = os.path.getsize(path) / 1024
    print(f"  {name:<28s} -> {path}  ({size_kb:.0f} KB)")

# Display each chart inline so the operator sees them without leaving Kaggle.
ordered = (sorted(k for k in written if k.startswith("phase"))
           + ["all_phases_overview", "all_phases_comparison"])
for key in ordered:
    if key not in written:
        continue
    display(Markdown(f"### {key.replace('_', ' ').title()}"))
    display(Image(filename=written[key]))
"""

CELL_16_PUSH_PLOTS = """\
# === CELL 16 — Push plots to HF Hub ===
if HF_PUSH_REPO and "<your-username>" not in HF_PUSH_REPO:
    push_checkpoint_to_hub(PLOTS_DIR, HF_PUSH_REPO,
                           commit_message="per-phase training plots")
    print(f"Plots pushed: https://huggingface.co/{HF_PUSH_REPO}/tree/main")
else:
    print("HF_PUSH_REPO not configured — plots stay only in /kaggle/working/.")
"""

CELL_17_INFER_MD = """\
## 17 · (Optional) Inference smoke-test on the trained model

Catches the classic 'merge path looked OK but the saved model emits garbage'
failure mode before the demo.
"""

CELL_17_INFER = """\
# === CELL 17 — Inference smoke-test on the trained model ===
from ER_MAP.training.train_grpo import generate_doctor_action, load_model_and_tokenizer
from peft import PeftModel

base_model, tok = load_model_and_tokenizer(model_name=MODEL_NAME)
trained = PeftModel.from_pretrained(base_model, FINAL_LORA_DIR)

test_obs = (
    '{"event":"episode_start","nurse_experience":"veteran",'
    '"message":"Patient with chest pain, HR 120, BP 90/60, vague history.",'
    '"soap_summary":{}}'
)
for i in range(3):
    print(f"\\n--- Sample {i+1} ---")
    print(generate_doctor_action(trained, tok, test_obs, max_new_tokens=160))
"""

CELL_18_COMPARE_MD = """\
## 18 · Trained-vs-Baseline comparison plot

Pulls the local `baseline_results.json` (uploaded to the HF model repo from
the dev box where matplotlib actually works) and the in-Kaggle
`training_metrics.json`, then renders the 2x2 comparison panel that goes
into the README and the blog post.
"""

CELL_18_COMPARE = """\
# === CELL 18 — Trained-vs-Baseline comparison plot ===
import os, json, urllib.request
from pathlib import Path

REPO_ROOT_LOCAL = REPO_ROOT  # set in cell 6

def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as e:
        print(f"  download {url} FAILED: {type(e).__name__}: {str(e)[:120]}")
        return False

# 1. Pull both baseline_results.json AND make_comparison_plot.py from the
#    HF model repo (uploaded from the dev box where matplotlib actually
#    works). The cloned GitHub repo doesn't have these.
plot_helper = Path(REPO_ROOT_LOCAL) / "docs" / "make_comparison_plot.py"
baseline_target = Path(REPO_ROOT_LOCAL) / "baseline_eval" / "baseline_results.json"

base = f"https://huggingface.co/{HF_PUSH_REPO}/resolve/main"
ok_helper   = _download(f"{base}/make_comparison_plot.py", plot_helper)
ok_baseline = _download(f"{base}/baseline_results.json",   baseline_target)

if ok_baseline:
    n = len(json.loads(baseline_target.read_text()))
    print(f"  -> baseline: {n} records")
if ok_helper:
    print(f"  -> plot helper: {plot_helper}")

if not ok_helper:
    print("Plot helper not found on HF Hub. Skipping comparison plot.")
else:
    # 2. Run the comparison plot helper
    metrics_path = f"{OUTPUT_DIR}/training_metrics.json"
    out_png  = f"{OUTPUT_DIR}/trained_vs_baseline.png"
    out_json = f"{OUTPUT_DIR}/trained_vs_baseline_summary.json"

    !python {plot_helper} \\
        --baseline {baseline_target} \\
        --metrics  {metrics_path} \\
        --out-png  {out_png} \\
        --out-json {out_json} \\
        --window   10

    # 3. Push the comparison plot to the HF model repo
    if HF_PUSH_REPO and "<your-username>" not in HF_PUSH_REPO:
        from huggingface_hub import upload_file
        for src in [out_png, out_json]:
            if os.path.isfile(src):
                try:
                    upload_file(
                        path_or_fileobj=src,
                        path_in_repo=os.path.basename(src),
                        repo_id=HF_PUSH_REPO,
                        repo_type="model",
                        commit_message=f"add {os.path.basename(src)} (trained vs baseline)",
                    )
                    print(f"Pushed {os.path.basename(src)} -> {HF_PUSH_REPO}")
                except Exception as e:
                    print(f"FAIL push {src}: {type(e).__name__}: {str(e)[:120]}")
"""


# ---------------------------------------------------------------------------
# Quickstart markdown (sibling file)
# ---------------------------------------------------------------------------

QUICKSTART_MD = """\
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

- **Cell 6:** `GIT_URL = "https://github.com/ankitraj8042/Meta.git"` (or your own fork)
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
| 6 | Clone / mount the repo | `OK. Repo at /kaggle/working/Meta` |
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
"""


# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------

def build_notebook() -> dict:
    cells = [
        md_cell(CELL_01_TITLE),                     # 0
        code_cell(CELL_02_SANITY),                  # 1
        code_cell(CELL_03_REPAIR),                  # 2
        md_cell(CELL_04_RESTART),                   # 3
        code_cell(CELL_05_VERIFY),                  # 4
        code_cell(CELL_06_REPO),                    # 5
        code_cell(CELL_07_SECRETS),                 # 6
        code_cell(CELL_08_HF),                      # 7
        code_cell(CELL_09_HYPERPARAMS),             # 8
        code_cell(CELL_10_PREFLIGHT),               # 9
        code_cell(CELL_11_DRYRUN),                  # 10
        code_cell(CELL_12_HOOK),                    # 11
        md_cell(CELL_13_TRAIN_MD),                  # 12
        code_cell(CELL_13_TRAIN),                   # 13
        code_cell(CELL_14_FINAL_PUSH),              # 14
        md_cell(CELL_15_PLOTS_MD),                  # 15
        code_cell(CELL_15_PLOTS),                   # 16
        code_cell(CELL_16_PUSH_PLOTS),              # 17
        md_cell(CELL_17_INFER_MD),                  # 18
        code_cell(CELL_17_INFER),                   # 19
        md_cell(CELL_18_COMPARE_MD),                # 20
        code_cell(CELL_18_COMPARE),                 # 21
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    here = Path(__file__).parent
    nb_path = here / "train_ermap_grpo_kaggle.ipynb"
    qs_path = here / "KAGGLE_QUICKSTART.md"

    nb = build_notebook()
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    qs_path.write_text(QUICKSTART_MD, encoding="utf-8")

    n_md = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"Wrote {nb_path}  ({len(nb['cells'])} cells: {n_md} md / {n_code} code)")
    print(f"Wrote {qs_path}  ({len(QUICKSTART_MD.splitlines())} lines)")


if __name__ == "__main__":
    main()
