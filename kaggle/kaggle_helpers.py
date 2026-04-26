"""
kaggle/kaggle_helpers.py
========================
Helpers that adapt ER-MAP training for Kaggle's free-tier session model.

Why this file exists
--------------------
Kaggle free-tier GPU sessions are capped at **12 hours per session** and
**30 GPU-hours per week**. The persistent storage (`/kaggle/working/`)
survives only as long as the notebook session — once the session
expires, anything that wasn't pushed somewhere external is lost.

This module provides three glue functions:

* ``load_kaggle_secrets()``        – reads Groq/HF/W&B keys out of
                                     Kaggle's `UserSecretsClient`
                                     and exports them as env vars so
                                     the rest of ER-MAP works unchanged.
* ``push_checkpoint_to_hub()``     – pushes the LoRA adapter directory
                                     produced by ``train_grpo.py`` to a
                                     private Hugging Face repo.
* ``download_checkpoint_from_hub()`` – pulls a previously-pushed
                                       checkpoint back into
                                       ``/kaggle/working/`` so a new
                                       session can resume training
                                       instead of starting from scratch.

All three are best-effort: missing secrets / network failures degrade
gracefully without crashing the notebook.
"""
from __future__ import annotations

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger("ER_MAP.kaggle")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Kaggle Secrets -> os.environ
# ---------------------------------------------------------------------------

# Map: Kaggle Secret Label  ->  env var name expected by ER-MAP code.
# These names match what dashboard.py and train_grpo.py already read.
_SECRET_TO_ENV = {
    # Groq — one secret per logical role (per-key rate-limit budget).
    "GROQ_DOCTOR_API_KEY":          "GROQ_DOCTOR_API_KEY",
    "GROQ_NURSE_API_KEY":           "GROQ_NURSE_API_KEY",
    "GROQ_PATIENT_API_KEY":         "GROQ_PATIENT_API_KEY",
    "GROQ_EMPATHY_JUDGE_API_KEY":   "GROQ_EMPATHY_JUDGE_API_KEY",
    "GROQ_MEDICAL_JUDGE_API_KEY":   "GROQ_MEDICAL_JUDGE_API_KEY",
    # Generic fallback — train_grpo.py reads this when no role-specific
    # key is set.
    "GROQ_API_KEY":                 "GROQ_API_KEY",
    # Hugging Face Hub — needed for push/pull of LoRA adapters.
    "HF_TOKEN":                     "HF_TOKEN",
    "HUGGINGFACE_TOKEN":            "HF_TOKEN",
    # Weights & Biases (optional logging).
    "WANDB_API_KEY":                "WANDB_API_KEY",
}


def load_kaggle_secrets(verbose: bool = True) -> dict:
    """
    Read every Kaggle Secret listed in ``_SECRET_TO_ENV`` and copy it
    into the corresponding environment variable so ER-MAP's existing
    ``os.environ.get(...)`` calls just work.

    Returns a dict of ``{env_var_name: True/False}`` indicating which
    secrets were found.
    """
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except ImportError:
        if verbose:
            print(
                "[kaggle_helpers] kaggle_secrets not importable — assuming "
                "non-Kaggle env. Skipping secret load."
            )
        return {}

    client = UserSecretsClient()
    loaded: dict = {}
    for secret_label, env_name in _SECRET_TO_ENV.items():
        if loaded.get(env_name):
            continue  # already populated by an alias
        try:
            value = client.get_secret(secret_label)
        except Exception:
            value = None
        if value:
            os.environ[env_name] = value
            loaded[env_name] = True
        else:
            loaded.setdefault(env_name, False)

    if verbose:
        alive = [k for k, v in loaded.items() if v]
        missing = [k for k, v in loaded.items() if not v]
        print(f"[kaggle_helpers] Loaded {len(alive)} secret(s): {alive}")
        if missing:
            print(f"[kaggle_helpers] Missing (optional): {missing}")
    return loaded


# ---------------------------------------------------------------------------
# HF Hub push / pull (so Kaggle session expiry doesn't nuke checkpoints)
# ---------------------------------------------------------------------------

def push_checkpoint_to_hub(
    local_dir: str,
    repo_id: str,
    *,
    commit_message: str = "ER-MAP checkpoint",
    private: bool = True,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Push a LoRA adapter folder (created by ``train_grpo.save_lora_adapters``)
    to a Hugging Face Hub repo. Creates the repo if it doesn't exist.

    Returns ``True`` on success, ``False`` on any failure (callers can
    keep training even if the upload fails).
    """
    src = Path(local_dir)
    if not src.exists() or not src.is_dir():
        logger.warning(f"push_checkpoint_to_hub: directory not found: {src}")
        return False

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.warning("push_checkpoint_to_hub: no HF_TOKEN available; skipping push.")
        return False

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.warning("huggingface_hub not installed; skipping push.")
        return False

    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            commit_message=commit_message,
            ignore_patterns=["*.tmp", "*.lock", "__pycache__"],
        )
        logger.info(f"Pushed {src} -> https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        logger.error(f"push_checkpoint_to_hub failed: {e}")
        return False


def push_file_to_hub(
    local_path: str,
    repo_id: str,
    path_in_repo: str,
    *,
    commit_message: str = "upload metrics / artifact",
    private: bool = True,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Upload a single file (e.g. ``training_metrics.json``) to a Hub repo
    at ``path_in_repo`` so the browser can show rolling reward without SSH.
    """
    p = Path(local_path)
    if not p.is_file():
        return False
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return False
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        return False
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )
        logger.info(f"Pushed {p.name} -> https://huggingface.co/{repo_id}/blob/main/{path_in_repo}")
        return True
    except Exception as e:
        logger.error(f"push_file_to_hub failed: {e}")
        return False


def download_checkpoint_from_hub(
    repo_id: str,
    local_dir: str,
    *,
    hf_token: Optional[str] = None,
) -> bool:
    """
    Snapshot-download a previously pushed adapter back into ``local_dir``
    so a fresh Kaggle session can resume training. Returns True on
    success.
    """
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed; cannot resume.")
        return False
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
        )
        logger.info(f"Downloaded {repo_id} -> {path}")
        return True
    except Exception as e:
        logger.warning(f"download_checkpoint_from_hub failed (this is fine on first run): {e}")
        return False


# ---------------------------------------------------------------------------
# Periodic-push hook
# ---------------------------------------------------------------------------

def make_hub_pusher(
    repo_id: str,
    push_every_episodes: int = 20,
    *,
    hf_token: Optional[str] = None,
):
    """
    Returns a closure that, when called with the current LoRA-adapter
    checkpoint folder + episode index, pushes to HF Hub every N
    episodes. Designed to be wired into ``train_grpo.train`` via a small
    monkey-patch (see the Kaggle notebook).
    """
    def _push(local_dir: str, episode_idx: int):
        if episode_idx % push_every_episodes != 0:
            return
        push_checkpoint_to_hub(
            local_dir,
            repo_id,
            commit_message=f"ER-MAP checkpoint @ episode {episode_idx}",
            hf_token=hf_token,
        )
    return _push


# ---------------------------------------------------------------------------
# Repo bootstrap (clones the project into Kaggle's working dir if needed)
# ---------------------------------------------------------------------------

def ensure_repo(
    target_dir: str = "/kaggle/working/Meta_Finals",
    *,
    git_url: str = "https://github.com/<your-fork>/Meta_Finals.git",
    branch: str = "main",
) -> str:
    """
    Clone the ER-MAP repo into ``/kaggle/working`` if not already present.
    If ``git_url`` is the placeholder, the function instead expects the
    repo to have been uploaded as a Kaggle Dataset (see KAGGLE.md).
    Returns the absolute path to the repo root.
    """
    target = Path(target_dir)
    if target.exists() and (target / "ER_MAP").is_dir():
        print(f"[kaggle_helpers] Repo already at {target}")
        return str(target)

    if "<your-fork>" in git_url:
        # The notebook ships a Kaggle-Dataset fallback, so this is OK.
        # See KAGGLE.md for instructions on uploading the repo as a dataset.
        print(
            f"[kaggle_helpers] git_url placeholder detected; expecting "
            f"the repo to be mounted as a Kaggle Dataset under /kaggle/input/."
        )
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    rc = os.system(f"git clone --depth 1 -b {branch} {git_url} {target}")
    if rc != 0:
        raise RuntimeError(f"git clone failed for {git_url}")
    return str(target)


def add_repo_to_path(repo_root: str) -> None:
    """Make `import ER_MAP` work from inside the notebook."""
    repo_root = str(Path(repo_root).resolve())
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    print(f"[kaggle_helpers] Added to sys.path: {repo_root}")


# ---------------------------------------------------------------------------
# Smoke-test: prints a 1-line summary of the Kaggle env so we can see at
# a glance whether GPU / internet / secrets are wired correctly.
# ---------------------------------------------------------------------------

def kaggle_env_summary() -> None:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(no GPU)"
        vram = (
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            if torch.cuda.is_available() else "—"
        )
    except Exception:
        gpu, vram = "(torch import failed)", "—"

    has_groq = bool(os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("GROQ_API_KEY"))
    has_hf = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))
    has_wandb = bool(os.environ.get("WANDB_API_KEY"))

    print("=" * 72)
    print("  KAGGLE ENVIRONMENT SUMMARY")
    print("=" * 72)
    print(f"  GPU         : {gpu}  ({vram})")
    print(f"  Groq keys   : {'YES' if has_groq else 'NO  (training will use mock LLMs!)'}")
    print(f"  HF token    : {'YES' if has_hf else 'NO  (checkpoints will not be pushed)'}")
    print(f"  W&B token   : {'YES' if has_wandb else 'no (optional)'}")
    print("=" * 72)
