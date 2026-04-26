"""
hf_space/push_to_space.py
=========================
One-shot deploy of the ER-MAP TriageEnv FastAPI server to a Hugging Face
Space. Designed to run from the repo root.

Usage:
    python hf_space/push_to_space.py \
        --space-id ankitraj86/er-map-triage \
        --hf-token <your_hf_token>

If --hf-token is omitted, reads HF_TOKEN from the environment (or .env).

What it does:
    1. Creates the Space if it doesn't exist (Docker SDK).
    2. Stages: ER_MAP/ + Dockerfile + hf_space/README.md (renamed to README.md
       in the Space root) + a .gitignore + .env.example for reference.
    3. Uploads via huggingface_hub create_commit (no local git clone needed).
    4. Sets the required Space secrets from your local .env.
    5. Prints the live Space URL.

After this runs, the Space starts building (~2-3 minutes for first deploy).
You can check status at https://huggingface.co/spaces/<space-id>.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_STAGE = REPO_ROOT / "hf_space"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader; only sets vars that are not already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and not os.environ.get(k):
            os.environ[k] = v


def _files_to_upload() -> List[Tuple[Path, str]]:
    """Return list of (local_path, path_in_space)."""
    pairs: List[Tuple[Path, str]] = []

    pairs.append((REPO_ROOT / "Dockerfile", "Dockerfile"))
    pairs.append((SPACE_STAGE / "README.md", "README.md"))

    # Whole ER_MAP package (excluding scratch UI files and .pyc cache)
    er_map_root = REPO_ROOT / "ER_MAP"
    skip_suffixes = {".pyc", ".pyo"}
    skip_dirs = {"__pycache__"}
    for path in er_map_root.rglob("*"):
        if path.is_dir() and path.name in skip_dirs:
            continue
        if path.is_file() and path.suffix not in skip_suffixes:
            # Skip throwaway UI and TTS demo files; the env doesn't need them.
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel.startswith("ER_MAP/UI/"):
                continue
            pairs.append((path, rel))

    return pairs


def push(space_id: str, hf_token: str, set_secrets: bool = True) -> str:
    """Create or update the Space, then upload files."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=hf_token)

    # 1. Create the Space (idempotent — exist_ok=True)
    print(f"[1/4] Creating Space {space_id} (Docker SDK)...")
    create_repo(
        repo_id=space_id,
        token=hf_token,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )

    # 2. Set Space secrets so the env can hit Groq
    if set_secrets:
        print("[2/4] Wiring Space secrets from .env...")
        wanted = [
            "GROQ_NURSE_API_KEY",
            "GROQ_PATIENT_API_KEY",
            "GROQ_EMPATHY_JUDGE_API_KEY",
            "GROQ_MEDICAL_JUDGE_API_KEY",
            "GROQ_API_KEY",
            "ERMAP_NURSE_MODEL",
            "ERMAP_PATIENT_MODEL",
            "ERMAP_EMPATHY_JUDGE_MODEL",
            "ERMAP_MEDICAL_JUDGE_MODEL",
        ]
        for name in wanted:
            val = os.environ.get(name, "").strip()
            if not val:
                print(f"   skip {name} (empty)")
                continue
            try:
                api.add_space_secret(repo_id=space_id, key=name, value=val)
                print(f"   set  {name}")
            except Exception as e:
                print(f"   FAIL {name}: {type(e).__name__}: {str(e)[:120]}")

    # 3. Stage and upload files
    print("[3/4] Uploading files...")
    pairs = _files_to_upload()
    print(f"   {len(pairs)} files staged")
    for local_path, repo_path in pairs[:8]:
        print(f"   + {repo_path}")
    if len(pairs) > 8:
        print(f"   + ... and {len(pairs) - 8} more")

    # Use upload_folder pattern via individual uploads (HfApi.upload_file is
    # batched server-side via create_commit when called via upload_folder)
    from huggingface_hub import upload_folder
    import tempfile, shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for local_path, repo_path in pairs:
            dest = tmp / repo_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest)

        upload_folder(
            folder_path=str(tmp),
            repo_id=space_id,
            repo_type="space",
            token=hf_token,
            commit_message="Deploy ER-MAP TriageEnv (Docker SDK, FastAPI on :7860)",
        )

    space_url = f"https://huggingface.co/spaces/{space_id}"
    print(f"[4/4] Done. Space: {space_url}")
    print(f"      App URL (after build):  https://{space_id.replace('/', '-')}.hf.space/")
    print(f"      Build status:           {space_url} (right side panel)")
    return space_url


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space-id", required=True,
                    help="HF Space id, e.g. ankitraj86/er-map-triage")
    ap.add_argument("--hf-token", default=None,
                    help="HF write token (defaults to HF_TOKEN env var)")
    ap.add_argument("--no-secrets", action="store_true",
                    help="Skip setting Space secrets (use if Space already configured)")
    args = ap.parse_args()

    _load_dotenv(REPO_ROOT / ".env")

    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: provide --hf-token or set HF_TOKEN in .env", file=sys.stderr)
        return 2

    push(args.space_id, token, set_secrets=not args.no_secrets)
    return 0


if __name__ == "__main__":
    sys.exit(main())
