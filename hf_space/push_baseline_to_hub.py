"""
hf_space/push_baseline_to_hub.py
================================
After running `python -m ER_MAP.evaluate_baseline` on the dev box,
this script ships the resulting `baseline_results.json` to the HF
model repo so the Kaggle training notebook can fetch it for the
trained-vs-baseline comparison plot.

Also uploads `docs/make_comparison_plot.py` so the Kaggle notebook can
download and run it without needing to clone our local-only changes.

Usage:
    python hf_space/push_baseline_to_hub.py --repo-id ankitraj86/ermap-doctor-lora
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="ankitraj86/ermap-doctor-lora",
                    help="HF model repo to upload baseline JSON + plot helper to")
    ap.add_argument("--hf-token", default=None,
                    help="HF write token (defaults to HF_TOKEN env var)")
    args = ap.parse_args()

    _load_dotenv(REPO_ROOT / ".env")
    token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: provide --hf-token or set HF_TOKEN in .env", file=sys.stderr)
        return 2

    baseline_json = REPO_ROOT / "baseline_eval" / "baseline_results.json"
    plot_helper = REPO_ROOT / "docs" / "make_comparison_plot.py"

    missing = [p for p in (baseline_json, plot_helper) if not p.is_file()]
    if missing:
        print(f"ERROR: missing files: {missing}", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import create_repo, upload_file
    except ImportError:
        print("ERROR: pip install huggingface_hub", file=sys.stderr)
        return 2

    create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type="model",
        exist_ok=True,
    )
    print(f"[1/3] Repo {args.repo_id} ready (model type)")

    upload_file(
        path_or_fileobj=str(baseline_json),
        path_in_repo="baseline_results.json",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message="upload local baseline_results.json (n=10/phase, fresh run)",
    )
    print(f"[2/3] Uploaded baseline_results.json")

    upload_file(
        path_or_fileobj=str(plot_helper),
        path_in_repo="make_comparison_plot.py",
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
        commit_message="upload trained-vs-baseline comparison plot helper",
    )
    print(f"[3/3] Uploaded make_comparison_plot.py")

    print(f"\nDone. View: https://huggingface.co/{args.repo_id}/tree/main")
    print("The Kaggle notebook (cell 18) will fetch both at the end of training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
