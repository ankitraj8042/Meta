# ER-MAP — 8-hour hackathon delivery runbook

This is the cheat-sheet for getting from "fresh repo + API keys" to "submission".
Everything here is what's already been done locally + what you have to do live.

## State already on disk

| Artifact | Path / URL | Status |
|---|---|---|
| `.env` with all keys | `.env` (git-ignored) | ✅ wired |
| Local sanity test | every Groq key + HF token | ✅ all PASS |
| 30-episode baseline | `baseline_eval/baseline_results.json` | ✅ saved |
| HF Space (live env server) | https://huggingface.co/spaces/ankitraj86/er-map-triage | ✅ RUNNING |
| HF Model repo (bootstrapped) | https://huggingface.co/ankitraj86/ermap-doctor-lora | ✅ has baseline + plot helper |
| Kaggle notebook | `kaggle/train_ermap_grpo_kaggle.ipynb` | ✅ patched, 22 cells |
| Mini-blog | `docs/BLOG.md` | ✅ drafted (fill in result numbers post-training) |
| 2-min video script | `docs/VIDEO_SCRIPT.md` | ✅ drafted (record after training plot lands) |
| Comparison plot helper | `docs/make_comparison_plot.py` | ✅ wired into Kaggle Cell 18 |
| Updated README | `README.md` | ✅ live submission links section, baseline results table |
| Dead code | `ER_MAP/training/train_ppo.py` | ✅ deleted |
| `server.py` judge wiring | full 4-LLM injection from env vars | ✅ fixed |

---

## Step 1 — Launch Kaggle training (you do this, ~10 min hands-on, ~6h training)

1. https://www.kaggle.com/code → New Notebook → **File → Upload Notebook** → select
   `kaggle/train_ermap_grpo_kaggle.ipynb` from this repo.
2. Right sidebar → Settings:
   - Accelerator: **GPU T4 ×2** (or P100)
   - Internet: **On**
   - Persistence: Files only
3. Add-ons → Secrets → add these 5 (label = exactly as shown):

   | Label | Value |
   |---|---|
   | `GROQ_NURSE_API_KEY` | your nurse Groq key |
   | `GROQ_PATIENT_API_KEY` | your patient Groq key |
   | `GROQ_EMPATHY_JUDGE_API_KEY` | your empathy-judge Groq key |
   | `GROQ_MEDICAL_JUDGE_API_KEY` | your medical-judge Groq key |
   | `HF_TOKEN` | your HF write token |

4. Run cells in this exact order: **2 → 3 → (restart kernel if cell 3 says so) → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 (the 6h training run) → 14 → 15 → 16 → 17 → 18**.
   - Cells 2–12 take ~5 min total; they're env setup + smoke tests.
   - Cell 10 must print **4× `[PASS]`** before you start cell 13.
   - Cell 11 must print **`Dry-run OK`** before you start cell 13.
   - Cell 13 is the actual training. Walk away. Early-stop usually fires around episode 80–120.
   - Cells 14–18 fire after training: save final adapter, push to HF, generate per-phase plots, generate trained-vs-baseline plot, upload all artifacts.

While Kaggle runs, do steps 2–4 below in any order.

---

## Step 2 — Decide your GitHub home (5 min, BLOCKS the README submission)

The README's "Live Submission Links" table currently points to the GitHub repo at
`VaibhavDeopa/Multi-Agents-for-Clinical-Decision-Making`. The hackathon needs a public
repo with our updated README, blog, plots, etc. Three options, pick one:

| Option | Pros | Cons | What you do |
|---|---|---|---|
| **A. Fork to `ankitraj86/...`** *(recommended)* | clean, you own it | you'll be the listed author | `gh repo fork --clone=false` then push our local diff to your fork |
| **B. Push to VaibhavDeopa's repo** | preserves co-authorship | needs VaibhavDeopa's permission | push to a feature branch, open a PR |
| **C. Brand new `ankitraj86/ER-MAP`** | totally clean slate | loses commit history | `gh repo create`, push everything |

Once you pick, push the local diff (~9 modified files + `docs/` + `hf_space/`) and
update the GitHub URL in `README.md` if you went with A or C.

---

## Step 3 — Record the demo video (~45 min)

Use `docs/VIDEO_SCRIPT.md` as the script. Walkthrough:

1. Run a phase 3 demo episode locally for the hook clip:
   ```powershell
   cd C:\Users\ankit\OneDrive\Desktop\meta\Multi-Agents-for-Clinical-Decision-Making
   .\.venv\Scripts\Activate.ps1
   python -m ER_MAP.autoplay --phase 3 --difficulty hard
   ```
   Capture the terminal with OBS.
2. For the Phase 3 contrast scene, record two episodes side by side:
   - Run 1: rude doctor lines (you can grep an autoplay run for AMA outcomes)
   - Run 2: empathetic doctor lines (run with `--difficulty easy` and let it succeed)
3. After Kaggle finishes, grab the trained-vs-baseline plot from
   https://huggingface.co/ankitraj86/ermap-doctor-lora/blob/main/trained_vs_baseline.png
   for the results scene.
4. Voice-over via ElevenLabs (you have the API key).
5. Edit in Clipchamp / DaVinci Resolve, export 1080p ≤ 100 MB.
6. Upload to YouTube unlisted → grab share link → paste into the README's
   "Live Submission Links" table.

---

## Step 4 — Final polish + submission (last ~30 min)

Once Kaggle's cell 18 finishes:

1. **Verify the trained-vs-baseline plot exists** at
   https://huggingface.co/ankitraj86/ermap-doctor-lora/blob/main/trained_vs_baseline.png
   and `trained_vs_baseline_summary.json` next to it.
2. **Read the summary JSON** for the headline number; e.g.:
   ```python
   import json, urllib.request
   data = json.loads(urllib.request.urlopen(
       "https://huggingface.co/ankitraj86/ermap-doctor-lora/resolve/main/trained_vs_baseline_summary.json"
   ).read())
   print(json.dumps(data, indent=2))
   ```
3. **Fill in result numbers in `docs/BLOG.md`** (search for "*(filled in after the Kaggle run finishes)*" and "*populated from `training_metrics.json`*"). Replace with actual numbers from step 2.
4. **Fill in result numbers in `README.md`** (the "Trained" row of the results table).
5. **Add the YouTube link** to the "Live Submission Links" table (top of README).
6. **Push everything** to GitHub (whichever option from Step 2 you picked).
7. **Submit** to the hackathon form using:
   - GitHub URL (your repo)
   - HF Space: https://huggingface.co/spaces/ankitraj86/er-map-triage
   - HF Model: https://huggingface.co/ankitraj86/ermap-doctor-lora
   - Blog: link to your README's `docs/BLOG.md` on GitHub
   - Video: YouTube unlisted URL

---

## Failure modes & escape hatches

- **Kaggle cell 13 dies on Groq 429**. The empathy judge is now on `llama-3.1-8b-instant` (5× quota). If you still hit limits, edit cell 9 of the notebook and set
  `os.environ["ERMAP_EMPATHY_JUDGE_MODEL"] = "llama-3.1-8b-instant"` (already done) AND/OR skip phase 3 by commenting out its phase budget in cell 13.
- **Kaggle session times out** (12h hard limit). The notebook saves checkpoints every
  20 episodes via `PUSH_EVERY_EPS=20` to your HF model repo. Restart the notebook,
  set `HF_RESUME_REPO = "ankitraj86/ermap-doctor-lora"` in cell 8, re-run.
- **Cell 18 plot fails**. The data is in `er_map_grpo_checkpoints/training_metrics.json`
  on Kaggle. Download it, run `python docs/make_comparison_plot.py --metrics training_metrics.json` from any matplotlib-working machine.
- **Local matplotlib is blocked by Windows AppLocker**. Don't run plotting locally.
  All plots come out of Kaggle's cell 15 + 18.

---

## What we explicitly skipped (and why)

- **Colab notebook** — Kaggle does the same job for free. The hackathon spec says
  "Colab notebook OR Kaggle notebook"; we picked Kaggle.
- **W&B logging** — disabled in cell 9 because the Kaggle base image has a protobuf
  conflict that's not worth fighting on the clock. Per-phase plots live in
  `er_map_grpo_checkpoints/plots/`.
- **Bumping baseline above n=10/phase** — n=30 total gives clean error bars and
  the trend (0/30 baseline wins) is unambiguous; more data wouldn't move the
  comparison story.
