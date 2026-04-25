# Training ER-MAP on Kaggle Free Tier

This guide walks you through training the ER-MAP **Doctor agent** with GRPO + 3-phase curriculum learning on Kaggle's free GPU tier — **zero dollars**, **30 GPU-hours/week**, **single Tesla T4 16 GB**.

## TL;DR — fastest path to a converged Doctor

1. **Fork** this repo on GitHub (it must be reachable from inside the Kaggle kernel).
2. Get **5 Groq API keys** from https://console.groq.com/keys (one per role gives you 5x the daily quota; you can also use one key for everything if you don't mind sharing the rate-limit budget).
3. Get one **HF write token** from https://huggingface.co/settings/tokens (fine-grained, scope: `write` to your own repos) — needed so checkpoints survive the 12-hour Kaggle session limit.
4. **New Notebook on Kaggle** → Settings → **Accelerator: GPU T4 x2** → **Internet: On**.
5. Add the secrets in the right sidebar (Add-ons → Secrets):
   - `GROQ_NURSE_API_KEY`, `GROQ_PATIENT_API_KEY`, `GROQ_EMPATHY_JUDGE_API_KEY`, `GROQ_MEDICAL_JUDGE_API_KEY`
   - `HF_TOKEN`
   - *(optional)* `WANDB_API_KEY`
6. Open `kaggle/train_ermap_grpo_kaggle.ipynb` from your fork inside Kaggle (File → Import Notebook → URL).
7. Edit the two URLs in cell 2 (`GIT_URL`) and cell 5 (`HF_PUSH_REPO`) to your fork / username.
8. **Run All**.

Training **stops automatically** as soon as the Doctor sustains target reward + win rate for 5 consecutive GRPO groups in Phase 3 (this is the "train until optimal rewards are constantly received" guarantee — see the *Train-until-optimal* section below). `NUM_EPISODES=120` is just a hard cap; on a healthy run you typically converge between episodes 60-100.

You'll see one full 6-panel dashboard PNG **per curriculum phase** land in `/kaggle/working/er_map_grpo_checkpoints/plots/` after training finishes (`phase1_dashboard.png`, `phase2_dashboard.png`, `phase3_dashboard.png`, plus `all_phases_overview.png` and `all_phases_comparison.png`), and your final LoRA adapter will be sitting on Hugging Face Hub at `<you>/ermap-doctor-lora`.

**What each per-phase dashboard shows:**

| Panel | What it tells you |
| --- | --- |
| Reward growth | raw episode reward + rolling mean + verified rolling mean |
| Rolling win rate (w=20) | did the policy actually get better in this phase? |
| Outcome distribution over time | stacked WIN/PARTIAL/INCORRECT/AMA_LOSS/FATAL_LOSS bars per ~5-episode bin |
| Reward components | mean of every reward component (process / treatment / empathy / labs / etc.) |
| GRPO update stats | per-group loss + KL — should *not* explode |
| Episode length | histogram of step counts — should rise from Phase 1 to Phase 3 |

---

## Hardware feasibility

| Resource | Kaggle Free Tier | What we use | Headroom |
|---|---|---|---|
| GPU | Tesla T4 16 GB | Llama-3.1-8B-4bit + LoRA(r=16) ≈ 7-9 GB | ~50% free |
| RAM | 13 GB system | base model + tokenizer + buffers ≈ 5 GB | OK |
| Disk | 73 GB | repo + checkpoints + cache ≈ 10 GB | OK |
| Session | 12 h max | typical full Phase-1+early-Phase-2 = 6-8 h | OK |
| Weekly | 30 GPU-h | one full curriculum run + a re-run = ~15-20 h | OK |
| Internet | allowed | Groq calls per env step | OK |

**Why Llama-3.1-8B over Qwen-3-4B (the other train_grpo.py default)?**
- 8B reasons noticeably better on multi-turn clinical dialogue
- 4-bit quant brings it to 5 GB — still fits on T4 with LoRA
- Groq hosts the same 8B (8B-instant) so the deployed inference path matches the training distribution exactly

If you ever need to fall back to a smaller model (e.g. for a P100 session), edit `MODEL_NAME` in cell 5 of the notebook to `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`. Everything else stays the same.

---

## Two ways to get the source onto Kaggle

### Option A — public GitHub fork (recommended)

In cell 2 of the notebook:
```python
GIT_URL = "https://github.com/YOUR_USERNAME/Meta_Finals.git"
BRANCH  = "main"
```
The cell does a shallow clone into `/kaggle/working/Meta_Finals` and you're done.

### Option B — upload as a Kaggle Dataset (no GitHub needed)

1. Locally:
   ```bash
   cd D:/Meta_Finals
   # Exclude heavy/regenerable folders before zipping.
   tar --exclude='.git' --exclude='__pycache__' --exclude='*.ipynb_checkpoints' \
       --exclude='er_map_grpo_checkpoints' \
       -czf ermap-source.tar.gz .
   ```
2. Kaggle → **Datasets** → **New Dataset** → upload `ermap-source.tar.gz` → name it **`ermap-source`** → save.
3. In your training notebook → right sidebar → **+ Add Data** → search for **`ermap-source`** → Add.
4. Cell 2 of the notebook detects `/kaggle/input/ermap-source/` and copies it into `/kaggle/working/Meta_Finals` automatically.

Use Option B when:
- Your fork is private and you don't want to expose the repo
- You have local edits not yet pushed
- Bandwidth from Kaggle to GitHub is flaky

---

## What happens when the 12-hour session ends mid-training

Without intervention you'd lose everything. The notebook prevents this:

1. **Periodic HF Hub push.** Cell 7 monkey-patches `save_lora_adapters()` so every checkpoint saved by the GRPO loop also pushes to your `HF_PUSH_REPO`. The training loop checkpoints every `group_size × 5` episodes (so every 10 episodes when `GROUP_SIZE=2`).
2. **Resume on the next session.** Set `HF_RESUME_REPO` in cell 4 of the notebook on the *new* Kaggle session. The latest LoRA adapter is downloaded to `/kaggle/working/checkpoints/resume/` before training starts — but **the current `train_grpo.py` doesn't auto-load this folder yet**; for now use it as a manual recovery (load the adapter and continue training in code). A future PR will wire the auto-resume into `load_model_and_tokenizer`.

In practice: a single 12-hour session is usually enough to clear Phase 1 and produce publishable per-phase dashboards, so resume is the safety net rather than the main path.

> **Re-render plots from any saved metrics file** (locally or in another Kaggle session):
> ```bash
> python -m ER_MAP.plotting \
>     --metrics er_map_grpo_checkpoints/training_metrics.json \
>     --out     er_map_grpo_checkpoints/plots
> ```
> This is the same call the notebook makes — handy if you want to regenerate the charts after training, or restyle them without re-running training.

---

## Per-role Groq keys vs. one shared key

The dashboard ships with 4 distinct Groq clients (Nurse, Patient, Empathy Judge, Medical Judge) and a fallback chain that walks across all four if any fails auth. Inside training:

- Each env step does **1 Nurse + 1 Patient + occasionally 1 Empathy Judge + 1 Medical Judge call** (judges fire mostly on terminal actions, so call ratio is roughly 4 : 4 : 1 : 1).
- 1 free Groq key = 14 400 req/day on 8B-instant or 6 000 req/day on 70B-versatile.
- 120-episode training × 8 avg steps × 2 conversational LLM calls = ~2 000 calls. **Even one key is enough for a single training run**, but if you split across 4 keys you have 4× the daily headroom for re-runs.

If you only have **one** Groq key, set just `GROQ_API_KEY` as a Kaggle Secret. Everything still works — the AgentRouter falls back to the same client for all roles.

---

## What the reward-growth curve should look like

If training is healthy, after ~80 episodes you should see:

- **Rolling avg reward** climbs from ≈ -0.4 (random baseline) toward +1.5+ (the early-stop target)
- **Rolling win rate** climbs from ~10% to 40%+
- A **vertical red dashed line** marks the Phase 1 → Phase 2 promotion (typically episode 30-60), and a second one marks Phase 2 → Phase 3 (typically episode 60-90)
- KL divergence stays in `[0.005, 0.05]` — if it spikes above 0.5 the model is drifting (lower `LEARNING_RATE` and re-run)

If the curve is flat or trending down:
- Check that Groq is actually responding (look for `Groq API error` lines in the log)
- Check that `rewards.std()` is non-zero across the group (cell logs print `adv_std=`; if it's < 1e-6 GRPO skips the update)
- Drop `GROUP_SIZE` from 2 → 1? **Don't** — group size 1 = no advantage signal = no GRPO update. Keep G ≥ 2.

---

## Train-until-optimal — the early-stopping policy

> *"I want training until certain optimal rewards are constantly received."*

The training loop tracks a **rolling buffer of the last `CONVERGENCE_WINDOW` GRPO groups** and stops the run the instant **all** of them satisfy:

- `rolling_avg_reward >= TARGET_ROLLING_REWARD`
- `rolling_win_rate   >= TARGET_WIN_RATE`
- the curriculum scheduler is in **phase ≥ `CONVERGENCE_MIN_PHASE`**

If even one group in the window slips below the bar, the counter resets — guaranteeing the policy is *constantly* hitting the target, not transiently. `NUM_EPISODES` becomes a hard safety cap, not a fixed budget.

### Default targets (in the notebook, section 5)

```python
EARLY_STOP_ENABLED      = True
TARGET_ROLLING_REWARD   = 1.5    # +1.5 average reward, sustained
TARGET_WIN_RATE         = 0.40   # 40% wins, sustained
CONVERGENCE_WINDOW      = 5      # for 5 consecutive GRPO groups (= 10 episodes at G=2)
CONVERGENCE_MIN_PHASE   = 3      # only count Phase-3 (Empathetic Negotiation) groups
```

### Reading the early-stop telemetry

After every GRPO group the log prints:

```
[Scheduler] Phase 3 (Empathetic Negotiation) | Win Rate: 42.0% | Avg Reward: +1.62 | Phase Episodes: 24
[EarlyStop] qualified 4/5 recent groups (need all 5)
```

When the buffer is full and every entry qualifies, you'll see:

```
************************************************************
  EARLY STOP: convergence reached after 84 episodes
  Last 5 groups all met target:
    rolling_avg_reward >= +1.50
    rolling_win_rate   >= 40%
    in phase           >= 3
************************************************************
```

…and the loop exits cleanly into the final-save / final-push / plotting cells.

### Tuning suggestions

| Goal | What to change |
|---|---|
| Smoke run (converge fast on a weak policy) | `TARGET_ROLLING_REWARD=0.5`, `TARGET_WIN_RATE=0.20`, `CONVERGENCE_MIN_PHASE=2` |
| Hackathon-grade Doctor | keep defaults |
| Aim for SOTA on this benchmark | `TARGET_ROLLING_REWARD=2.0`, `TARGET_WIN_RATE=0.55`, `CONVERGENCE_WINDOW=8` |
| Disable entirely (run full 120 episodes regardless) | `EARLY_STOP_ENABLED=False` |
| Resuming from a partial run | targets are unchanged — the scheduler's rolling window is rebuilt from the new session's episodes, so nothing weird happens |

### What NOT to do

- Don't set `CONVERGENCE_MIN_PHASE=1`. The scheduler promotes out of Phase 1 with a 50% win-rate trigger, so the win-rate condition is trivially satisfied by easy cases — you'd stop training before the model has even seen Phase-3 difficulty.
- Don't set `CONVERGENCE_WINDOW=1`. A single lucky group can pass the bar even on a weak policy. 5 groups (= 10 episodes at G=2) is the smallest window that's stable.
- Don't lower `TARGET_WIN_RATE` to 0 while keeping `CONVERGENCE_MIN_PHASE=3`. The "minimum phase" check needs the policy to first survive Phase 2's promotion bar (60% rolling win), so the floor is already implicit.

---

## Common Kaggle gotchas

| Symptom | Fix |
|---|---|
| `Groq API error: 401 invalid_api_key` | Regenerate the key (Groq auto-revokes keys posted publicly). Update the Kaggle Secret. |
| `OutOfMemoryError` on T4 | Drop `MAX_SEQ_LENGTH` from 2048 to 1536 inside `load_model_and_tokenizer`, or switch to `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`. |
| `unsloth import` failed | Restart kernel after `pip install` — Unsloth pins `xformers` versions and the running kernel keeps the old import cached. |
| Checkpoints not appearing on HF Hub | Verify `HF_PUSH_REPO` doesn't still contain the `<your-username>` placeholder, and that `HF_TOKEN` has `write` scope. |
| "Internet off" warning | Right sidebar → Settings → toggle Internet to **On**. (Default is off for new accounts.) |

---

## Cost summary

- **Kaggle**: free
- **Groq API (training)**: free (within free-tier daily quotas, ~2 000 calls per full run)
- **Hugging Face Hub**: free for the LoRA adapter (~50 MB) + free for the merged fp16 (~16 GB on a public repo, free up to 1 TB total)
- **Wandb**: free for personal projects
- **Total**: $0
