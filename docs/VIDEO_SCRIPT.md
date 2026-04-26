# ER-MAP — 2-minute demo video script

**Target length:** 2 min total (~110 s of voice over). Cuts at 0:00, 0:18, 0:40, 1:05, 1:30.

**Recording stack (zero-cost):**
- Screen capture: OBS (free) at 1080p / 30 fps, scene preset "Display Capture (full)"
- Voice over: ElevenLabs TTS (`v3_alpha` or any natural voice) — script below feeds straight in
- Editor: Clipchamp (built into Windows) or DaVinci Resolve free
- Music: bensound.com → "Memories" or "Tenderness" (CC-BY)
- Final: 1080p MP4 ≤ 100 MB, upload to YouTube (unlisted) or Loom

---

## SCENE 1 — Hook (0:00–0:18, 18 s)

**Voice over:**
> "Most medical AI benchmarks are multiple-choice tests. Real ER triage is a five-minute conversation with a panicking patient, an overworked nurse, and an EMR full of half-truths. ER-MAP turns that into a multi-agent reinforcement learning environment, on the OpenEnv standard."

**On-screen:**
- 0:00–0:06: Title card. **"ER-MAP"** large, subtitle "Emergency Room Multi-Agent Protocol — OpenEnv Hackathon, April 2026."
- 0:06–0:18: Cut to the autoplay terminal. A patient saying "I'm scared, what's wrong with me?" appears. Doctor responds. Nurse status banner shows.

---

## SCENE 2 — The environment (0:18–0:40, 22 s)

**Voice over:**
> "The Doctor is the RL agent. Inside the env we run a Nurse, a Patient, an empathy judge and a medical judge — all LLMs. The Doctor never sees the diagnosis. It has to read the SOAP note, talk to the patient, order labs, and earn consent — and the reward function grades each clinical step as it happens, not just the final answer."

**On-screen:**
- 0:18–0:30: The architecture diagram from the README (the GRPO ⇄ TriageEnv ⇄ 4 LLMs box). Highlight each LLM agent in turn with a glow.
- 0:30–0:40: Quick cuts of the action-space JSON examples — `speak_to`, `order_lab`, `read_soap`, `update_soap`, `terminal_discharge`. Cap on each.

---

## SCENE 3 — The hard part: empathy + consent (0:40–1:05, 25 s)

**Voice over:**
> "Phase 3 is where it gets interesting. Patients can refuse treatment if you've been dismissive. We track trust and anxiety state explicitly. Walk in cold, you lose to AMA — Against Medical Advice. The same diagnosis, with empathy, agrees and gets treated."

**On-screen:**
- 0:40–0:52: Side-by-side two-episode replay. Left panel: Doctor says "Just calm down, we need to do this test." → Patient trust drops to 18 → REFUSE → episode ends with -1.5 reward.
- 0:52–1:05: Right panel: Doctor says "I know waiting is scary. Let me explain why this test matters." → Patient trust climbs to 78 → AGREE → +2.0 terminal reward, full treatment delivered.
- Show the trust/anxiety bars animating in the corner.

---

## SCENE 4 — Training results (1:05–1:30, 25 s)

**Voice over:**
> "We trained an 8-billion-parameter Llama on a single Kaggle T4 — no paid GPUs, no HF credits. GRPO with group size two, a three-phase curriculum that promotes automatically when rolling reward clears the gate. After about a hundred episodes total, the same baseline model — same prompt, no LoRA — wins X percent. With our training, Y."

**On-screen:**
- 1:05–1:15: Pan across the trained-vs-baseline reward histogram. (`docs/plots/trained_vs_baseline.png` — generated post-training.)
- 1:15–1:25: Cut to the per-phase reward curve chart from `er_map_grpo_checkpoints/plots/phase_*_rewards.png`.
- 1:25–1:30: Single number on screen: **"Win rate: baseline X% → trained Y%"**.

> **Note:** fill X / Y from `er_map_grpo_checkpoints/training_metrics.json` (post-training mean group reward) and `baseline_eval/baseline_metrics.json` (baseline win rate). The plot helper at `docs/make_comparison_plot.py` renders both panels at once.

---

## SCENE 5 — Try it live (1:30–1:55, 25 s)

**Voice over:**
> "The whole environment is live on Hugging Face Spaces — Docker SDK, FastAPI, OpenEnv-compatible. You can hit slash-reset and slash-step from anywhere with a couple of lines of Python. The trained adapter is on the Hub. Source, Kaggle notebook, and the full writeup are linked. That's ER-MAP. Thanks for watching."

**On-screen:**
- 1:30–1:42: Browser tab → https://huggingface.co/spaces/ankitraj86/er-map-triage → click "App" → Swagger UI / interactive demo.
- 1:42–1:50: Code window with the 5-line `requests.post` example from the README.
- 1:50–1:55: Closing card with all four links: Space / model / GitHub / blog.

---

## SCENE 6 — Outro card (1:55–2:00, 5 s)

**On-screen (no voice over, just music tail):**
```
ER-MAP
A multi-agent OpenEnv environment for ER triage RL.

Built by Vaibhav Deopa & Ankit Raj
Meta × PyTorch OpenEnv Hackathon · April 2026

github.com/VaibhavDeopa/Multi-Agents-for-Clinical-Decision-Making
```

---

## Production checklist

- [ ] Record SCENE 1 hook with autoplay terminal (run `python -m ER_MAP.autoplay --phase 3 --difficulty hard`)
- [ ] Capture SCENE 3 contrast clip (run autoplay twice with `--seed 42`, manually inject hostile vs empathetic doctor lines, OR scripted demo episodes)
- [ ] Generate plots from training run: `python docs/make_comparison_plot.py`
- [ ] Record voice over (ElevenLabs, take ~3 passes per scene, pick best)
- [ ] Cut in editor with the 5 SCENE chunks back-to-back, light music underneath at -18 dB
- [ ] Export 1080p MP4 ≤ 100 MB
- [ ] Upload (YouTube unlisted or Loom) → grab share link → paste into README top section
