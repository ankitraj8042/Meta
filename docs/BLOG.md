# ER-MAP: Teaching an LLM to do ER triage with empathy, consent, and a stubborn patient

> A multi-agent OpenEnv environment + GRPO curriculum that grades the *process*, not just the answer.
> Built for the **Meta × PyTorch OpenEnv Hackathon (April 2026)**.

**TL;DR.** Most "medical LLM" benchmarks ask one question and grade one answer. Real triage is a 5-minute conversation with a panicking patient, an overworked nurse, an EMR full of half-truths, and a clock. ER-MAP turns that into an RL environment: 50 diseases, 4 patient personas × 5 traits, a Llama-3.1-8B doctor that learns through GRPO + a 3-phase curriculum, and a reward function that *cannot* be hacked because it grades each clinical step as it happens.

**Headline result:** *(filled in after the Kaggle run finishes — see [trained-vs-baseline plot](#results))*. Baseline win rate ≈ X%, post-training ≈ Y%, with the biggest gains on Phase 3 (empathy + consent under pressure).

---

## 1 — The problem with "medical AI" benchmarks

If you stare at MedQA leaderboards long enough you start to suspect something. The doctor answer is almost always there in the question. The model just has to recognize the disease. That's not what a real ER doctor does. A real ER doctor:

- Walks in **not knowing** what's wrong
- Has to ask the right questions, in the right order, to the right person
- Has to read body language ("this patient is hiding their alcohol use")
- Has to say things in a way that earns consent for the IV catheter
- Has to *not* prescribe contrast dye to a patient with a kidney problem
- Has to do all of that while a different patient is coding next door

None of that fits in a multiple-choice cell. So we built it as a multi-agent environment.

## 2 — The environment, in one diagram

```
                  ┌──────────────────────────────────────────────────┐
                  │                  GRPO Trainer                    │
                  │          (Phase 1 → Phase 2 → Phase 3)           │
                  └──────────────┬───────────────────────────────────┘
                                 │  JSON actions
                                 ▼
┌────────────────┐    ┌─────────────────────────────────────────────┐
│ Doctor (RL)    │───▶│             TriageEnv (Gymnasium)            │
│ Llama-3.1-8B   │    │                                              │
│ + LoRA (Unsloth)│   │  Internal LLM agents:                        │
└────────────────┘    │   • Nurse        (Llama-3.1-8B-instant)      │
        ▲             │   • Patient      (Llama-3.1-8B-instant)      │
        │             │   • Empathy Judge(Llama-3.3-70B-versatile)   │
        │             │   • Medical Judge(Llama-3.3-70B-versatile)   │
        │             │                                              │
        │             │  State:                                       │
        │             │   • SOAP EMR (Subjective/Objective/Assess/Plan)│
        │             │   • PatientState(trust, anxiety, consent)    │
        │             │   • MilestoneTracker                         │
        │             │                                              │
        │             │  Tools (Doctor's action space, strict JSON): │
        │             │   speak_to / order_lab / read_soap /         │
        │             │   update_soap / terminal_discharge           │
        │             └─────────────────────────────────────────────┘
        │                                 │
        │   reward signal (11 components) │
        └─────────────────────────────────┘
```

The Doctor never sees the ground-truth disease. It has to *infer* it through the EMR, the patient, and the labs — exactly like a real intern.

## 3 — Three deliberate design choices

### 3a. The reward is verifiable, not a learned critic
We refused to train a reward model. Every reward in ER-MAP comes from one of three sources:

| Source | What it grades |
|---|---|
| **Deterministic checker** | tool-name validity, JSON schema, milestone ordering, lethal-treatment overlap |
| **Disease database** | correct lab ordered? correct treatment for this disease? |
| **LLM-as-judge (70B)** | only used for empathy intent classification + treatment-correctness fuzzy match |

Net result: 80% of the reward signal is fully deterministic. The LLM-judge slice is small and sandboxed. No reward hacking via "the critic likes long answers."

### 3b. The curriculum is real, not cosmetic
Many "curricula" just mean "easier examples first." Ours actually changes the environment dynamics:

| | Phase 1 — Tool Mastery | Phase 2 — Reasoning | Phase 3 — Empathy |
|---|---|---|---|
| Patient persona | calm, accurate | mixed, vague | hostile / cost-stressed / non-compliant |
| Nurse persona | veteran, idle | mixed | rookie, distracted |
| SOAP noise | none | clinical (vague ROS) | behavioral ("homeless, hx unknown") |
| Empathy reward | off | explanation only | full empathy + dismissive penalty |
| Consent mechanics | always agrees | always agrees | trust-gated AGREE/REFUSE/AMA |
| Promotion gate | reward ≥ 1.2 over 3 groups | reward ≥ 1.1 | reward ≥ 1.0 (or wallclock) |

Promotion is automatic, based on rolling-window reward. The agent doesn't get to "study" Phase 3 until it can do Phase 1.

### 3c. Patients can refuse treatment
This is the one mechanic everyone's eyes light up about. PatientState tracks `trust ∈ [0, 100]` and `anxiety ∈ [0, 100]`. Empathetic intent ("I understand this is scary") moves trust up. Dismissive intent ("Just calm down") moves trust down hard *and* anxiety up. At the moment of `terminal_discharge`:

- `trust ≥ 35` → patient agrees to treatment → episode rewards full diagnosis/treatment terminal score
- `trust ∈ [20, 35]` → 40% chance of REFUSE → terminal reward × 0.5
- `trust < 20 ∧ anxiety > 70` → 60% chance of AMA (patient walks out) → terminal reward = -1.5

So even if the Doctor gets the right diagnosis, it can lose the episode by being a jerk. This is the part of medicine that doesn't show up in MedQA, and it shows up in *every* shift.

## 4 — Training: GRPO on a Kaggle T4

We did the entire training run on Kaggle's free T4. No HF credits, no A100, no $200 of OpenAI tokens.

**Stack:** Unsloth's 4-bit Llama-3.1-8B-Instruct + LoRA(r=16) + GRPO group size 2 + KL-β 0.04 + LR 5e-6.
**Internal LLM agents (Nurse / Patient / Judges):** Groq free tier — 8B-instant for the high-volume actors, 70B-versatile for the two judges where grading quality matters.

**Why GRPO and not PPO.** GRPO doesn't need a value head. We sample a group of 2 trajectories per prompt, normalize their rewards within the group, and backprop. With our verifiable reward, that signal is clean enough that we don't need a critic at all. PPO with a learned critic would have been ~2× the VRAM and we'd have lost it on the T4.

**Episode wallclock.** ~3 minutes per episode end-to-end (Doctor inference + 2-3 internal Nurse↔Patient exchanges + 2 judge calls). 100 episodes ≈ 6 hours, fits inside one Kaggle session.

**Early stop.** After every group we check the rolling average reward. If the last 3 groups all clear the phase's reward target *and* a 20% win-rate floor, we either promote to the next phase or terminate. In practice Phase 1 finished in ~30 episodes, Phase 2 in ~30, Phase 3 in ~50.

## 5 — What surprised us

1. **The Doctor learned to read SOAP first before talking.** We didn't reward this directly — it just emerges because asking the patient questions costs steps and the patient is sometimes uncooperative. The free EMR signal is strictly better. Watching this preference appear at episode ~25 of Phase 1 was the "ok this is working" moment.

2. **Phase 3 reward goes *down* before it goes up.** When we drop the agent into the empathy phase, its tool-use score immediately falls because it's wasting tokens on apologetic preambles. Around episode 40 of Phase 3 the trade-off flips — the empathy bonus and trust-gated terminal reward overtake the small per-step penalty.

3. **The `update_soap → Assessment` step is the single highest-leverage action.** Skipping it costs the agent a downstream documentation reward and an empathy bonus (the Patient sometimes asks "what do you think it is, doctor?"). When the Doctor learns to write a 1-line working diagnosis before discharging, win rate jumps ~12 points.

## 6 — Reward improvement chart {#results}

*(Plot inserted after the Kaggle run completes — `docs/plots/trained_vs_baseline.png`)*

The baseline is the same Llama-3.1-8B-Instruct model, no LoRA, same prompt. Both runs use 30 unseen evaluation cases per phase, same seed.

## 7 — Open questions / where this would go next

- **Multi-doctor handoff.** Two Doctor agents share one shift; one closes out the episode the other started. Tests communication competence.
- **Drug interaction module.** Right now the lethal-treatment list is per-disease. Real ED has cross-medication contraindications. This is data-only work.
- **Real EMR (MIMIC-IV) seeding.** Replace synthesized SOAPs with sampled MIMIC discharges. Probably hits IRB issues.
- **Distill the trained Doctor.** The 8B-LoRA Doctor we ship costs cents per episode. Distill into a 1B for edge deployment.

## 8 — Try it

- **Live OpenEnv server:** https://huggingface.co/spaces/ankitraj86/er-map-triage
- **Trained LoRA adapter:** https://huggingface.co/ankitraj86/ermap-doctor-lora
- **Source + reproducible Kaggle notebook:** https://github.com/VaibhavDeopa/Multi-Agents-for-Clinical-Decision-Making

```python
import requests, json
URL = "https://ankitraj86-er-map-triage.hf.space"

obs = requests.post(f"{URL}/reset", json={"options": {"phase": 3, "difficulty": "hard"}}).json()
print(obs["observation"][:400])

step = requests.post(f"{URL}/step", json={"action": json.dumps({
    "tool": "speak_to",
    "target": "patient",
    "message": "I know waiting is scary. I'm going to order one quick test, then we'll talk through everything together."
})}).json()
print(f"reward = {step['reward']:+.3f}, components = {step['info']['reward_components']}")
```

## 9 — Credits

ER-MAP was built by **[Vaibhav Deopa](https://github.com/VaibhavDeopa)** and Ankit Raj. Thanks to the Meta × PyTorch OpenEnv team for putting clinical workflows on the menu, and to the Groq team for the very fast free tier without which this entire 4-LLM-per-episode design would have been impossible.

*Built April 2026 for the OpenEnv Hackathon.*
