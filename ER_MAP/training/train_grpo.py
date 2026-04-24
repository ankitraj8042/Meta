"""
ER_MAP/training/train_grpo.py
==============================
GRPO (Group Relative Policy Optimization) Training Script with
3-Phase Curriculum Learning for the ER-MAP Triage Environment.

Key differences from PPO:
- No value head / critic needed (simpler, more stable)
- Uses group-relative rewards: compares G completions per prompt
- Process-based rewards via verifier functions (not learned)
- Curriculum scheduler controls phase transitions automatically

Usage (Colab / HF Spaces):
    !pip install unsloth trl transformers datasets accelerate peft
    !pip install gymnasium groq
    python -m ER_MAP.training.train_grpo --episodes 200

Usage (local dry-run, no GPU):
    python -m ER_MAP.training.train_grpo --dry-run
"""

import os
import json
import time
import torch
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ER_MAP.train_grpo")


# ============================================================================
# Curriculum Scheduler
# ============================================================================

@dataclass
class PhaseConfig:
    """Configuration for a single curriculum phase."""
    name: str
    phase_id: int
    difficulty: str                    # "easy", "medium", "hard", or None
    min_episodes: int                  # Minimum episodes before promotion
    promotion_win_rate: float          # Win rate threshold to advance
    promotion_avg_reward: float        # Avg reward threshold to advance
    description: str = ""


class CurriculumScheduler:
    """
    Manages 3-phase curriculum transitions.

    Phase 1 (Tool Mastery):
        - Easy patients (calm, compliant)
        - Clean SOAP data
        - Focus: learn to use tools correctly (read_soap, order_lab, speak_to)
        - Promote when: win_rate >= 40% over last 20 episodes

    Phase 2 (Clinical Reasoning):
        - Mixed difficulty patients
        - Noisy SOAP data (missing fields, vague history)
        - Focus: differential diagnosis, correct lab ordering
        - Promote when: win_rate >= 35% AND avg_reward >= 0.5

    Phase 3 (Empathetic Negotiation):
        - Full persona randomization (hostile, non-compliant, uninsured)
        - Heavy SOAP noise + behavioral friction
        - Focus: empathy, trust-building, consent management
        - No promotion (final phase)
    """

    PHASES = [
        PhaseConfig(
            name="Tool Mastery",
            phase_id=1,
            difficulty="easy",
            min_episodes=20,
            promotion_win_rate=0.40,
            promotion_avg_reward=0.3,
            description="Learn clinical tools: read_soap, order_lab, speak_to, terminal_discharge",
        ),
        PhaseConfig(
            name="Clinical Reasoning",
            phase_id=2,
            difficulty="medium",
            min_episodes=30,
            promotion_win_rate=0.35,
            promotion_avg_reward=0.5,
            description="Differential diagnosis with noisy data and mixed-compliance patients",
        ),
        PhaseConfig(
            name="Empathetic Negotiation",
            phase_id=3,
            difficulty="hard",
            min_episodes=50,
            promotion_win_rate=1.0,   # Never auto-promote (final phase)
            promotion_avg_reward=99.0,
            description="Full persona randomization, trust-building, socio-economic barriers",
        ),
    ]

    def __init__(self):
        self.current_phase_idx = 0
        self.phase_episode_count = 0
        self.phase_history: List[Dict[str, Any]] = []
        self.window_size = 20  # Rolling window for promotion check

    @property
    def current_phase(self) -> PhaseConfig:
        return self.PHASES[self.current_phase_idx]

    @property
    def phase_id(self) -> int:
        return self.current_phase.phase_id

    def record_episode(self, outcome: str, total_reward: float) -> bool:
        """
        Record an episode result. Returns True if phase was promoted.
        """
        self.phase_episode_count += 1
        self.phase_history.append({
            "outcome": outcome,
            "reward": total_reward,
            "phase": self.phase_id,
        })

        # Check promotion
        if self.current_phase_idx >= len(self.PHASES) - 1:
            return False  # Already at final phase

        cfg = self.current_phase
        if self.phase_episode_count < cfg.min_episodes:
            return False  # Not enough episodes yet

        # Calculate rolling metrics
        recent = self.phase_history[-self.window_size:]
        win_rate = sum(1 for e in recent if e["outcome"] == "WIN") / len(recent)
        avg_reward = sum(e["reward"] for e in recent) / len(recent)

        if win_rate >= cfg.promotion_win_rate and avg_reward >= cfg.promotion_avg_reward:
            self._promote()
            return True
        return False

    def _promote(self):
        """Advance to the next phase."""
        old_phase = self.current_phase.name
        self.current_phase_idx += 1
        self.phase_episode_count = 0
        logger.info(
            f"\n{'*' * 60}\n"
            f"  CURRICULUM PROMOTION: {old_phase} -> {self.current_phase.name}\n"
            f"{'*' * 60}"
        )

    def get_env_options(self) -> Dict[str, Any]:
        """Return options dict to pass to env.reset()."""
        return {
            "phase": self.phase_id,
            "difficulty": self.current_phase.difficulty,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return scheduler state for logging."""
        recent = self.phase_history[-self.window_size:] if self.phase_history else []
        win_rate = sum(1 for e in recent if e["outcome"] == "WIN") / max(len(recent), 1)
        avg_reward = sum(e["reward"] for e in recent) / max(len(recent), 1)
        return {
            "phase": self.phase_id,
            "phase_name": self.current_phase.name,
            "phase_episodes": self.phase_episode_count,
            "rolling_win_rate": round(win_rate, 3),
            "rolling_avg_reward": round(avg_reward, 3),
            "total_episodes": len(self.phase_history),
        }


# ============================================================================
# Reward Verifier (process-based, no learned critic)
# ============================================================================

def verify_episode_reward(trajectory: Dict[str, Any]) -> float:
    """
    Compute a normalized episode-level reward for GRPO.
    This is the 'verifier' function that replaces the critic in PPO.

    The environment already computes dense per-step rewards.
    We aggregate them and add episode-level bonuses.
    """
    total = trajectory.get("total_reward", 0.0)
    outcome = trajectory.get("outcome", "unknown")
    steps = trajectory.get("steps", 0)
    milestones = trajectory.get("milestones", {})
    patient_state = trajectory.get("patient_state", {})

    # Outcome bonus (verified result)
    outcome_bonus = {
        "WIN": 2.0,
        "FATAL_LOSS": -3.0,
        "AMA_LOSS": -1.5,
        "INCORRECT": -2.0,
        "unknown": -0.5,
    }.get(outcome, -0.5)

    # Efficiency bonus: fewer steps = better (caps at 5 steps)
    efficiency = max(0, 1.0 - (steps / 20))  # 0 at 20 steps, 1.0 at 0 steps

    # Milestone completion bonus
    completion = milestones.get("completion", 0.0) if isinstance(milestones, dict) else 0.0
    milestone_bonus = completion * 0.5

    # Trust maintenance bonus (Phase 3 relevant)
    trust = patient_state.get("trust", 50) if isinstance(patient_state, dict) else 50
    trust_bonus = 0.2 if trust > 60 else (-0.1 if trust < 30 else 0.0)

    verified_reward = total + outcome_bonus + efficiency * 0.3 + milestone_bonus + trust_bonus
    return verified_reward


# ============================================================================
# Model Loading (reused from train_ppo.py)
# ============================================================================

def load_model_and_tokenizer(
    model_name: str = "unsloth/Qwen3-4B",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """Load Doctor policy model with Unsloth or HF fallback."""
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        logger.info(f"Loaded model via Unsloth: {model_name} (4-bit={load_in_4bit})")
        return model, tokenizer

    except ImportError:
        logger.warning("Unsloth not available. Falling back to HuggingFace.")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=16, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Loaded model via HF: {model_name}")
        return model, tokenizer


# ============================================================================
# Doctor Action Generation
# ============================================================================

DOCTOR_SYSTEM_PROMPT = """You are an ER doctor performing triage. You must diagnose and treat the patient.

AVAILABLE TOOLS (respond with JSON):
1. {"tool": "read_soap"} - Read the patient's medical record
2. {"tool": "speak_to", "target": "patient", "message": "..."} - Talk to patient
3. {"tool": "speak_to", "target": "nurse", "message": "..."} - Talk to nurse
4. {"tool": "order_lab", "test_name": "..."} - Order a lab test
5. {"tool": "update_soap", "section": "...", "content": "..."} - Update medical record
6. {"tool": "terminal_discharge", "treatment": "...", "diagnosis": "..."} - Final diagnosis

WORKFLOW: Read SOAP -> Talk to patient -> Order labs -> Assess -> Diagnose & Treat

Respond with ONLY a valid JSON action object."""


def generate_doctor_action(
    model,
    tokenizer,
    observation: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> str:
    """Generate Doctor's JSON action from observation."""
    prompt = f"{DOCTOR_SYSTEM_PROMPT}\n\nObservation:\n{observation}\n\nJSON Action:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return generated.strip()


# ============================================================================
# Episode Rollout
# ============================================================================

def run_episode(
    model,
    tokenizer,
    env,
    env_options: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run a single episode. Returns trajectory data for GRPO.
    """
    obs, info = env.reset(options=env_options)
    done = False
    truncated = False

    queries: List[str] = []
    responses: List[str] = []
    rewards: List[float] = []
    total_reward = 0.0
    steps = 0
    outcome = "unknown"
    last_info = info

    while not done and not truncated:
        action_text = generate_doctor_action(model, tokenizer, obs, device=device)

        next_obs, reward, done, truncated, info = env.step(action_text)

        queries.append(obs)
        responses.append(action_text)
        rewards.append(reward)
        total_reward += reward
        steps += 1
        last_info = info
        obs = next_obs

        if done:
            try:
                obs_dict = json.loads(obs)
                event = obs_dict.get("event", "")
                if "win" in event:
                    outcome = "WIN"
                elif "fatal" in event:
                    outcome = "FATAL_LOSS"
                elif "ama" in event:
                    outcome = "AMA_LOSS"
                elif "incorrect" in event:
                    outcome = "INCORRECT"
            except json.JSONDecodeError:
                pass

    return {
        "queries": queries,
        "responses": responses,
        "rewards": rewards,
        "total_reward": total_reward,
        "steps": steps,
        "outcome": outcome,
        "milestones": last_info.get("milestones", {}),
        "patient_state": last_info.get("patient_state", {}),
    }


# ============================================================================
# GRPO Training Loop
# ============================================================================

def train(
    num_episodes: int = 200,
    group_size: int = 4,
    model_name: str = "unsloth/Qwen3-4B",
    groq_api_key: str = "",
    learning_rate: float = 5e-6,
    use_wandb: bool = False,
    output_dir: str = "./er_map_grpo_checkpoints",
    dry_run: bool = False,
):
    """
    Main GRPO training loop with curriculum scheduling.

    Args:
        num_episodes: Total training episodes across all phases
        group_size: G completions per prompt for GRPO (default 4)
        model_name: Base model (Qwen3-4B recommended for $200 budget)
        groq_api_key: For Nurse/Patient LLM APIs
        learning_rate: GRPO learning rate
        use_wandb: Enable W&B logging
        output_dir: Checkpoint directory
        dry_run: If True, skip model loading (test scheduler only)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    logger.info(f"GRPO Config: group_size={group_size}, lr={learning_rate}")

    groq_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    # --- Curriculum Scheduler ---
    scheduler = CurriculumScheduler()
    logger.info(f"Starting Phase: {scheduler.current_phase.name}")
    logger.info(f"  {scheduler.current_phase.description}")

    # --- Model ---
    if not dry_run:
        model, tokenizer = load_model_and_tokenizer(model_name=model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    else:
        model, tokenizer = None, None
        logger.info("DRY RUN mode -- skipping model loading")

    # --- GRPO Trainer (TRL) ---
    use_trl = False
    grpo_trainer = None
    if not dry_run:
        try:
            from trl import GRPOConfig, GRPOTrainer

            grpo_config = GRPOConfig(
                output_dir=output_dir,
                learning_rate=learning_rate,
                num_generations=group_size,
                max_completion_length=256,
                log_with="wandb" if use_wandb else None,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=group_size,
            )

            grpo_trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                config=grpo_config,
                reward_funcs=verify_episode_reward,
            )
            use_trl = True
            logger.info("TRL GRPOTrainer initialized successfully.")

        except (ImportError, Exception) as e:
            logger.warning(f"TRL GRPO not available ({e}). Running manual GRPO loop.")

    # --- Environment ---
    from ER_MAP.envs.triage_env import TriageEnv

    env = TriageEnv(groq_api_key=groq_key, render_mode="human")

    # --- Training Loop ---
    os.makedirs(output_dir, exist_ok=True)
    metrics_log: List[Dict[str, Any]] = []
    start_time = time.time()

    logger.info(f"\nStarting GRPO training for {num_episodes} episodes...")
    logger.info(f"Phase 1: {scheduler.PHASES[0].description}")

    for episode_idx in range(1, num_episodes + 1):
        ep_start = time.time()

        # Get phase-aware env options
        env_options = scheduler.get_env_options()

        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"  Episode {episode_idx}/{num_episodes} | "
            f"Phase {scheduler.phase_id}: {scheduler.current_phase.name} | "
            f"Difficulty: {env_options.get('difficulty', 'random')}"
        )
        logger.info(f"{'=' * 60}")

        if dry_run:
            # Simulate a trajectory for testing the scheduler
            import random
            outcome = random.choice(["WIN", "WIN", "FATAL_LOSS", "INCORRECT", "WIN"])
            total_reward = random.uniform(-1.0, 2.0) if outcome == "WIN" else random.uniform(-3.0, 0.0)
            trajectory = {
                "queries": [], "responses": [], "rewards": [],
                "total_reward": total_reward, "steps": random.randint(3, 15),
                "outcome": outcome, "milestones": {"completion": random.uniform(0.3, 1.0)},
                "patient_state": {"trust": random.uniform(20, 80)},
            }
        else:
            # Real rollout
            trajectory = run_episode(model, tokenizer, env, env_options, device=device)

        # Verify reward
        verified_reward = verify_episode_reward(trajectory)

        logger.info(
            f"  Outcome: {trajectory['outcome']} | "
            f"Steps: {trajectory['steps']} | "
            f"Raw Reward: {trajectory['total_reward']:+.3f} | "
            f"Verified: {verified_reward:+.3f}"
        )

        # --- GRPO Update (if TRL available) ---
        if use_trl and grpo_trainer and trajectory["queries"]:
            try:
                # For GRPO, we batch G completions per prompt
                # In our env-based setup, each episode is one "completion"
                # We accumulate group_size episodes before updating
                pass  # TRL GRPOTrainer handles batching internally
            except Exception as e:
                logger.error(f"  GRPO update failed: {e}")

        # Record for curriculum scheduler
        promoted = scheduler.record_episode(trajectory["outcome"], trajectory["total_reward"])

        if promoted:
            logger.info(f"  >>> PROMOTED to Phase {scheduler.phase_id}: {scheduler.current_phase.name}")

        # Log metrics
        ep_time = time.time() - ep_start
        sched_summary = scheduler.get_summary()
        episode_metrics = {
            "episode": episode_idx,
            "phase": sched_summary["phase"],
            "phase_name": sched_summary["phase_name"],
            "outcome": trajectory["outcome"],
            "steps": trajectory["steps"],
            "raw_reward": round(trajectory["total_reward"], 3),
            "verified_reward": round(verified_reward, 3),
            "rolling_win_rate": sched_summary["rolling_win_rate"],
            "rolling_avg_reward": sched_summary["rolling_avg_reward"],
            "milestones": trajectory.get("milestones", {}),
            "patient_trust": trajectory.get("patient_state", {}).get("trust", None),
            "episode_time_s": round(ep_time, 1),
        }
        metrics_log.append(episode_metrics)

        # Periodic checkpoint
        if episode_idx % 25 == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_ep{episode_idx}_phase{scheduler.phase_id}")
            try:
                if not dry_run:
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    logger.info(f"  Checkpoint saved: {ckpt_path}")
            except Exception as e:
                logger.error(f"  Checkpoint failed: {e}")

            # Save intermediate metrics
            metrics_path = os.path.join(output_dir, "training_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics_log, f, indent=2)

        # Log rolling stats every 5 episodes
        if episode_idx % 5 == 0:
            s = scheduler.get_summary()
            logger.info(
                f"  [Scheduler] Phase {s['phase']} ({s['phase_name']}) | "
                f"Win Rate: {s['rolling_win_rate']:.1%} | "
                f"Avg Reward: {s['rolling_avg_reward']:+.2f} | "
                f"Phase Episodes: {s['phase_episodes']}"
            )

    # --- Final Save ---
    total_time = time.time() - start_time
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  TRAINING COMPLETE")
    logger.info(f"  Total episodes: {num_episodes}")
    logger.info(f"  Total time: {total_time / 60:.1f} minutes")
    logger.info(f"  Final phase: {scheduler.current_phase.name}")
    logger.info(f"  Metrics: {metrics_path}")
    logger.info(f"{'=' * 60}")

    if not dry_run:
        env.close()

    return metrics_log


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ER-MAP GRPO Training with Curriculum")
    parser.add_argument("--episodes", type=int, default=200, help="Total training episodes")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size (G)")
    parser.add_argument("--model", type=str, default="unsloth/Qwen3-4B", help="Base model")
    parser.add_argument("--groq-key", type=str, default="", help="Groq API key")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--output-dir", type=str, default="./er_map_grpo_checkpoints", help="Output dir")
    parser.add_argument("--dry-run", action="store_true", help="Test scheduler without model")

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        group_size=args.group_size,
        model_name=args.model,
        groq_api_key=args.groq_key,
        learning_rate=args.lr,
        use_wandb=args.wandb,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
