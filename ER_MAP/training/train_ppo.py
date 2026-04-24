"""
ER_MAP/training/train_ppo.py
=============================
Minimal PPO Training Script for the ER-MAP Triage Environment.
Designed to run in Google Colab with Unsloth + HuggingFace TRL.

Usage (Colab):
    !pip install unsloth trl transformers datasets accelerate peft
    !pip install gymnasium groq
    %run train_ppo.py

Usage (Local):
    python -m ER_MAP.training.train_ppo
"""

import os
import json
import torch
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ER_MAP.train_ppo")

# ---------------------------------------------------------------------------
# 1. Load the Base Policy Model with Unsloth (4-bit quantization)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str = "unsloth/llama-3-8b-Instruct",
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
):
    """
    Load the Doctor policy model using Unsloth for efficient 4-bit inference.
    Falls back to standard HuggingFace loading if Unsloth is unavailable.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            dtype=None,  # auto-detect
        )

        # Apply LoRA adapters for PPO fine-tuning
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
        logger.warning("Unsloth not available. Falling back to HuggingFace Transformers.")
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from peft import get_peft_model, LoraConfig  # type: ignore

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Loaded model via HF Transformers: {model_name}")
        return model, tokenizer


# ---------------------------------------------------------------------------
# 2. Doctor Action Generation
# ---------------------------------------------------------------------------

def generate_doctor_action(
    model,
    tokenizer,
    observation: str,
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> str:
    """
    Given an observation string from the environment, generate the Doctor's
    JSON action using the policy model.
    """
    prompt = f"""You are an ER doctor performing triage. Based on the observation below, 
respond with a valid JSON action.

Valid tools: speak_to, order_lab, terminal_discharge
Valid targets: nurse, patient

Observation:
{observation}

Respond ONLY with a JSON object:
{{"thought": "your reasoning", "tool": "...", "target": "...", "message": "...", "test_name": "...", "treatment": "..."}}

JSON Action:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


# ---------------------------------------------------------------------------
# 3. Rollout: Play One Episode
# ---------------------------------------------------------------------------

def run_episode(
    model,
    tokenizer,
    env,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run a single episode of the Doctor interacting with the TriageEnv.
    Collects trajectory data for PPO training.

    Returns:
        dict with keys: queries, responses, rewards, total_reward, steps, outcome
    """
    obs, info = env.reset()
    done = False
    truncated = False

    queries: List[str] = []
    responses: List[str] = []
    rewards: List[float] = []
    total_reward = 0.0
    steps = 0
    outcome = "unknown"

    while not done and not truncated:
        # Generate Doctor action
        action_text = generate_doctor_action(model, tokenizer, obs, device=device)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action_text)

        queries.append(obs)
        responses.append(action_text)
        rewards.append(reward)
        total_reward += reward
        steps += 1

        obs = next_obs

        # Determine outcome
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
    }


# ---------------------------------------------------------------------------
# 4. PPO Training Loop
# ---------------------------------------------------------------------------

def train(
    num_episodes: int = 100,
    model_name: str = "unsloth/llama-3-8b-Instruct",
    groq_api_key: str = "",
    learning_rate: float = 1.41e-5,
    batch_size: int = 4,
    mini_batch_size: int = 1,
    ppo_epochs: int = 4,
    use_wandb: bool = False,
    output_dir: str = "./er_map_checkpoints",
):
    """
    Main PPO training loop:
    1. Load Doctor policy model (Unsloth, 4-bit).
    2. Initialize TRL PPOTrainer.
    3. Roll out episodes in TriageEnv.
    4. Update policy via PPO.
    """
    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    groq_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")

    # --- Load Model ---
    model, tokenizer = load_model_and_tokenizer(model_name=model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Initialize TRL PPO ---
    try:
        from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead  # type: ignore

        ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            ppo_epochs=ppo_epochs,
            log_with="wandb" if use_wandb else None,
            output_dir=output_dir,
        )

        # Wrap model with value head for PPO
        model_with_value_head = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        
        ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=model_with_value_head,
            tokenizer=tokenizer,
        )
        logger.info("TRL PPOTrainer initialized successfully.")
        use_trl = True

    except ImportError:
        logger.warning("TRL not available. Running in evaluation-only mode (no PPO updates).")
        use_trl = False

    # --- Initialize Environment ---
    from ER_MAP.envs.triage_env import TriageEnv

    env = TriageEnv(groq_api_key=groq_key, render_mode="human")

    # --- Training Loop ---
    os.makedirs(output_dir, exist_ok=True)
    metrics_log: List[Dict[str, Any]] = []

    logger.info(f"Starting training for {num_episodes} episodes...")
    for episode_idx in range(1, num_episodes + 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"  Episode {episode_idx}/{num_episodes}")
        logger.info(f"{'='*50}")

        # Run one episode
        trajectory = run_episode(model, tokenizer, env, device=device)

        logger.info(
            f"  Outcome: {trajectory['outcome']} | "
            f"Steps: {trajectory['steps']} | "
            f"Total Reward: {trajectory['total_reward']:+.3f}"
        )

        # --- PPO Update ---
        if use_trl and trajectory["queries"]:
            try:
                # Tokenize queries and responses for PPO
                query_tensors = [
                    tokenizer.encode(q, return_tensors="pt", truncation=True, max_length=512).squeeze().to(device)
                    for q in trajectory["queries"]
                ]
                response_tensors = [
                    tokenizer.encode(r, return_tensors="pt", truncation=True, max_length=256).squeeze().to(device)
                    for r in trajectory["responses"]
                ]
                reward_tensors = [torch.tensor([r], device=device) for r in trajectory["rewards"]]

                # PPO step
                stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensors)
                logger.info(f"  PPO Loss: {stats.get('ppo/loss/total', 'N/A')}")
            except Exception as e:
                logger.error(f"  PPO update failed: {e}")

        # Log metrics
        episode_metrics = {
            "episode": episode_idx,
            "outcome": trajectory["outcome"],
            "steps": trajectory["steps"],
            "total_reward": trajectory["total_reward"],
        }
        metrics_log.append(episode_metrics)

        # Periodic checkpoint
        if episode_idx % 10 == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint_ep{episode_idx}")
            try:
                if use_trl:
                    ppo_trainer.save_pretrained(ckpt_path)
                else:
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                logger.info(f"  Checkpoint saved: {ckpt_path}")
            except Exception as e:
                logger.error(f"  Failed to save checkpoint: {e}")

    # --- Save Final Metrics ---
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_log, f, indent=2)
    logger.info(f"\nTraining complete! Metrics saved to {metrics_path}")

    env.close()
    return metrics_log


# ---------------------------------------------------------------------------
# 5. Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ER-MAP PPO Training")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--model", type=str, default="unsloth/llama-3-8b-Instruct", help="Base model name")
    parser.add_argument("--groq-key", type=str, default="", help="Groq API key")
    parser.add_argument("--lr", type=float, default=1.41e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="PPO batch size")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--output-dir", type=str, default="./er_map_checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        model_name=args.model,
        groq_api_key=args.groq_key,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        use_wandb=args.wandb,
        output_dir=args.output_dir,
    )
