import os
import sys
import json
import time
import argparse
from colorama import init, Fore, Style

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ER_MAP.envs.triage_env import TriageEnv

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

init(autoreset=True)

def print_header(title):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*60}")
    print(f"{Fore.CYAN}{Style.BRIGHT} {title}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'='*60}")

def run_automated_cli(phase: int):
    groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("GROQ_PATIENT_API_KEY")
    if not groq_key or not GROQ_AVAILABLE:
        print(f"{Fore.RED}ERROR: GROQ_API_KEY environment variable required and 'groq' package must be installed.")
        return

    client = Groq(api_key=groq_key)
    
    print(f"{Fore.YELLOW}Initializing Environment (Phase {phase})...")
    env = TriageEnv(render_mode="human")
    
    obs_json, info = env.reset(options={"phase": phase})
    gt = env.ground_truth

    print_header(f"GROUND TRUTH GENERATED (PHASE {phase})")
    print(f"{Fore.MAGENTA}Disease: {gt['disease']['true_disease']} | Diff: {gt['disease']['difficulty']} | Emergency: {gt['disease'].get('is_emergency', False)}")
    print(f"Correct Tx: {gt['disease']['correct_treatment']}")
    
    print(f"\n{Fore.BLUE}Patient: {gt['patient']['communication']}, {gt['patient']['compliance']}, {gt['patient']['literacy']}")
    print(f"{Fore.GREEN}Nurse: {gt['nurse']['experience']}, {gt['nurse']['bandwidth']}, {gt['nurse']['empathy']}")
    
    print_header("AUTOMATED DOCTOR AGENT RUNNING (70B)")
    
    messages = [
        {"role": "system", "content": "You are the Doctor. You must output ONLY valid JSON matching the exact schema requested in your prompt. Tools available: speak_to, order_lab, read_soap, update_soap, terminal_discharge."}
    ]
    
    step_count = 0
    max_steps = 30
    cumulative_reward = 0.0
    
    while step_count < max_steps:
        # Provide the observation
        messages.append({"role": "user", "content": obs_json})
        
        print(f"\n{Fore.YELLOW}--- Step {step_count + 1} ---")
        print(f"{Fore.WHITE}Doctor 70B is thinking...")
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            raw_action = completion.choices[0].message.content
            messages.append({"role": "assistant", "content": raw_action})
            
            # Print Action
            try:
                action_parsed = json.loads(raw_action)
                print(f"{Fore.CYAN}💭 Thought: {action_parsed.get('thought', '')}")
                print(f"{Fore.GREEN}🛠️  Action:  {action_parsed.get('tool', '')} -> {json.dumps({k:v for k,v in action_parsed.items() if k not in ['thought', 'tool']})}")
            except:
                print(f"{Fore.GREEN}🛠️  Action (Raw): {raw_action}")
                
            # Execute step
            obs_json, reward, done, truncated, info = env.step(raw_action)
            cumulative_reward += reward
            print(f"{Fore.MAGENTA}🪙  Step Reward: {reward:.2f} | Total: {cumulative_reward:.2f}")
            
            if done or truncated:
                print(f"\n{Fore.RED}{Style.BRIGHT}=== EPISODE FINISHED ===")
                try:
                    final_obs = json.loads(obs_json)
                    print(f"Outcome: {final_obs.get('event')}")
                    print(f"Message: {final_obs.get('message')}")
                    if 'match_ratio' in final_obs:
                        print(f"Match Ratio: {final_obs['match_ratio']:.0%}")
                except:
                    print(f"Final Obs: {obs_json}")
                break
                
        except Exception as e:
            print(f"{Fore.RED}LLM Error: {e}")
            break
            
        step_count += 1
        time.sleep(1)

    if step_count >= max_steps:
        print(f"\n{Fore.RED}{Style.BRIGHT}=== MAX STEPS REACHED ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER-MAP Automated CLI Tester")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2, 3], help="Curriculum phase (1-3)")
    args = parser.parse_args()
    
    try:
        run_automated_cli(args.phase)
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
