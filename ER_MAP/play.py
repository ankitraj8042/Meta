"""
ER_MAP/play.py
==============
Interactive manual play mode. YOU are the Doctor.
Diagnose and treat the patient by typing JSON commands.
Nurse and Patient respond with realistic emotion-induced voice.

Usage:
    cd d:/Meta_Finals
    python -m ER_MAP.play
    python -m ER_MAP.play --no-voice
"""

import json
import os
import sys
import argparse

def print_banner():
    print()
    print("=" * 64)
    print("   ER-MAP: You Are The Doctor")
    print("   Diagnose the patient. Save a life.")
    print("=" * 64)
    print()
    print("TOOLS AVAILABLE:")
    print("  1. speak_to  - Talk to the nurse or patient")
    print("  2. order_lab - Order a lab test (CBC, troponin, ABG, etc.)")
    print("  3. terminal_discharge - Discharge with a treatment (ENDS GAME)")
    print("  4. read_soap - Read the patient's SOAP note / EMR")
    print("  5. update_soap - Update SOAP Assessment or Plan")
    print()
    print("QUICK COMMANDS (type the number instead of JSON):")
    print("  1  = Speak to nurse")
    print("  2  = Speak to patient")
    print("  3  = Order a lab")
    print("  4  = Discharge with treatment")
    print("  5  = Read SOAP note")
    print("  6  = Update SOAP (Assessment/Plan)")
    print("  q  = Quit")
    print("  s  = Show current state (labs ordered, consent, etc.)")
    print("-" * 64)


def build_action_from_shortcut(shortcut):
    """Convert quick-command shortcuts into full JSON actions."""
    if shortcut == "1":
        msg = input("  >> What do you say to the NURSE? > ")
        return json.dumps({
            "thought": "Doctor speaking to nurse.",
            "tool": "speak_to",
            "target": "nurse",
            "message": msg,
        })
    elif shortcut == "2":
        msg = input("  >> What do you say to the PATIENT? > ")
        return json.dumps({
            "thought": "Doctor speaking to patient directly.",
            "tool": "speak_to",
            "target": "patient",
            "message": msg,
        })
    elif shortcut == "3":
        test = input("  >> Which lab test? (e.g. CBC, troponin, ABG, BMP, D-dimer, CSF) > ")
        return json.dumps({
            "thought": "Ordering a lab test.",
            "tool": "order_lab",
            "target": "nurse",
            "test_name": test,
        })
    elif shortcut == "4":
        treatment = input("  >> Your treatment plan? > ")
        return json.dumps({
            "thought": "Discharging with treatment.",
            "tool": "terminal_discharge",
            "treatment": treatment,
        })
    elif shortcut == "5":
        section = input("  >> Which section? (blank=ALL, or: Subjective, Objective, Assessment, Plan) > ").strip()
        return json.dumps({
            "thought": "Reading SOAP note.",
            "tool": "read_soap",
            "section": section,
        })
    elif shortcut == "6":
        section = input("  >> Section to update (Assessment, Plan, Subjective.HPI, etc.) > ").strip()
        content = input("  >> Content > ")
        return json.dumps({
            "thought": "Updating SOAP note.",
            "tool": "update_soap",
            "section": section,
            "content": content,
        })
    return None


def pretty_print_obs(obs_str):
    """Print observation in a readable format."""
    try:
        obs = json.loads(obs_str)
        event = obs.get("event", "unknown")
        print()
        print(f"  [{event.upper()}]")

        if event == "episode_start":
            print(f"  Nurse Experience: {obs.get('nurse_experience')}")
            print(f"  {obs.get('message', '')}")

        elif event == "nurse_report":
            print(f"  Nurse says: {obs.get('nurse_message', '')}")
            print(f"  Nurse status: {obs.get('nurse_status', '')}")
            print(f"  Patient status: {obs.get('patient_status', '')}")
            exchanges = obs.get("internal_exchanges", [])
            if exchanges:
                print(f"  --- Internal Nurse/Patient exchanges ---")
                for ex in exchanges:
                    for k, v in ex.items():
                        print(f"    {k}: {v}")

        elif event == "patient_response":
            print(f"  Patient says: {obs.get('patient_message', '')}")
            print(f"  Patient status: {obs.get('patient_status', '')}")

        elif event == "lab_result":
            print(f"  Test: {obs.get('test_name', '')}")
            print(f"  Result: {obs.get('result', '')}")
            if obs.get("redundant"):
                print(f"  ** DUPLICATE ORDER **")

        elif event == "terminal_win":
            print(f"  CORRECT! Patient stabilized.")
            print(f"  Disease was: {obs.get('ground_truth', '')}")

        elif event == "terminal_fatal":
            print(f"  FATAL ERROR. Patient died.")
            print(f"  Disease was: {obs.get('ground_truth', '')}")

        elif event == "terminal_incorrect":
            print(f"  WRONG treatment.")
            print(f"  Disease was: {obs.get('ground_truth', '')}")
            print(f"  Correct treatment: {obs.get('correct_treatment', '')}")

        elif event == "terminal_ama":
            print(f"  Patient LEFT against medical advice!")
            print(f"  Patient said: {obs.get('patient_message', '')}")

        elif event == "soap_read":
            section = obs.get('section', 'ALL')
            print(f"  --- SOAP NOTE ({section}) ---")
            content = obs.get('content', {})
            if isinstance(content, dict):
                _print_soap_dict(content)
            else:
                print(f"  {content}")
            print(f"  --- END SOAP ---")

        elif event == "soap_updated":
            print(f"  SOAP Updated: {obs.get('section', '')}")
            print(f"  {obs.get('message', '')}")

        elif event == "system_error":
            print(f"  ERROR: {obs.get('message', '')}")

        else:
            for k, v in obs.items():
                print(f"  {k}: {v}")
    except json.JSONDecodeError:
        print(f"  {obs_str}")


def _print_soap_dict(d, indent=2):
    """Recursively print a SOAP dict with indentation."""
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{prefix}{k}:")
            _print_soap_dict(v, indent + 1)
        elif isinstance(v, str) and v:
            # Wrap long strings
            if len(v) > 80:
                print(f"{prefix}{k}:")
                print(f"{prefix}  {v}")
            else:
                print(f"{prefix}{k}: {v}")
        elif v:  # non-empty non-string
            print(f"{prefix}{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="ER-MAP: You Are The Doctor")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile",
                        help="Groq model for Nurse/Patient (default: llama-3.3-70b-versatile)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable TTS voice output")
    args = parser.parse_args()

    from ER_MAP.envs.triage_env import TriageEnv

    # Initialize TTS Engine (Nurse/Patient speak aloud, Doctor is the human)
    tts = None
    if not args.no_voice:
        try:
            from ER_MAP.tts_engine import TTSEngine
            tts = TTSEngine()
            print(f"  🔊 Voice: {'ElevenLabs' if tts.use_elevenlabs else 'Edge-TTS'}")
        except Exception as e:
            print(f"  [TTS init failed: {e}] Running without voice.")

    nurse_key = os.environ.get("GROQ_NURSE_API_KEY", "")
    patient_key = os.environ.get("GROQ_PATIENT_API_KEY", "")
    shared_key = os.environ.get("GROQ_API_KEY", "")

    has_any_key = nurse_key or patient_key or shared_key

    print_banner()
    if has_any_key:
        print(f"  Nurse:   {'LIVE' if (nurse_key or shared_key) else 'MOCK'}")
        print(f"  Patient: {'LIVE' if (patient_key or shared_key) else 'MOCK'}")
    else:
        print("  Mode: MOCK (offline)")
        print("  Tip: set GROQ_NURSE_API_KEY and GROQ_PATIENT_API_KEY for real LLM responses")
    print(f"  Model:   {args.model}")
    print(f"  Voice:   {'ON' if tts else 'OFF'}")
    print()

    env = TriageEnv(
        groq_api_key=shared_key,
        nurse_api_key=nurse_key,
        patient_api_key=patient_key,
        model=args.model,
    )
    obs, info = env.reset()

    # Show persona traits
    gt = env.ground_truth
    print(f"  [SECRET - Disease: {info.get('ground_truth_disease', '???')}]")
    print(f"  [Difficulty: {gt.get('difficulty', 'random').upper()}]")
    print()
    print("  PATIENT PERSONA:")
    for trait, val in gt["patient"].items():
        print(f"    {trait:20s} : {val}")
    print()
    print("  NURSE PERSONA:")
    for trait, val in gt["nurse"].items():
        print(f"    {trait:20s} : {val}")
    print()
    pretty_print_obs(obs)

    total_reward = 0.0
    step = 0

    while True:
        print()
        print(f"--- Step {step + 1} | Total Reward: {total_reward:+.2f} ---")
        user_input = input("  >> Your action (1-6/q/s or raw JSON): ").strip()

        if user_input.lower() == "q":
            print("\n  Quitting. Final reward: {:.2f}".format(total_reward))
            break

        if user_input.lower() == "s":
            print(f"  Labs ordered: {list(env.ordered_labs)}")
            print(f"  Consent given: {env.consent_given}")
            print(f"  Patient status: {env.last_patient_status}")
            print(f"  Steps taken: {env.step_count}")
            continue

        # Build action
        if user_input in ("1", "2", "3", "4", "5", "6"):
            action = build_action_from_shortcut(user_input)
            if action is None:
                continue
        elif user_input.startswith("{"):
            action = user_input
        else:
            print("  Invalid input. Use 1-4, q, s, or paste raw JSON.")
            continue

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        print(f"  Reward this step: {reward:+.2f}")
        pretty_print_obs(obs)

        # 🔊 Speak Nurse/Patient responses (Doctor is the human)
        if tts:
            tts.speak_observation(obs, gt)

        if done:
            print()
            print("=" * 64)
            print(f"  GAME OVER | Total Reward: {total_reward:+.2f}")
            print(f"  Disease was: {env.ground_truth['disease']['true_disease']}")
            print(f"  Correct treatment: {env.ground_truth['disease']['correct_treatment']}")
            print("=" * 64)
            break

        if truncated:
            print()
            print("  TRUNCATED: Max steps reached.")
            print(f"  Final reward: {total_reward:+.2f}")
            break

    if tts:
        tts.close()
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
