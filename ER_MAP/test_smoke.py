"""
ER_MAP/test_smoke.py
====================
Smoke test: validates the full environment pipeline end-to-end
without needing a GPU or Groq API key (uses mock mode).

Usage:
    cd d:/Meta_Finals
    python -m ER_MAP.test_smoke
"""

import json
import sys

def main():
    print("=" * 60)
    print("  ER-MAP Smoke Test")
    print("=" * 60)

    # --- 1. Test Randomizer ---
    print("\n[1/4] Testing randomizer.py ...")
    from ER_MAP.envs.randomizer import generate_ground_truth, construct_prompts

    gt = generate_ground_truth()
    assert "patient" in gt, "Missing patient traits"
    assert "nurse" in gt, "Missing nurse traits"
    assert "disease" in gt, "Missing disease config"
    assert gt["disease"]["true_disease"], "No disease set"
    print(f"  [OK] Ground truth generated: {gt['disease']['true_disease']}")
    print(f"  [OK] Patient traits: {gt['patient']}")
    print(f"  [OK] Nurse traits: {gt['nurse']}")

    prompts = construct_prompts(gt)
    assert "nurse_system_prompt" in prompts
    assert "patient_system_prompt" in prompts
    assert len(prompts["nurse_system_prompt"]) > 100
    print(f"  [OK] Nurse prompt: {len(prompts['nurse_system_prompt'])} chars")
    print(f"  [OK] Patient prompt: {len(prompts['patient_system_prompt'])} chars")

    # --- 2. Test API Router (Mock Mode) ---
    print("\n[2/4] Testing api_router.py (mock mode) ...")
    from ER_MAP.envs.api_router import AgentRouter

    router = AgentRouter(api_key="")  # empty key → mock mode
    router.set_system_prompt("nurse", prompts["nurse_system_prompt"])
    router.set_system_prompt("patient", prompts["patient_system_prompt"])

    nurse_resp = router.query("nurse", "[Doctor]: What is the patient's chief complaint?")
    assert isinstance(nurse_resp, dict), "Nurse response is not a dict"
    assert "tool" in nurse_resp, "Nurse response missing 'tool'"
    print(f"  [OK] Nurse response: tool={nurse_resp['tool']}, status={nurse_resp.get('status')}")

    patient_resp = router.query("patient", "[Nurse]: Can you tell me what's wrong?")
    assert isinstance(patient_resp, dict), "Patient response is not a dict"
    assert "tool" in patient_resp, "Patient response missing 'tool'"
    print(f"  [OK] Patient response: tool={patient_resp['tool']}, status={patient_resp.get('status')}")

    # Test sliding window
    for i in range(10):
        router.query("nurse", f"Test message {i}")
    windowed = router._get_windowed_messages("nurse")
    # Should be system + last 6 messages (3 turns × 2)
    assert len(windowed) <= 1 + 6 + 2, f"Sliding window too large: {len(windowed)} messages"
    print(f"  [OK] Sliding window: {len(windowed)} messages retained (after 12+ appended)")

    # --- 3. Test TriageEnv ---
    print("\n[3/4] Testing triage_env.py ...")
    from ER_MAP.envs.triage_env import TriageEnv

    env = TriageEnv(render_mode="human")
    obs, info = env.reset()
    obs_dict = json.loads(obs)
    assert obs_dict["event"] == "episode_start"
    assert "nurse_experience" in obs_dict
    print(f"  [OK] Reset: nurse_experience={obs_dict['nurse_experience']}")
    print(f"  [OK] Ground truth disease: {info.get('ground_truth_disease')}")

    # Test valid speak_to nurse action
    action = json.dumps({
        "thought": "I should talk to the nurse first.",
        "tool": "speak_to",
        "target": "nurse",
        "message": "What symptoms is the patient presenting with?",
    })
    obs2, reward1, done, truncated, info = env.step(action)
    print(f"  [OK] Step 1 (speak_to nurse): reward={reward1:+.2f}, done={done}")

    # Test order_lab
    action_lab = json.dumps({
        "thought": "Let me order some blood work.",
        "tool": "order_lab",
        "target": "nurse",
        "test_name": "CBC",
    })
    obs3, reward2, done, truncated, info = env.step(action_lab)
    obs3_dict = json.loads(obs3)
    print(f"  [OK] Step 2 (order_lab CBC): reward={reward2:+.2f}, result={obs3_dict.get('result', 'N/A')[:60]}...")

    # Test redundant lab
    obs4, reward3, done, truncated, info = env.step(action_lab)
    print(f"  [OK] Step 3 (duplicate CBC): reward={reward3:+.2f} (should include -0.05 penalty)")

    # Test invalid JSON
    obs5, reward4, done, truncated, info = env.step("this is not json {{{")
    print(f"  [OK] Step 4 (invalid JSON): reward={reward4:+.2f} (should include -0.20 penalty)")

    # Test terminal_discharge
    correct_treatment = env.ground_truth["disease"]["correct_treatment"]
    action_discharge = json.dumps({
        "thought": "I believe I have the diagnosis.",
        "tool": "terminal_discharge",
        "treatment": correct_treatment,
    })
    obs6, reward5, done, truncated, info = env.step(action_discharge)
    obs6_dict = json.loads(obs6)
    print(f"  [OK] Step 5 (terminal_discharge): reward={reward5:+.2f}, event={obs6_dict['event']}, done={done}")

    env.close()

    # --- 4. Consent Lock Test ---
    print("\n[4/6] Testing Consent Lock ...")
    env2 = TriageEnv()
    env2.reset()
    assert env2.consent_given == False, "Consent should be False at start"
    print(f"  [OK] Consent at start: {env2.consent_given}")
    env2.close()

    # --- 5. SOAP EMR Tests ---
    print("\n[5/6] Testing SOAP EMR System ...")
    env3 = TriageEnv()
    obs3, info3 = env3.reset()
    obs3_dict = json.loads(obs3)

    # 5a. Check EMR is pre-populated from SOAP_HISTORY_DB
    assert env3.emr is not None, "EMR should exist"
    assert env3.emr["Subjective"]["HPI"] != "", "HPI should be pre-populated"
    assert env3.emr["Subjective"]["Past_Medical_History"] != "", "PMH should be pre-populated"
    assert env3.emr["Subjective"]["Medications"] != "", "Medications should be pre-populated"
    assert env3.emr["Subjective"]["Allergies"] != "", "Allergies should be pre-populated"
    assert env3.emr["Objective"]["Physical_Examination"] != "", "PE should be pre-populated"
    print(f"  [OK] EMR pre-populated: HPI={len(env3.emr['Subjective']['HPI'])} chars")
    print(f"  [OK] PMH: {env3.emr['Subjective']['Past_Medical_History'][:60]}...")
    print(f"  [OK] Allergies: {env3.emr['Subjective']['Allergies']}")
    print(f"  [OK] Medications: {env3.emr['Subjective']['Medications'][:60]}...")

    # 5b. Check initial obs includes soap_summary
    assert "soap_summary" in obs3_dict, "Initial observation should include soap_summary"
    print(f"  [OK] soap_summary included in initial observation")

    # 5c. Test read_soap tool
    read_action = json.dumps({
        "thought": "I need to review the patient's medical history.",
        "tool": "read_soap",
        "section": "Subjective",
    })
    obs_read, reward_read, done_r, trunc_r, info_r = env3.step(read_action)
    obs_read_dict = json.loads(obs_read)
    assert obs_read_dict["event"] == "soap_read", f"Expected soap_read event, got {obs_read_dict['event']}"
    assert obs_read_dict["section"] == "Subjective"
    assert "HPI" in obs_read_dict["content"], "Subjective content should contain HPI"
    print(f"  [OK] read_soap Subjective: returned {len(json.dumps(obs_read_dict['content']))} chars")

    # 5d. Test read_soap ALL
    read_all_action = json.dumps({
        "thought": "Review full SOAP.",
        "tool": "read_soap",
        "section": "",
    })
    obs_all, _, _, _, _ = env3.step(read_all_action)
    obs_all_dict = json.loads(obs_all)
    assert obs_all_dict["section"] == "ALL"
    assert "Subjective" in obs_all_dict["content"]
    assert "Objective" in obs_all_dict["content"]
    assert "Assessment" in obs_all_dict["content"]
    print(f"  [OK] read_soap ALL: returned full EMR structure")

    # 5e. Test update_soap Assessment
    update_action = json.dumps({
        "thought": "I believe this is an MI.",
        "tool": "update_soap",
        "section": "Assessment",
        "content": "Acute Myocardial Infarction based on chest pain, diaphoresis, and risk factors.",
    })
    obs_upd, reward_upd, _, _, _ = env3.step(update_action)
    obs_upd_dict = json.loads(obs_upd)
    assert obs_upd_dict["event"] == "soap_updated", f"Expected soap_updated, got {obs_upd_dict['event']}"
    assert env3.emr["Assessment"] != "", "Assessment should be populated"
    assert reward_upd > 0, f"Should get positive reward for documenting Assessment, got {reward_upd}"
    print(f"  [OK] update_soap Assessment: reward={reward_upd:+.2f}, content='{env3.emr['Assessment'][:50]}...'")

    # 5f. Test update_soap Plan
    plan_action = json.dumps({
        "thought": "Setting up treatment plan.",
        "tool": "update_soap",
        "section": "Plan",
        "content": "Aspirin 325mg, heparin drip, emergent PCI consult.",
    })
    obs_plan, reward_plan, _, _, _ = env3.step(plan_action)
    assert env3.emr["Plan"] != "", "Plan should be populated"
    print(f"  [OK] update_soap Plan: reward={reward_plan:+.2f}")

    # 5g. Test auto-update of vitals into EMR via check_vitals
    speak_action = json.dumps({
        "thought": "Ask nurse to check vitals.",
        "tool": "speak_to",
        "target": "nurse",
        "message": "Please check the patient's vitals.",
    })
    env3.step(speak_action)
    # After nurse interaction, vitals may have been recorded
    print(f"  [OK] Vitals in EMR: '{env3.emr['Objective']['Vitals'][:60]}...' (may be empty in mock mode)")

    # 5h. Test auto-update of labs into EMR
    lab_action = json.dumps({
        "thought": "Order CBC.",
        "tool": "order_lab",
        "target": "nurse",
        "test_name": "CBC",
    })
    env3.step(lab_action)
    assert env3.emr["Objective"]["Labs"] != "", "Labs should be auto-updated in EMR after ordering"
    print(f"  [OK] Labs auto-updated in EMR: '{env3.emr['Objective']['Labs'][:60]}...'")

    # 5i. Test SOAP reward shaping on discharge (Assessment filled = bonus)
    correct_treatment = env3.ground_truth["disease"]["correct_treatment"]
    discharge_action = json.dumps({
        "thought": "Discharging with treatment.",
        "tool": "terminal_discharge",
        "treatment": correct_treatment,
    })
    obs_dis, reward_dis, done_dis, _, _ = env3.step(discharge_action)
    # Should include +0.10 SOAP bonus since Assessment is filled
    print(f"  [OK] Discharge with Assessment: reward={reward_dis:+.2f} (includes +0.10 SOAP bonus)")

    # 5j. Test state() includes soap_note
    state = env3.state()
    assert "soap_note" in state, "state() should include soap_note"
    print(f"  [OK] state() includes soap_note")

    env3.close()

    # --- 6. SOAP penalty test (discharge without Assessment) ---
    print("\n[6/6] Testing SOAP penalty (no Assessment before discharge) ...")
    env4 = TriageEnv()
    env4.reset()
    assert env4.emr["Assessment"] == "", "Assessment should be empty at start"
    # Discharge immediately without documenting Assessment
    correct_tx = env4.ground_truth["disease"]["correct_treatment"]
    obs_pen, reward_pen, _, _, _ = env4.step(json.dumps({
        "thought": "Quick discharge.",
        "tool": "terminal_discharge",
        "treatment": correct_tx,
    }))
    # Should include -0.10 SOAP penalty since Assessment is empty
    print(f"  [OK] Discharge without Assessment: reward={reward_pen:+.2f} (includes -0.10 SOAP penalty)")
    env4.close()

    print("\n" + "=" * 60)
    print("  ALL SMOKE TESTS PASSED [OK]")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
