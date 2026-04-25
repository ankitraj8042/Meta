"""
ER_MAP/envs/randomizer.py
=========================
Ground Truth Generator & System Prompt Builder.
Handles domain randomization for Patient/Nurse persona traits and
pairs them with a randomly selected Disease configuration.
Supports 50 diseases (10 classes x 5), 17,280+ persona combos,
3 difficulty tiers, and 3-phase curriculum noise injection.
"""

import random
import copy
from typing import Dict, Any, Optional, List

# Import the 50-disease database
from ER_MAP.envs.disease_db import (
    DISEASES_DB as _DISEASES_DB,
    VITALS_DB as _VITALS_DB,
    LAB_RESULTS_DB as _LAB_RESULTS_DB,
    SOAP_HISTORY_DB as _SOAP_HISTORY_DB,
)

# ---------------------------------------------------------------------------
# Patient Persona Trait Arrays
# ---------------------------------------------------------------------------
PATIENT_FINANCIAL = ["poor_uninsured", "average", "wealthy_insured"]
PATIENT_COMMUNICATION = [
    "hostile_aggressive",
    "anxious_panicked",
    "calm_stoic",
    "disorganized_confused",
]
PATIENT_COMPLIANCE = [
    "fully_compliant",
    "partially_compliant",
    "cost_constrained",
    "non_compliant",
]
PATIENT_LITERACY = ["high_expert", "webmd_warrior", "low_basic", "nil_clueless"]
PATIENT_SYMPTOM_STYLE = [
    "accurate_precise",
    "vague_under_reported",
    "exaggerated_catastrophic",
    "storyteller_oversharer",
]

# ---------------------------------------------------------------------------
# Nurse Persona Trait Arrays
# ---------------------------------------------------------------------------
NURSE_EXPERIENCE = ["rookie", "standard", "veteran"]
NURSE_BANDWIDTH = ["idle_fast", "overworked_exhausted", "distracted"]
NURSE_COMMUNICATION = ["concise_robotic", "verbose_panicked", "skeptical_questioning"]
NURSE_EMPATHY = ["high_empathy", "cold_clinical", "impatient_abrasive"]

# ---------------------------------------------------------------------------
# Difficulty Tier Definitions
# ---------------------------------------------------------------------------
# Each tier defines weighted probability pools for patient traits.
# EASY patients cooperate. HARD patients fight you at every step.

DIFFICULTY_TIERS = {
    "easy": {
        "financial": ["average", "wealthy_insured", "wealthy_insured"],
        "communication": ["calm_stoic", "calm_stoic", "anxious_panicked"],
        "compliance": ["fully_compliant", "fully_compliant", "partially_compliant"],
        "literacy": ["high_expert", "high_expert", "webmd_warrior"],
        "symptom_style": ["accurate_precise", "accurate_precise", "storyteller_oversharer"],
        "nurse_experience": ["veteran", "standard", "standard"],
        "nurse_bandwidth": ["idle_fast", "idle_fast", "overworked_exhausted"],
        "nurse_empathy": ["high_empathy", "high_empathy", "cold_clinical"],
    },
    "medium": {
        "financial": ["poor_uninsured", "average", "average"],
        "communication": ["anxious_panicked", "anxious_panicked", "disorganized_confused"],
        "compliance": ["partially_compliant", "cost_constrained", "partially_compliant"],
        "literacy": ["webmd_warrior", "low_basic", "webmd_warrior"],
        "symptom_style": ["vague_under_reported", "exaggerated_catastrophic", "storyteller_oversharer"],
        "nurse_experience": ["standard", "rookie", "standard"],
        "nurse_bandwidth": ["overworked_exhausted", "distracted", "idle_fast"],
        "nurse_empathy": ["cold_clinical", "high_empathy", "impatient_abrasive"],
    },
    "hard": {
        "financial": ["poor_uninsured", "poor_uninsured", "average"],
        "communication": ["hostile_aggressive", "hostile_aggressive", "disorganized_confused"],
        "compliance": ["non_compliant", "non_compliant", "cost_constrained"],
        "literacy": ["nil_clueless", "low_basic", "nil_clueless"],
        "symptom_style": ["vague_under_reported", "exaggerated_catastrophic", "vague_under_reported"],
        "nurse_experience": ["rookie", "rookie", "standard"],
        "nurse_bandwidth": ["overworked_exhausted", "distracted", "distracted"],
        "nurse_empathy": ["impatient_abrasive", "cold_clinical", "impatient_abrasive"],
    },
}

# ---------------------------------------------------------------------------
# Phase-Based Persona Constraints
# Phase 1: compliant patients, standard nurses, no friction
# Phase 2: mixed compliance, clinical noise in SOAP
# Phase 3: full persona randomization, behavioral friction, socio-economic
# ---------------------------------------------------------------------------
PHASE_PERSONA_CONSTRAINTS = {
    1: {
        "financial": ["average", "wealthy_insured"],
        "communication": ["calm_stoic"],
        "compliance": ["fully_compliant"],
        "literacy": ["high_expert", "webmd_warrior"],
        "symptom_style": ["accurate_precise"],
        "nurse_experience": ["standard", "veteran"],
        "nurse_bandwidth": ["idle_fast"],
        "nurse_empathy": ["high_empathy"],
    },
    2: {
        "financial": ["poor_uninsured", "average", "wealthy_insured"],
        "communication": ["calm_stoic", "anxious_panicked", "disorganized_confused"],
        "compliance": ["fully_compliant", "partially_compliant", "cost_constrained"],
        "literacy": ["high_expert", "webmd_warrior", "low_basic"],
        "symptom_style": ["accurate_precise", "vague_under_reported", "storyteller_oversharer"],
        "nurse_experience": ["rookie", "standard", "veteran"],
        "nurse_bandwidth": ["idle_fast", "overworked_exhausted"],
        "nurse_empathy": ["high_empathy", "cold_clinical"],
    },
    3: {  # Full randomization -- all traits available
        "financial": PATIENT_FINANCIAL,
        "communication": PATIENT_COMMUNICATION,
        "compliance": PATIENT_COMPLIANCE,
        "literacy": PATIENT_LITERACY,
        "symptom_style": PATIENT_SYMPTOM_STYLE,
        "nurse_experience": NURSE_EXPERIENCE,
        "nurse_bandwidth": NURSE_BANDWIDTH,
        "nurse_empathy": NURSE_EMPATHY,
    },
}

# ---------------------------------------------------------------------------
# SOAP Noise Injection Engine
# Phase 1: Clean SOAP -- all data accurate, no diagnosis hints
# Phase 2: Moderate noise -- missing fields, vague data, contradictions
# Phase 3: Heavy noise -- behavioral data injected, unreliable history
# ---------------------------------------------------------------------------

def _apply_soap_noise(soap: Dict[str, Any], phase: int, is_emergency: bool = False) -> Dict[str, Any]:
    """Apply phase-dependent noise to SOAP history. Returns a copy."""
    soap = copy.deepcopy(soap)
    
    if phase <= 1:
        # Phase 1: Clean SOAP but NEVER reveal diagnosis/treatment
        # Remove any mention of the actual disease from HPI
        return soap
    
    if phase >= 2:
        # Phase 2: Introduce clinical noise
        # Randomly degrade some fields
        if random.random() < 0.4:
            soap["Allergies"] = random.choice([
                "Unknown -- patient unable to recall",
                "States 'maybe penicillin, not sure'",
                "Family unsure, no records available",
            ])
        if random.random() < 0.3:
            soap["Medications"] = soap.get("Medications", "") + " (patient unsure of doses, no medication list available)"
        if random.random() < 0.35:
            soap["Past_Medical_History"] = soap.get("Past_Medical_History", "") + ". NOTE: limited records, patient provides inconsistent timeline."
        if random.random() < 0.3 and not is_emergency:
            # Vague ROS (skip for emergencies where ROS is critical)
            ros = soap.get("ROS", {})
            if isinstance(ros, dict) and ros:
                key = random.choice(list(ros.keys()))
                ros[key] = "patient gives vague response, difficult to characterize"
                soap["ROS"] = ros
    
    if phase >= 3:
        # Phase 3: Behavioral and socio-economic noise
        if is_emergency:
            behavioral_notes = [
                "EMS handoff was chaotic and rushed, some details missing.",
                "Family members are screaming in the hallway, making it hard to obtain history.",
                "Patient is too unstable to provide complete history; relying on bystander accounts.",
                "Police brought patient in, no ID or medical records available.",
                "Multiple traumas arriving simultaneously, triage is overwhelmed.",
                "Patient unresponsive, unable to obtain comprehensive review of systems.",
            ]
        else:
            behavioral_notes = [
                "Patient appears anxious about cost of treatment, asks repeatedly about billing.",
                "Patient's family member is hostile, demanding immediate answers.",
                "Patient is tearful, expressing fear of losing job if hospitalized.",
                "Patient requests to leave AMA, states cannot afford to miss work.",
                "Language barrier noted -- communicating through teenage child as interpreter.",
                "Patient appears intoxicated, history unreliable per triage nurse.",
                "Patient is homeless, uncertain of medication history.",
                "Patient brought by police from shelter, no ID or insurance card.",
            ]
        soap["Social_History"] = soap.get("Social_History", "") + ". " + random.choice(behavioral_notes)
        
        # Inject conflicting/misleading physical exam findings
        if random.random() < 0.3 and not is_emergency:
            soap["Physical_Examination"] = soap.get("Physical_Examination", "") + " Patient is uncooperative with portions of exam."
        
        # Degrade HPI reliability
        if random.random() < 0.4:
            if is_emergency:
                hpi_noise = random.choice([
                    " Historian is an unknown bystander, limited knowledge of medical history.",
                    " HPI obtained rapidly during active resuscitation.",
                    " Pre-hospital intervention details unclear from EMS."
                ])
            else:
                hpi_noise = random.choice([
                    " Historian is patient's neighbor, limited knowledge of medical history.",
                    " Patient gives contradictory timeline, unclear onset.",
                    " History obtained through interpreter, possible miscommunication.",
                ])
            soap["HPI"] = soap.get("HPI", "") + hpi_noise
    
    return soap


# ---------------------------------------------------------------------------
# 50-Disease Pool (converted from disease_db.py)
# ---------------------------------------------------------------------------
DISEASE_POOL = list(_DISEASES_DB.values())

# Re-export databases for backward compatibility
LAB_RESULTS_DB = _LAB_RESULTS_DB
VITALS_DB = _VITALS_DB
SOAP_HISTORY_DB = _SOAP_HISTORY_DB

# Legacy inline databases removed -- now imported from disease_db.py
# Original 15 diseases have been superseded by 50-disease database.

# Backward compat: keep old variable name
_LEGACY_DISEASE_POOL_REPLACED = True  # marker


def generate_ground_truth(
    difficulty: Optional[str] = None,
    phase: int = 1,
) -> Dict[str, Any]:
    """
    Build a complete ground truth dict by randomly sampling one trait from
    every Patient and Nurse axis, then pairing with a random disease config.

    Args:
        difficulty: "easy", "medium", "hard", or None (fully random).
                    Controls the probability distribution of patient/nurse traits.
        phase: Curriculum phase (1=tool mastery, 2=clinical reasoning, 3=empathy).
               Controls persona constraints and SOAP noise injection.
    """
    disease = random.choice(DISEASE_POOL)
    disease_name = disease["true_disease"]
    phase_constraints = PHASE_PERSONA_CONSTRAINTS.get(phase, PHASE_PERSONA_CONSTRAINTS[3])

    if difficulty and difficulty in DIFFICULTY_TIERS:
        tier = DIFFICULTY_TIERS[difficulty]
        # Phase constraints override difficulty tier for specific axes
        ground_truth: Dict[str, Any] = {
            "patient": {
                "financial": random.choice(phase_constraints["financial"]),
                "communication": random.choice(phase_constraints["communication"]),
                "compliance": random.choice(phase_constraints["compliance"]),
                "literacy": random.choice(phase_constraints.get("literacy", tier["literacy"])),
                "symptom_style": random.choice(phase_constraints.get("symptom_style", tier["symptom_style"])),
            },
            "nurse": {
                "experience": random.choice(phase_constraints["nurse_experience"]),
                "bandwidth": random.choice(phase_constraints["nurse_bandwidth"]),
                "communication": random.choice(NURSE_COMMUNICATION),
                "empathy": random.choice(phase_constraints["nurse_empathy"]),
            },
            "disease": {
                "true_disease": disease_name,
                "true_symptoms": disease["true_symptoms"],
                "medical_history": disease.get("medical_history", ""),
                "correct_treatment": disease["correct_treatment"],
                "lethal_treatments": disease.get("lethal_treatments", []),
                "critical_labs": disease.get("critical_labs", []),
                "difficulty": disease.get("difficulty", "medium"),
                "is_emergency": disease.get("is_emergency", False),
            },
            "difficulty": difficulty,
            "phase": phase,
        }
    else:
        ground_truth = {
            "patient": {
                "financial": random.choice(phase_constraints["financial"]),
                "communication": random.choice(phase_constraints["communication"]),
                "compliance": random.choice(phase_constraints["compliance"]),
                "literacy": random.choice(phase_constraints.get("literacy", PATIENT_LITERACY)),
                "symptom_style": random.choice(phase_constraints.get("symptom_style", PATIENT_SYMPTOM_STYLE)),
            },
            "nurse": {
                "experience": random.choice(phase_constraints["nurse_experience"]),
                "bandwidth": random.choice(phase_constraints["nurse_bandwidth"]),
                "communication": random.choice(NURSE_COMMUNICATION),
                "empathy": random.choice(phase_constraints["nurse_empathy"]),
            },
            "disease": {
                "true_disease": disease_name,
                "true_symptoms": disease["true_symptoms"],
                "medical_history": disease.get("medical_history", ""),
                "correct_treatment": disease["correct_treatment"],
                "lethal_treatments": disease.get("lethal_treatments", []),
                "critical_labs": disease.get("critical_labs", []),
                "difficulty": disease.get("difficulty", "medium"),
                "is_emergency": disease.get("is_emergency", False),
            },
            "difficulty": "random",
            "phase": phase,
        }

    # Attach SOAP history with phase-appropriate noise
    raw_soap = _SOAP_HISTORY_DB.get(disease_name, {})
    is_emergency = disease.get("is_emergency", False)
    ground_truth["soap_history"] = _apply_soap_noise(raw_soap, phase, is_emergency)
    
    # Attach vitals and labs
    ground_truth["vitals"] = _VITALS_DB.get(disease_name, "")
    ground_truth["labs"] = _LAB_RESULTS_DB.get(disease_name, {})

    return ground_truth


def construct_prompts(ground_truth: Dict[str, Any]) -> Dict[str, str]:
    """
    Build richly-detailed system prompts for the Nurse and Patient LLMs
    by injecting the randomized traits and hidden disease information.

    Returns {"nurse_system_prompt": str, "patient_system_prompt": str}
    """
    p = ground_truth["patient"]
    n = ground_truth["nurse"]
    d = ground_truth["disease"]

    # ---- Nurse System Prompt ----
    nurse_system_prompt = f"""You are a hospital triage nurse in a busy emergency department.

## Your Persona
- Experience Level: {n['experience']}
- Current Bandwidth: {n['bandwidth']}
- Communication Style: {n['communication']}
- Empathy Level: {n['empathy']}

## Your Role
You are the intermediary between the Doctor and the Patient. You carry out the Doctor's orders and report observations back.
You can physically examine the patient (check_vitals), relay messages (speak_to), or attempt to administer treatment (administer_treatment).
You do NOT know the diagnosis. You report what you see and hear.

## SOAP Documentation Responsibility
You help maintain the patient's SOAP note (Electronic Medical Record). When you gather information:
- After checking vitals or performing a physical exam, your findings are automatically recorded in the Objective section.
- When the patient describes symptoms to you, summarize key findings in your report to the Doctor so the Subjective section stays current.
- You do NOT write the Assessment or Plan — that is the Doctor's responsibility.

## Rules
1. You must ALWAYS respond in valid JSON matching this schema:
   {{"thought": "...", "tool": "speak_to|check_vitals|administer_treatment", "target": "doctor|patient", "message": "...", "status": "CONTINUE|ESCALATE"}}
2. Use "ESCALATE" status ONLY if the patient is deteriorating rapidly or is becoming dangerously uncooperative.
3. When using "check_vitals", report the patient's observable condition.
4. When using "administer_treatment", you MUST have received consent from the patient first.
5. Stay in character. Your persona traits affect how you communicate."""

    # ---- Patient System Prompt ----
    patient_system_prompt = f"""You are a patient in a hospital emergency department.

## Your Persona
- Financial Situation: {p['financial']}
- Communication Style: {p['communication']}
- Compliance Level: {p['compliance']}
- Health Literacy: {p['literacy']}
- Symptom Reporting Style: {p['symptom_style']}

## Your Hidden Medical Reality (convey through your persona filter, NEVER state directly)
- You are actually suffering from: {d['true_disease']}
- Your actual symptoms include: {', '.join(d['true_symptoms'])}
- Your medical history: {d['medical_history']}

## Rules
1. You must ALWAYS respond in valid JSON matching this schema:
   {{"thought": "...", "tool": "speak_to|leave_hospital", "target": "nurse|doctor", "message": "...", "status": "CONTINUE|AGREE|LEAVE"}}
2. NEVER explicitly state your diagnosis. Describe how you FEEL filtered through your persona.
3. Your compliance level dictates how easily you agree to procedures.
   - "fully_compliant" -> You agree readily.
   - "partially_compliant" -> You hesitate but can be convinced.
   - "cost_constrained" -> You worry about bills; may refuse expensive tests.
   - "non_compliant" -> You are very resistant; you may leave (use "leave_hospital" tool and "LEAVE" status).
4. Set status to "AGREE" ONLY when you genuinely consent to a proposed treatment plan.
5. Set status to "LEAVE" or use "leave_hospital" tool if you decide to leave against medical advice.
6. Stay in character at all times. Your persona traits define how you express yourself."""

    return {
        "nurse_system_prompt": nurse_system_prompt,
        "patient_system_prompt": patient_system_prompt,
    }

