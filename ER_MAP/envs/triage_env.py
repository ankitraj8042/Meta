"""
ER_MAP/envs/triage_env.py
=========================
Core OpenEnv Gymnasium environment for ER-MAP.
The Doctor (RL agent) interacts with this env. Nurse and Patient are
internal environment actors driven by LLMs via the AgentRouter.
"""

import json
import logging
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

from .randomizer import (
    generate_ground_truth,
    construct_prompts,
    LAB_RESULTS_DB,
    VITALS_DB,
    SOAP_HISTORY_DB,
)
from .empathy_engine import (
    compute_empathy_reward,
    PatientState,
    MilestoneTracker,
)
from .api_router import AgentRouter

logger = logging.getLogger("ER_MAP.triage_env")

# ---------------------------------------------------------------------------
# Valid tool sets per role (for hallucination detection)
# ---------------------------------------------------------------------------
DOCTOR_TOOLS = {"speak_to", "order_lab", "terminal_discharge", "read_soap", "update_soap"}
NURSE_TOOLS = {"speak_to", "check_vitals", "administer_treatment"}
PATIENT_TOOLS = {"speak_to", "leave_hospital"}

# ---------------------------------------------------------------------------
# Max constants  (overridable via env vars for demo / training tuning)
#   ERMAP_MAX_EPISODE_STEPS         — total Doctor turns before truncation
#   ERMAP_MAX_INTERNAL_EXCHANGES    — Nurse ↔ Patient loop cap per Doctor step
# ---------------------------------------------------------------------------
MAX_EPISODE_STEPS = int(os.environ.get("ERMAP_MAX_EPISODE_STEPS", "20"))
MAX_INTERNAL_EXCHANGES = int(os.environ.get("ERMAP_MAX_INTERNAL_EXCHANGES", "5"))

# ---------------------------------------------------------------------------
# Per-episode reward caps (anti-farming; keeps process signal balanced
# against terminal signal so the model cannot learn to spam empathy or
# milestones to mask a wrong diagnosis).
# ---------------------------------------------------------------------------
EMPATHY_REWARD_CAP_POS = 0.30   # max positive empathy reward per episode
EMPATHY_REWARD_CAP_NEG = -0.40  # max negative empathy reward per episode

# ---------------------------------------------------------------------------
# Stop-words used by the cheap process-level keyword verifier (intermediate
# diagnosis bonus + terminal keyword overlap).
# ---------------------------------------------------------------------------
_VERIFIER_STOP_WORDS = {
    "for", "if", "or", "and", "with", "to", "of", "the", "a", "an", "in",
    "on", "at", "by", "from", "unable", "po", "signs", "is", "are", "then",
    "above", "below", "due", "via", "per", "as", "be", "no", "not", "any",
    "some", "this", "that", "these", "those", "patient", "patients",
}


def _keyword_set(text: str) -> set:
    """Lowercase tokenize and drop stop words."""
    if not text:
        return set()
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {t for t in tokens if t not in _VERIFIER_STOP_WORDS}


def _fuzzy_overlap(target_kw: set, candidate_kw: set) -> float:
    """
    Substring-aware overlap ratio in [0, 1].
    Uses both exact and substring matches (>=4 chars) so that
    'thrombolytic' counts against 'thrombolytics'.
    """
    if not target_kw:
        return 0.0
    matched = set()
    for t in target_kw:
        if t in candidate_kw:
            matched.add(t)
            continue
        for c in candidate_kw:
            if len(t) >= 4 and (t in c or c in t):
                matched.add(t)
                break
    return len(matched) / max(len(target_kw), 1)


class TriageEnv(gym.Env):
    """
    ER-MAP Triage Environment.

    Observation: JSON string visible to the Doctor.
    Action: JSON string produced by the Doctor.

    Internal loop each step:
        Doctor action → env dispatches to Nurse/Patient APIs (≤3 internal
        exchanges) → env computes dense reward → returns observation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        nurse_api_key: Optional[str] = None,
        patient_api_key: Optional[str] = None,
        empathy_judge_api_key: Optional[str] = None,
        medical_judge_api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        nurse_model: Optional[str] = None,
        patient_model: Optional[str] = None,
        empathy_judge_model: Optional[str] = None,
        medical_judge_model: Optional[str] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # Gymnasium spaces — text-based (large Discrete placeholders)
        self.observation_space = spaces.Text(
            min_length=1, max_length=8192
        )
        self.action_space = spaces.Text(
            min_length=1, max_length=2048
        )

        # Internal API router — supports up to 4 distinct Groq clients
        # (nurse, patient, empathy_judge, medical_judge), each with its
        # own key and model. Older callers passing only nurse/patient
        # continue to work — judges silently fall back to those clients.
        self.router = AgentRouter(
            api_key=groq_api_key,
            nurse_api_key=nurse_api_key,
            patient_api_key=patient_api_key,
            empathy_judge_api_key=empathy_judge_api_key,
            medical_judge_api_key=medical_judge_api_key,
            model=model,
            nurse_model=nurse_model,
            patient_model=patient_model,
            empathy_judge_model=empathy_judge_model,
            medical_judge_model=medical_judge_model,
        )

        # Episode state
        self.ground_truth: Dict[str, Any] = {}
        self.step_count: int = 0
        self.done: bool = False
        self.consent_given: bool = False
        self.ordered_labs: set = set()
        self.episode_log: list = []
        self.render_mode = render_mode
        self.last_patient_status: str = "CONTINUE"

        # SOAP EMR (Electronic Medical Record)
        self.emr: Dict[str, Any] = self._create_empty_emr()

        # Phase-based systems (initialized on reset)
        self.phase: int = 1
        self.patient_state: Optional[PatientState] = None
        self.milestone_tracker: Optional[MilestoneTracker] = None

        # Per-episode reward bookkeeping (filled in reset())
        self.reward_components: Dict[str, float] = {}
        self.intermediate_diagnosis_awarded: bool = False
        self.intermediate_plan_awarded: bool = False

    # ==================================================================
    # reset()
    # ==================================================================

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Start a new episode.
        - Generate ground truth (disease + persona traits).
        - Initialize Nurse/Patient LLMs with system prompts.
        - Return the Doctor's initial observation (only sees Nurse experience).

        If `seed` is provided, the global `random` module is also seeded so
        that scenario generation (which uses `random.choice`) is deterministic.
        This is required for GRPO group rollouts where G completions must
        share the same scenario.
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        # 1. Generate ground truth with phase-aware constraints
        self.phase = (options or {}).get("phase", 1)
        difficulty = (options or {}).get("difficulty", None)
        self.ground_truth = generate_ground_truth(
            difficulty=difficulty,
            phase=self.phase,
        )

        # 2. Build and set system prompts for the environment actors
        prompts = construct_prompts(self.ground_truth)
        self.router.reset_memory()
        self.router.set_system_prompt("nurse", prompts["nurse_system_prompt"])
        self.router.set_system_prompt("patient", prompts["patient_system_prompt"])

        # 3. Reset episode state
        self.step_count = 0
        self.done = False
        self.consent_given = False
        self.ordered_labs = set()
        self.episode_log = []
        self.last_patient_status = "CONTINUE"

        # 3a. Reset per-component reward tracker (used for logging,
        #     anti-farming caps, and GRPO advantage diagnostics).
        self.reward_components = {
            "process": 0.0,        # JSON validity, tool legality, basic ops
            "milestones": 0.0,     # ordered clinical workflow
            "labs": 0.0,           # lab ordering (critical vs generic)
            "empathy": 0.0,        # bounded by EMPATHY_REWARD_CAP_*
            "consent": 0.0,        # AGREE / AMA outcomes
            "diagnosis": 0.0,      # intermediate Assessment keyword bonus
            "plan": 0.0,           # intermediate Plan keyword bonus
            "documentation": 0.0,  # SOAP filled vs empty at discharge
            "emergency_id": 0.0,   # is_emergency correctly classified
            "treatment": 0.0,      # terminal judge + keyword combined
            "penalties": 0.0,      # turn cost, redundancy, early discharge
        }
        self.intermediate_diagnosis_awarded = False
        self.intermediate_plan_awarded = False

        # 4. Initialize SOAP EMR and pre-populate with patient history
        self.emr = self._create_empty_emr()
        self._populate_emr_from_history()

        # 5. Initialize phase-based systems
        self.patient_state = PatientState(self.ground_truth["patient"])
        is_emergency = self.ground_truth.get("disease", {}).get("is_emergency", False)
        self.milestone_tracker = MilestoneTracker(phase=self.phase, is_emergency=is_emergency)

        # 5. Build initial observation for Doctor
        #    Doctor sees nurse experience level + the pre-populated SOAP note.
        initial_obs = json.dumps({
            "event": "episode_start",
            "nurse_experience": self.ground_truth["nurse"]["experience"],
            "message": (
                "You are an ER doctor beginning a new triage case. "
                "A nurse is available to assist. A patient has just arrived. "
                "Use your tools to diagnose and treat the patient.\n"
                "TOOLS: speak_to, order_lab, read_soap, update_soap, terminal_discharge.\n"
                "When using 'terminal_discharge', you MUST include an 'is_emergency' boolean field (true/false) to indicate if this is a time-critical emergency.\n"
                "The patient's prior medical history and initial presentation "
                "have been recorded in the SOAP note. Use 'read_soap' to review it. "
                "Update the Assessment and Plan sections before discharging."
            ),
            "soap_summary": self._get_soap_summary(),
        })

        info: Dict[str, Any] = {"ground_truth_disease": self.ground_truth["disease"]["true_disease"]}
        return initial_obs, info

    # ==================================================================
    # step()
    # ==================================================================

    def _add(self, component: str, value: float) -> float:
        """Accumulate reward into a tracked component bucket."""
        self.reward_components[component] = (
            self.reward_components.get(component, 0.0) + value
        )
        return value

    def _add_empathy(self, value: float) -> float:
        """
        Accumulate empathy reward with a per-episode cap on both ends to
        block reward farming (e.g. spam-empathy-then-misdiagnose).
        Returns the actually applied value (may be 0 if cap reached).
        """
        current = self.reward_components.get("empathy", 0.0)
        if value > 0 and current >= EMPATHY_REWARD_CAP_POS:
            return 0.0
        if value < 0 and current <= EMPATHY_REWARD_CAP_NEG:
            return 0.0
        # Clip the increment so we never overshoot the cap
        if value > 0:
            value = min(value, EMPATHY_REWARD_CAP_POS - current)
        else:
            value = max(value, EMPATHY_REWARD_CAP_NEG - current)
        self.reward_components["empathy"] = current + value
        return value

    def step(self, action: str) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        """
        Process one Doctor action, run internal Nurse/Patient exchange loop,
        compute dense reward, return next observation.

        Reward design (rebalanced for process > terminal):
            process (per step):     ~ +0.05 valid JSON, +0.07 milestones, +0.20 critical lab,
                                      +0.08 update_soap, +0.20-0.30 intermediate diagnosis
            terminal (one-shot):    smoothed treatment judge in [-0.30, +0.50],
                                    emergency id ±0.30, SOAP doc ±0.20
            empathy (per step):     capped at +0.30 / -0.40 across the full episode

        Total max process ≈ +1.5, total max terminal ≈ +1.0 → process-dominant.
        """
        reward = 0.0
        self.step_count += 1
        truncated = False
        info: Dict[str, Any] = {}

        # --- Turn penalty (softened so that early decisive actions are not
        #     wiped out by accumulated turn cost in long episodes). ---
        is_emergency = self.ground_truth.get("disease", {}).get("is_emergency", False)
        if is_emergency:
            reward += self._add("penalties", -0.10)  # was -0.15
        else:
            reward += self._add("penalties", -0.01)

        # --- Parse Doctor's JSON action ---
        doctor_action = self._parse_doctor_action(action)
        if doctor_action is None:
            reward += self._add("penalties", -0.15)  # was -0.20
            obs = json.dumps({
                "event": "system_error",
                "message": "Your last action was not valid JSON. Please respond with a properly formatted JSON action.",
            })
            self.episode_log.append({"role": "system", "content": "Doctor sent invalid JSON"})
            info["reward_components"] = dict(self.reward_components)
            return obs, reward, self.done, self._check_truncated(), info

        tool = doctor_action.get("tool", "")
        target = doctor_action.get("target", "")

        # --- Hallucinated tool check ---
        if tool not in DOCTOR_TOOLS:
            reward += self._add("penalties", -0.15)  # was -0.20
            obs = json.dumps({
                "event": "system_error",
                "message": f"Unknown tool '{tool}'. Valid tools: speak_to, order_lab, terminal_discharge.",
            })
            info["reward_components"] = dict(self.reward_components)
            return obs, reward, self.done, self._check_truncated(), info

        # --- Valid JSON bonus ---
        reward += self._add("process", 0.05)

        self.episode_log.append({"role": "doctor", "action": doctor_action})

        # ============================================================
        # Handle each Doctor tool
        # ============================================================

        if tool == "speak_to":
            obs, step_reward = self._handle_speak_to(doctor_action, target)
            reward += step_reward
            message = doctor_action.get("message", "")
            if message and self.patient_state:
                intent = self.router.evaluate_empathy(message)
                self.patient_state.update(intent)
                emp_r = compute_empathy_reward(intent, self.patient_state, self.phase)
                reward += self._add_empathy(emp_r)
            if target == "patient" and self.milestone_tracker:
                reward += self._add(
                    "milestones", self.milestone_tracker.mark("PATIENT_CONTACT")
                )

        elif tool == "order_lab":
            obs, step_reward = self._handle_order_lab(doctor_action)
            reward += step_reward
            if self.milestone_tracker:
                reward += self._add(
                    "milestones", self.milestone_tracker.mark("LABS")
                )

        elif tool == "read_soap":
            obs, step_reward = self._handle_read_soap(doctor_action)
            reward += step_reward
            if self.milestone_tracker:
                reward += self._add(
                    "milestones", self.milestone_tracker.mark("READ_SOAP")
                )

        elif tool == "update_soap":
            obs, step_reward = self._handle_update_soap(doctor_action)
            reward += step_reward

        elif tool == "terminal_discharge":
            obs, step_reward = self._handle_terminal_discharge(doctor_action)
            reward += step_reward
            if self.milestone_tracker:
                reward += self._add(
                    "milestones", self.milestone_tracker.mark("DISCHARGE")
                )

        else:
            obs = json.dumps({"event": "system_error", "message": "Unhandled tool."})

        # Check for truncation (max steps)
        truncated = self._check_truncated()
        if truncated and not self.done:
            reward += self._add("penalties", -0.30)  # was -0.50
            info["truncation_reason"] = "max_episode_steps_reached"

        info["step_count"] = self.step_count
        info["step_reward"] = round(reward, 4)
        info["reward_components"] = dict(self.reward_components)
        info["consent_given"] = self.consent_given
        info["patient_status"] = self.last_patient_status
        if self.patient_state:
            info["patient_state"] = self.patient_state.get_state_summary()
        if self.milestone_tracker:
            info["milestones"] = self.milestone_tracker.get_summary()

        # Auto-inject SOAP summary into every observation so Doctor
        # always sees the current EMR state without needing read_soap
        try:
            obs_dict = json.loads(obs)
            if "soap_summary" not in obs_dict and obs_dict.get("event") not in ("soap_read",):
                obs_dict["soap_summary"] = self._get_soap_summary()
                obs = json.dumps(obs_dict)
        except (json.JSONDecodeError, TypeError):
            pass

        if self.render_mode == "human":
            self._render_step(doctor_action, obs, reward)

        return obs, reward, self.done, truncated, info

    # ==================================================================
    # Tool Handlers
    # ==================================================================

    def _handle_speak_to(
        self, doctor_action: Dict[str, Any], target: str
    ) -> Tuple[str, float]:
        """Handle Doctor using 'speak_to' tool."""
        reward = 0.0
        message = doctor_action.get("message", "")

        if target == "nurse":
            # Doctor → Nurse, then Nurse ↔ Patient internal loop
            nurse_response = self.router.query(
                "nurse",
                f"[Doctor says to you]: {message}",
            )
            self.episode_log.append({"role": "nurse", "action": nurse_response})

            # Run internal Nurse ↔ Patient loop (max 3 exchanges)
            internal_results = self._run_internal_loop(nurse_response)
            reward += internal_results.get("reward", 0.0)

            obs = json.dumps({
                "event": "nurse_report",
                "nurse_message": nurse_response.get("message", ""),
                "nurse_status": nurse_response.get("status", "CONTINUE"),
                "internal_exchanges": internal_results.get("summary", []),
                "patient_status": self.last_patient_status,
            })

        elif target == "patient":
            # Doctor speaks directly to patient
            patient_response = self.router.query(
                "patient",
                f"[Doctor says to you]: {message}",
            )
            self.episode_log.append({"role": "patient", "action": patient_response})

            self.last_patient_status = patient_response.get("status", "CONTINUE")

            # Check for de-escalation success
            if self.last_patient_status == "AGREE":
                reward += self._add("consent", 0.25)  # was 0.30
                self.consent_given = True

            # Check for AMA LOSS
            if (
                self.last_patient_status == "LEAVE"
                or patient_response.get("tool") == "leave_hospital"
            ):
                reward += self._add("consent", -0.50)  # was -0.75
                self.done = True
                obs = json.dumps({
                    "event": "terminal_ama",
                    "message": "The patient has left against medical advice.",
                    "patient_message": patient_response.get("message", ""),
                })
                return obs, reward

            obs = json.dumps({
                "event": "patient_response",
                "patient_message": patient_response.get("message", ""),
                "patient_status": self.last_patient_status,
            })

        else:
            obs = json.dumps({
                "event": "system_error",
                "message": f"Invalid speak_to target '{target}'. Valid targets: nurse, patient.",
            })

        return obs, reward

    def _handle_order_lab(self, doctor_action: Dict[str, Any]) -> Tuple[str, float]:
        """
        Handle Doctor using 'order_lab' tool.

        Reward design:
            redundant order:        -0.20  (was -0.25)
            critical lab (in disease.critical_labs): +0.20
            generic informative lab (matches disease lab DB): +0.08
            non-informative lab (no match):                 +0.02
        Critical-lab bonus is the key process-distributed signal of correct
        differential reasoning — without it, lab ordering felt random.
        """
        reward = 0.0
        test_name = doctor_action.get("test_name", "").strip().lower()

        if not test_name:
            obs = json.dumps({
                "event": "system_error",
                "message": "order_lab requires a 'test_name' field.",
            })
            return obs, reward

        # Redundancy check
        if test_name in self.ordered_labs:
            reward += self._add("penalties", -0.20)
            obs = json.dumps({
                "event": "lab_result",
                "test_name": test_name,
                "result": "DUPLICATE ORDER — results already available from prior draw.",
                "redundant": True,
            })
            return obs, reward

        self.ordered_labs.add(test_name)

        # Look up lab results from the ground truth disease
        disease_name = self.ground_truth["disease"]["true_disease"]
        disease_labs = LAB_RESULTS_DB.get(disease_name, {})
        critical_labs = [
            c.lower() for c in self.ground_truth["disease"].get("critical_labs", [])
        ]

        # Is this a critical (diagnosis-clinching) lab?
        is_critical = any(
            test_name in c or c in test_name for c in critical_labs if len(c) >= 3
        )

        # Fuzzy match: check if test_name substring-matches any key in the lab DB
        result_text = None
        for lab_key, lab_value in disease_labs.items():
            if test_name in lab_key.lower() or lab_key.lower() in test_name:
                result_text = lab_value
                break

        if is_critical:
            reward += self._add("labs", 0.20)
        elif result_text:
            reward += self._add("labs", 0.08)
        else:
            reward += self._add("labs", 0.02)

        if result_text:
            obs = json.dumps({
                "event": "lab_result",
                "test_name": test_name,
                "result": result_text,
                "redundant": False,
                "is_critical_lab": is_critical,
            })
            self._emr_append("Objective", "Labs", f"[{test_name.upper()}] {result_text}")
        else:
            result_normal = f"Lab '{test_name}' results: within normal limits. No significant findings."
            obs = json.dumps({
                "event": "lab_result",
                "test_name": test_name,
                "result": result_normal,
                "redundant": False,
                "is_critical_lab": is_critical,
            })
            self._emr_append("Objective", "Labs", f"[{test_name.upper()}] {result_normal}")

        self.episode_log.append({"role": "system", "content": f"Lab ordered: {test_name}"})
        return obs, reward

    def _handle_read_soap(
        self, doctor_action: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Handle Doctor using 'read_soap' tool — returns the full EMR."""
        reward = 0.0
        section = doctor_action.get("section", "").strip()

        if section and section in self.emr:
            # Read a specific section
            content = self.emr[section]
            obs = json.dumps({
                "event": "soap_read",
                "section": section,
                "content": content,
            })
        else:
            # Return full SOAP note
            obs = json.dumps({
                "event": "soap_read",
                "section": "ALL",
                "content": self.emr,
            })

        self.episode_log.append({"role": "system", "content": f"Doctor read SOAP: {section or 'ALL'}"})
        return obs, reward

    def _handle_update_soap(
        self, doctor_action: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Handle Doctor using 'update_soap' tool.

        Reward design (KEY rebalance):
          - Base documentation reward: +0.08 (was +0.05).
          - Intermediate diagnosis bonus (one-shot per episode):
              +0.20 if the Assessment text fuzzy-overlaps the true_disease
                    name and/or true_symptoms by >= 30%
              +0.30 if overlap >= 60% (strong, clinically reasoned)
          - Intermediate plan bonus (one-shot per episode):
              +0.15 if the Plan text fuzzy-overlaps the correct_treatment
                    keyword set by >= 30%
              +0.25 if overlap >= 60%

        These bonuses pull the diagnosis-correctness signal *out of the
        terminal step* and distribute it across the trajectory, which is
        critical for stable GRPO learning on long horizons. They are
        one-shot per episode so they cannot be farmed by repeatedly
        re-writing the same Assessment.
        """
        reward = 0.0
        section = doctor_action.get("section", "").strip()
        content = doctor_action.get("content", "").strip()

        if not section or not content:
            obs = json.dumps({
                "event": "system_error",
                "message": "update_soap requires 'section' and 'content' fields. Valid sections: Assessment, Plan, Subjective.HPI, Subjective.ROS",
            })
            return obs, reward

        # Parse dotted notation (e.g., "Subjective.HPI")
        parts = section.split(".")
        updated = False
        section_kind = None  # "Assessment" | "Plan" | "Subjective" | "Objective"

        if len(parts) == 1 and parts[0] in ("Assessment", "Plan"):
            self.emr[parts[0]] = content
            updated = True
            section_kind = parts[0]
            reward += self._add("documentation", 0.08)
        elif len(parts) == 2 and parts[0] == "Subjective" and parts[1] in self.emr.get("Subjective", {}):
            self.emr["Subjective"][parts[1]] = content
            updated = True
            section_kind = "Subjective"
            reward += self._add("documentation", 0.08)
        elif len(parts) == 2 and parts[0] == "Objective" and parts[1] in self.emr.get("Objective", {}):
            self.emr["Objective"][parts[1]] = content
            updated = True
            section_kind = "Objective"
            reward += self._add("documentation", 0.08)
        else:
            obs = json.dumps({
                "event": "system_error",
                "message": f"Invalid SOAP section '{section}'. Valid: Assessment, Plan, Subjective.HPI, Subjective.ROS, Objective.Physical_Examination",
            })
            return obs, reward

        # ----- Intermediate diagnosis bonus (Assessment) -----
        diag_overlap = 0.0
        plan_overlap = 0.0
        if section_kind == "Assessment" and not self.intermediate_diagnosis_awarded:
            disease = self.ground_truth.get("disease", {})
            target_text = (
                disease.get("true_disease", "") + " " +
                " ".join(disease.get("true_symptoms", []))
            )
            target_kw = _keyword_set(target_text)
            content_kw = _keyword_set(content)
            diag_overlap = _fuzzy_overlap(target_kw, content_kw)
            if diag_overlap >= 0.60:
                reward += self._add("diagnosis", 0.30)
                self.intermediate_diagnosis_awarded = True
            elif diag_overlap >= 0.30:
                reward += self._add("diagnosis", 0.20)
                self.intermediate_diagnosis_awarded = True

        # ----- Intermediate plan bonus (Plan) -----
        if section_kind == "Plan" and not self.intermediate_plan_awarded:
            target_kw = _keyword_set(
                self.ground_truth.get("disease", {}).get("correct_treatment", "")
            )
            content_kw = _keyword_set(content)
            plan_overlap = _fuzzy_overlap(target_kw, content_kw)
            if plan_overlap >= 0.60:
                reward += self._add("plan", 0.25)
                self.intermediate_plan_awarded = True
            elif plan_overlap >= 0.30:
                reward += self._add("plan", 0.15)
                self.intermediate_plan_awarded = True

        if updated:
            obs = json.dumps({
                "event": "soap_updated",
                "section": section,
                "message": f"SOAP note '{section}' updated successfully.",
                "soap_summary": self._get_soap_summary(),
                "intermediate_signals": {
                    "diagnosis_overlap": round(diag_overlap, 2),
                    "plan_overlap": round(plan_overlap, 2),
                },
            })
            self.episode_log.append({"role": "doctor", "content": f"Updated SOAP {section}: {content[:100]}"})

        return obs, reward

    def _handle_terminal_discharge(
        self, doctor_action: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Handle Doctor using 'terminal_discharge' tool. Ends the episode.

        REWARD REBALANCE (terminal-light, process-heavy):
          1. Emergency identification:   correct ±0.30, neg correct +0.05,
                                        false-pos -0.20, missed-true -0.30
                                        (was ±0.50 / +0.10 / -0.30 / -0.50)
          2. SOAP Assessment documented: +0.20  /  empty: -0.30  (was -0.50)
          3. Practiced blind (no read_soap, non-emergency): -0.30 (was -0.50)
          4. Early discharge:            -0.10 * shortfall   (was -0.15)
          5. Treatment outcome (the BIG change):
                Independent dual-verifier: LLM judge + keyword overlap.
                Final score   = 0.6 * judge_score + 0.4 * keyword_overlap
                                (when both available; falls back gracefully)
                Lethal-treatment hard penalty: -0.80 (was -1.50). The dual
                check requires *both* signals to agree before declaring
                lethality unilaterally.
                Treatment terminal reward is now SMOOTH:
                    base = -0.30 + final_score * 0.90    -> [-0.30, +0.60]
                    if lethal:  base += -0.80
                This eliminates the +1.0/-1.0 cliff, distributing partial
                credit linearly so the policy gradient is well-shaped.
        """
        reward = 0.0
        treatment = doctor_action.get("treatment", "").strip().lower()
        declared_emergency = bool(doctor_action.get("is_emergency", False))
        is_actual_emergency = self.ground_truth.get("disease", {}).get("is_emergency", False)
        self.done = True

        # --- 1. Emergency Identification (reduced magnitudes) ---
        if declared_emergency and is_actual_emergency:
            reward += self._add("emergency_id", 0.30)
        elif not declared_emergency and not is_actual_emergency:
            reward += self._add("emergency_id", 0.05)
        elif declared_emergency and not is_actual_emergency:
            reward += self._add("emergency_id", -0.20)
        elif not declared_emergency and is_actual_emergency:
            reward += self._add("emergency_id", -0.30)

        # --- 2. SOAP Assessment documented ---
        assessment = self.emr.get("Assessment", "").strip()
        if assessment:
            reward += self._add("documentation", 0.20)
        else:
            reward += self._add("documentation", -0.30)

        # --- 3. Read patient history before treating (non-emergency) ---
        is_emergency = is_actual_emergency
        if (
            self.milestone_tracker
            and not self.milestone_tracker.achieved.get("READ_SOAP", False)
            and not is_emergency
        ):
            reward += self._add("documentation", -0.30)

        # --- 4. Early discharge penalty (softened) ---
        if not is_emergency:
            min_steps_map = {1: 4, 2: 6, 3: 8}
            min_steps = min_steps_map.get(self.phase, 5)
            if self.step_count < min_steps:
                shortfall = min_steps - self.step_count
                reward += self._add("penalties", -0.10 * shortfall)

        # Auto-record the treatment in SOAP Plan
        self.emr["Plan"] = treatment

        correct_treatment = self.ground_truth["disease"]["correct_treatment"].lower()
        lethal_treatments = [
            t.lower() for t in self.ground_truth["disease"]["lethal_treatments"]
        ]
        disease_name = self.ground_truth["disease"]["true_disease"]

        # --- 5. INDEPENDENT DUAL VERIFIER ---
        #     (a) Cheap keyword overlap (always available)
        correct_kw = _keyword_set(correct_treatment)
        treatment_kw = _keyword_set(treatment)
        keyword_overlap = _fuzzy_overlap(correct_kw, treatment_kw)
        keyword_lethal = any(
            lethal_kw and lethal_kw in treatment for lethal_kw in lethal_treatments
        )

        #     (b) LLM-as-a-Judge (70B) — semantic grading
        judge_result = None
        if self.router:
            try:
                judge_result = self.router.evaluate_treatment(
                    prescribed_treatment=treatment,
                    correct_treatment=self.ground_truth["disease"]["correct_treatment"],
                    lethal_treatments=self.ground_truth["disease"]["lethal_treatments"],
                    disease_name=disease_name,
                )
            except Exception as e:
                logger.error(f"LLM Judge failed: {e}")
                judge_result = None

        judge_score: Optional[float] = None
        judge_lethal: bool = False
        judge_reasoning: str = ""
        if judge_result and judge_result.get("score") is not None:
            judge_score = float(judge_result["score"])
            judge_lethal = bool(judge_result.get("is_lethal", False))
            judge_reasoning = str(judge_result.get("reasoning", ""))

        # ----- Combine the two independent signals -----
        # If both available: weighted blend (judge gets 60% as the semantic
        # signal, keyword gets 40% as the grounding signal). This satisfies
        # the §7 / §8 guidance: 'multiple independent reward functions' so
        # the model cannot exploit either one alone.
        if judge_score is not None:
            final_score = 0.6 * judge_score + 0.4 * keyword_overlap
            # Lethal triggers only when BOTH signals agree, OR one signal
            # is overwhelming (judge_lethal AND keyword_lethal, OR a single
            # very strong signal). This blocks judge-hallucinated lethality.
            is_lethal = (judge_lethal and keyword_lethal) or (
                judge_lethal and judge_score < 0.20
            ) or (
                keyword_lethal and judge_score < 0.40
            )
        else:
            # Judge unavailable: fall back to keyword-only with a wider band
            final_score = keyword_overlap
            is_lethal = keyword_lethal

        # ----- Smooth treatment reward (no cliffs) -----
        # Linear in [-0.30, +0.60] across final_score in [0, 1].
        smooth_treatment = -0.30 + final_score * 0.90
        if is_lethal:
            smooth_treatment += -0.80   # additional lethal penalty (was -1.50 hard)
        reward += self._add("treatment", smooth_treatment)

        # ----- Build observation -----
        # Choose a label for telemetry / curriculum scheduler:
        #   WIN if final_score >= 0.65, FATAL if lethal, INCORRECT if <0.30,
        #   PARTIAL otherwise.
        if is_lethal:
            event = "terminal_fatal"
            message = "CRITICAL ERROR: Lethal treatment administered."
        elif final_score >= 0.65:
            event = "terminal_win"
            message = "Correct diagnosis and treatment! Patient stabilized."
        elif final_score >= 0.30:
            event = "terminal_partial"
            message = f"Partially correct treatment ({final_score:.0%})."
        else:
            event = "terminal_incorrect"
            message = "Incorrect treatment. Patient outcome: adverse."

        obs = json.dumps({
            "event": event,
            "message": message,
            "ground_truth": disease_name,
            "correct_treatment": self.ground_truth["disease"]["correct_treatment"],
            "prescribed_treatment": treatment,
            "verifier": {
                "judge_score": judge_score,
                "judge_lethal": judge_lethal,
                "judge_reasoning": judge_reasoning,
                "keyword_overlap": round(keyword_overlap, 3),
                "keyword_lethal": keyword_lethal,
                "final_score": round(final_score, 3),
                "is_lethal_combined": is_lethal,
            },
            "soap_note": self.emr,
        })

        return obs, reward

    # ==================================================================
    # Internal Nurse ↔ Patient Loop
    # ==================================================================

    def _run_internal_loop(
        self, initial_nurse_action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the internal Nurse ↔ Patient dialogue loop.
        Max MAX_INTERNAL_EXCHANGES exchanges.
        Enforces the CONSENT LOCK on administer_treatment.

        Returns dict with 'reward' and 'summary' keys.
        """
        reward = 0.0
        summary = []
        exchanges = 0
        current_nurse_action = initial_nurse_action

        while exchanges < MAX_INTERNAL_EXCHANGES:
            nurse_tool = current_nurse_action.get("tool", "speak_to")
            nurse_target = current_nurse_action.get("target", "patient")
            nurse_message = current_nurse_action.get("message", "")
            nurse_status = current_nurse_action.get("status", "CONTINUE")

            # --- Nurse: check_vitals ---
            if nurse_tool == "check_vitals":
                disease_name = self.ground_truth["disease"]["true_disease"]
                vitals = VITALS_DB.get(disease_name, "Vitals: stable, no acute distress")
                summary.append({"nurse_action": "check_vitals", "result": vitals})

                # Auto-update SOAP Objective with vitals
                self._emr_append("Objective", "Vitals", vitals)

                # Feed vitals back to nurse as system feedback
                self.router._append_to_memory(
                    "nurse", "user", f"[VITALS RESULT]: {vitals}"
                )
                exchanges += 1
                break  # vitals check completes this loop

            # --- Nurse: administer_treatment (CONSENT LOCK) ---
            if nurse_tool == "administer_treatment":
                if not self.consent_given:
                    # CONSENT LOCK: Reject and force speak_to
                    logger.info("CONSENT LOCK: Treatment rejected — no patient consent.")
                    summary.append({
                        "nurse_action": "administer_treatment_BLOCKED",
                        "reason": "Patient has not given consent (status != AGREE).",
                    })
                    # Force nurse to speak_to patient instead
                    forced_msg = (
                        "[SYSTEM]: Treatment attempt blocked — patient consent required. "
                        "Please speak to the patient to obtain consent first."
                    )
                    current_nurse_action = self.router.query("nurse", forced_msg)
                    self.episode_log.append({"role": "nurse", "action": current_nurse_action})
                    exchanges += 1
                    continue
                else:
                    summary.append({"nurse_action": "administer_treatment", "consent": True})
                    exchanges += 1
                    break  # treatment administered, loop done

            # --- Nurse: speak_to patient ---
            if nurse_target == "patient" and nurse_tool == "speak_to":
                patient_response = self.router.query(
                    "patient",
                    f"[Nurse says to you]: {nurse_message}",
                )
                self.episode_log.append({"role": "patient", "action": patient_response})

                patient_status = patient_response.get("status", "CONTINUE")
                self.last_patient_status = patient_status

                summary.append({
                    "nurse_said": nurse_message,
                    "patient_said": patient_response.get("message", ""),
                    "patient_status": patient_status,
                })

                # Update consent
                if patient_status == "AGREE":
                    self.consent_given = True

                # AMA check
                if (
                    patient_status == "LEAVE"
                    or patient_response.get("tool") == "leave_hospital"
                ):
                    self.last_patient_status = "LEAVE"
                    break

                # If nurse was delegating to handle uncooperative patient and failed
                if patient_status != "AGREE" and patient_status != "CONTINUE":
                    reward += -0.10  # Blind delegation penalty

                exchanges += 1

                # Nurse needs to report back; no further internal exchange needed
                if nurse_status == "ESCALATE":
                    break

                # Get nurse's next action
                nurse_followup_msg = (
                    f"[Patient responded]: {patient_response.get('message', '')} "
                    f"(Patient status: {patient_status})"
                )
                current_nurse_action = self.router.query("nurse", nurse_followup_msg)
                self.episode_log.append({"role": "nurse", "action": current_nurse_action})

            elif nurse_target == "doctor":
                # Nurse reporting to doctor — this exits the internal loop
                summary.append({"nurse_report_to_doctor": nurse_message})
                break
            else:
                break

        return {"reward": reward, "summary": summary}

    # ==================================================================
    # Helpers
    # ==================================================================

    def _parse_doctor_action(self, action: str) -> Optional[Dict[str, Any]]:
        """Parse and validate the Doctor's JSON action."""
        try:
            parsed = json.loads(action.strip())
        except (json.JSONDecodeError, TypeError):
            # Attempt regex extraction
            match = re.search(r"\{.*\}", action, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Basic schema validation
        if not isinstance(parsed, dict) or "tool" not in parsed:
            return None

        return parsed

    def _check_truncated(self) -> bool:
        """Check if episode has exceeded max steps."""
        return self.step_count >= MAX_EPISODE_STEPS

    def _render_step(
        self, doctor_action: Dict[str, Any], obs: str, reward: float
    ) -> None:
        """Render a human-readable step summary to stdout."""
        print(f"\n{'='*60}")
        print(f"  STEP {self.step_count}  |  Reward: {reward:+.2f}  |  Done: {self.done}")
        print(f"{'='*60}")
        print(f"  Doctor tool: {doctor_action.get('tool')}")
        print(f"  Doctor target: {doctor_action.get('target', 'N/A')}")
        print(f"  Doctor message: {doctor_action.get('message', doctor_action.get('treatment', 'N/A'))}")
        try:
            obs_dict = json.loads(obs)
            print(f"  Observation event: {obs_dict.get('event')}")
        except Exception:
            pass
        print(f"{'='*60}\n")

    # ==================================================================
    # OpenEnv state/close
    # ==================================================================

    # ==================================================================
    # SOAP EMR Helpers
    # ==================================================================

    @staticmethod
    def _create_empty_emr() -> Dict[str, Any]:
        """Create a blank SOAP EMR structure."""
        return {
            "Subjective": {
                "HPI": "",
                "ROS": {},
                "Past_Medical_History": "",
                "Medications": "",
                "Allergies": "",
                "Social_History": "",
            },
            "Objective": {
                "Vitals": "",
                "Physical_Examination": "",
                "Labs": "",
            },
            "Assessment": "",
            "Plan": "",
        }

    def _populate_emr_from_history(self) -> None:
        """
        Pre-populate the SOAP EMR with the patient's prior medical history
        from the SOAP_HISTORY_DB. This gives the Doctor structured data
        to analyze at episode start.
        """
        disease_name = self.ground_truth["disease"]["true_disease"]
        history = SOAP_HISTORY_DB.get(disease_name)

        if history:
            self.emr["Subjective"]["HPI"] = history.get("HPI", "")
            self.emr["Subjective"]["ROS"] = history.get("ROS", {})
            self.emr["Subjective"]["Past_Medical_History"] = history.get("Past_Medical_History", "")
            self.emr["Subjective"]["Medications"] = history.get("Medications", "")
            self.emr["Subjective"]["Allergies"] = history.get("Allergies", "")
            self.emr["Subjective"]["Social_History"] = history.get("Social_History", "")
            self.emr["Objective"]["Physical_Examination"] = history.get("Physical_Examination", "")
        else:
            # Fallback: use the basic medical_history string
            self.emr["Subjective"]["Past_Medical_History"] = self.ground_truth["disease"].get("medical_history", "")

    def _emr_append(self, section: str, field: str, content: str) -> None:
        """
        Append content to a specific EMR field. Used for auto-updating
        Objective.Vitals and Objective.Labs during the episode.
        """
        if section in self.emr and field in self.emr[section]:
            existing = self.emr[section][field]
            if existing:
                self.emr[section][field] = existing + "\n" + content
            else:
                self.emr[section][field] = content

    def _get_soap_summary(self) -> Dict[str, Any]:
        """
        Return a compact summary of the current SOAP note state.
        Used to include in observations so the Doctor sees the EMR status.
        """
        def _trunc(s, n=120):
            if isinstance(s, dict):
                return {k: v[:n] if isinstance(v, str) and len(v) > n else v for k, v in s.items()}
            return s[:n] + "..." if isinstance(s, str) and len(s) > n else s

        return {
            "Subjective": {
                "HPI": _trunc(self.emr["Subjective"]["HPI"]),
                "ROS": self.emr["Subjective"]["ROS"],
                "PMH": _trunc(self.emr["Subjective"]["Past_Medical_History"]),
                "Medications": _trunc(self.emr["Subjective"]["Medications"]),
                "Allergies": _trunc(self.emr["Subjective"]["Allergies"]),
            },
            "Objective": {
                "Vitals": _trunc(self.emr["Objective"]["Vitals"]),
                "PE": _trunc(self.emr["Objective"]["Physical_Examination"]),
                "Labs": _trunc(self.emr["Objective"]["Labs"]),
            },
            "Assessment": _trunc(self.emr["Assessment"]) or "[NOT YET DOCUMENTED]",
            "Plan": _trunc(self.emr["Plan"]) or "[NOT YET DOCUMENTED]",
        }

    # ==================================================================
    # OpenEnv state/close
    # ==================================================================

    def state(self) -> Dict[str, Any]:
        """Return the full internal state (for OpenEnv compatibility)."""
        return {
            "step_count": self.step_count,
            "done": self.done,
            "consent_given": self.consent_given,
            "ordered_labs": list(self.ordered_labs),
            "patient_status": self.last_patient_status,
            "ground_truth": self.ground_truth,
            "episode_log": self.episode_log,
            "soap_note": self.emr,
        }

    def close(self) -> None:
        """Cleanup resources."""
        self.router.reset_memory()
