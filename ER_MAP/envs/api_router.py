"""
ER_MAP/envs/api_router.py
=========================
External API handler for Nurse and Patient LLM actors.
Uses Groq inference API with Llama-3-8B-Instruct.
Maintains local episode memory with a sliding window to prevent VRAM bloat.
Enforces strict JSON output parsing with graceful failure handling.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger("ER_MAP.api_router")

# ---------------------------------------------------------------------------
# Attempt to import the Groq client. Falls back gracefully if unavailable.
# ---------------------------------------------------------------------------
try:
    from groq import Groq  # type: ignore
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed. API calls will use mock responses.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_SLIDING_WINDOW_TURNS = 3  # Keep system prompt + last 3 exchanges
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7

# ---------------------------------------------------------------------------
# JSON Extraction Helpers
# ---------------------------------------------------------------------------

def _extract_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to extract a valid JSON object from raw LLM output.
    Tries direct parse first, then regex extraction.
    """
    # --- Attempt 1: Direct parse ---
    try:
        return json.loads(raw_text.strip())
    except (json.JSONDecodeError, TypeError):
        pass

    # --- Attempt 2: Find JSON block within markdown fences ---
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # --- Attempt 3: Find first { ... } block via greedy regex ---
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw_text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _make_failure_response(role: str) -> Dict[str, Any]:
    """
    Return a safe programmatic failure action when JSON parsing fails.
    """
    if role == "nurse":
        return {
            "thought": "SYSTEM: JSON parse failure from Nurse LLM.",
            "tool": "speak_to",
            "target": "doctor",
            "message": "I'm sorry, I'm having trouble processing that. Could you repeat?",
            "status": "CONTINUE",
            "_parse_failed": True,
        }
    else:  # patient
        return {
            "thought": "SYSTEM: JSON parse failure from Patient LLM.",
            "tool": "speak_to",
            "target": "nurse",
            "message": "...I... what? I don't understand what's happening.",
            "status": "CONTINUE",
            "_parse_failed": True,
        }


# ---------------------------------------------------------------------------
# AgentRouter: Manages Nurse/Patient API sessions
# ---------------------------------------------------------------------------

class AgentRouter:
    """
    Manages LLM inference for Nurse and Patient agents via the Groq API.

    Supports separate API keys per role for independent rate limits.
    Each agent maintains its own conversation memory with a sliding-window
    strategy: the system prompt is always retained at position 0, and only
    the last `MAX_SLIDING_WINDOW_TURNS` user/assistant exchanges are kept.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        nurse_api_key: Optional[str] = None,
        patient_api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.model = model

        # Resolve per-role API keys (explicit > role-specific env > shared)
        shared_key = api_key or os.environ.get("GROQ_API_KEY", "")
        nurse_key = nurse_api_key or os.environ.get("GROQ_NURSE_API_KEY", "") or shared_key
        patient_key = patient_api_key or os.environ.get("GROQ_PATIENT_API_KEY", "") or shared_key

        # Create per-role Groq clients
        self._clients: Dict[str, Any] = {"nurse": None, "patient": None}

        if GROQ_AVAILABLE:
            if nurse_key:
                self._clients["nurse"] = Groq(api_key=nurse_key)
                logger.info("Nurse client: LIVE (Groq API)")
            else:
                logger.warning("No API key for Nurse. Using mock mode.")

            if patient_key:
                self._clients["patient"] = Groq(api_key=patient_key)
                logger.info("Patient client: LIVE (Groq API)")
            else:
                logger.warning("No API key for Patient. Using mock mode.")
        else:
            logger.warning("groq package not installed. Both agents in mock mode.")

        # Episode conversation histories: {"nurse": [...], "patient": [...]}
        self._memory: Dict[str, List[Dict[str, str]]] = {
            "nurse": [],
            "patient": [],
        }

    # ----- Memory Management -----

    def reset_memory(self) -> None:
        """Clear all conversation memory for a new episode."""
        self._memory = {"nurse": [], "patient": []}

    def set_system_prompt(self, role: str, system_prompt: str) -> None:
        """
        Initialize conversation memory with the system prompt for a role.
        """
        self._memory[role] = [{"role": "system", "content": system_prompt}]

    def _get_windowed_messages(self, role: str) -> List[Dict[str, str]]:
        """
        Return the sliding-window view of conversation history.
        System prompt (index 0) + last MAX_SLIDING_WINDOW_TURNS * 2 messages.
        """
        history = self._memory[role]
        if len(history) <= 1:
            return list(history)

        system_msg = history[0]
        dialogue = history[1:]

        # Each "turn" = 1 user + 1 assistant = 2 messages
        max_msgs = MAX_SLIDING_WINDOW_TURNS * 2
        if len(dialogue) > max_msgs:
            dialogue = dialogue[-max_msgs:]

        return [system_msg] + dialogue

    def _append_to_memory(self, role: str, msg_role: str, content: str) -> None:
        """Append a message to the specified agent's conversation memory."""
        self._memory[role].append({"role": msg_role, "content": content})

    # ----- Inference -----

    def query(
        self,
        agent_role: str,
        user_message: str,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Dict[str, Any]:
        """
        Send a message to the specified agent (nurse/patient) and return
        the parsed JSON action dict. Falls back to a failure response if
        JSON parsing fails or the API is unavailable.

        Args:
            agent_role: "nurse" or "patient"
            user_message: The incoming message (from Doctor or other agent)
            temperature: LLM sampling temperature
            max_tokens: Max tokens for the response

        Returns:
            Parsed JSON action dict with the agent's response.
        """
        # Append incoming user message
        self._append_to_memory(agent_role, "user", user_message)

        # Build windowed context
        messages = self._get_windowed_messages(agent_role)

        # Pick the correct client for this role
        client = self._clients.get(agent_role)

        # --- API Call ---
        if client is not None:
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text = completion.choices[0].message.content or ""
            except Exception as e:
                logger.error(f"Groq API error for {agent_role}: {e}")
                raw_text = ""
        else:
            # --- Mock Mode: return a canned response ---
            raw_text = self._mock_response(agent_role)

        # --- Parse JSON ---
        parsed = _extract_json_from_text(raw_text)

        if parsed is None:
            logger.warning(f"JSON parse failure for {agent_role}. Raw: {raw_text[:200]}")
            parsed = _make_failure_response(agent_role)
        else:
            parsed["_parse_failed"] = False

        # Store assistant response in memory
        self._append_to_memory(agent_role, "assistant", json.dumps(parsed))

        return parsed

    # ----- LLM-as-a-Judge Empathy Evaluation -----

    def evaluate_empathy(self, message: str) -> dict:
        """
        Use a 70B LLM to grade the Doctor's empathy and communication style.
        Returns dict with scores 0.0-1.0 for: empathy, explanation, dismissive, acknowledgment.

        Routing: prefers the **patient** client so the Empathy Judge lives on
        the same API key as the Doctor (Key A), leaving the Medical Judge
        on the Nurse client (Key B) — which keeps both rate-limit budgets
        independent.
        """
        client = self._clients.get("patient") or self._clients.get("nurse")
        default_scores = {"empathy": 0.0, "explanation": 0.0, "dismissive": 0.0, "acknowledgment": 0.0}
        
        if client is None or not message:
            return default_scores

        judge_prompt = (
            "Analyze the following message from a Doctor to a patient/nurse.\n"
            f"Message: \"{message}\"\n\n"
            "Grade the message on a scale of 0.0 to 1.0 for each of these four intents:\n"
            "- empathy: Shows understanding, concern, reassurance, compassion.\n"
            "- explanation: Educates the patient, explains reasoning, outlines plans.\n"
            "- acknowledgment: Actively listens, asks clarifying questions, says 'I see' or 'tell me more'.\n"
            "- dismissive: Curt, ignores concerns, rude, rushing, invalidating.\n\n"
            "Respond ONLY in valid JSON format:\n"
            '{"empathy": <float>, "explanation": <float>, "acknowledgment": <float>, "dismissive": <float>}'
        )

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a communication analysis AI. Output ONLY valid JSON."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.1,
                max_tokens=128,
                response_format={"type": "json_object"},
            )
            raw_text = completion.choices[0].message.content or ""
            parsed = _extract_json_from_text(raw_text)

            if parsed:
                return {
                    "empathy": max(0.0, min(1.0, float(parsed.get("empathy", 0.0)))),
                    "explanation": max(0.0, min(1.0, float(parsed.get("explanation", 0.0)))),
                    "acknowledgment": max(0.0, min(1.0, float(parsed.get("acknowledgment", 0.0)))),
                    "dismissive": max(0.0, min(1.0, float(parsed.get("dismissive", 0.0)))),
                }
            return default_scores
        except Exception as e:
            logger.error(f"Empathy Judge API error: {e}")
            return default_scores

    # ----- LLM-as-a-Judge Treatment Evaluation -----

    def evaluate_treatment(
        self,
        prescribed_treatment: str,
        correct_treatment: str,
        lethal_treatments: list,
        disease_name: str,
    ) -> dict:
        """
        Use a 70B LLM as a Chief Medical Officer to grade the Doctor's
        prescribed treatment against the ground truth.

        Returns:
            {"score": float 0.0-1.0, "is_lethal": bool, "reasoning": str}
        """
        # Treatment / Medical Judge: prefers nurse client → lives on Key B.
        # Empathy Judge prefers patient client → lives on Key A. This keeps
        # the two judges on independent API key buckets.
        client = self._clients.get("nurse") or self._clients.get("patient")
        if client is None:
            logger.warning("No API client available for LLM Judge. Returning default score.")
            return {"score": 0.5, "is_lethal": False, "reasoning": "No API client available."}

        lethal_str = ", ".join(lethal_treatments) if lethal_treatments else "None"

        judge_prompt = (
            "You are the Chief Medical Officer reviewing an ER Doctor's treatment plan.\n\n"
            f"## Disease (Hidden from Doctor during episode)\n{disease_name}\n\n"
            f"## Gold-Standard Correct Treatment\n{correct_treatment}\n\n"
            f"## Known Lethal / Contraindicated Treatments\n{lethal_str}\n\n"
            f"## Doctor's Prescribed Treatment\n{prescribed_treatment}\n\n"
            "## Your Task\n"
            "Grade the Doctor's treatment plan on a scale of 0.0 to 1.0:\n"
            "- 1.0 = Covers all critical interventions, correct drugs, correct dosing approach\n"
            "- 0.75 = Correct diagnosis and most key interventions, minor omissions\n"
            "- 0.5 = Right diagnosis but significant treatment gaps\n"
            "- 0.25 = Partially relevant but missing critical life-saving steps\n"
            "- 0.0 = Completely wrong or dangerous treatment\n\n"
            "IMPORTANT GRADING RULES:\n"
            "- Accept synonyms and equivalent drugs (e.g., 'broad-spectrum antibiotics' partially covers 'cefoxitin')\n"
            "- Accept equivalent procedures described differently (e.g., 'emergency surgery' = 'emergent surgical exploration')\n"
            "- Penalize missing CRITICAL steps (e.g., missing 'NPO' before surgery, missing specific drug classes)\n"
            "- Heavily penalize if a lethal/contraindicated treatment was prescribed\n\n"
            'Respond ONLY in valid JSON:\n'
            '{"score": <float 0.0-1.0>, "is_lethal": <bool>, "reasoning": "<brief explanation>"}'
        )

        try:
            completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical evaluation AI. Output ONLY valid JSON."},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.1,  # Low temp for consistent grading
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            raw_text = completion.choices[0].message.content or ""
            parsed = _extract_json_from_text(raw_text)

            if parsed and "score" in parsed:
                score = max(0.0, min(1.0, float(parsed["score"])))
                is_lethal = bool(parsed.get("is_lethal", False))
                reasoning = parsed.get("reasoning", "")
                logger.info(f"LLM Judge: score={score:.2f}, lethal={is_lethal}, reason={reasoning}")
                return {"score": score, "is_lethal": is_lethal, "reasoning": reasoning}
            else:
                logger.warning(f"LLM Judge returned unparseable response: {raw_text[:200]}")
                return {"score": 0.5, "is_lethal": False, "reasoning": "Judge response unparseable."}

        except Exception as e:
            logger.error(f"LLM Judge API error: {e}")
            return {"score": 0.5, "is_lethal": False, "reasoning": f"API error: {e}"}

    # ----- Mock Responses (for testing without API) -----

    @staticmethod
    def _mock_response(agent_role: str) -> str:
        """Return a valid mock JSON string for testing without the Groq API."""
        if agent_role == "nurse":
            return json.dumps({
                "thought": "Mock nurse: I should check on the patient.",
                "tool": "speak_to",
                "target": "patient",
                "message": "Hello, can you tell me what's bothering you today?",
                "status": "CONTINUE",
            })
        else:
            return json.dumps({
                "thought": "Mock patient: I should describe my symptoms.",
                "tool": "speak_to",
                "target": "nurse",
                "message": "I'm not feeling well. I have pain and feel dizzy.",
                "status": "CONTINUE",
            })
