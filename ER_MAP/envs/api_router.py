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
        empathy_judge_api_key: Optional[str] = None,
        medical_judge_api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        nurse_model: Optional[str] = None,
        patient_model: Optional[str] = None,
        empathy_judge_model: Optional[str] = None,
        medical_judge_model: Optional[str] = None,
    ):
        # Default / shared model (used as fallback when a role-specific
        # model is not provided). Kept as ``self.model`` for backward
        # compatibility with any caller that reads it directly.
        self.model = model

        # Per-role models: each role can run a different Groq model so the
        # Doctor (e.g. 8B for speed/quota) is independent of the Judges
        # (70B for nuanced grading).
        self._models: Dict[str, str] = {
            "nurse":          nurse_model          or model,
            "patient":        patient_model        or model,
            "empathy_judge":  empathy_judge_model  or model,
            "medical_judge":  medical_judge_model  or model,
        }

        # Resolve per-role API keys (explicit > role-specific env > shared)
        shared_key      = api_key or os.environ.get("GROQ_API_KEY", "")
        nurse_key       = nurse_api_key         or os.environ.get("GROQ_NURSE_API_KEY", "")          or shared_key
        patient_key     = patient_api_key       or os.environ.get("GROQ_PATIENT_API_KEY", "")        or shared_key
        empathy_key     = empathy_judge_api_key or os.environ.get("GROQ_EMPATHY_JUDGE_API_KEY", "")  or ""
        medical_key     = medical_judge_api_key or os.environ.get("GROQ_MEDICAL_JUDGE_API_KEY", "")  or ""

        # Create per-role Groq clients (4 logical roles).
        # If a judge has no dedicated key, its client is left None and the
        # corresponding evaluator silently falls back to the
        # nurse / patient client (legacy behaviour) via _pick_live_client.
        self._clients: Dict[str, Any] = {
            "nurse":          None,
            "patient":        None,
            "empathy_judge":  None,
            "medical_judge":  None,
        }

        if GROQ_AVAILABLE:
            for role_name, role_key in [
                ("nurse",         nurse_key),
                ("patient",       patient_key),
                ("empathy_judge", empathy_key),
                ("medical_judge", medical_key),
            ]:
                if role_key:
                    self._clients[role_name] = Groq(api_key=role_key)
                    logger.info(
                        f"{role_name} client: LIVE (Groq API, model={self._models[role_name]})"
                    )
                else:
                    if role_name in {"empathy_judge", "medical_judge"}:
                        logger.info(
                            f"No dedicated key for {role_name}; will reuse "
                            f"{'patient' if role_name == 'empathy_judge' else 'nurse'} client."
                        )
                    else:
                        logger.warning(f"No API key for {role_name}. Using mock mode.")
        else:
            logger.warning("groq package not installed. All agents in mock mode.")

        # Episode conversation histories: only the conversational roles
        # (nurse / patient) actually carry a per-episode chat memory; the
        # judges are stateless evaluators.
        self._memory: Dict[str, List[Dict[str, str]]] = {
            "nurse": [],
            "patient": [],
        }

        # Tracks which Groq clients have permanently failed auth (HTTP 401
        # or invalid_api_key) during this process. Once a client is in
        # this set, the judges automatically route through the surviving
        # client instead of retrying a dead key on every step.
        self._dead_clients: set = set()

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

        # Build a fallback chain across ALL 4 client roles. The conversational
        # role's own client is tried first; if dead, we walk through the
        # remaining live clients in priority order (peer conversational role
        # first, then the two judges) so the demo survives 3-of-4 dead keys.
        if agent_role == "nurse":
            attempt_order = ["nurse", "patient", "medical_judge", "empathy_judge"]
        else:
            attempt_order = ["patient", "nurse", "empathy_judge", "medical_judge"]

        # Each role can run a different Groq model — pick the model
        # registered for the *original* requested role even if we ended up
        # using another client's key, so personas stay consistent.
        role_model = self._models.get(agent_role, self.model)

        raw_text = ""
        attempted_any = False
        for attempt_role in attempt_order:
            if attempt_role in self._dead_clients:
                continue
            attempt_client = self._clients.get(attempt_role)
            if attempt_client is None:
                continue
            attempted_any = True
            try:
                completion = attempt_client.chat.completions.create(
                    model=role_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_text = completion.choices[0].message.content or ""
                break  # success → stop the cascade
            except Exception as e:
                logger.error(
                    f"Groq API error for {agent_role} (via {attempt_role} client, model={role_model}): {e}"
                )
                # Persist a 401 so subsequent steps don't keep retrying.
                if self._mark_dead_if_auth_error(attempt_role, e):
                    continue  # auth error → try the next client in the chain
                break  # non-auth error → don't burn rate budget on every key

        if not attempted_any:
            # All four clients are dead or never configured.
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

    # ----- Live-client picker (auto-fallback for dead keys) -----

    def _pick_live_client(self, primary: str, fallback: str):
        """
        Return the first non-dead Groq client in [primary, fallback].

        When one of the demo Groq keys gets revoked / hits 401, the
        corresponding role's client is added to ``self._dead_clients`` and
        the judges silently route through the surviving key.
        """
        order = [primary, fallback]
        for role in order:
            if role in self._dead_clients:
                continue
            client = self._clients.get(role)
            if client is not None:
                return client, role
        return None, None

    def _mark_dead_if_auth_error(self, role: str, err: Exception) -> bool:
        """
        Inspect the exception. If it indicates an auth failure, blacklist
        the client for the rest of this process and return True.
        """
        msg = str(err).lower()
        if (
            "401" in msg
            or "invalid_api_key" in msg
            or "invalid api key" in msg
            or "unauthorized" in msg
        ):
            if role not in self._dead_clients:
                logger.error(
                    f"Groq client '{role}' failed auth (401). "
                    f"Blacklisting for the rest of this session; "
                    f"judges will route through the surviving key."
                )
                self._dead_clients.add(role)
            return True
        return False

    # ----- LLM-as-a-Judge Empathy Evaluation -----

    def evaluate_empathy(self, message: str) -> dict:
        """
        Use a 70B LLM to grade the Doctor's empathy and communication style.
        Returns dict with scores 0.0-1.0 for: empathy, explanation, dismissive, acknowledgment.

        Client routing (in order of preference):
            1. ``empathy_judge`` — dedicated client (own Groq key)
            2. ``patient``       — legacy fallback (shares Key A with Doctor)
            3. ``nurse``         — last-resort fallback

        Each fallback step is tried automatically if the previous client is
        blacklisted (401) or hits an auth error mid-call.
        """
        default_scores = {"empathy": 0.0, "explanation": 0.0, "dismissive": 0.0, "acknowledgment": 0.0}
        if not message:
            return default_scores

        # Routing order — dedicated client first, then legacy fallbacks.
        attempt_order = ["empathy_judge", "patient", "nurse"]
        # Models: empathy_judge uses its own model; legacy fallbacks use
        # whatever model is registered for that role.
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

        for attempt_role in attempt_order:
            if attempt_role in self._dead_clients:
                continue
            attempt_client = self._clients.get(attempt_role)
            if attempt_client is None:
                continue
            attempt_model = self._models.get(attempt_role, self.model)
            try:
                completion = attempt_client.chat.completions.create(
                    model=attempt_model,
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
                logger.error(f"Empathy Judge API error (via {attempt_role}, model={attempt_model}): {e}")
                if self._mark_dead_if_auth_error(attempt_role, e):
                    continue  # try the next role in attempt_order
                return default_scores

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
        # Routing order for the Medical Judge:
        #     1. ``medical_judge`` — dedicated Groq key (preferred)
        #     2. ``nurse``         — legacy fallback (shares Key B)
        #     3. ``patient``       — last-resort fallback
        # Each step is skipped if its client has been blacklisted (401).
        attempt_order = ["medical_judge", "nurse", "patient"]
        live_clients = [
            r for r in attempt_order
            if r not in self._dead_clients and self._clients.get(r) is not None
        ]
        if not live_clients:
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

        for attempt_role in live_clients:
            if attempt_role in self._dead_clients:
                continue
            attempt_client = self._clients.get(attempt_role)
            if attempt_client is None:
                continue
            attempt_model = self._models.get(attempt_role, self.model)
            try:
                completion = attempt_client.chat.completions.create(
                    model=attempt_model,
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
                    logger.info(
                        f"LLM Judge (via {attempt_role}, model={attempt_model}): "
                        f"score={score:.2f}, lethal={is_lethal}, reason={reasoning}"
                    )
                    return {"score": score, "is_lethal": is_lethal, "reasoning": reasoning}
                else:
                    logger.warning(f"LLM Judge returned unparseable response: {raw_text[:200]}")
                    return {"score": 0.5, "is_lethal": False, "reasoning": "Judge response unparseable."}

            except Exception as e:
                logger.error(f"LLM Judge API error (via {attempt_role}, model={attempt_model}): {e}")
                if self._mark_dead_if_auth_error(attempt_role, e):
                    continue  # try the next role in attempt_order
                return {"score": 0.5, "is_lethal": False, "reasoning": f"API error: {e}"}

        return {"score": 0.5, "is_lethal": False, "reasoning": "All judge clients unavailable."}

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
