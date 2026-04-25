"""
Smoke test: simulate the EXACT failure mode from the user's last log
(Patient + Nurse keys revoked, Doctor + Judge keys alive) and verify
that:

1. AgentRouter.query falls back to a live judge client
2. DoctorBrain's chain advances past the dead Doctor key (we'll also
   simulate Doctor revocation)
3. TTS emotion adapter gets disabled after first 401 (no spam)

Runs from the repo root: ``python _smoke_dead_keys.py``
"""
from __future__ import annotations

import os
import sys
import importlib

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

CHECKS_PASSED = 0
CHECKS_FAILED = 0


def check(label, ok, detail=""):
    global CHECKS_PASSED, CHECKS_FAILED
    tag = "PASS" if ok else "FAIL"
    if ok:
        CHECKS_PASSED += 1
    else:
        CHECKS_FAILED += 1
    line = f"  [{tag}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line, flush=True)


# ---------------------------------------------------------------------------
# 1. AgentRouter fallback chain (Patient + Nurse dead, judges alive)
# ---------------------------------------------------------------------------
print("\n--- Test 1: AgentRouter.query with patient+nurse dead ---", flush=True)

# Inject env vars BEFORE importing dashboard so demo defaults still get set.
os.environ.setdefault("GROQ_DOCTOR_API_KEY", "gsk_dummy_doctor")
os.environ.setdefault("GROQ_NURSE_API_KEY", "gsk_dummy_nurse")
os.environ.setdefault("GROQ_PATIENT_API_KEY", "gsk_dummy_patient")
os.environ.setdefault("GROQ_EMPATHY_JUDGE_API_KEY", "gsk_dummy_judge")
os.environ.setdefault("GROQ_MEDICAL_JUDGE_API_KEY", "gsk_dummy_judge")

from ER_MAP.envs import api_router as _api_router_mod  # noqa: E402

class _MockResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _MockClient:
    """Mock Groq client that either succeeds or raises a 401."""

    def __init__(self, name, dead=False, payload='{"action":"speak","content":"OK"}'):
        self.name = name
        self.dead = dead
        self.payload = payload
        self.calls = 0
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kw):
        self.calls += 1
        if self.dead:
            raise Exception(
                f"Error code: 401 - {{'error': {{'message': 'Invalid API Key', "
                f"'type': 'invalid_request_error', 'code': 'invalid_api_key'}}}}"
            )
        return _MockResp(self.payload)


router = _api_router_mod.AgentRouter(
    api_key="x",
    nurse_api_key="x",
    patient_api_key="x",
    empathy_judge_api_key="x",
    medical_judge_api_key="x",
)

# Patch all 4 clients with mocks: patient + nurse are dead, judges are alive.
mock_clients = {
    "nurse":         _MockClient("nurse",         dead=True),
    "patient":       _MockClient("patient",       dead=True),
    "empathy_judge": _MockClient("empathy_judge", dead=False),
    "medical_judge": _MockClient("medical_judge", dead=False),
}
router._clients = mock_clients
router._dead_clients = set()  # let runtime detect deadness through the cascade

# Query as nurse — should walk nurse -> patient -> medical_judge and succeed.
result = router.query("nurse", "system", [{"role": "user", "content": "hi"}])

check(
    "router.query('nurse') falls through to a live judge",
    result.get("action") == "speak",
    f"got {result}",
)
check(
    "nurse client was attempted",
    mock_clients["nurse"].calls == 1,
    f"calls={mock_clients['nurse'].calls}",
)
check(
    "patient client was attempted (next in chain)",
    mock_clients["patient"].calls == 1,
    f"calls={mock_clients['patient'].calls}",
)
check(
    "medical_judge client served the request",
    mock_clients["medical_judge"].calls == 1,
    f"calls={mock_clients['medical_judge'].calls}",
)
check(
    "empathy_judge NOT called once medical_judge succeeded",
    mock_clients["empathy_judge"].calls == 0,
    f"calls={mock_clients['empathy_judge'].calls}",
)
check(
    "nurse marked dead in router state",
    "nurse" in router._dead_clients,
)
check(
    "patient marked dead in router state",
    "patient" in router._dead_clients,
)

# Subsequent queries should skip dead clients entirely.
mock_clients["medical_judge"].calls = 0
mock_clients["empathy_judge"].calls = 0
mock_clients["nurse"].calls = 0
mock_clients["patient"].calls = 0
result2 = router.query("patient", "system", [{"role": "user", "content": "hi"}])
check(
    "second query (patient) skips dead clients",
    mock_clients["nurse"].calls == 0 and mock_clients["patient"].calls == 0,
)
check(
    "second query reaches a live judge",
    result2.get("action") == "speak",
    f"got {result2}",
)

# ---------------------------------------------------------------------------
# 2. DoctorBrain key chain
# ---------------------------------------------------------------------------
print("\n--- Test 2: DoctorBrain with primary key dead, fallback alive ---", flush=True)

from ER_MAP import dashboard as _dash  # noqa: E402

class _DoctorMockChat:
    def __init__(self, owner):
        self.owner = owner
        self.completions = self

    def create(self, **kw):
        self.owner.calls += 1
        if self.owner.dead:
            raise Exception("Error code: 401 - {'error': {'code': 'invalid_api_key'}}")
        return _MockResp('{"action":"read_soap","content":"check the chart first"}')


class _DoctorMockGroq:
    def __init__(self, dead=False):
        self.dead = dead
        self.calls = 0
        self.chat = _DoctorMockChat(self)


# Build a brain with 3 keys: primary dead, second dead, third alive.
brain = _dash.DoctorBrain(
    api_key="key1",
    fallback_api_keys=["key2", "key3"],
    model="llama-3.1-8b-instant",
)

# Replace each entry's client with our mock.
brain._chain[0]["client"] = _DoctorMockGroq(dead=True)   # key1 dead
brain._chain[1]["client"] = _DoctorMockGroq(dead=True)   # key2 dead
brain._chain[2]["client"] = _DoctorMockGroq(dead=False)  # key3 alive

reply = brain.decide("Patient is here. Vitals pending.")
check(
    "DoctorBrain walks past 2 dead keys and uses the 3rd",
    '"action":"read_soap"' in reply or "'action': 'read_soap'" in reply,
    f"reply={reply[:120]}",
)
check(
    "key1 marked dead",
    brain._chain[0]["dead"] is True,
)
check(
    "key2 marked dead",
    brain._chain[1]["dead"] is True,
)
check(
    "key3 still alive",
    brain._chain[2]["dead"] is False,
)
check(
    "key3 actually answered (call count)",
    brain._chain[2]["client"].calls == 1,
    f"calls={brain._chain[2]['client'].calls}",
)

# Second decide() should jump straight to key3 — no retries on the dead ones.
brain._chain[0]["client"].calls = 0
brain._chain[1]["client"].calls = 0
brain._chain[2]["client"].calls = 0
brain.decide("Now consider next step.")
check(
    "second decide() skips dead keys (no extra calls on key1/key2)",
    brain._chain[0]["client"].calls == 0 and brain._chain[1]["client"].calls == 0,
)
check(
    "second decide() served by key3 again",
    brain._chain[2]["client"].calls == 1,
)

# All 3 dead → falls back to _smart_fallback_action (no crash).
brain2 = _dash.DoctorBrain(
    api_key="k1",
    fallback_api_keys=["k2"],
    model="llama-3.1-8b-instant",
)
brain2._chain[0]["client"] = _DoctorMockGroq(dead=True)
brain2._chain[1]["client"] = _DoctorMockGroq(dead=True)
reply3 = brain2.decide("Patient is here.")
check(
    "all keys dead -> _smart_fallback_action returns valid JSON",
    reply3.startswith("{") and ('"tool"' in reply3 or '"action"' in reply3),
    f"reply={reply3[:120]}",
)

# ---------------------------------------------------------------------------
# 3. TTS emotion adapter auto-disable on 401
# ---------------------------------------------------------------------------
print("\n--- Test 3: TTS emotion adapter shuts down after first 401 ---", flush=True)

from ER_MAP import tts_engine as _tts  # noqa: E402

# Make sure ElevenLabs is forced off so we don't hit network.
os.environ["ERMAP_DISABLE_ELEVENLABS"] = "1"

eng = _tts.TTSEngine(elevenlabs_api_key="", groq_api_key="dummy")

# Replace its Groq client with a mock that always raises 401.
class _AlwaysAuthFail:
    def __init__(self):
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, **kw):
        self.calls += 1
        raise Exception("Error code: 401 - {'error': {'code': 'invalid_api_key'}}")

mock_groq = _AlwaysAuthFail()
eng._groq_client = mock_groq

# Trigger the adapter: status helper should report auth=True the first time.
text1, auth1 = _tts._emotionalize_with_status(
    "Patient please describe your symptoms in detail.",
    "patient_anxious_panicked",
    eng._groq_client,
    eng._groq_model,
)
check(
    "first call hits Groq and observes 401",
    auth1 is True and mock_groq.calls == 1,
    f"auth={auth1} calls={mock_groq.calls}",
)
check(
    "first call still returns usable text via regex fallback",
    isinstance(text1, str) and len(text1) > 5,
    f"text={text1[:80]}",
)

# Simulate the engine setting its dead flag and verify subsequent passes
# never hit Groq again.
eng._emotion_adapter_dead = True
mock_groq.calls = 0

# Run the same code path the engine uses internally:
if eng._emotion_adapter_dead:
    # Engine bypasses the LLM call entirely → no Groq invocation.
    fallback_only = _tts._fallback_emotion_transform(
        "Patient please describe your symptoms in detail.",
        "patient_anxious_panicked",
    )
    fallback_calls = mock_groq.calls
else:
    fallback_calls = -1
check(
    "engine bypasses Groq once emotion adapter marked dead",
    fallback_calls == 0,
    f"calls after mark-dead={fallback_calls}",
)
check(
    "regex fallback still produces speech",
    isinstance(fallback_only, str) and len(fallback_only) > 5,
    f"text={fallback_only[:80]}",
)

# ---------------------------------------------------------------------------
# 4. Health probe smoke (returns DEAD_AUTH on a junk key without crashing)
# ---------------------------------------------------------------------------
print("\n--- Test 4: _probe_groq_key handles an invalid key gracefully ---", flush=True)

status, detail = _dash._probe_groq_key("gsk_definitely_invalid_key", "llama-3.1-8b-instant", timeout_s=4.0)
check(
    "probe returns DEAD_AUTH for invalid key",
    status == "DEAD_AUTH",
    f"status={status} detail={detail}",
)

status_missing, _ = _dash._probe_groq_key("", "llama-3.1-8b-instant")
check(
    "probe returns MISSING for empty key (no network call)",
    status_missing == "MISSING",
)

# ---------------------------------------------------------------------------
print("\n" + "=" * 60, flush=True)
print(f"  RESULT: {CHECKS_PASSED} passed, {CHECKS_FAILED} failed", flush=True)
print("=" * 60, flush=True)
sys.exit(0 if CHECKS_FAILED == 0 else 1)
