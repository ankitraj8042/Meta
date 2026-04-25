"""
ER_MAP/dashboard.py
===================
Visual Dashboard for ER-MAP multi-agent triage simulation.
Shows God's View of all agent parameters + live conversation flow.

Usage:
    cd d:/Meta_Finals
    python -m ER_MAP.dashboard
    Open http://localhost:5050 in browser
"""

import json
import os
import sys
import time
import asyncio
import io
import threading
from flask import Flask, jsonify, request, Response, send_file

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Global state
ENV = None
DOCTOR = None
EPISODE_STATE = {
    "active": False,
    "ground_truth": {},
    "conversation": [],
    "metrics": {"total_reward": 0, "step": 0, "outcome": None},
    "obs": "",
    "done": False,
}


def get_env():
    global ENV
    if ENV is None:
        from ER_MAP.envs.triage_env import TriageEnv
        nurse_key = os.environ.get("GROQ_NURSE_API_KEY", "")
        patient_key = os.environ.get("GROQ_PATIENT_API_KEY", "")
        model = os.environ.get("ERMAP_MODEL", "llama-3.3-70b-versatile")
        ENV = TriageEnv(nurse_api_key=nurse_key, patient_api_key=patient_key, model=model)
    return ENV


def get_doctor():
    global DOCTOR
    if DOCTOR is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_DOCTOR_API_KEY", "") or os.environ.get("GROQ_PATIENT_API_KEY", "")
        model = os.environ.get("ERMAP_MODEL", "llama-3.3-70b-versatile")
        DOCTOR = DoctorBrain(api_key=api_key, model=model)
    return DOCTOR


# ---------------------------------------------------------------------------
# Doctor Brain
# ---------------------------------------------------------------------------
DOCTOR_SYSTEM_PROMPT = """You are an AI learning to be an emergency room doctor through reinforcement learning. You have a patient in the ER.

## Available Tools (respond with STRICT JSON)
1. speak_to: {"thought":"...","tool":"speak_to","target":"nurse or patient","message":"..."}
2. order_lab: {"thought":"...","tool":"order_lab","target":"nurse","test_name":"lab name"}
3. read_soap: {"thought":"...","tool":"read_soap","section":"Subjective or Objective or ALL"}
4. update_soap: {"thought":"...","tool":"update_soap","section":"Assessment or Plan","content":"..."}
5. terminal_discharge: {"thought":"...","tool":"terminal_discharge","treatment":"detailed treatment plan"}

## Objective
Your goal is to figure out the correct diagnosis, document your findings, and prescribe a comprehensive treatment plan.
You have NO predefined instructions on how to behave, what order to do things, or how to speak to patients. 
You must learn how to gather information, interact with the patient and nurse, and manage the environment entirely through trial and error. 
The environment will reward you for good clinical processes, effective communication, and correct medical outcomes. It will penalize you for mistakes, rushing, or losing the patient's trust.

RESPOND ONLY WITH VALID JSON."""


class DoctorBrain:
    def __init__(self, api_key, model="llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model
        self.history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

    def reset(self):
        self.history = [{"role": "system", "content": DOCTOR_SYSTEM_PROMPT}]

    def decide(self, observation):
        self.history.append({"role": "user", "content": f"Observation:\n{observation}"})
        if len(self.history) > 17:
            self.history = [self.history[0]] + self.history[-16:]
        try:
            c = self.client.chat.completions.create(
                model=self.model, messages=self.history,
                temperature=0.6, max_tokens=300,
                response_format={"type": "json_object"},
            )
            resp = c.choices[0].message.content or ""
        except Exception as e:
            resp = json.dumps({"thought": f"API error: {e}", "tool": "speak_to", "target": "nurse", "message": "Update me"})
        self.history.append({"role": "assistant", "content": resp})
        return resp


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML_PAGE


# ---------------------------------------------------------------------------
# TTS Engine — uses shared tts_engine module
# ---------------------------------------------------------------------------
from ER_MAP.tts_engine import TTSEngine, get_voice_key, clean_text_for_speech

# Lazy-initialized shared TTS engine for the dashboard
_tts_engine = None

def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine


@app.route("/api/speak", methods=["POST"])
def speak():
    """Generate neural TTS audio for a message."""
    data = request.json or {}
    text = data.get("text", "")
    agent = data.get("agent", "system")

    if not text or len(text.strip()) < 2:
        return jsonify({"error": "no text"}), 400

    print(f"  [TTS] agent={agent} text={text[:120]}", flush=True)

    gt = EPISODE_STATE.get("ground_truth", {})
    tts = _get_tts_engine()

    try:
        # pre_cleaned=True because frontend JS already cleaned the text
        audio_buf = tts.generate(text, agent, gt, pre_cleaned=True)
        if audio_buf is None:
            print(f"  [TTS] generate() returned None for agent={agent}", flush=True)
            return jsonify({"error": "generation failed"}), 500
        buf_size = audio_buf.getbuffer().nbytes
        print(f"  [TTS] Success: agent={agent} buf_size={buf_size}", flush=True)
        return send_file(audio_buf, mimetype="audio/mpeg")
    except Exception as e:
        import traceback
        print(f"  [TTS ERROR] agent={agent}: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/new_episode", methods=["POST"])
def new_episode():
    global EPISODE_STATE
    env = get_env()
    doctor = get_doctor()
    doctor.reset()

    # Accept phase from frontend (default 1)
    req_data = request.json or {}
    phase = req_data.get("phase", 1)
    difficulty_map = {1: "easy", 2: "medium", 3: "hard"}
    options = {"phase": phase, "difficulty": difficulty_map.get(phase, None)}
    print(f"  [ENV] Starting episode: phase={phase}, difficulty={options['difficulty']}", flush=True)

    obs, info = env.reset(options=options)
    gt = env.ground_truth

    EPISODE_STATE = {
        "active": True,
        "ground_truth": gt,
        "conversation": [],
        "metrics": {"total_reward": 0, "step": 0, "outcome": None},
        "obs": obs,
        "done": False,
    }

    # Parse initial obs
    try:
        obs_data = json.loads(obs)
        EPISODE_STATE["conversation"].append({
            "agent": "system",
            "type": "episode_start",
            "message": f"New patient arrived. Nurse experience: {obs_data.get('nurse_experience', '?')}",
            "thought": None,
        })
    except:
        pass

    return jsonify({
        "status": "ok",
        "ground_truth": gt,
        "conversation": EPISODE_STATE["conversation"],
        "metrics": EPISODE_STATE["metrics"],
    })


@app.route("/api/step", methods=["POST"])
def step():
    global EPISODE_STATE
    if not EPISODE_STATE["active"] or EPISODE_STATE["done"]:
        return jsonify({"status": "no_active_episode"})

    env = get_env()
    doctor = get_doctor()

    # Doctor decides
    action_str = doctor.decide(EPISODE_STATE["obs"])
    try:
        action = json.loads(action_str)
    except:
        action = {"thought": "parse error", "tool": "speak_to", "target": "nurse", "message": "Update me"}

    # Log doctor action
    EPISODE_STATE["conversation"].append({
        "agent": "doctor",
        "type": action.get("tool", "speak_to"),
        "target": action.get("target", ""),
        "message": action.get("message", action.get("treatment", action.get("test_name", ""))),
        "thought": action.get("thought", ""),
    })

    # Step environment
    obs, reward, done, truncated, info = env.step(action_str)
    EPISODE_STATE["obs"] = obs
    EPISODE_STATE["metrics"]["total_reward"] = round(EPISODE_STATE["metrics"]["total_reward"] + reward, 2)
    EPISODE_STATE["metrics"]["step"] += 1
    EPISODE_STATE["metrics"]["last_reward"] = round(reward, 2)

    # Parse observation and log responses
    def extract_spoken_text(raw):
        """Extract just the human-readable message from a raw LLM response."""
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed.get("message", parsed.get("reason", raw))
        except (json.JSONDecodeError, TypeError):
            pass
        return raw

    def extract_thought(raw):
        """Extract the thought field from a raw LLM response."""
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed.get("thought", None)
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    try:
        obs_data = json.loads(obs)
        event = obs_data.get("event", "")

        if event == "nurse_report":
            # Log internal exchanges
            for ex in obs_data.get("internal_exchanges", []):
                if "nurse_said" in ex:
                    nurse_raw = ex.get("nurse_said", "")
                    patient_raw = ex.get("patient_said", "")
                    EPISODE_STATE["conversation"].append({
                        "agent": "nurse", "type": "speak_to", "target": "patient",
                        "message": extract_spoken_text(nurse_raw),
                        "thought": extract_thought(nurse_raw),
                    })
                    EPISODE_STATE["conversation"].append({
                        "agent": "patient", "type": "speak_to", "target": "nurse",
                        "message": extract_spoken_text(patient_raw),
                        "thought": extract_thought(patient_raw),
                        "status": ex.get("patient_status", ""),
                    })
                elif "nurse_action" in ex:
                    EPISODE_STATE["conversation"].append({
                        "agent": "nurse", "type": ex.get("nurse_action", ""),
                        "target": "patient",
                        "message": ex.get("result", ex.get("reason", "")),
                        "thought": None,
                    })
            # Nurse report to doctor
            nurse_msg_raw = obs_data.get("nurse_message", "")
            EPISODE_STATE["conversation"].append({
                "agent": "nurse", "type": "report", "target": "doctor",
                "message": extract_spoken_text(nurse_msg_raw),
                "thought": extract_thought(nurse_msg_raw),
            })

        elif event == "patient_response":
            patient_raw = obs_data.get("patient_message", "")
            EPISODE_STATE["conversation"].append({
                "agent": "patient", "type": "speak_to", "target": "doctor",
                "message": extract_spoken_text(patient_raw),
                "thought": extract_thought(patient_raw),
                "status": obs_data.get("patient_status", ""),
            })

        elif event == "lab_result":
            EPISODE_STATE["conversation"].append({
                "agent": "system", "type": "lab_result",
                "message": f"[{obs_data.get('test_name','')}] {obs_data.get('result','')}",
                "thought": None, "redundant": obs_data.get("redundant", False),
            })

        elif "terminal" in event:
            outcome_map = {"terminal_win": "WIN", "terminal_fatal": "FATAL",
                           "terminal_incorrect": "WRONG", "terminal_ama": "AMA"}
            outcome = outcome_map.get(event, event)
            EPISODE_STATE["metrics"]["outcome"] = outcome
            msg = obs_data.get("patient_message", obs_data.get("correct_treatment", ""))
            EPISODE_STATE["conversation"].append({
                "agent": "system", "type": event,
                "message": f"GAME OVER: {outcome}. {extract_spoken_text(msg)}",
                "thought": None,
            })

    except:
        pass

    if done or truncated:
        EPISODE_STATE["done"] = True
        if not EPISODE_STATE["metrics"]["outcome"]:
            EPISODE_STATE["metrics"]["outcome"] = "TRUNCATED"

    return jsonify({
        "status": "ok",
        "conversation": EPISODE_STATE["conversation"],
        "metrics": EPISODE_STATE["metrics"],
        "done": EPISODE_STATE["done"],
        "reward": round(reward, 2),
    })


@app.route("/api/state")
def state():
    return jsonify(EPISODE_STATE)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ER-MAP Mission Control</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 40%, #24243e 100%);
    min-height: 100vh;
    color: #e0e0e0;
    overflow-x: hidden;
}

/* Header */
.header {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #fda085 100%);
    padding: 18px 30px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 30px rgba(240, 147, 251, 0.3);
}
.header h1 {
    font-size: 22px;
    font-weight: 700;
    color: white;
    text-shadow: 0 2px 10px rgba(0,0,0,0.3);
    letter-spacing: 1px;
}
.header .controls {
    display: flex;
    gap: 10px;
}
.btn {
    padding: 10px 22px;
    border: none;
    border-radius: 25px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.btn-new {
    background: rgba(255,255,255,0.95);
    color: #f5576c;
}
.btn-new:hover { transform: scale(1.05); box-shadow: 0 4px 20px rgba(255,255,255,0.3); }
.btn-step {
    background: rgba(255,255,255,0.2);
    color: white;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.3);
}
.btn-step:hover { background: rgba(255,255,255,0.3); }
.btn-auto {
    background: rgba(46, 160, 67, 0.8);
    color: white;
}
.btn-auto:hover { background: rgba(46, 160, 67, 1); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }

/* Main Layout */
.main {
    display: grid;
    grid-template-columns: 280px 1fr 260px;
    gap: 16px;
    padding: 16px;
    height: calc(100vh - 70px);
}

/* Glass Panel */
.panel {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 20px;
    overflow-y: auto;
}
.panel::-webkit-scrollbar { width: 6px; }
.panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }

.panel-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #fda085;
    margin-bottom: 16px;
    font-weight: 600;
}

/* God's View Panel */
.god-section {
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.god-section:last-child { border-bottom: none; }
.god-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #888;
    margin-bottom: 6px;
}
.god-value {
    font-size: 13px;
    font-weight: 500;
    color: #f0f0f0;
    margin-bottom: 4px;
}
.god-disease {
    font-size: 16px;
    font-weight: 700;
    background: linear-gradient(135deg, #f093fb, #f5576c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.god-difficulty {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.diff-easy { background: rgba(46,160,67,0.2); color: #2ea043; }
.diff-medium { background: rgba(240,136,62,0.2); color: #f0883e; }
.diff-hard { background: rgba(248,81,73,0.2); color: #f85149; }
.diff-random { background: rgba(88,166,255,0.2); color: #58a6ff; }

.trait-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
}
.trait-key {
    font-size: 11px;
    color: #888;
}
.trait-val {
    font-size: 11px;
    font-weight: 500;
    color: #c9d1d9;
    background: rgba(255,255,255,0.05);
    padding: 2px 8px;
    border-radius: 8px;
}
.trait-val.bad { color: #f85149; background: rgba(248,81,73,0.1); }
.trait-val.good { color: #2ea043; background: rgba(46,160,67,0.1); }
.trait-val.warn { color: #f0883e; background: rgba(240,136,62,0.1); }

/* Conversation Panel */
.conversation {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-bottom: 20px;
}
.msg {
    max-width: 85%;
    padding: 12px 16px;
    border-radius: 16px;
    position: relative;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

.msg-doctor {
    align-self: flex-end;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-bottom-right-radius: 4px;
}
.msg-nurse {
    align-self: flex-start;
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: #0a2e1a;
    border-bottom-left-radius: 4px;
}
.msg-patient {
    align-self: flex-start;
    background: linear-gradient(135deg, #fc5c7d 0%, #6a82fb 100%);
    border-bottom-left-radius: 4px;
}
.msg-system {
    align-self: center;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.1);
    text-align: center;
    font-size: 12px;
    color: #aaa;
    max-width: 95%;
}
.msg-system.terminal_win { border-color: #2ea043; color: #2ea043; background: rgba(46,160,67,0.1); }
.msg-system.terminal_fatal, .msg-system.terminal_incorrect { border-color: #f85149; color: #f85149; background: rgba(248,81,73,0.1); }
.msg-system.terminal_ama { border-color: #f0883e; color: #f0883e; background: rgba(240,136,62,0.1); }

.msg-agent {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
    margin-bottom: 4px;
    font-weight: 600;
}
.msg-text {
    font-size: 13px;
    line-height: 1.5;
}
.msg-thought {
    font-size: 11px;
    font-style: italic;
    opacity: 0.6;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid rgba(255,255,255,0.15);
}
.msg-meta {
    font-size: 10px;
    opacity: 0.5;
    margin-top: 4px;
}

/* Metrics Panel */
.metric-card {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}
.metric-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #888;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
}
.metric-value.positive { color: #2ea043; }
.metric-value.negative { color: #f85149; }
.metric-value.neutral { color: #58a6ff; }

.reward-history {
    display: flex;
    gap: 4px;
    margin-top: 10px;
    flex-wrap: wrap;
}
.reward-dot {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 9px;
    font-weight: 600;
}

/* Waiting animation */
.waiting {
    display: flex;
    gap: 6px;
    padding: 16px;
    align-self: center;
}
.waiting .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #fda085;
    animation: pulse 1.2s infinite;
}
.waiting .dot:nth-child(2) { animation-delay: 0.2s; }
.waiting .dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes pulse { 0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); } 40% { opacity: 1; transform: scale(1.2); } }

/* Empty state */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #555;
}
.empty-state .icon { font-size: 48px; margin-bottom: 16px; }
.empty-state p { font-size: 14px; }
</style>
</head>
<body>

<div class="header">
    <h1>ER-MAP MISSION CONTROL</h1>
    <div class="controls">
        <button class="btn btn-voice" id="btnVoice" onclick="toggleVoice()" style="background:rgba(168,85,247,0.8);color:white;">🔊 Voice ON</button>
        <select id="phaseSelect" style="padding:6px 12px;border-radius:8px;background:rgba(30,30,60,0.9);color:#fff;border:1px solid rgba(168,85,247,0.5);font-size:13px;cursor:pointer;">
            <option value="1">Phase 1: Tool Mastery</option>
            <option value="2">Phase 2: Clinical Reasoning</option>
            <option value="3">Phase 3: Empathy + Chaos</option>
        </select>
        <button class="btn btn-new" onclick="newEpisode()">New Episode</button>
        <button class="btn btn-step" id="btnStep" onclick="step()" disabled>Next Step</button>
        <button class="btn btn-auto" id="btnAuto" onclick="toggleAuto()" disabled>Auto-Play</button>
    </div>
</div>

<div class="main">
    <!-- Left: God's View -->
    <div class="panel" id="godPanel">
        <div class="panel-title">God's View</div>
        <div id="godContent">
            <div class="empty-state">
                <div class="icon">🏥</div>
                <p>Start an episode to see parameters</p>
            </div>
        </div>
    </div>

    <!-- Center: Conversation -->
    <div class="panel" id="convPanel">
        <div class="panel-title">Agent Interaction</div>
        <div class="conversation" id="conversation">
            <div class="empty-state">
                <div class="icon">💬</div>
                <p>Start an episode to see the conversation</p>
            </div>
        </div>
    </div>

    <!-- Right: Metrics -->
    <div class="panel" id="metricsPanel">
        <div class="panel-title">Episode Metrics</div>
        <div id="metricsContent">
            <div class="metric-card">
                <div class="metric-label">Total Reward</div>
                <div class="metric-value neutral" id="metricReward">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Step</div>
                <div class="metric-value neutral" id="metricStep">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Outcome</div>
                <div class="metric-value neutral" id="metricOutcome">—</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Step Rewards</div>
                <div class="reward-history" id="rewardHistory"></div>
            </div>
        </div>
    </div>
</div>

<script>
let autoPlay = false;
let autoTimer = null;
let convCount = 0;
let voiceEnabled = true;
let currentPatientComm = 'calm_stoic';
let currentNurseExp = 'standard';

const badTraits = ['hostile_aggressive','non_compliant','nil_clueless','poor_uninsured','overworked_exhausted','impatient_abrasive','vague_under_reported','rookie','distracted'];
const goodTraits = ['fully_compliant','calm_stoic','high_expert','wealthy_insured','veteran','idle_fast','high_empathy','accurate_precise'];
const warnTraits = ['cost_constrained','anxious_panicked','partially_compliant','webmd_warrior','low_basic','overworked_exhausted','cold_clinical','disorganized_confused'];

// Clean text for TTS: extract only natural language, strip all code/JSON
function cleanForSpeech(text) {
    if (!text) return '';
    // Try to parse as JSON and extract message
    try {
        const parsed = JSON.parse(text);
        if (parsed && typeof parsed === 'object' && parsed.message) return parsed.message;
    } catch(e) {}
    // Strip JSON-like content
    let clean = text;
    clean = clean.replace(/[{}\[\]]/g, '');
    clean = clean.replace(/"/g, '');
    clean = clean.replace(/'/g, '');
    // Remove field names
    clean = clean.replace(/\b(thought|tool|target|status|test_name|message|speak_to|order_lab|terminal_discharge|check_vitals|leave_hospital|administer_treatment|nurse_message|patient_message|event|nurse_report|patient_response|lab_result|nurse|patient|doctor)\s*:/gi, '');
    // Remove standalone keywords
    clean = clean.replace(/\b(CONTINUE|ESCALATE|AGREE|LEAVE|speak_to|order_lab|terminal_discharge|check_vitals|null|true|false|undefined)\b/gi, '');
    // Remove commas between removed fields
    clean = clean.replace(/\s*,\s*/g, ' ');
    // Collapse whitespace
    clean = clean.replace(/\s+/g, ' ').trim();
    return clean;
}
// --------------- VOICE / TTS ENGINE (Neural) ---------------
let audioQueue = [];
let isPlaying = false;

async function speakMessage(text, agent) {
    if (!voiceEnabled || !text) return;
    audioQueue.push({ text, agent });
    processAudioQueue();
}

async function processAudioQueue() {
    if (isPlaying || audioQueue.length === 0) return;
    isPlaying = true;
    const item = audioQueue.shift();
    try {
        const res = await fetch('/api/speak', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ text: item.text, agent: item.agent })
        });
        if (!res.ok) { isPlaying = false; processAudioQueue(); return; }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        // Timeout fallback — if audio doesn't end in 60s, move on
        const timeout = setTimeout(() => {
            URL.revokeObjectURL(url);
            isPlaying = false;
            processAudioQueue();
        }, 60000);
        audio.onended = () => { clearTimeout(timeout); URL.revokeObjectURL(url); isPlaying = false; processAudioQueue(); };
        audio.onerror = () => { clearTimeout(timeout); URL.revokeObjectURL(url); isPlaying = false; processAudioQueue(); };
        // Handle autoplay policy
        const playPromise = audio.play();
        if (playPromise !== undefined) {
            playPromise.catch(() => { clearTimeout(timeout); isPlaying = false; processAudioQueue(); });
        }
    } catch(e) {
        console.error('TTS error:', e);
        isPlaying = false;
        processAudioQueue();
    }
}

function toggleVoice() {
    voiceEnabled = !voiceEnabled;
    const btn = document.getElementById('btnVoice');
    if (voiceEnabled) {
        btn.textContent = '🔊 Voice ON';
        btn.style.background = 'rgba(168,85,247,0.8)';
    } else {
        audioQueue = [];
        isPlaying = false;
        btn.textContent = '🔇 Voice OFF';
        btn.style.background = 'rgba(100,100,100,0.5)';
    }
}

function traitClass(val) {
    if (badTraits.includes(val)) return 'bad';
    if (goodTraits.includes(val)) return 'good';
    if (warnTraits.includes(val)) return 'warn';
    return '';
}

function diffClass(d) {
    if (d === 'easy') return 'diff-easy';
    if (d === 'medium') return 'diff-medium';
    if (d === 'hard') return 'diff-hard';
    return 'diff-random';
}

function renderGod(gt) {
    const d = gt.disease || {};
    const p = gt.patient || {};
    const n = gt.nurse || {};
    const diff = gt.difficulty || 'random';

    let html = `
        <div class="god-section">
            <div class="god-label">Disease (Hidden from Doctor)</div>
            <div class="god-disease">${d.true_disease || '???'}</div>
            <span class="god-difficulty ${diffClass(diff)}">${diff}</span>
        </div>
        <div class="god-section">
            <div class="god-label">Correct Treatment</div>
            <div class="god-value" style="font-size:11px; color:#2ea043;">${d.correct_treatment || '—'}</div>
        </div>
        <div class="god-section">
            <div class="god-label">Patient Persona</div>
            ${Object.entries(p).map(([k,v]) => `
                <div class="trait-row">
                    <span class="trait-key">${k}</span>
                    <span class="trait-val ${traitClass(v)}">${v}</span>
                </div>
            `).join('')}
        </div>
        <div class="god-section">
            <div class="god-label">Nurse Persona</div>
            ${Object.entries(n).map(([k,v]) => `
                <div class="trait-row">
                    <span class="trait-key">${k}</span>
                    <span class="trait-val ${traitClass(v)}">${v}</span>
                </div>
            `).join('')}
        </div>
        <div class="god-section">
            <div class="god-label">True Symptoms</div>
            ${(d.true_symptoms || []).map(s => `<div class="god-value" style="font-size:11px;">• ${s}</div>`).join('')}
        </div>
    `;
    document.getElementById('godContent').innerHTML = html;
}

function addMessage(msg) {
    const conv = document.getElementById('conversation');
    if (convCount === 0) conv.innerHTML = '';

    const div = document.createElement('div');
    let cls = 'msg msg-' + (msg.agent || 'system');
    if (msg.type && msg.type.startsWith('terminal')) cls += ' ' + msg.type;
    div.className = cls;

    let label = '';
    if (msg.agent === 'doctor') label = '🩺 DOCTOR';
    else if (msg.agent === 'nurse') label = '👩‍⚕️ NURSE';
    else if (msg.agent === 'patient') label = '🤒 PATIENT';
    else label = '⚡ SYSTEM';

    let targetInfo = '';
    if (msg.target) targetInfo = ` → ${msg.target}`;
    if (msg.type === 'order_lab') targetInfo = ' → LAB ORDER';
    if (msg.type === 'terminal_discharge') targetInfo = ' → DISCHARGE';
    if (msg.type === 'check_vitals') label = '👩‍⚕️ NURSE (VITALS)';
    if (msg.type === 'lab_result') label = '🧪 LAB RESULT';

    let html = `<div class="msg-agent">${label}${targetInfo}</div>`;
    html += `<div class="msg-text">${msg.message || ''}</div>`;
    if (msg.thought) html += `<div class="msg-thought">💭 ${msg.thought}</div>`;
    if (msg.status && msg.status !== 'CONTINUE') html += `<div class="msg-meta">Status: ${msg.status}</div>`;

    div.innerHTML = html;
    conv.appendChild(div);
    conv.scrollTop = conv.scrollHeight;
    convCount++;

    // Speak ONLY actual dialogue — no labs, no system, no code
    if (msg.message && msg.agent !== 'system') {
        const spokenTypes = ['speak_to', 'report', 'terminal_discharge'];
        if (spokenTypes.includes(msg.type)) {
            const clean = cleanForSpeech(msg.message);
            if (clean.length > 3) speakMessage(clean, msg.agent);
        }
    }
}

function updateMetrics(m) {
    const r = m.total_reward || 0;
    const el = document.getElementById('metricReward');
    el.textContent = (r >= 0 ? '+' : '') + r.toFixed(2);
    el.className = 'metric-value ' + (r > 0 ? 'positive' : r < 0 ? 'negative' : 'neutral');
    document.getElementById('metricStep').textContent = m.step || 0;

    const outcome = m.outcome;
    const oel = document.getElementById('metricOutcome');
    if (outcome) {
        oel.textContent = outcome;
        if (outcome === 'WIN') oel.className = 'metric-value positive';
        else if (outcome === 'AMA') oel.className = 'metric-value negative';
        else oel.className = 'metric-value negative';
    } else {
        oel.textContent = 'In Progress...';
        oel.className = 'metric-value neutral';
    }
}

function addRewardDot(reward) {
    const container = document.getElementById('rewardHistory');
    const dot = document.createElement('div');
    dot.className = 'reward-dot';
    dot.textContent = (reward >= 0 ? '+' : '') + reward.toFixed(1);
    if (reward >= 1.5) { dot.style.background = 'rgba(46,160,67,0.3)'; dot.style.color = '#2ea043'; }
    else if (reward > 0) { dot.style.background = 'rgba(88,166,255,0.2)'; dot.style.color = '#58a6ff'; }
    else if (reward > -0.5) { dot.style.background = 'rgba(240,136,62,0.2)'; dot.style.color = '#f0883e'; }
    else { dot.style.background = 'rgba(248,81,73,0.2)'; dot.style.color = '#f85149'; }
    container.appendChild(dot);
}

function showWaiting() {
    const conv = document.getElementById('conversation');
    const w = document.createElement('div');
    w.className = 'waiting';
    w.id = 'waitingDots';
    w.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    conv.appendChild(w);
    conv.scrollTop = conv.scrollHeight;
}

function hideWaiting() {
    const w = document.getElementById('waitingDots');
    if (w) w.remove();
}

async function newEpisode() {
    convCount = 0;
    document.getElementById('rewardHistory').innerHTML = '';
    document.getElementById('conversation').innerHTML = '';
    document.getElementById('metricOutcome').textContent = '—';
    document.getElementById('metricOutcome').className = 'metric-value neutral';

    showWaiting();
    const phase = parseInt(document.getElementById('phaseSelect').value) || 1;
    const res = await fetch('/api/new_episode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ phase: phase })
    });
    const data = await res.json();
    hideWaiting();

    renderGod(data.ground_truth);
    // Update voice persona settings from ground truth
    if (data.ground_truth.patient) currentPatientComm = data.ground_truth.patient.communication || 'calm_stoic';
    if (data.ground_truth.nurse) currentNurseExp = data.ground_truth.nurse.experience || 'standard';
    data.conversation.forEach(addMessage);
    updateMetrics(data.metrics);

    document.getElementById('btnStep').disabled = false;
    document.getElementById('btnAuto').disabled = false;
}

async function step() {
    document.getElementById('btnStep').disabled = true;
    showWaiting();

    const res = await fetch('/api/step', { method: 'POST' });
    const data = await res.json();
    hideWaiting();

    if (data.status === 'no_active_episode') return;

    // Render only new messages
    const rendered = document.querySelectorAll('.msg').length;
    data.conversation.slice(rendered).forEach(addMessage);
    updateMetrics(data.metrics);
    if (data.reward !== undefined) addRewardDot(data.reward);

    if (data.done) {
        document.getElementById('btnStep').disabled = true;
        document.getElementById('btnAuto').disabled = true;
        stopAuto();
    } else {
        document.getElementById('btnStep').disabled = false;
    }
}

function toggleAuto() {
    if (autoPlay) {
        stopAuto();
    } else {
        autoPlay = true;
        document.getElementById('btnAuto').textContent = 'Stop';
        document.getElementById('btnAuto').style.background = 'rgba(248,81,73,0.8)';
        autoStep();
    }
}

async function autoStep() {
    if (!autoPlay) return;
    await step();
    // Wait for all audio to finish before next step
    await waitForAudio();
    if (autoPlay) autoTimer = setTimeout(autoStep, 2000);
}

function waitForAudio() {
    return new Promise(resolve => {
        function check() {
            if (!isPlaying && audioQueue.length === 0) { resolve(); return; }
            setTimeout(check, 500);
        }
        check();
    });
}

function stopAuto() {
    autoPlay = false;
    if (autoTimer) clearTimeout(autoTimer);
    document.getElementById('btnAuto').textContent = 'Auto-Play';
    document.getElementById('btnAuto').style.background = 'rgba(46,160,67,0.8)';
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  ER-MAP Dashboard: http://localhost:5050\n", flush=True)
    app.run(host="0.0.0.0", port=5050, debug=False)
