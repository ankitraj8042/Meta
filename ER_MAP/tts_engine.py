"""
ER_MAP/tts_engine.py
====================
Standalone emotion-induced TTS engine for all three ER-MAP agents.
ElevenLabs (premium, realistic) with Edge-TTS (free) fallback.

Each agent gets a unique voice mapped to their persona traits:
  - Patient voice varies by communication style (hostile/anxious/calm/confused)
  - Nurse voice varies by experience level (rookie/standard/veteran)
  - Doctor voice is calm and authoritative

Usage:
    from ER_MAP.tts_engine import TTSEngine
    tts = TTSEngine()
    tts.speak("I have chest pain", "patient", ground_truth)
"""

import os
import io
import re
import json
import random
import logging
import tempfile
import time
from typing import Optional, Dict, Any

logger = logging.getLogger("ER_MAP.tts_engine")

# ---------------------------------------------------------------------------
# ElevenLabs Voice Configurations (per persona trait)
# ---------------------------------------------------------------------------
ELEVEN_VOICES = {
    "doctor": {
        "voice_id": "onwK4e9ZLuTAKqWW03F9",   # Daniel — British, calm, deep
        "settings": {"stability": 0.70, "similarity_boost": 0.80, "style": 0.35, "speed": 0.92},
    },
    "patient_hostile_aggressive": {
        "voice_id": "N2lVS1w4EtoT3dr4eOWO",   # Callum — intense, assertive
        "settings": {"stability": 0.25, "similarity_boost": 0.85, "style": 0.85, "speed": 1.15},
    },
    "patient_anxious_panicked": {
        "voice_id": "pFZP5JQG7iQjIQuC4Bku",   # Lily — young, nervous
        "settings": {"stability": 0.20, "similarity_boost": 0.75, "style": 0.90, "speed": 1.20},
    },
    "patient_calm_stoic": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",   # Adam — deep, composed
        "settings": {"stability": 0.75, "similarity_boost": 0.80, "style": 0.25, "speed": 0.88},
    },
    "patient_disorganized_confused": {
        "voice_id": "XB0fDUnXU5powFXDhCwa",   # Charlotte — uncertain
        "settings": {"stability": 0.30, "similarity_boost": 0.70, "style": 0.65, "speed": 0.92},
    },
    "nurse_rookie": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",   # Sarah — young, unsure
        "settings": {"stability": 0.30, "similarity_boost": 0.75, "style": 0.70, "speed": 1.08},
    },
    "nurse_standard": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",   # Rachel — professional
        "settings": {"stability": 0.55, "similarity_boost": 0.80, "style": 0.35, "speed": 1.00},
    },
    "nurse_veteran": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",   # Rachel — confident, steady
        "settings": {"stability": 0.80, "similarity_boost": 0.85, "style": 0.20, "speed": 0.88},
    },
}

# ---------------------------------------------------------------------------
# Edge-TTS Fallback Voice Configurations
# ---------------------------------------------------------------------------
EDGE_VOICE_MAP = {
    "doctor":                        {"voice": "en-US-GuyNeural",     "rate": "-5%",  "pitch": "-5Hz"},
    "patient_hostile_aggressive":    {"voice": "en-US-ChristopherNeural", "rate": "+10%", "pitch": "-8Hz"},
    "patient_anxious_panicked":      {"voice": "en-US-AndrewNeural",  "rate": "+15%", "pitch": "+3Hz"},
    "patient_calm_stoic":            {"voice": "en-US-BrianNeural",   "rate": "-8%",  "pitch": "-3Hz"},
    "patient_disorganized_confused": {"voice": "en-US-AnaNeural",     "rate": "-5%",  "pitch": "+3Hz"},
    "nurse_rookie":                  {"voice": "en-US-AriaNeural",    "rate": "+10%", "pitch": "+5Hz"},
    "nurse_standard":                {"voice": "en-US-JennyNeural",   "rate": "+0%",  "pitch": "+0Hz"},
    "nurse_veteran":                 {"voice": "en-US-JennyNeural",   "rate": "-8%",  "pitch": "-3Hz"},
}

# ---------------------------------------------------------------------------
# Emotional Text Transforms
#   Tier 1: LLM-powered rewrite (preferred — natural, varied, persona-aware)
#   Tier 2: Simple random transforms (fallback if no API client)
# ---------------------------------------------------------------------------

# ---- Persona instructions for the LLM emotion adapter ----
_PERSONA_INSTRUCTIONS = {
    "patient_hostile_aggressive": (
        "You are an angry, hostile patient in an ER. "
        "Rewrite this text to sound aggressive, impatient, confrontational. "
        "Add sharp exclamations, interrupting interjections (Look!, Listen!, I said!), "
        "short frustrated sentences. Replace periods with exclamation marks where natural. "
        "Include ElevenLabs tags: [frustrated], [sigh], and use '—' for sharp pauses. "
        "Make every rewrite UNIQUE — never repeat the same pattern twice. "
        "Vary the intensity: sometimes seething quiet rage, sometimes explosive anger."
    ),
    "patient_anxious_panicked": (
        "You are a terrified, panicking patient in an ER. "
        "Rewrite this text to sound breathless, scared, and rushed. "
        "Add stuttering (w-what, I-I can't), filler words (um, oh god, please), "
        "trailing off (using ...), and voice breaks. "
        "Include ElevenLabs tags: [nervous], [gasps], [stammers], [short pause]. "
        "Vary the panic level: sometimes hyperventilating terror, sometimes "
        "quiet trembling fear, sometimes tearful pleading. Never repeat the same opener."
    ),
    "patient_calm_stoic": (
        "You are a calm, stoic patient in an ER. "
        "Rewrite this text with measured, composed speech. "
        "Add thoughtful pauses (...), understated reactions. "
        "Include ElevenLabs tags: [calm], [short pause]. "
        "Keep the tone steady and slightly detached, but natural — not robotic."
    ),
    "patient_disorganized_confused": (
        "You are a confused, disoriented patient in an ER. "
        "Rewrite this text to sound scattered and lost. "
        "Add mid-sentence restarts (wait... what was I...), long pauses, "
        "jumbled word order, and trailing thoughts. "
        "Include ElevenLabs tags: [stammers], [long pause], [short pause]. "
        "Vary confusion style: sometimes foggy, sometimes tangential rambling, "
        "sometimes childlike simplicity. Never use the same filler twice in a row."
    ),
    "nurse_rookie": (
        "You are a nervous rookie nurse in an ER. "
        "Rewrite this text with hedging language (I think, it seems like, um), "
        "slight uncertainty, and over-explanation. "
        "Include ElevenLabs tags: [clears throat], [nervous]. "
        "Sometimes confident on facts but unsure on interpretation."
    ),
    "nurse_standard": (
        "You are a professional ER nurse. "
        "Rewrite this text to sound crisp and efficient. "
        "Minimal emotional color — just professional delivery. "
        "You may add brief [short pause] between medical facts."
    ),
    "nurse_veteran": (
        "You are a veteran ER nurse who has seen everything. "
        "Rewrite this text with calm authority, maybe slight weariness. "
        "Occasional [sigh] before delivering routine information. "
        "Direct, no-nonsense phrasing."
    ),
    "doctor": (
        "You are a calm, authoritative ER doctor. "
        "Rewrite this text with measured, professional delivery. "
        "Add [calm] and [short pause] tags for thoughtful pacing. "
        "Warm but clinical."
    ),
}


def emotionalize_for_tts(text: str, voice_key: str, groq_client=None, model: str = "llama-3.3-70b-versatile") -> str:
    """
    LLM-powered emotion adapter for ElevenLabs TTS.

    Takes clean, clinical LLM agent text and rewrites it into emotionally
    expressive natural speech WITH ElevenLabs audio tags.

    This function is ONLY called in the TTS pipeline — the emotionalized
    text never reaches the RL agents, so agent behavior is unaffected.

    Args:
        text: Clean agent message text
        voice_key: Persona voice key (e.g. "patient_anxious_panicked")
        groq_client: Optional Groq API client. Falls back to regex transforms if None.
        model: Model to use for rewriting (default: 70B)

    Returns:
        Emotionally rewritten text with ElevenLabs audio tags.
    """
    if not text or len(text.strip()) < 5:
        return text

    persona_instruction = _PERSONA_INSTRUCTIONS.get(voice_key)
    if not persona_instruction or groq_client is None:
        # Fallback to simple regex transforms
        return _fallback_emotion_transform(text, voice_key)

    prompt = (
        f"{persona_instruction}\n\n"
        f"ORIGINAL TEXT:\n\"{text}\"\n\n"
        f"RULES:\n"
        f"- Output ONLY the rewritten speech text. No quotes, no labels, no explanations.\n"
        f"- Keep the medical content and meaning EXACTLY the same.\n"
        f"- Only change HOW it is said, not WHAT is said.\n"
        f"- Add ElevenLabs audio tags like [sigh], [gasps], [nervous], [calm], etc.\n"
        f"- Use '...' for trailing off and '—' for sharp pauses.\n"
        f"- Keep the output under {len(text) + 80} characters.\n"
        f"- Make it sound like a REAL person talking, not a script."
    )

    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You rewrite clinical text into emotionally expressive natural speech for text-to-speech synthesis. Output ONLY the rewritten text."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,  # High temp for variety — prevents repetition
            max_tokens=256,
        )
        result = (completion.choices[0].message.content or "").strip()
        # Strip any accidental quotes the model wraps around the output
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        if result and len(result) > 5:
            return result
        return _fallback_emotion_transform(text, voice_key)
    except Exception as e:
        logger.warning(f"LLM emotion adapter failed ({voice_key}): {e}")
        return _fallback_emotion_transform(text, voice_key)


def _fallback_emotion_transform(text: str, voice_key: str) -> str:
    """Simple regex-based fallback when LLM adapter is unavailable."""
    if voice_key == "patient_hostile_aggressive":
        text = text.replace('. ', '! ')
        if not text.endswith('!') and not text.endswith('?'):
            text = text.rstrip('.') + '!'
        interjections = ["Look, ", "Listen! ", "I said, ", "For God's sake, "]
        if random.random() < 0.4 and len(text) > 20:
            text = random.choice(interjections) + text[0].lower() + text[1:]
    elif voice_key == "patient_anxious_panicked":
        words = text.split()
        result = []
        for i, word in enumerate(words):
            if i < 3 and random.random() < 0.3 and len(word) > 2:
                result.append(word[0] + '-' + word)
            elif random.random() < 0.15:
                result.append(random.choice(['um,', 'uh,', 'oh god,']))
                result.append(word)
            else:
                result.append(word)
        text = ' '.join(result)
    elif voice_key == "patient_disorganized_confused":
        if random.random() < 0.5:
            text = 'I... ' + text[0].lower() + text[1:]
    elif voice_key == "nurse_rookie":
        hedges = ['I think ', 'It looks like ', 'Um, ']
        if random.random() < 0.4:
            text = random.choice(hedges) + text[0].lower() + text[1:]
    return text


# ---------------------------------------------------------------------------
# Natural Speech Markers — TTS-ONLY preprocessing (never fed to LLM agents)
# These markers are interpreted by ElevenLabs v2 as natural vocal behaviors.
# ---------------------------------------------------------------------------

def _inject_speech_markers(text: str, voice_key: str) -> str:
    """
    Inject natural speech markers into text BEFORE sending to ElevenLabs.
    This function is ONLY called in the TTS pipeline — the markers never
    reach the LLM agents, so agent behavior is unaffected.

    ElevenLabs v3 Audio Tags (bracketed cues):
      - [sigh], [laughs], [gasps], [clears throat]  — non-verbal reactions
      - [nervous], [excited], [frustrated], [calm]   — emotional states
      - [whispers], [stammers]                       — delivery style
      - [short pause], [long pause]                  — timing control
      - '...'  and '—'                               — natural pacing
    """
    if not text or len(text) < 10:
        return text

    if voice_key == "patient_hostile_aggressive":
        # Frustrated, aggressive delivery
        if random.random() < 0.5:
            text = '[frustrated] ' + text
        sentences = text.split('. ')
        marked = []
        for i, s in enumerate(sentences):
            if random.random() < 0.3:
                s = s.replace(',', ' —')  # sharp pause
            if i > 0 and random.random() < 0.25:
                s = '[sigh] ' + s
            marked.append(s)
        text = '. '.join(marked)
        if random.random() < 0.3:
            text += ' [sigh]'

    elif voice_key == "patient_anxious_panicked":
        # Nervous, gasping, rapid
        opener = random.choice(['[nervous] ', '[gasps] ', '[stammers] '])
        if random.random() < 0.6:
            text = opener + text
        text = text.replace('. ', '... ')
        if 'pain' in text.lower():
            text = text.replace('pain', '[gasps] pain', 1)
        if random.random() < 0.4:
            text += '... [short pause] [nervous]'

    elif voice_key == "patient_calm_stoic":
        # Measured, composed
        text = text.replace('. ', '... ')
        if random.random() < 0.3:
            text = '[calm] ' + text

    elif voice_key == "patient_disorganized_confused":
        # Losing train of thought, long pauses
        sentences = text.split('. ')
        marked = []
        for i, s in enumerate(sentences):
            if i > 0 and random.random() < 0.4:
                filler = random.choice(['[long pause]', '... [short pause]', '[stammers]'])
                marked.append(filler + ' ' + s)
            else:
                marked.append(s)
        text = '. '.join(marked)
        if random.random() < 0.3:
            text += '... [long pause] I lost my train of thought.'

    elif voice_key == "nurse_rookie":
        # Nervous, uncertain
        if random.random() < 0.4:
            text = '[clears throat] ' + text
        if random.random() < 0.3:
            text = '[nervous] ' + text
        text = text.replace('. ', '... ')

    elif voice_key == "nurse_veteran":
        # Efficient, maybe tired
        if random.random() < 0.2:
            text = '[sigh] ' + text

    elif voice_key == "nurse_standard":
        # Professional, minimal markers
        pass

    elif voice_key == "doctor":
        # Calm, authoritative, measured pauses
        if random.random() < 0.25:
            text = '[calm] ' + text
        if random.random() < 0.3:
            text = text.replace('. ', '. [short pause] ')

    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_voice_key(agent: str, ground_truth: Dict) -> str:
    """Resolve agent + persona traits → voice lookup key."""
    if agent == "doctor":
        return "doctor"
    elif agent == "patient":
        comm = ground_truth.get("patient", {}).get("communication", "calm_stoic")
        return f"patient_{comm}"
    elif agent == "nurse":
        exp = ground_truth.get("nurse", {}).get("experience", "standard")
        return f"nurse_{exp}"
    return "doctor"


def clean_text_for_speech(text: str) -> str:
    """Strip JSON/code artifacts, extract only natural spoken language."""
    if not text:
        return ""
    # Try to parse as JSON and extract message field
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = parsed.get("message", parsed.get("patient_said",
                    parsed.get("nurse_said", str(parsed))))
    except (json.JSONDecodeError, TypeError):
        pass
    # Remove JSON syntax
    text = re.sub(r'[{}\[\]]', '', text)
    text = text.replace('"', '').replace("'", '')
    # Remove JSON field names
    text = re.sub(
        r'\b(thought|tool|target|status|test_name|message|speak_to|order_lab|'
        r'terminal_discharge|check_vitals|leave_hospital|administer_treatment|'
        r'nurse|CONTINUE|ESCALATE|AGREE|LEAVE)\s*:', '', text
    )
    # Remove standalone keywords
    text = re.sub(
        r'\b(speak_to|order_lab|terminal_discharge|check_vitals|'
        r'CONTINUE|ESCALATE|null|true|false|undefined)\b', '', text
    )
    text = re.sub(r'\\n|\\t|\\r', ' ', text)
    text = re.sub(r'\s*,\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:300]


# ---------------------------------------------------------------------------
# TTSEngine — main class
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    Emotion-induced neural TTS engine for ER-MAP agents.

    Two-layer emotion system:
      Layer 1: LLM Emotion Adapter (70B) — rewrites clean text into
               emotionally expressive natural speech with ElevenLabs tags.
               Only used for TTS; RL agents never see the rewritten text.
      Layer 2: ElevenLabs voice settings — stability, style, speed tuned
               per persona for vocal quality.

    Supports ElevenLabs (premium, ultra-realistic) with automatic
    Edge-TTS fallback (free, unlimited). Each agent gets a unique
    voice mapped to their persona traits.
    """

    def __init__(self, elevenlabs_api_key: Optional[str] = None, groq_api_key: Optional[str] = None):
        self.api_key = elevenlabs_api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        self.use_elevenlabs = False
        self._eleven_client = None
        self._pygame = None
        self._has_pygame = False
        self._groq_client = None
        self._groq_model = "llama-3.3-70b-versatile"
        # Initialize ElevenLabs
        if self.api_key:
            try:
                from elevenlabs.client import ElevenLabs
                self._eleven_client = ElevenLabs(api_key=self.api_key)
                self.use_elevenlabs = True
                logger.info("TTS Engine: ElevenLabs (premium voices)")
            except ImportError:
                logger.warning("elevenlabs package not installed. Using Edge-TTS.")

        if not self.use_elevenlabs:
            logger.info("TTS Engine: Edge-TTS (free fallback)")

        # Initialize Groq client for LLM Emotion Adapter
        _groq_key = groq_api_key or os.environ.get("GROQ_NURSE_API_KEY") or os.environ.get("GROQ_API_KEY", "")
        if _groq_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=_groq_key)
                logger.info("TTS Emotion Adapter: LLM-powered (70B)")
            except ImportError:
                logger.warning("groq package not installed. Using regex emotion fallback.")
        else:
            logger.info("TTS Emotion Adapter: regex fallback (no GROQ key)")

        # Initialize pygame for audio playback
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
            self._pygame = pygame
            self._has_pygame = True
            logger.info("Audio playback: pygame mixer")
        except Exception as e:
            logger.warning(f"pygame not available ({e}). Audio playback disabled.")

    # ----- Core Generation -----

    def generate(self, text: str, agent: str, ground_truth: Dict, pre_cleaned: bool = False) -> Optional[io.BytesIO]:
        """
        Generate speech audio from text.

        Args:
            text: Raw text (may contain JSON — will be cleaned)
            agent: "doctor", "nurse", or "patient"
            ground_truth: Episode ground truth dict (for persona lookup)
            pre_cleaned: If True, skip text cleaning (already done by caller)

        Returns:
            BytesIO containing MP3 audio, or None on failure.
        """
        if not pre_cleaned:
            text = clean_text_for_speech(text)
        if not text or len(text.strip()) < 3:
            logger.warning(f"TTS skip: text too short after cleaning for {agent}")
            return None

        voice_key = get_voice_key(agent, ground_truth)

        # Layer 1: LLM Emotion Adapter — rewrites clean text into
        # emotionally expressive natural speech. Falls back to regex
        # transforms if no Groq client is available.
        text = emotionalize_for_tts(text, voice_key, self._groq_client, self._groq_model)
        logger.info(f"TTS [{voice_key}]: {text[:120]}")

        try:
            if self.use_elevenlabs:
                # Layer 2: ElevenLabs audio tags ([sigh], [nervous])
                # added on top of the LLM-rewritten text
                text_el = _inject_speech_markers(text, voice_key)
                return self._generate_elevenlabs(text_el, voice_key)
            else:
                # Edge-TTS does NOT support bracketed tags — strip them
                import re as _re
                clean_for_edge = _re.sub(r'\[.*?\]', '', text).strip()
                if len(clean_for_edge) < 3:
                    clean_for_edge = text
                return self._generate_edge(clean_for_edge, voice_key)
        except Exception as e:
            logger.error(f"TTS generation failed ({voice_key}): {e}")
            print(f"  [TTS ERROR] voice_key={voice_key} agent={agent}: {e}", flush=True)
            # Fallback to Edge-TTS if ElevenLabs fails
            if self.use_elevenlabs:
                try:
                    print(f"  [TTS] Falling back to Edge-TTS for {agent}...", flush=True)
                    # Strip any bracketed tags before sending to Edge-TTS
                    import re as _re
                    clean_for_edge = _re.sub(r'\[.*?\]', '', text).strip()
                    if len(clean_for_edge) < 3:
                        clean_for_edge = text  # safety fallback
                    return self._generate_edge(clean_for_edge, voice_key)
                except Exception as e2:
                    logger.error(f"Edge-TTS fallback also failed: {e2}")
            return None

    def _generate_elevenlabs(self, text: str, voice_key: str) -> io.BytesIO:
        """Generate audio using ElevenLabs API."""
        from elevenlabs.types import VoiceSettings

        config = ELEVEN_VOICES.get(voice_key, ELEVEN_VOICES["doctor"])
        s = config["settings"]

        audio_iter = self._eleven_client.text_to_speech.convert(
            voice_id=config["voice_id"],
            text=text,
            model_id="eleven_v3",  # v3 supports [sigh], [nervous], [gasps] audio tags
            voice_settings=VoiceSettings(
                stability=s["stability"],
                similarity_boost=s["similarity_boost"],
                style=s["style"],
                speed=s.get("speed", 1.0),
                use_speaker_boost=True,
            ),
        )

        buf = io.BytesIO()
        for chunk in audio_iter:
            buf.write(chunk)
        buf.seek(0)
        return buf

    def _generate_edge(self, text: str, voice_key: str) -> io.BytesIO:
        """Generate audio using Edge-TTS (free fallback)."""
        import asyncio
        import edge_tts

        config = EDGE_VOICE_MAP.get(voice_key, EDGE_VOICE_MAP["doctor"])
        # Fallback voice if primary fails
        FALLBACK_VOICE = "en-US-GuyNeural"

        async def _gen(voice, rate, pitch):
            comm = edge_tts.Communicate(
                text, voice,
                rate=rate, pitch=pitch,
            )
            buf = io.BytesIO()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk["data"])
            buf.seek(0)
            return buf

        loop = asyncio.new_event_loop()
        try:
            buf = loop.run_until_complete(_gen(config["voice"], config["rate"], config["pitch"]))
            # Check if we actually got audio data
            if buf.getbuffer().nbytes < 100:
                logger.warning(f"Edge-TTS: voice {config['voice']} returned no audio, retrying with {FALLBACK_VOICE}")
                buf = loop.run_until_complete(_gen(FALLBACK_VOICE, "+0%", "+0Hz"))
            if buf.getbuffer().nbytes < 100:
                raise RuntimeError("No audio was received from Edge-TTS even with fallback voice.")
            return buf
        finally:
            loop.close()

    # ----- Playback -----

    def speak(self, text: str, agent: str, ground_truth: Dict, label: str = ""):
        """
        Generate speech and play through speakers. Blocks until done.

        Args:
            text: Text to speak (cleaned automatically)
            agent: "doctor", "nurse", or "patient"
            ground_truth: Episode ground truth dict
            label: Optional console label (e.g. "NURSE→PATIENT")
        """
        clean = clean_text_for_speech(text)
        if not clean or len(clean.strip()) < 3:
            return

        voice_key = get_voice_key(agent, ground_truth)
        engine = "ElevenLabs" if self.use_elevenlabs else "Edge-TTS"
        tag = f" ({label})" if label else ""
        print(f"  🔊 [{agent.upper()}{tag}] {clean[:100]}{'...' if len(clean)>100 else ''}", flush=True)
        print(f"     voice={voice_key} engine={engine}", flush=True)

        buf = self.generate(text, agent, ground_truth)
        if buf is None:
            return

        if self._has_pygame:
            self._play_with_pygame(buf)
        else:
            logger.warning("No audio backend available for playback.")

    def _play_with_pygame(self, audio_buf: io.BytesIO):
        """Play audio bytes through pygame mixer (blocking)."""
        tmp_path = None
        try:
            # Write to temp file (pygame needs a file path for MP3)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(audio_buf.read())
                tmp_path = f.name

            self._pygame.mixer.music.load(tmp_path)
            self._pygame.mixer.music.play()
            while self._pygame.mixer.music.get_busy():
                self._pygame.time.wait(100)

        except Exception as e:
            logger.error(f"Playback error: {e}")
        finally:
            # Cleanup temp file
            try:
                self._pygame.mixer.music.stop()
            except:
                pass
            if tmp_path:
                time.sleep(0.1)  # Small delay before delete
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    # ----- High-Level Helpers -----

    def speak_doctor_action(self, action_str: str, ground_truth: Dict):
        """Parse a Doctor JSON action and speak the appropriate text."""
        try:
            a = json.loads(action_str)
        except (json.JSONDecodeError, TypeError):
            return

        tool = a.get("tool", "")
        if tool == "speak_to":
            target = a.get("target", "")
            msg = a.get("message", "")
            self.speak(msg, "doctor", ground_truth, label=f"→{target}")
        elif tool == "order_lab":
            test = a.get("test_name", "")
            self.speak(f"Nurse, I need a {test} ordered stat.", "doctor", ground_truth, label="→nurse")
        elif tool == "terminal_discharge":
            tx = a.get("treatment", "")
            self.speak(f"Discharge plan: {tx}", "doctor", ground_truth, label="DISCHARGE")

    def speak_observation(self, obs_str: str, ground_truth: Dict):
        """Parse an environment observation and speak all agent messages."""
        try:
            obs = json.loads(obs_str)
        except (json.JSONDecodeError, TypeError):
            return

        event = obs.get("event", "")

        if event == "nurse_report":
            # Speak internal Nurse ↔ Patient exchanges
            for ex in obs.get("internal_exchanges", []):
                if "nurse_said" in ex:
                    self.speak(ex["nurse_said"], "nurse", ground_truth, label="→patient")
                    time.sleep(0.3)
                    self.speak(ex["patient_said"], "patient", ground_truth, label="→nurse")
                    time.sleep(0.3)
                elif "nurse_action" in ex:
                    action = ex.get("nurse_action", "")
                    result = ex.get("result", "")
                    if result:
                        self.speak(result, "nurse", ground_truth, label=action)
            # Speak nurse's final report to doctor
            nurse_msg = obs.get("nurse_message", "")
            if nurse_msg:
                self.speak(nurse_msg, "nurse", ground_truth, label="→doctor")

        elif event == "patient_response":
            patient_msg = obs.get("patient_message", "")
            if patient_msg:
                self.speak(patient_msg, "patient", ground_truth, label="→doctor")

        elif event == "lab_result":
            test = obs.get("test_name", "")
            result = obs.get("result", "")
            self.speak(
                f"Lab results for {test}: {result}",
                "nurse", ground_truth, label="LAB"
            )

        elif event == "terminal_win":
            self.speak(
                "Correct diagnosis and treatment. The patient has been stabilized.",
                "doctor", ground_truth, label="WIN"
            )

        elif event == "terminal_fatal":
            self.speak(
                "Critical error. Lethal treatment administered. The patient did not survive.",
                "doctor", ground_truth, label="FATAL"
            )

        elif event == "terminal_incorrect":
            correct = obs.get("correct_treatment", "")
            self.speak(
                f"Incorrect treatment. The correct treatment was: {correct}",
                "doctor", ground_truth, label="WRONG"
            )

        elif event == "terminal_ama":
            patient_msg = obs.get("patient_message", "I'm leaving this hospital!")
            self.speak(patient_msg, "patient", ground_truth, label="LEAVING")

    def close(self):
        """Cleanup resources."""
        if self._has_pygame:
            try:
                self._pygame.mixer.quit()
            except:
                pass
