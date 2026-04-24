"""
ER_MAP/envs/empathy_engine.py
=============================
Intent-based empathy detection and patient trust/anxiety model.

Flow:
    Doctor message -> classify_intent() -> update_patient_state() -> consent_decision()

This replaces naive keyword matching with a causal chain:
    Empathy -> Trust -> Consent -> Treatment Success -> Reward
"""

import re
import random
from typing import Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Intent Classification (heuristic, no LLM call needed)
# ---------------------------------------------------------------------------

# Empathetic phrases -- Doctor shows understanding, concern, reassurance
EMPATHY_PATTERNS = [
    r"\bi understand\b", r"\bi hear you\b", r"\bthat must be\b",
    r"\bi can see\b", r"\bi know this is\b", r"\byou.?re doing great\b",
    r"\bdon.?t worry\b", r"\bwe.?re going to\b", r"\bwe.?ll take care\b",
    r"\bi.?m here\b", r"\byou.?re safe\b", r"\btake your time\b",
    r"\bthat sounds\b.*\b(scary|difficult|painful|frightening)\b",
    r"\bi.?m sorry\b.*\b(going through|dealing|feeling|hear)\b",
    r"\bhow are you feeling\b", r"\bare you comfortable\b",
    r"\blet me help\b", r"\bwe.?ll work together\b",
    r"\bthat.?s understandable\b", r"\bit.?s okay\b",
    r"\bi want to make sure\b.*\b(comfortable|safe|okay)\b",
    r"\bthank you for\b.*\b(telling|sharing|trusting|coming)\b",
]

# Explanatory phrases -- Doctor educates, explains reasoning
EXPLAIN_PATTERNS = [
    r"\blet me explain\b", r"\bwhat this means\b", r"\bthe reason\b",
    r"\bbecause\b.*\bneed\b", r"\bthis test will\b", r"\bthis helps us\b",
    r"\bwhat we.?re doing\b", r"\bhere.?s (the|my) plan\b",
    r"\bthe results show\b", r"\bbased on\b.*\bfindings\b",
    r"\bi recommend\b.*\bbecause\b", r"\bthis is important because\b",
    r"\bto rule out\b", r"\bto make sure\b", r"\bso we can\b",
    r"\bthink of it as\b", r"\bin simple terms\b",
]

# Dismissive phrases -- Doctor is curt, ignores patient concerns
DISMISSIVE_PATTERNS = [
    r"\bjust do\b", r"\bjust take\b", r"\bjust calm\b",
    r"\bthat.?s not important\b", r"\bdoesn.?t matter\b",
    r"\bi don.?t have time\b", r"\bwe.?re busy\b",
    r"\bjust sign\b", r"\bjust let me\b.*\bjob\b",
    r"\bstop (complaining|worrying|asking)\b",
    r"\byou.?re fine\b", r"\bit.?s nothing\b",
    r"\bnext patient\b", r"\bhurry up\b",
]

# Acknowledgment phrases -- Doctor actively listens
ACKNOWLEDGE_PATTERNS = [
    r"\bi see\b", r"\bgo on\b", r"\btell me more\b",
    r"\bwhat else\b", r"\band then\b", r"\bwhen did\b",
    r"\bhow long\b", r"\bcan you describe\b",
    r"\bwhat happened\b", r"\bwalk me through\b",
]


def classify_intent(message: str) -> Dict[str, float]:
    """
    Classify a Doctor message into intent scores.
    Returns dict with scores for: empathy, explanation, dismissive, acknowledgment.
    All scores are 0.0-1.0. Multiple intents can co-occur.
    """
    msg_lower = message.lower()
    
    def _score(patterns):
        hits = sum(1 for p in patterns if re.search(p, msg_lower))
        # Normalize: 1 match = 0.5, 2+ = 0.8, 3+ = 1.0
        if hits == 0: return 0.0
        if hits == 1: return 0.5
        if hits == 2: return 0.8
        return 1.0
    
    return {
        "empathy": _score(EMPATHY_PATTERNS),
        "explanation": _score(EXPLAIN_PATTERNS),
        "dismissive": _score(DISMISSIVE_PATTERNS),
        "acknowledgment": _score(ACKNOWLEDGE_PATTERNS),
    }


# ---------------------------------------------------------------------------
# Patient Trust / Anxiety State Model
# ---------------------------------------------------------------------------

class PatientState:
    """
    Tracks patient's internal emotional state during an episode.
    
    trust: 0-100 (starts at 50, modified by doctor's behavior)
    anxiety: 0-100 (starts based on persona, modified by interactions)
    
    These states influence:
    - Whether patient agrees to treatment (consent)
    - Whether patient provides accurate symptom info
    - Whether patient leaves AMA
    """
    
    def __init__(self, persona: Dict[str, str]):
        """Initialize based on patient persona traits."""
        comm = persona.get("communication", "calm_stoic")
        compliance = persona.get("compliance", "fully_compliant")
        financial = persona.get("financial", "average")
        
        # Base trust depends on communication style
        trust_map = {
            "calm_stoic": 60,
            "anxious_panicked": 40,
            "hostile_aggressive": 25,
            "disorganized_confused": 45,
        }
        self.trust = trust_map.get(comm, 50)
        
        # Base anxiety depends on communication + financial stress
        anxiety_map = {
            "calm_stoic": 20,
            "anxious_panicked": 75,
            "hostile_aggressive": 55,
            "disorganized_confused": 50,
        }
        self.anxiety = anxiety_map.get(comm, 40)
        
        # Financial stress increases anxiety
        if financial == "poor_uninsured":
            self.anxiety = min(100, self.anxiety + 20)
            self.trust = max(0, self.trust - 10)
        
        # Compliance affects trust baseline
        if compliance == "non_compliant":
            self.trust = max(0, self.trust - 15)
        elif compliance == "cost_constrained":
            self.anxiety = min(100, self.anxiety + 10)
        
        # Track interaction history
        self.empathy_count = 0
        self.dismissive_count = 0
        self.explanation_count = 0
        self.total_interactions = 0
    
    def update(self, intent: Dict[str, float]) -> Dict[str, float]:
        """
        Update trust/anxiety based on Doctor's intent classification.
        Returns the delta values for reward computation.
        """
        self.total_interactions += 1
        trust_delta = 0.0
        anxiety_delta = 0.0
        
        emp = intent.get("empathy", 0.0)
        expl = intent.get("explanation", 0.0)
        dismiss = intent.get("dismissive", 0.0)
        ack = intent.get("acknowledgment", 0.0)
        
        # Empathy increases trust, decreases anxiety
        if emp > 0:
            trust_delta += emp * 8
            anxiety_delta -= emp * 6
            self.empathy_count += 1
        
        # Explanation increases trust (patient feels informed)
        if expl > 0:
            trust_delta += expl * 5
            anxiety_delta -= expl * 3
            self.explanation_count += 1
        
        # Dismissiveness damages trust, increases anxiety
        if dismiss > 0:
            trust_delta -= dismiss * 12
            anxiety_delta += dismiss * 10
            self.dismissive_count += 1
        
        # Acknowledgment is a mild positive
        if ack > 0:
            trust_delta += ack * 3
            anxiety_delta -= ack * 2
        
        # Apply deltas with clamping
        self.trust = max(0, min(100, self.trust + trust_delta))
        self.anxiety = max(0, min(100, self.anxiety + anxiety_delta))
        
        return {"trust_delta": trust_delta, "anxiety_delta": anxiety_delta}
    
    def consent_decision(self) -> str:
        """
        Decide patient response based on current trust/anxiety.
        Returns: "AGREE", "REFUSE", "AMA" (against medical advice)
        """
        # AMA threshold: very low trust + very high anxiety
        if self.trust < 20 and self.anxiety > 70:
            if random.random() < 0.6:
                return "AMA"
        
        # Refuse: low trust or very high anxiety
        if self.trust < 35:
            if random.random() < 0.4:
                return "REFUSE"
        
        if self.anxiety > 80 and self.trust < 50:
            if random.random() < 0.3:
                return "REFUSE"
        
        # Default: agree
        return "AGREE"
    
    def get_state_summary(self) -> Dict[str, any]:
        """Return current state for logging/reward computation."""
        return {
            "trust": round(self.trust, 1),
            "anxiety": round(self.anxiety, 1),
            "empathy_count": self.empathy_count,
            "dismissive_count": self.dismissive_count,
            "explanation_count": self.explanation_count,
            "total_interactions": self.total_interactions,
        }


# ---------------------------------------------------------------------------
# Phase-Specific Reward Computation
# ---------------------------------------------------------------------------

def compute_empathy_reward(
    intent: Dict[str, float],
    patient_state: "PatientState",
    phase: int,
) -> float:
    """
    Compute empathy-related reward based on phase.
    
    Phase 1: No empathy rewards (focus on clinical workflow)
    Phase 2: Small empathy bonus for explanation (+0.02)
    Phase 3: Full empathy reward chain (+0.05 empathy, +0.03 explain, -0.08 dismissive)
    """
    reward = 0.0
    
    if phase <= 1:
        # Phase 1: Zero empathy reward -- focus on tool mastery
        return 0.0
    
    if phase >= 2:
        # Phase 2: Reward explanations to patients
        if intent.get("explanation", 0) > 0:
            reward += 0.02 * intent["explanation"]
    
    if phase >= 3:
        # Phase 3: Full empathy reward chain
        emp = intent.get("empathy", 0)
        dismiss = intent.get("dismissive", 0)
        
        if emp > 0:
            reward += 0.05 * emp
        if dismiss > 0:
            reward -= 0.08 * dismiss
        
        # Bonus for maintaining high trust
        if patient_state.trust > 70:
            reward += 0.02
        # Penalty for critically low trust
        if patient_state.trust < 25:
            reward -= 0.03
    
    return reward


# ---------------------------------------------------------------------------
# Milestone Tracker (Clinical Workflow State Machine)
# ---------------------------------------------------------------------------

class MilestoneTracker:
    """
    Tracks whether the Doctor follows the correct clinical workflow.
    
    Expected milestone order (Phase 1 strict, Phase 2-3 relaxed):
    1. READ_SOAP    -- Review patient history
    2. PATIENT_CONTACT -- Speak to patient (history taking)
    3. VITALS       -- Check vitals (via nurse or observation)
    4. LABS         -- Order relevant labs
    5. ASSESSMENT   -- Update SOAP Assessment
    6. DISCHARGE    -- Terminal discharge with treatment
    """
    
    MILESTONES = [
        "READ_SOAP",
        "PATIENT_CONTACT",
        "VITALS",
        "LABS",
        "ASSESSMENT",
        "DISCHARGE",
    ]
    
    def __init__(self, phase: int = 1):
        self.phase = phase
        self.achieved: Dict[str, bool] = {m: False for m in self.MILESTONES}
        self.order: list = []  # Track achievement order
    
    def mark(self, milestone: str) -> float:
        """
        Mark a milestone as achieved. Returns milestone reward.
        
        Phase 1: Strict ordering -- reward only if in correct sequence
        Phase 2: Semi-strict -- reward for completion, small penalty for wrong order
        Phase 3: Relaxed -- reward for completion only, no ordering constraint
        """
        if milestone not in self.MILESTONES:
            return 0.0
        
        if self.achieved.get(milestone, False):
            return 0.0  # Already achieved, no double reward
        
        self.achieved[milestone] = True
        self.order.append(milestone)
        
        expected_idx = self.MILESTONES.index(milestone)
        actual_idx = len(self.order) - 1
        
        if self.phase == 1:
            # Phase 1: Strict ordering enforcement
            if actual_idx == expected_idx:
                return 0.05  # Correct order bonus
            elif actual_idx < expected_idx:
                return 0.02  # Done but out of order -- small reward
            else:
                return 0.01  # Late but still gets credit
        
        elif self.phase == 2:
            # Phase 2: Moderate enforcement
            if actual_idx <= expected_idx + 1:
                return 0.04  # Close enough to correct order
            return 0.02  # Still rewarded for completion
        
        else:
            # Phase 3: Just reward completion
            return 0.03  # Maintenance reward for clinical competence
    
    def completion_ratio(self) -> float:
        """Return fraction of milestones completed (0.0-1.0)."""
        return sum(self.achieved.values()) / len(self.MILESTONES)
    
    def missing_milestones(self) -> list:
        """Return list of milestones not yet achieved."""
        return [m for m, done in self.achieved.items() if not done]
    
    def get_summary(self) -> Dict[str, any]:
        """Return tracker state for logging."""
        return {
            "achieved": {k: v for k, v in self.achieved.items()},
            "order": self.order,
            "completion": self.completion_ratio(),
            "missing": self.missing_milestones(),
        }
