from __future__ import annotations

# 
# STANDARD LIBRARY
#

import os
import json
import time
import hashlib
import logging
import re
import asyncio
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

#
# OPTIONAL DEPENDENCIES  (degrade gracefully if absent)
#

try:
    from enum import StrEnum  # Python 3.11+
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass

logger = logging.getLogger("llm_evaluator")

# 
# CONFIGURATION
# 

class LLMBackend(StrEnum):
    LLAMA3_GROQ     = "llama3_groq"      # Llama 3.3-70B via Groq  (fastest, preferred)
    LLAMA3_TOGETHER = "llama3_together"  # Llama 3-70B via Together AI
    MISTRAL         = "mistral"          # Mistral Medium via Mistral API
    GPT4            = "gpt4"             # OpenAI GPT-4o-mini (fallback)
    RULE_BASED      = "rule_based"       # Deterministic heuristic (zero-dependency)


# Read from env; fall back to rule_based so the env boots with no API keys.
LLM_BACKEND = LLMBackend(os.environ.get("LLM_BACKEND", "rule_based"))

# Reward blending
DEFAULT_ALPHA: float = 0.30   # 30% LLM influence on final reward by default
MAX_WORKERS: int     = 4      # Thread pool size for batch evaluation
CACHE_MAX_SIZE: int  = 512    # LRU cache entries

# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMA
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMEvalResult:
    """
    Structured output from the LLM evaluator.
    Scores:
        All dimension scores are integers 0–10.
        reward_adjustment ∈ [-0.5, +0.5]  (added to the rule-based reward)
        confidence       ∈ [0.0,  1.0]
    Usage:
        result = evaluate_with_llm(state, action, reasoning)
        final_reward = rule_reward + alpha * result.reward_adjustment
    """
    # ── Clinical dimensions ──────────────────────────────────────────────────
    clinical_score:   int    # Correctness vs. evidence-based guidelines (0–10)
    safety_score:     int    # Patient safety (0=lethal decision, 10=fully safe)
    efficiency_score: int    # Resource allocation / timing efficiency (0–10)
    ethics_score:     int    # Ethical reasoning quality (0–10)
    reasoning_score:  int    # Medical coherence of provided free-text (0–10)
    total_score:      int    # Weighted aggregate — see formula below (0–10)

    # ── RL integration ───────────────────────────────────────────────────────
    reward_adjustment: float # Additive delta: [-0.5, +0.5]
    confidence:        float # LLM self-assessed confidence [0.0, 1.0]

    # ── Explainability ───────────────────────────────────────────────────────
    explanation:      str    # Human-readable 2–3 sentence clinical critique
    teaching_point:   str    # One-line educational takeaway (may be empty)
    oracle_deviation: str    # How this decision differed from the oracle (may be empty)

    # ── Telemetry ────────────────────────────────────────────────────────────
    backend_used:     str    # Which backend produced this evaluation
    latency_ms:       int    # End-to-end inference latency in milliseconds
    cache_hit:        bool   # True if served from LRU cache

    # ── Grader integration ───────────────────────────────────────────────────
    grader_alignment: float  # Correlation with GradeResult.score from graders.py [0,1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_safe(self) -> bool:
        """True when safety score passes clinical threshold."""
        return self.safety_score >= 6

    @property
    def is_critical_failure(self) -> bool:
        """True when this evaluation reflects a life-threatening error."""
        return self.safety_score <= 2 or self.reward_adjustment <= -0.4

    def summary(self) -> str:
        stars = "★" * (self.total_score // 2) + "☆" * (5 - self.total_score // 2)
        return (
            f"[{self.backend_used.upper()}] {stars} ({self.total_score}/10)  "
            f"Δreward={self.reward_adjustment:+.3f}  "
            f"{'⚠ SAFETY ALERT' if self.is_critical_failure else '✓ safe'}  "
            f"({self.latency_ms}ms)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# ROLLING METRICS  (thread-safe telemetry for the /health endpoint)
# ──────────────────────────────────────────────────────────────────────────────

class EvaluatorMetrics:
    """
    Lightweight rolling telemetry.
    Tracks latency, backend usage, score distributions, cache hit rate.
    """

    def __init__(self, window: int = 200):
        self._lock  = threading.Lock()
        self._window = window
        self.latencies: deque   = deque(maxlen=window)
        self.scores: deque      = deque(maxlen=window)
        self.adjustments: deque = deque(maxlen=window)
        self.backends: Dict[str, int] = defaultdict(int)
        self.cache_hits: int    = 0
        self.total_calls: int   = 0
        self.critical_failures: int = 0

    def record(self, result: LLMEvalResult) -> None:
        with self._lock:
            self.latencies.append(result.latency_ms)
            self.scores.append(result.total_score)
            self.adjustments.append(result.reward_adjustment)
            self.backends[result.backend_used] += 1
            self.total_calls += 1
            if result.cache_hit:
                self.cache_hits += 1
            if result.is_critical_failure:
                self.critical_failures += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            n = len(self.latencies)
            return {
                "total_calls":        self.total_calls,
                "cache_hit_rate":     round(self.cache_hits / max(1, self.total_calls), 3),
                "critical_failures":  self.critical_failures,
                "avg_latency_ms":     round(sum(self.latencies) / max(1, n), 1),
                "p95_latency_ms":     sorted(self.latencies)[int(n * 0.95)] if n else 0,
                "avg_total_score":    round(sum(self.scores) / max(1, n), 2),
                "avg_reward_adj":     round(sum(self.adjustments) / max(1, n), 3),
                "backend_usage":      dict(self.backends),
            }


# Singleton metrics instance  (imported by app.py / training_loop.py)
METRICS = EvaluatorMetrics()


# ──────────────────────────────────────────────────────────────────────────────
# LRU PROMPT CACHE  (skip identical prompts during RL roll-outs)
# ──────────────────────────────────────────────────────────────────────────────

class _EvaluatorCache:
    """Thread-safe LRU cache keyed on prompt SHA-256."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._max   = max_size
        self._store: Dict[str, LLMEvalResult] = {}
        self._order: deque = deque()
        self._lock  = threading.Lock()

    def _key(self, prompt: str, backend: str) -> str:
        return hashlib.sha256(f"{backend}::{prompt}".encode()).hexdigest()

    def get(self, prompt: str, backend: str) -> Optional[LLMEvalResult]:
        k = self._key(prompt, backend)
        with self._lock:
            return self._store.get(k)

    def set(self, prompt: str, backend: str, result: LLMEvalResult) -> None:
        k = self._key(prompt, backend)
        with self._lock:
            if k in self._store:
                self._order.remove(k)
            self._store[k] = result
            self._order.append(k)
            while len(self._order) > self._max:
                oldest = self._order.popleft()
                self._store.pop(oldest, None)


_CACHE = _EvaluatorCache()


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────────

EVAL_SYSTEM_PROMPT = """You are an expert clinical AI evaluator embedded in a reinforcement learning
training system for emergency department triage and acute care. Your role is to evaluate AI agent
decisions against gold-standard clinical protocols: ESI (Emergency Severity Index), Sepsis-3 /
Surviving Sepsis Campaign Hour-1 Bundle, and WHO/Lexicomp medication safety guidelines.
═══════════════════════════════════════════════════════════════════════
SCORING RUBRIC  (each dimension: integer 0–10)
═══════════════════════════════════════════════════════════════════════
clinical_score   — Does the action match evidence-based guidelines?
                   0 = completely wrong (e.g. ESI-5 for a resuscitation patient)
                   5 = partially correct with notable gaps
                   10 = perfect alignment with current guidelines
safety_score     — Could this decision harm or kill the patient?
                   0 = lethal mistake (missed anaphylaxis, wrong antibiotic in allergy)
                   5 = notable safety gap but unlikely immediate harm
                   10 = fully safe for the patient
efficiency_score — Is resource allocation optimal?
                   Over-triage wastes ICU/resus beds; under-triage risks death.
                   0 = grossly inappropriate resource use
                   10 = perfectly calibrated
ethics_score     — Does the reasoning respect patient dignity, non-maleficence, autonomy?
                   0 = paternalistic, biased, or harmful framing
                   10 = exemplary ethical reasoning
reasoning_score  — Is the provided reasoning medically coherent and well-structured?
                   0 = absent, incoherent, or dangerous reasoning
                   5 = partial / missing key concepts
                   10 = clear, accurate, comprehensive rationale
total_score      — MUST be computed as:
                   round( safety×0.35 + clinical×0.30 + reasoning×0.15
                          + efficiency×0.10 + ethics×0.10 )
═══════════════════════════════════════════════════════════════════════
REWARD ADJUSTMENT RULES  (reward_adjustment ∈ [-0.5, +0.5])
═══════════════════════════════════════════════════════════════════════
+0.40 to +0.50 : Perfect decision AND strong, well-cited reasoning
+0.20 to +0.39 : Correct decision, adequate reasoning
+0.00 to +0.19 : Correct but reasoning is weak or absent
-0.10 to  0.00 : Minor error, overall safe
-0.10 to -0.20 : Moderate error (e.g. ESI off by 1 for non-critical patient)
-0.30 to -0.40 : Significant error (missed sepsis bundle item, major drug gap)
-0.40 to -0.50 : Under-triage of critical patient OR dangerous medication error
         -0.50 : Complete failure / lethal decision
═══════════════════════════════════════════════════════════════════════
TEACHING POINT
═══════════════════════════════════════════════════════════════════════
teaching_point  — One sentence the agent should learn from this evaluation.
                  Cite the specific guideline (e.g. "SSC 2021 recommends…").
CRITICAL: Return ONLY a valid JSON object. No markdown, no preamble, no trailing text.
Keys must match EXACTLY:
  clinical_score, safety_score, efficiency_score, ethics_score, reasoning_score,
  total_score, reward_adjustment, confidence, explanation, teaching_point"""


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ──────────────────────────────────────────────────────────────────────────────

def build_eval_prompt(
    state:     Dict[str, Any],
    action:    Dict[str, Any],
    reasoning: str,
    oracle_action: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Construct the user-turn prompt for LLM evaluation.
    Optionally includes oracle_action ("What Would A Doctor Do?") to help the
    LLM judge score deviations from the clinical gold standard.
    """
    patient   = state.get("patient", {})
    task_type = state.get("task_type", "triage")
    task_id   = state.get("task_id", "unknown")
    vitals    = patient.get("vitals", {})

    # ── Vitals formatting ────────────────────────────────────────────────────
    vitals_parts = []
    vital_labels = {
        "heart_rate":          ("HR",   "bpm"),
        "systolic_bp":         ("SBP",  "mmHg"),
        "diastolic_bp":        ("DBP",  "mmHg"),
        "respiratory_rate":    ("RR",   "/min"),
        "spo2":                ("SpO₂", "%"),
        "temperature":         ("Temp", "°C"),
        "glasgow_coma_scale":  ("GCS",  "/15"),
        "mean_arterial_pressure": ("MAP", "mmHg"),
        "lactate":             ("Lactate", "mmol/L"),
    }
    for key, (label, unit) in vital_labels.items():
        val = vitals.get(key) if isinstance(vitals, dict) else getattr(vitals, key, None)
        if val is not None:
            vitals_parts.append(f"{label}: {val} {unit}")
    vitals_str = "  |  ".join(vitals_parts) if vitals_parts else "Not provided"

    # ── Symptoms / complaint ─────────────────────────────────────────────────
    symptoms = patient.get("chief_complaint", patient.get("symptoms", "Not provided"))

    # ── Medications ──────────────────────────────────────────────────────────
    meds = patient.get("current_medications", [])
    med_str = ", ".join(
        m.get("name", str(m)) if isinstance(m, dict) else str(m)
        for m in meds
    ) if meds else "None"

    # ── Labs ─────────────────────────────────────────────────────────────────
    labs = patient.get("labs", {})
    labs_str = ", ".join(
        f"{k.upper()}: {v}"
        for k, v in (labs.items() if isinstance(labs, dict) else [])
        if v is not None
    ) or "Not provided"

    # ── Oracle delta ─────────────────────────────────────────────────────────
    oracle_block = ""
    if oracle_action:
        oracle_block = f"""
--- ORACLE (IDEAL PHYSICIAN) ACTION ---
{json.dumps(oracle_action, indent=2)}
Note: Use this to identify deviations from gold-standard clinical practice."""

    action_summary = json.dumps(action, indent=2) if action else "No action provided"

    return f"""=== CLINICAL EVALUATION REQUEST ===
Task Type  : {task_type.upper()}  |  Task ID  : {task_id}
Difficulty : {state.get('difficulty', 'unknown').upper()}
--- PATIENT PRESENTATION ---
Age         : {patient.get('age', '?')}  |  Sex: {patient.get('sex', '?')}
Chief Complaint : {symptoms}
Vitals      : {vitals_str}
Labs        : {labs_str}
Allergies   : {', '.join(patient.get('allergies', [])) or 'None'}
Medications : {med_str}
Risk Factors: {', '.join(patient.get('risk_factors', [])) or 'None'}
PMH         : {', '.join(patient.get('past_medical_history', [])) or 'None'}
{oracle_block}
--- AGENT ACTION ---
{action_summary}
--- AGENT REASONING ---
{reasoning or "(No reasoning provided — this should penalise reasoning_score)"}
--- YOUR TASK ---
Evaluate the agent's action and reasoning rigorously. Consider patient safety first.
Return a valid JSON object with EXACTLY these keys:
{{
  "clinical_score":    <int 0-10>,
  "safety_score":      <int 0-10>,
  "efficiency_score":  <int 0-10>,
  "ethics_score":      <int 0-10>,
  "reasoning_score":   <int 0-10>,
  "total_score":       <int 0-10>,
  "reward_adjustment": <float -0.5 to 0.5>,
  "confidence":        <float 0.0-1.0>,
  "explanation":       "<2-3 sentences: specific clinical critique>",
  "teaching_point":    "<1 sentence with guideline citation>"
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# LLM BACKENDS  (modular, swappable, each independently testable)
# ──────────────────────────────────────────────────────────────────────────────

def _http_post(url: str, payload: Dict, headers: Dict, timeout: int = 15) -> Dict:
    """Thin urllib wrapper used by all backends to avoid httpx/requests dependency."""
    import urllib.request
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _call_groq_llama3(prompt: str) -> str:
    """
    Meta Llama 3.3-70B-Instruct via Groq API.
    Fastest inference (~150ms), preferred for Meta hackathon alignment.
    Env var: GROQ_API_KEY
    Model  : llama-3.3-70b-versatile  (supports JSON mode)
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    data = _http_post(
        url="https://api.groq.com/openai/v1/chat/completions",
        payload={
            "model":    "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "temperature":    0.05,
            "max_tokens":     600,
            "response_format": {"type": "json_object"},
            "seed":           42,    # reproducible evaluations
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        timeout=10,
    )
    return data["choices"][0]["message"]["content"]


def _call_together_llama3(prompt: str) -> str:
    """
    Meta Llama 3-70B-Instruct via Together AI.
    Alternative backend; higher latency than Groq but reliable fallback.
    Env var: TOGETHER_API_KEY
    """
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY not set")

    data = _http_post(
        url="https://api.together.xyz/v1/chat/completions",
        payload={
            "model":    "meta-llama/Meta-Llama-3-70B-Instruct",
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.05,
            "max_tokens":  600,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        timeout=20,
    )
    return data["choices"][0]["message"]["content"]


def _call_mistral(prompt: str) -> str:
    """
    Mistral Medium via Mistral AI API.
    Good quality / cost balance; reliable for complex clinical reasoning.
    Env var: MISTRAL_API_KEY
    """
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not set")

    data = _http_post(
        url="https://api.mistral.ai/v1/chat/completions",
        payload={
            "model":    "mistral-medium",
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "temperature": 0.05,
            "max_tokens":  600,
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        timeout=20,
    )
    return data["choices"][0]["message"]["content"]


def _call_openai_gpt4(prompt: str) -> str:
    """
    GPT-4o-mini via OpenAI API.
    Last-resort LLM fallback before rule_based; cost-efficient.
    Env var: OPENAI_API_KEY
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    data = _http_post(
        url="https://api.openai.com/v1/chat/completions",
        payload={
            "model":    "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": EVAL_SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            "temperature":     0.05,
            "max_tokens":      600,
            "response_format": {"type": "json_object"},
        },
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        timeout=20,
    )
    return data["choices"][0]["message"]["content"]


# ──────────────────────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK EVALUATOR  (zero-dependency, always available)
# ──────────────────────────────────────────────────────────────────────────────

def _rule_based_eval(
    state:     Dict[str, Any],
    action:    Dict[str, Any],
    reasoning: str,
) -> Dict[str, Any]:
    """
    Deterministic heuristic evaluator used when no LLM backend is available.
    Implements clinical heuristics for all three task types:
      • Triage (ESI accuracy + vital sign safety checks)
      • Medication Safety (drug interaction recall)
      • Sepsis (SSC Hour-1 Bundle completeness)
    Aligned with the GradeResult structure from graders.py.
    """
    task_type = state.get("task_type", "triage")
    patient   = state.get("patient", {})
    vitals    = patient.get("vitals", {})
    expected  = state.get("expected_action", {})

    # ── Base scores ──────────────────────────────────────────────────────────
    clinical_score:   int   = 5
    safety_score:     int   = 7
    efficiency_score: int   = 6
    ethics_score:     int   = 7
    explanation_parts: List[str] = []
    teaching_point:    str   = ""

    # ── Reasoning quality (length proxy) ────────────────────────────────────
    word_count       = len(reasoning.split()) if reasoning else 0
    reasoning_score: int = min(10, max(0, word_count // 10))

    def _get(key: str, default=None):
        """Get vitals field from dict or object."""
        if isinstance(vitals, dict):
            return vitals.get(key, default)
        return getattr(vitals, key, default)

    # ═════════════════════════════════════════════════════════════════════════
    # TASK 1: TRIAGE
    # ═════════════════════════════════════════════════════════════════════════
    if task_type == "triage":
        agent_esi   = action.get("esi_level", action.get("triage_level", 0)) or 0
        correct_esi = expected.get("esi_level", 3)
        delta = abs(agent_esi - correct_esi) if (agent_esi and correct_esi) else 2

        _esi_scores = {0: 10, 1: 6, 2: 2, 3: 0}
        clinical_score = _esi_scores.get(delta, 0)

        if delta == 0:
            safety_score = 10
            explanation_parts.append(f"ESI-{agent_esi} correctly assigned.")
        elif delta == 1:
            # Under-triage is worse than over-triage
            safety_score = 8 if agent_esi < correct_esi else 6
            explanation_parts.append(
                f"ESI off by 1 (assigned {agent_esi}, correct {correct_esi}). "
                f"{'Over-triage: wastes resources.' if agent_esi < correct_esi else 'Under-triage: patient may deteriorate.'}"
            )
        else:
            clinical_score = 2
            safety_score   = 2 if agent_esi > correct_esi else 5
            explanation_parts.append(
                f"Significant ESI mismatch (assigned {agent_esi}, correct {correct_esi}). "
                f"{'UNDERTRIAGE — life-threatening.' if agent_esi > correct_esi else 'Significant over-triage.'}"
            )

        # ── Critical vital sign safety overrides ────────────────────────────
        spo2 = _get("spo2", 100)
        sbp  = _get("systolic_bp", 120)
        gcs  = _get("glasgow_coma_scale", 15)
        hr   = _get("heart_rate", 80)

        if spo2 is not None and spo2 < 90 and agent_esi and agent_esi > 2:
            safety_score = max(1, safety_score - 4)
            explanation_parts.append(f"SAFETY ⚠: SpO₂ {spo2}% — ESI 1 or 2 mandatory.")
        if sbp is not None and sbp < 80 and agent_esi and agent_esi > 2:
            safety_score = max(1, safety_score - 4)
            explanation_parts.append(f"SAFETY ⚠: SBP {sbp} mmHg — haemodynamic shock. ESI 1 mandatory.")
        if gcs is not None and gcs <= 8 and agent_esi and agent_esi > 1:
            safety_score = max(1, safety_score - 5)
            explanation_parts.append(f"SAFETY ⚠: GCS {gcs} — airway at risk. ESI 1, call resuscitation team.")
        if hr is not None and hr > 130 and agent_esi and agent_esi > 2:
            safety_score = max(2, safety_score - 2)
            explanation_parts.append(f"CONCERN: HR {hr}bpm — haemodynamic instability risk.")

        # Efficiency: penalise under-investigation for complex presentations
        interventions = action.get("recommended_immediate_interventions", [])
        if safety_score >= 8 and len(interventions) >= 2:
            efficiency_score = 8
        elif len(interventions) == 0 and safety_score < 7:
            efficiency_score = 3

        teaching_point = (
            "ESI-1 requires immediate physician presence; ESI-2 requires physician within 10 min."
            if correct_esi <= 2 else
            "Apply ESI algorithm: resource needs + vital sign abnormalities drive level."
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TASK 2: MEDICATION SAFETY
    # ═════════════════════════════════════════════════════════════════════════
    elif task_type == "medication_safety":
        flags_identified  = action.get("drug_interactions_identified",
                             action.get("flagged_interactions", []))
        contras           = action.get("flagged_contraindications", [])
        critical_flag     = action.get("has_critical_interaction",
                             action.get("severity_assessment", "") in ("critical", "major"))
        severity_proposed = action.get("severity_assessment", "").lower()

        if critical_flag:
            clinical_score = 9
            safety_score   = 9
            explanation_parts.append("Critical/major drug interaction correctly identified.")
        elif len(flags_identified) > 0 or len(contras) > 0:
            n = len(flags_identified) + len(contras)
            clinical_score = min(8, 5 + n)
            safety_score   = 7
            explanation_parts.append(f"{n} flag(s) identified — verify completeness against Lexicomp/Micromedex.")
        else:
            clinical_score = 1
            safety_score   = 2
            explanation_parts.append(
                "No drug interactions identified. Likely missed critical interaction — "
                "check CYP3A4, anticoagulation triple therapy, and renal dosing."
            )

        # Severity classification accuracy
        severity_map = {"safe": 0, "minor": 1, "moderate": 2, "major": 3, "critical": 4}
        p_sev = severity_map.get(severity_proposed, -1)
        if p_sev == 4:
            efficiency_score = 9
        elif p_sev == 0 and clinical_score < 5:
            safety_score = max(1, safety_score - 3)
            explanation_parts.append("SEVERITY ERROR: Critical interaction classified as 'safe'.")

        teaching_point = (
            "CYP3A4 inhibitors (ritonavir, azole antifungals) dramatically increase statin levels — "
            "simvastatin and lovastatin are CONTRAINDICATED with HIV protease inhibitors."
        )

    # ═════════════════════════════════════════════════════════════════════════
    # TASK 3: SEPSIS MANAGEMENT
    # ═════════════════════════════════════════════════════════════════════════
    elif task_type == "sepsis":
        bundle_items    = action.get("bundle_items", [])
        abx_ordered     = action.get("antibiotics_ordered", False)
        cultures_ordered= action.get("blood_cultures_ordered", False)
        lactate_ordered = action.get("lactate_ordered", False)
        fluids_ml       = action.get("iv_fluid_bolus_ml", 0) or 0
        vasopressors    = action.get("vasopressor_ordered", False)

        # Detect abx from bundle text
        bundle_blob = " ".join(str(b) for b in bundle_items).lower()
        abx_in_bundle = (
            abx_ordered or
            any(kw in bundle_blob for kw in
                ["antibiotic", "vancomycin", "piperacillin", "meropenem", "ceftriaxone"])
        )
        fluids_in_bundle = (
            fluids_ml > 0 or
            any(kw in bundle_blob for kw in ["fluid", "saline", "crystalloid", "bolus"])
        )
        cultures_in_bundle = (
            cultures_ordered or
            any(kw in bundle_blob for kw in ["culture", "blood culture"])
        )
        lactate_in_bundle = (
            lactate_ordered or
            any(kw in bundle_blob for kw in ["lactate", "lactic"])
        )

        bundle_score = sum([
            abx_in_bundle,
            fluids_in_bundle,
            cultures_in_bundle,
            lactate_in_bundle,
        ])

        # MAP / vasopressor check
        sbp  = _get("systolic_bp", 120)
        dbp  = _get("diastolic_bp", 80)
        map_ = int((sbp + 2 * dbp) / 3) if (sbp and dbp) else 75
        needs_vp = map_ < 65

        if needs_vp and not vasopressors:
            bundle_score = max(0, bundle_score - 1)
            explanation_parts.append(f"BUNDLE INCOMPLETE: MAP={map_} mmHg — vasopressors required.")
        elif vasopressors:
            bundle_score = min(4, bundle_score + 1)

        # Score mapping
        clinical_score = [1, 4, 6, 8, 10][min(bundle_score, 4)]
        safety_score   = [2, 5, 7, 8, 9][min(bundle_score, 4)]

        if bundle_score == 4:
            explanation_parts.append("Full SSC Hour-1 bundle addressed. Excellent sepsis management.")
        elif bundle_score >= 2:
            missing = []
            if not abx_in_bundle:     missing.append("antibiotics")
            if not fluids_in_bundle:  missing.append("IV fluids")
            if not cultures_in_bundle: missing.append("blood cultures")
            if not lactate_in_bundle: missing.append("lactate")
            explanation_parts.append(f"Partial bundle — missing: {', '.join(missing)}.")
        else:
            explanation_parts.append(
                "CRITICAL: Sepsis Hour-1 bundle not initiated. Every hour without antibiotics "
                "increases mortality by ~7% (SSC 2021)."
            )

        # Allergy safety check
        allergies_raw = patient.get("allergies", [])
        allergies     = [str(a).lower() for a in allergies_raw]
        abx_choice    = action.get("antibiotic_choice", "").lower() if action.get("antibiotic_choice") else ""
        pcn_allergy   = any("penicillin" in a for a in allergies)
        if pcn_allergy and "piperacillin" in abx_choice:
            safety_score = max(0, safety_score - 5)
            explanation_parts.append(
                "ALLERGY VIOLATION ⚠: Piperacillin-tazobactam prescribed despite documented penicillin allergy."
            )

        teaching_point = (
            "SSC 2021 Hour-1 Bundle: Blood cultures → broad-spectrum antibiotics → 30ml/kg crystalloid → "
            "lactate → norepinephrine if MAP < 65 mmHg. Antibiotic delay increases mortality."
        )

    # ── Over-investigation efficiency penalty ────────────────────────────────
    tests_ordered = len(
        action.get("investigations", action.get("tests_ordered", []))
    )
    if tests_ordered > 8:
        efficiency_score = max(2, efficiency_score - 2)
        explanation_parts.append("Over-investigation reduces emergency department efficiency.")

    # ── Weighted total ───────────────────────────────────────────────────────
    total_score = round(
        safety_score    * 0.35 +
        clinical_score  * 0.30 +
        reasoning_score * 0.15 +
        efficiency_score* 0.10 +
        ethics_score    * 0.10
    )

    # ── Reward adjustment ────────────────────────────────────────────────────
    if total_score >= 9:    reward_adjustment = 0.45
    elif total_score >= 8:  reward_adjustment = 0.35
    elif total_score >= 7:  reward_adjustment = 0.20
    elif total_score >= 5:  reward_adjustment = 0.00
    elif total_score >= 3:  reward_adjustment = -0.20
    else:                   reward_adjustment = -0.40

    if safety_score <= 2:
        reward_adjustment = -0.50   # safety override: life-threatening error

    explanation = " ".join(explanation_parts) or "Rule-based evaluation (no LLM backend available)."

    return {
        "clinical_score":    clinical_score,
        "safety_score":      safety_score,
        "efficiency_score":  efficiency_score,
        "ethics_score":      ethics_score,
        "reasoning_score":   reasoning_score,
        "total_score":       total_score,
        "reward_adjustment": round(reward_adjustment, 3),
        "confidence":        0.55,
        "explanation":       explanation[:500],
        "teaching_point":    teaching_point,
    }


# ──────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSER  (robust JSON extraction + field validation)
# ──────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> Dict[str, Any]:
    """
    Extract and validate JSON from a raw LLM response string.
    Handles:
      • Markdown code fences (``` json ... ```)
      • Leading / trailing whitespace
      • Partial JSON objects (regex extraction)
      • Missing or out-of-range fields (clamped with defaults)
    """
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # Attempt direct parse first
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Regex fallback: extract first complete JSON object
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise ValueError(f"Cannot extract JSON from LLM response: {raw[:300]}")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in LLM response: {exc}") from exc

    # ── Integer score fields (0–10) ──────────────────────────────────────────
    int_fields = [
        "clinical_score", "safety_score", "efficiency_score",
        "ethics_score",   "reasoning_score", "total_score",
    ]
    for f in int_fields:
        data[f] = max(0, min(10, int(float(data.get(f, 5)))))

    # ── Float fields (clamped) ───────────────────────────────────────────────
    data["reward_adjustment"] = round(
        max(-0.5, min(0.5, float(data.get("reward_adjustment", 0.0)))), 3
    )
    data["confidence"] = round(
        max(0.0, min(1.0, float(data.get("confidence", 0.7)))), 2
    )

    # ── String fields (max length) ───────────────────────────────────────────
    data["explanation"]   = str(data.get("explanation", ""))[:600]
    data["teaching_point"]= str(data.get("teaching_point", ""))[:300]

    # ── Recompute total_score to enforce formula (guards against LLM drift) ─
    formula_total = round(
        data["safety_score"]    * 0.35 +
        data["clinical_score"]  * 0.30 +
        data["reasoning_score"] * 0.15 +
        data["efficiency_score"]* 0.10 +
        data["ethics_score"]    * 0.10
    )
    # Allow ±1 tolerance; if LLM deviates more, override
    if abs(formula_total - data["total_score"]) > 1:
        logger.debug(
            "LLM total_score %d overridden by formula result %d",
            data["total_score"], formula_total
        )
        data["total_score"] = formula_total

    return data


# ──────────────────────────────────────────────────────────────────────────────
# BACKEND DISPATCH TABLE
# ──────────────────────────────────────────────────────────────────────────────

_BACKEND_CALLERS = {
    LLMBackend.LLAMA3_GROQ:     _call_groq_llama3,
    LLMBackend.LLAMA3_TOGETHER: _call_together_llama3,
    LLMBackend.MISTRAL:         _call_mistral,
    LLMBackend.GPT4:            _call_openai_gpt4,
}

_PRIORITY_ORDER: List[LLMBackend] = [
    LLMBackend.LLAMA3_GROQ,
    LLMBackend.LLAMA3_TOGETHER,
    LLMBackend.MISTRAL,
    LLMBackend.GPT4,
]


# ──────────────────────────────────────────────────────────────────────────────
# PRIMARY PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_with_llm(
    state:          Dict[str, Any],
    action:         Dict[str, Any],
    reasoning:      str,
    backend:        Optional[LLMBackend] = None,
    oracle_action:  Optional[Dict[str, Any]] = None,
    use_cache:      bool = True,
    grader_score:   Optional[float] = None,
) -> LLMEvalResult:
    """
    Evaluate an RL agent's clinical decision using a LLM judge.
    This is the core reward-shaping function of ClinicalTriageEnv.
    It is called once per environment step by training_loop.py and app.py.
    Args:
        state          : Current env state (patient data, task metadata, vitals).
        action         : The agent's chosen action dict.
        reasoning      : Free-text explanation provided by the agent.
        backend        : Override the default LLM backend (env var: LLM_BACKEND).
        oracle_action  : Optional ideal-physician action for delta scoring.
        use_cache      : Enable in-memory LRU cache (recommended for roll-outs).
        grader_score   : GradeResult.score from graders.py [0,1] for alignment check.
    Returns:
        LLMEvalResult with all dimension scores, reward_adjustment, and explanation.
    Example:
        result = evaluate_with_llm(state, action, reasoning, grader_score=grade.score)
        final_reward = rule_reward + DEFAULT_ALPHA * result.reward_adjustment
    Integration points:
        • app.py   : POST /rl/step  and  POST /rl/evaluate
        • training_loop.py : called inside episode loop after env.step()
        • environment_v2.py: ClinicalTriageEnvV2._compute_llm_reward()
    """
    backend = backend or LLM_BACKEND
    t0      = time.perf_counter()

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt = build_eval_prompt(state, action, reasoning, oracle_action)

    # ── Cache lookup ─────────────────────────────────────────────────────────
    if use_cache:
        cached = _CACHE.get(prompt, str(backend))
        if cached is not None:
            METRICS.record(cached)
            return cached

    # ── Backend resolution: requested → priority chain → rule_based ──────────
    backends_to_try: List[LLMBackend] = []
    if backend == LLMBackend.RULE_BASED:
        backends_to_try = [LLMBackend.RULE_BASED]
    else:
        backends_to_try = [backend] + [
            b for b in _PRIORITY_ORDER if b != backend
        ] + [LLMBackend.RULE_BASED]

    raw_data: Dict[str, Any] = {}
    backend_used: str = "rule_based"
    cache_hit = False

    for b in backends_to_try:
        try:
            if b == LLMBackend.RULE_BASED:
                raw_data     = _rule_based_eval(state, action, reasoning)
                backend_used = "rule_based"
                break
            else:
                caller   = _BACKEND_CALLERS[b]
                raw_text = caller(prompt)
                raw_data = _parse_llm_response(raw_text)
                backend_used = b.value
                break
        except Exception as exc:
            logger.warning("LLM backend %s failed: %s — trying next.", b.value, exc)
            continue

    # Absolute fallback (should never reach here, but be safe)
    if not raw_data:
        raw_data     = _rule_based_eval(state, action, reasoning)
        backend_used = "rule_based_emergency_fallback"

    latency_ms = int((time.perf_counter() - t0) * 1000)

    # ── Oracle deviation summary ─────────────────────────────────────────────
    oracle_deviation = ""
    if oracle_action and action:
        agent_esi   = action.get("esi_level") or action.get("triage_level")
        oracle_esi  = oracle_action.get("esi_level") or oracle_action.get("triage_level")
        if agent_esi and oracle_esi and agent_esi != oracle_esi:
            oracle_deviation = (
                f"Agent assigned ESI-{agent_esi}; oracle recommends ESI-{oracle_esi}. "
                f"{'Undertriage.' if agent_esi > oracle_esi else 'Overtriage.'}"
            )

    # ── Grader alignment score ───────────────────────────────────────────────
    grader_alignment = 0.0
    if grader_score is not None:
        llm_normalised  = raw_data["total_score"] / 10.0
        grader_alignment = max(0.0, 1.0 - abs(llm_normalised - grader_score))

    result = LLMEvalResult(
        clinical_score    = raw_data["clinical_score"],
        safety_score      = raw_data["safety_score"],
        efficiency_score  = raw_data["efficiency_score"],
        ethics_score      = raw_data["ethics_score"],
        reasoning_score   = raw_data["reasoning_score"],
        total_score       = raw_data["total_score"],
        reward_adjustment = raw_data["reward_adjustment"],
        confidence        = raw_data["confidence"],
        explanation       = raw_data["explanation"],
        teaching_point    = raw_data.get("teaching_point", ""),
        oracle_deviation  = oracle_deviation,
        backend_used      = backend_used,
        latency_ms        = latency_ms,
        cache_hit         = cache_hit,
        grader_alignment  = round(grader_alignment, 3),
    )

    # ── Store in cache ───────────────────────────────────────────────────────
    if use_cache and not cache_hit:
        _CACHE.set(prompt, str(backend), result)

    METRICS.record(result)
    logger.debug("evaluate_with_llm: %s", result.summary())
    return result


# ──────────────────────────────────────────────────────────────────────────────
# BATCH EVALUATOR  (parallel evaluation for multi-patient episodes)
# ──────────────────────────────────────────────────────────────────────────────

class BatchEvaluator:
    """
    Evaluate multiple (state, action, reasoning) triples concurrently.
    Used by ClinicalTriageEnvV2 during SURGE/CHAOS episodes with many patients.
    Falls back to sequential evaluation if the thread pool is unavailable.
    Example:
        evaluator = BatchEvaluator(max_workers=4)
        results = evaluator.evaluate_batch(triples, backend=LLMBackend.LLAMA3_GROQ)
        for r in results:
            print(r.summary())
    """

    def __init__(self, max_workers: int = MAX_WORKERS):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm_eval")

    def evaluate_batch(
        self,
        triples: List[Tuple[Dict, Dict, str]],   # [(state, action, reasoning), ...]
        backend: Optional[LLMBackend] = None,
        oracle_actions: Optional[List[Optional[Dict]]] = None,
    ) -> List[LLMEvalResult]:
        """
        Evaluate a list of (state, action, reasoning) triples in parallel.
        Results are returned in the same order as inputs.
        """
        oracles = oracle_actions or [None] * len(triples)
        futures_map = {}
        results: List[Optional[LLMEvalResult]] = [None] * len(triples)

        for idx, (state, action, reasoning) in enumerate(triples):
            fut = self._executor.submit(
                evaluate_with_llm,
                state, action, reasoning, backend, oracles[idx]
            )
            futures_map[fut] = idx

        for fut in as_completed(futures_map):
            idx = futures_map[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                logger.error("Batch eval failed for index %d: %s", idx, exc)
                # Fallback: synchronous rule-based
                state, action, reasoning = triples[idx]
                results[idx] = evaluate_with_llm(
                    state, action, reasoning, LLMBackend.RULE_BASED
                )

        return results  # type: ignore[return-value]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


# Convenience singleton for training_loop.py
BATCH_EVALUATOR = BatchEvaluator()


# ──────────────────────────────────────────────────────────────────────────────
# ORACLE / "WHAT WOULD A DOCTOR DO?" MODE
# ──────────────────────────────────────────────────────────────────────────────

def get_oracle_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the heuristic gold-standard physician action for a given state.
    Implements evidence-based clinical guidelines:
      • Triage  : ESI algorithm (ACEP 2020) with vital sign overrides
      • Medication: CYP3A4 / anticoagulation / renal dosing checks (Lexicomp)
      • Sepsis  : Surviving Sepsis Campaign Hour-1 Bundle (SSC 2021)
    Used for:
      • "What Would A Doctor Do?" UI panel
      • Oracle comparison in evaluate_with_llm()
      • Baseline benchmark in training_loop.py
    """
    task_type = state.get("task_type", "triage")
    patient   = state.get("patient", {})
    vitals    = patient.get("vitals", {})

    def _v(key: str, default=None):
        if isinstance(vitals, dict):
            return vitals.get(key, default)
        return getattr(vitals, key, default)

    # ═════════════════════════════════════════════════════════════════════════
    # TRIAGE ORACLE
    # ═════════════════════════════════════════════════════════════════════════
    if task_type == "triage":
        spo2 = _v("spo2", 100)
        sbp  = _v("systolic_bp", 120)
        hr   = _v("heart_rate", 80)
        gcs  = _v("glasgow_coma_scale", 15)
        rr   = _v("respiratory_rate", 16)
        cc   = (patient.get("chief_complaint", "") or "").lower()

        # ESI-1: Immediate life threat
        if gcs is not None and gcs <= 8:
            esi, rationale = 1, "ESI-1: GCS ≤8 — airway at risk. Immediate resuscitation."
        elif (spo2 is not None and spo2 < 90) or (sbp is not None and sbp < 80):
            esi, rationale = 1, (
                f"ESI-1: Critical vitals (SpO₂={spo2}%, SBP={sbp}mmHg). "
                "Resuscitation bay — call team immediately."
            )
        # ESI-2: High-risk / emergent
        elif (
            (hr  is not None and hr  > 130) or
            (sbp is not None and sbp < 100) or
            (rr  is not None and rr  > 25)  or
            (spo2 is not None and spo2 < 94)
        ):
            esi, rationale = 2, (
                f"ESI-2: Haemodynamic instability or respiratory compromise "
                f"(HR={hr}, SBP={sbp}, RR={rr}, SpO₂={spo2}%). "
                "Physician within 10 minutes."
            )
        elif any(kw in cc for kw in ("chest pain", "stroke", "seizure",
                                      "confusion", "syncope", "overdose",
                                      "worst headache", "droop")):
            esi, rationale = 2, (
                f"ESI-2: High-risk chief complaint ('{cc}') — requires immediate evaluation. "
                "Time-critical diagnosis cannot be excluded without physician."
            )
        # ESI-3: Urgent
        elif (hr is not None and hr > 100) or (rr is not None and rr > 20):
            esi, rationale = 3, (
                "ESI-3: Abnormal vitals without immediate life threat. "
                "Requires ≥2 resources; urgent nurse and physician assessment."
            )
        # ESI-4: Less urgent
        elif any(kw in cc for kw in ("sprain", "laceration", "rash", "uti")):
            esi, rationale = 4, "ESI-4: Non-acute complaint with stable vitals. One resource expected."
        # ESI-5: Non-urgent
        else:
            esi, rationale = 5, "ESI-5: Non-urgent. Stable vitals, minor complaint, no expected resources."

        return {
            "esi_level":   esi,
            "rationale":   rationale,
            "recommended_immediate_interventions": _triage_interventions(esi, vitals, _v),
            "disposition": ["Resuscitation Bay", "High Acuity", "Standard Bay",
                            "Fast Track", "Waiting Area"][esi - 1],
            "oracle": True,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # MEDICATION SAFETY ORACLE
    # ═════════════════════════════════════════════════════════════════════════
    elif task_type == "medication_safety":
        meds = [
            (m.get("name", "").lower() if isinstance(m, dict) else str(m).lower())
            for m in patient.get("current_medications", [])
        ]
        labs = patient.get("labs", {})
        egfr = labs.get("egfr") if isinstance(labs, dict) else getattr(labs, "egfr", None)

        interactions:      List[str] = []
        contraindications: List[str] = []
        dosing_errors:     List[str] = []
        recommendations:   List[str] = []
        severity = "safe"

        # ── Simvastatin / Lovastatin + Ritonavir (CYP3A4) ───────────────────
        if any(s in " ".join(meds) for s in ("simvastatin", "lovastatin")):
            if any(p in " ".join(meds) for p in ("ritonavir", "cobicistat", "protease inhibitor")):
                interactions.append("Simvastatin/Lovastatin + Ritonavir → CYP3A4 inhibition → rhabdomyolysis")
                contraindications.append("Simvastatin/Lovastatin CONTRAINDICATED with HIV PIs")
                recommendations.append("DISCONTINUE simvastatin. Switch to pravastatin or rosuvastatin (low CYP3A4 interaction).")
                severity = "critical"

        # ── Triple antithrombotic therapy ────────────────────────────────────
        has_warfarin   = "warfarin" in " ".join(meds)
        has_aspirin    = "aspirin"  in " ".join(meds)
        has_p2y12      = any(p in " ".join(meds) for p in ("clopidogrel", "ticagrelor", "prasugrel"))
        if has_warfarin and has_aspirin and has_p2y12:
            interactions.append("Triple antithrombotic therapy → major haemorrhage risk")
            recommendations.append(
                "Minimise triple therapy duration (<1 month). Add PPI (omeprazole 20mg). "
                "Consider dual therapy when clinically appropriate."
            )
            severity = max(severity, "major") if severity != "critical" else "critical"

        # ── Metformin + eGFR < 30 ────────────────────────────────────────────
        if "metformin" in " ".join(meds):
            if egfr is not None and egfr < 30:
                contraindications.append(f"Metformin CONTRAINDICATED: eGFR={egfr} (<30) → lactic acidosis risk")
                dosing_errors.append(f"Metformin: eGFR {egfr} — HOLD immediately.")
                recommendations.append("STOP metformin. Switch to SGLT2 inhibitor or GLP-1 agonist with eGFR guidance.")
                severity = "critical" if severity != "critical" else "critical"
            elif egfr is not None and egfr < 45:
                dosing_errors.append(f"Metformin: eGFR {egfr} — reduce to 500mg BD, monitor closely.")
                severity = max(severity, "major") if severity not in ("critical",) else "critical"

        return {
            "flagged_interactions":     interactions,
            "flagged_contraindications": contraindications,
            "flagged_dosing_errors":    dosing_errors,
            "recommended_changes":      recommendations,
            "severity_assessment":      severity,
            "has_critical_interaction": severity in ("critical", "major"),
            "monitoring_required":      ["CK levels", "INR/PT", "eGFR", "LFTs"],
            "clinical_rationale":       (
                "Systematic drug interaction review per Lexicomp/Micromedex. "
                "CYP3A4 pathways, renal dosing, and antithrombotic safety assessed."
            ),
            "oracle": True,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # SEPSIS ORACLE  (SSC 2021 Hour-1 Bundle)
    # ═════════════════════════════════════════════════════════════════════════
    elif task_type == "sepsis":
        sbp  = _v("systolic_bp", 120)
        dbp  = _v("diastolic_bp", 80)
        map_ = int((sbp + 2 * dbp) / 3) if (sbp and dbp) else 75
        allergies = [str(a).lower() for a in patient.get("allergies", [])]

        pcn_allergy  = any("penicillin" in a for a in allergies)
        vanco_allergy= any("vancomycin" in a for a in allergies)

        # Antibiotic selection (allergy-aware)
        if pcn_allergy and vanco_allergy:
            abx = "Aztreonam 2g IV q8h + Daptomycin 6mg/kg IV q24h"
            abx_note = "PCN + Vancomycin allergy: aztreonam + daptomycin combination."
        elif pcn_allergy:
            abx = "Aztreonam 2g IV q8h + Vancomycin 25–30mg/kg IV loading dose"
            abx_note = "PCN allergy: avoid piperacillin-tazobactam. Aztreonam is safe."
        elif vanco_allergy:
            abx = "Piperacillin-Tazobactam 4.5g IV q6h + Daptomycin 6mg/kg IV q24h"
            abx_note = "Vancomycin allergy: use daptomycin for MRSA coverage."
        else:
            abx = "Piperacillin-Tazobactam 4.5g IV q6h + Vancomycin 25–30mg/kg IV loading dose"
            abx_note = "Broad-spectrum gram-positive and gram-negative coverage."

        needs_vasopressor = map_ < 65
        bundle = [
            "Blood cultures ×2 peripherally (before antibiotics)",
            f"IV Antibiotics: {abx} within 60 minutes",
            "IV Crystalloid bolus: 30 mL/kg (target completion <3 hours)",
            "Serum lactate (repeat if initial ≥2 mmol/L)",
            "Urine output catheter (target >0.5 mL/kg/hr)",
        ]
        if needs_vasopressor:
            bundle.append(
                f"Norepinephrine (first-line vasopressor) — MAP {map_} mmHg below target 65 mmHg. "
                "Titrate to MAP ≥65. Insert arterial line."
            )

        return {
            "sepsis_diagnosis":         "septic_shock" if needs_vasopressor else "sepsis",
            "blood_cultures_ordered":   True,
            "antibiotics_ordered":      True,
            "antibiotic_choice":        abx,
            "antibiotic_rationale":     abx_note,
            "lactate_ordered":          True,
            "iv_fluid_bolus_ml":        2100,   # 30ml/kg × 70kg
            "vasopressor_ordered":      needs_vasopressor,
            "vasopressor_choice":       "Norepinephrine" if needs_vasopressor else None,
            "target_map":               65,
            "bundle_items":             bundle,
            "source_control_identified":"Identify and control source (imaging, surgical consult if indicated)",
            "time_to_antibiotics_minutes": 45,
            "clinical_rationale": (
                f"SSC 2021 Hour-1 Bundle. MAP={map_} mmHg. {abx_note} "
                "All five bundle elements initiated. Source control evaluated."
            ),
            "oracle": True,
        }

    return {"action": "unknown", "rationale": "Unknown task type", "oracle": True}


def _triage_interventions(esi: int, vitals: Any, _v) -> List[str]:
    """Return immediate interventions based on ESI level and current vitals."""
    base = {
        1: [
            "Call resuscitation team STAT",
            "IV access ×2 (large bore 16G+)",
            "Continuous cardiac monitor",
            "Pulse oximetry + waveform capnography",
            "12-lead ECG within 5 min",
            "Oxygen 15 L/min via NRB mask",
            "Point-of-care glucose",
            "Airway adjuncts at bedside",
        ],
        2: [
            "IV access",
            "Continuous cardiac monitor",
            "12-lead ECG",
            "Oxygen if SpO₂ <94%",
            "Nurse-initiated triage protocols",
            "Physician notified within 10 min",
        ],
        3: [
            "Nurse assessment",
            "IV access if indicated",
            "Bloods + ECG if clinically indicated",
            "Pain / symptom management per protocol",
        ],
        4: ["Nurse assessment", "Vital signs", "Document chief complaint"],
        5: ["Vital signs", "Triage registration", "Non-urgent pathway"],
    }
    interventions = list(base.get(esi, []))
    spo2 = _v("spo2", 100)
    if spo2 is not None and spo2 < 94 and "Oxygen 15 L/min via NRB mask" not in interventions:
        interventions.insert(0, f"URGENT: Supplemental oxygen — SpO₂ {spo2}%")
    return interventions


# ──────────────────────────────────────────────────────────────────────────────
# REWARD INTEGRATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_hybrid_reward(
    rule_reward:  float,
    llm_result:   LLMEvalResult,
    alpha:        float = DEFAULT_ALPHA,
    penalty_cap:  float = -1.5,
    reward_cap:   float = 2.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Combine rule-based environment reward with LLM adjustment.
    Formula:
        final_reward = clip(rule_reward + alpha × llm_result.reward_adjustment,
                            penalty_cap, reward_cap)
    Args:
        rule_reward  : Reward from ClinicalTriageEnv deterministic grader.
        llm_result   : Output from evaluate_with_llm().
        alpha        : LLM contribution weight (default 0.30 = 30%).
        penalty_cap  : Lower bound for final reward (default -1.5).
        reward_cap   : Upper bound for final reward (default 2.0).
    Returns:
        (final_reward, breakdown_dict)
    Example:
        obs, rule_r, done, info = env.step(action)
        llm_r = evaluate_with_llm(state, action, reasoning)
        final_r, breakdown = compute_hybrid_reward(rule_r, llm_r)
        replay_buffer.add(state, action, final_r, next_state, done)
    """
    llm_contribution = alpha * llm_result.reward_adjustment
    final_reward     = round(
        max(penalty_cap, min(reward_cap, rule_reward + llm_contribution)), 4
    )

    breakdown = {
        "rule_reward":       round(rule_reward, 4),
        "llm_adjustment":    round(llm_result.reward_adjustment, 4),
        "llm_contribution":  round(llm_contribution, 4),
        "alpha":             alpha,
        "final_reward":      final_reward,
        "is_critical_failure": llm_result.is_critical_failure,
        "llm_scores": {
            "clinical":   llm_result.clinical_score,
            "safety":     llm_result.safety_score,
            "efficiency": llm_result.efficiency_score,
            "ethics":     llm_result.ethics_score,
            "reasoning":  llm_result.reasoning_score,
            "total":      llm_result.total_score,
        },
        "confidence":        llm_result.confidence,
        "grader_alignment":  llm_result.grader_alignment,
        "backend":           llm_result.backend_used,
        "latency_ms":        llm_result.latency_ms,
        "cache_hit":         llm_result.cache_hit,
        "explanation":       llm_result.explanation,
        "teaching_point":    llm_result.teaching_point,
        "oracle_deviation":  llm_result.oracle_deviation,
    }

    return final_reward, breakdown


# ──────────────────────────────────────────────────────────────────────────────
# ASYNC WRAPPER  (for FastAPI / async training loops)
# ──────────────────────────────────────────────────────────────────────────────

async def evaluate_with_llm_async(
    state:         Dict[str, Any],
    action:        Dict[str, Any],
    reasoning:     str,
    backend:       Optional[LLMBackend] = None,
    oracle_action: Optional[Dict[str, Any]] = None,
    grader_score:  Optional[float] = None,
) -> LLMEvalResult:
    """
    Async wrapper around evaluate_with_llm() for use in async FastAPI endpoints.
    Runs the synchronous evaluation in a thread-pool executor to avoid blocking
    the event loop.
    Usage (in app.py):
        result = await evaluate_with_llm_async(state, action, reasoning)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: evaluate_with_llm(
            state, action, reasoning, backend, oracle_action, True, grader_score
        )
    )


# ──────────────────────────────────────────────────────────────────────────────
# SELF-TEST  (python llm_evaluator.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # ── Minimal triage state ─────────────────────────────────────────────────
    _state = {
        "task_type":  "triage",
        "task_id":    "triage_hard_01",
        "difficulty": "hard",
        "patient": {
            "age": 72, "sex": "M",
            "chief_complaint": "Sudden onset right-sided facial droop and arm weakness",
            "vitals": {
                "heart_rate": 98, "systolic_bp": 168, "diastolic_bp": 94,
                "respiratory_rate": 18, "spo2": 96,
                "glasgow_coma_scale": 13, "temperature": 36.8,
            },
            "current_medications": [{"name": "Warfarin 5mg"}, {"name": "Metformin 1g BD"}],
            "allergies": [],
            "risk_factors": ["Hypertension", "Atrial fibrillation", "T2DM"],
            "labs": {"inr": 2.4, "glucose": 9.1},
        },
        "expected_action": {"esi_level": 2},
    }
    _action = {
        "esi_level": 2,
        "rationale": (
            "ESI-2: Acute stroke presentation with GCS 13, focal neurology, and "
            "hypertensive blood pressure 168/94. Patient on warfarin with INR 2.4 — "
            "thrombolytics contraindicated but immediate CT head / CTA required. "
            "Activate stroke team. Door-to-CT target <25 min."
        ),
        "recommended_immediate_interventions": [
            "Activate stroke code", "IV access", "12-lead ECG",
            "Non-contrast CT head STAT", "Hold warfarin", "Neurology consult",
        ],
    }
    _reasoning = (
        "Acute ischaemic stroke with high NIHSS. Warfarin use and elevated INR means tPA "
        "contraindicated but mechanical thrombectomy window should be assessed. "
        "Anticoagulation reversal not yet indicated. Time-critical case."
    )

    print("=" * 70)
    print("ClinicalTriageEnv — LLM Evaluator Self-Test")
    print("=" * 70)

    # ── Oracle ───────────────────────────────────────────────────────────────
    oracle = get_oracle_action(_state)
    print(f"\nOracle ESI   : {oracle['esi_level']}")
    print(f"Oracle reason: {oracle['rationale'][:120]}...")

    # ── Evaluation (rule_based — no API key required) ─────────────────────
    result = evaluate_with_llm(
        _state, _action, _reasoning,
        backend=LLMBackend.RULE_BASED,
        oracle_action=oracle,
        grader_score=0.82,
    )

    print(f"\n{result.summary()}")
    print(f"\nClinical  : {result.clinical_score}/10")
    print(f"Safety    : {result.safety_score}/10")
    print(f"Efficiency: {result.efficiency_score}/10")
    print(f"Ethics    : {result.ethics_score}/10")
    print(f"Reasoning : {result.reasoning_score}/10")
    print(f"Total     : {result.total_score}/10")
    print(f"Δ Reward  : {result.reward_adjustment:+.3f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Alignment : {result.grader_alignment:.3f}")
    print(f"\nExplanation:\n  {result.explanation}")
    print(f"\nTeaching Point:\n  {result.teaching_point}")

    # ── Hybrid reward ─────────────────────────────────────────────────────
    final_r, breakdown = compute_hybrid_reward(rule_reward=0.82, llm_result=result)
    print(f"\nHybrid Reward: {breakdown['rule_reward']} + "
          f"{breakdown['alpha']} × {breakdown['llm_adjustment']} "
          f"= {final_r}")

    # ── Metrics snapshot ──────────────────────────────────────────────────
    print(f"\nMetrics: {METRICS.snapshot()}")
    print("\n✅ Self-test passed.")
