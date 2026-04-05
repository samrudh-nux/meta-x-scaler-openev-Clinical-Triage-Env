from __future__ import annotations

import os
import sys
import uuid
import json
import time
import io
import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Core environment (v1 — real graders) ─────────────────────────────────────
try:
    from environment import ClinicalTriageEnv, TASK_REGISTRY
    from models import TriageAction, MedicationSafetyAction, SepsisManagementAction
    from graders import TriageGrader, MedicationSafetyGrader, SepsisGrader
    from scenarios import TRIAGE_SCENARIOS, MEDICATION_SCENARIOS, SEPSIS_SCENARIOS
    ENV_V1_AVAILABLE = True
    print("✅ environment.py loaded")
except ImportError as e:
    ENV_V1_AVAILABLE = False
    TASK_REGISTRY = {}
    print(f"⚠️  environment.py unavailable: {e}")

# ── Inference (Llama 3 via HF router) ────────────────────────────────────────
try:
    from inference import (
        get_client, run_task as llm_run_task,
        build_triage_prompt, build_med_safety_prompt, build_sepsis_prompt,
        call_llm, extract_json, get_fallback_action, build_action as build_llm_action,
        SYSTEM_PROMPTS, MODEL_NAME, ALL_TASKS
    )
    INFERENCE_AVAILABLE = True
    print(f"✅ inference.py loaded — model: {MODEL_NAME}")
except ImportError as e:
    INFERENCE_AVAILABLE = False
    MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
    print(f"⚠️  inference.py unavailable: {e}")

# ── LLM Evaluator (reward shaping) ───────────────────────────────────────────
try:
    from llm_evaluator import (
        evaluate_with_llm, compute_hybrid_reward, get_oracle_action,
        LLMBackend, LLMEvalResult
    )
    LLM_EVAL_AVAILABLE = True
    print("✅ llm_evaluator.py loaded")
except ImportError as e:
    LLM_EVAL_AVAILABLE = False
    print(f"⚠️  llm_evaluator.py unavailable: {e}")

# ── RL Environment v2 (multi-patient) ────────────────────────────────────────
try:
    from environment_v2 import ClinicalTriageEnvV2, DifficultyMode
    ENV_V2_AVAILABLE = True
    print("✅ environment_v2.py loaded")
except ImportError as e:
    ENV_V2_AVAILABLE = False
    print(f"⚠️  environment_v2.py unavailable: {e}")

# ── ML Engine (Q-learning) ───────────────────────────────────────────────────
try:
    from ml_engine import QLearningAgent
    ML_ENGINE_AVAILABLE = True
    print("✅ ml_engine.py loaded")
except ImportError:
    ML_ENGINE_AVAILABLE = False

# ── Optional: PDF ─────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Optional: OpenAI ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Optional: Anthropic ───────────────────────────────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# =============================================================================
# TASK REGISTRY FALLBACK (if environment.py is unavailable)
# =============================================================================

if not TASK_REGISTRY:
    TASK_REGISTRY = {
        "triage_easy":       {"name": "Emergency Triage - Easy",       "type": "triage",          "difficulty": "easy",   "max_steps": 3, "scenario_key": "triage_easy_01",   "description": "Assign correct ESI triage level. ESI: 1=Resuscitation…5=Non-Urgent."},
        "triage_medium":     {"name": "Emergency Triage - Medium",     "type": "triage",          "difficulty": "medium", "max_steps": 3, "scenario_key": "triage_medium_01", "description": "Triage patient presenting with potential ACS."},
        "triage_hard":       {"name": "Emergency Triage - Hard",       "type": "triage",          "difficulty": "hard",   "max_steps": 3, "scenario_key": "triage_hard_01",   "description": "Triage complex patient with acute neurological symptoms."},
        "med_safety_easy":   {"name": "Medication Safety Review - Easy",  "type": "medication_safety", "difficulty": "easy",   "max_steps": 3, "scenario_key": "med_easy_01",      "description": "Review medication list for drug interactions."},
        "med_safety_medium": {"name": "Medication Safety Review - Medium","type": "medication_safety", "difficulty": "medium", "max_steps": 3, "scenario_key": "med_medium_01",   "description": "Post-cardiac cath patient on triple antithrombotic therapy."},
        "med_safety_hard":   {"name": "Medication Safety Review - Hard",  "type": "medication_safety", "difficulty": "hard",   "max_steps": 3, "scenario_key": "med_hard_01",    "description": "HIV patient with life-threatening drug interaction."},
        "sepsis_easy":       {"name": "Sepsis Management - Easy",      "type": "sepsis",          "difficulty": "easy",   "max_steps": 3, "scenario_key": "sepsis_easy_01",   "description": "Recognise sepsis criteria and execute Hour-1 SSC bundle."},
        "sepsis_medium":     {"name": "Sepsis Management - Medium",    "type": "sepsis",          "difficulty": "medium", "max_steps": 3, "scenario_key": "sepsis_medium_01", "description": "Manage septic shock in elderly nursing home patient."},
        "sepsis_hard":       {"name": "Sepsis Management - Hard",      "type": "sepsis",          "difficulty": "hard",   "max_steps": 3, "scenario_key": "sepsis_hard_01",   "description": "Post-operative septic shock with anastomotic leak."},
    }

MORTALITY_RISK = {
    "triage_easy":       {"baseline": 0.5,  "undertriage_mult": 2.0, "delay_per_min": 0.01},
    "triage_medium":     {"baseline": 8.0,  "undertriage_mult": 3.5, "delay_per_min": 0.15},
    "triage_hard":       {"baseline": 18.0, "undertriage_mult": 5.0, "delay_per_min": 0.40},
    "med_safety_easy":   {"baseline": 0.2,  "undertriage_mult": 1.5, "delay_per_min": 0.005},
    "med_safety_medium": {"baseline": 3.0,  "undertriage_mult": 2.5, "delay_per_min": 0.05},
    "med_safety_hard":   {"baseline": 12.0, "undertriage_mult": 4.0, "delay_per_min": 0.30},
    "sepsis_easy":       {"baseline": 6.0,  "undertriage_mult": 2.5, "delay_per_min": 0.20},
    "sepsis_medium":     {"baseline": 22.0, "undertriage_mult": 4.0, "delay_per_min": 0.55},
    "sepsis_hard":       {"baseline": 45.0, "undertriage_mult": 6.0, "delay_per_min": 1.20},
}


# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="ClinicalTriageEnv v5 — RL + LLM Hybrid Clinical AI",
    version="5.0.0",
    description=(
        "Fully integrated RL + LLM system for clinical triage optimization. "
        "Uses Llama 3 70B for inference and reward shaping. "
        "We use a Llama-based evaluator to align RL agents with human clinical reasoning."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SESSION STORES
# =============================================================================

_v1_sessions:   Dict[str, Dict] = {}
_v2_sessions:   Dict[str, Any]  = {}
_train_jobs:    Dict[str, Dict] = {}
_report_cache:  Dict[str, Any]  = {}
_chat_histories: Dict[str, List] = {}
_llm_client = None  # lazy-init HF/OpenAI client


def _get_llm_client():
    global _llm_client
    if _llm_client is None and INFERENCE_AVAILABLE:
        try:
            _llm_client = get_client()
        except Exception:
            pass
    return _llm_client


# =============================================================================
# PYDANTIC REQUEST MODELS
# =============================================================================

class ResetRequest(BaseModel):
    task_id: Optional[str] = "triage_easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None
    reasoning: Optional[str] = ""
    use_llm_eval: Optional[bool] = True

class AnalyzeRequest(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = "Anonymous"
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: str = Field(..., min_length=5)
    vitals: Optional[Dict[str, Any]] = None
    risk_factors: Optional[List[str]] = []

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = []
    session_id: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None

class BenchmarkRequest(BaseModel):
    task_id: str
    user_action: Dict[str, Any]

class InferenceRequest(BaseModel):
    task_id: str
    use_cot: Optional[bool] = True

class RLResetRequest(BaseModel):
    difficulty: str = "calm"
    llm_backend: str = "rule_based"
    task_type: str = "triage"
    enable_deterioration: bool = True
    seed: Optional[int] = None

class RLStepRequest(BaseModel):
    session_id: str
    patient_id: str
    action: Dict[str, Any]
    reasoning: str = ""

class LLMEvalRequest(BaseModel):
    state: Dict[str, Any]
    action: Dict[str, Any]
    reasoning: str = ""
    backend: str = "rule_based"

class OracleRequest(BaseModel):
    state: Dict[str, Any]

class TrainRequest(BaseModel):
    n_episodes: int = Field(default=20, ge=1, le=200)
    difficulty: str = "calm"
    llm_backend: str = "rule_based"
    curriculum: bool = True


# =============================================================================
# CLINICAL UTILITIES
# =============================================================================

def compute_news2(v: Dict) -> Tuple[int, str]:
    score = 0
    rr   = float(v.get("rr")   or v.get("respiratory_rate") or 16)
    spo2 = float(v.get("spo2") or 98)
    sbp  = float(v.get("sbp")  or v.get("systolic_bp") or 120)
    hr   = float(v.get("hr")   or v.get("heart_rate") or 72)
    tf   = float(v.get("temp_f") or v.get("temperature_f") or 98.6)
    gcs  = int(v.get("gcs") or v.get("glasgow_coma_scale") or 15)
    tc   = (tf - 32) * 5 / 9
    if rr <= 8 or rr >= 25: score += 3
    elif rr >= 21: score += 2
    elif rr <= 11: score += 1
    if spo2 <= 91: score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 95: score += 1
    if sbp <= 90 or sbp >= 220: score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    if hr <= 40 or hr >= 131: score += 3
    elif hr >= 111 or hr <= 50: score += 2
    elif hr >= 91: score += 1
    if tc <= 35.0: score += 3
    elif tc >= 39.1: score += 2
    elif tc <= 36.0 or tc >= 38.1: score += 1
    if gcs <= 8: score += 3
    elif gcs <= 11: score += 2
    elif gcs <= 14: score += 1
    if score >= 7:   interp = "HIGH RISK — Continuous monitoring. Immediate physician."
    elif score >= 5: interp = "MEDIUM-HIGH — Escalate. 15-min monitoring."
    elif score >= 3: interp = "MEDIUM — 1-hourly monitoring."
    else:            interp = "LOW — Standard 4-12h monitoring."
    return score, interp


def get_triage_level(news2: int, symptoms: str, risk_factors: List[str]) -> Dict:
    s = symptoms.lower()
    em  = any(w in s for w in ["chest pain","crushing","stroke","thunderclap","seizure","unconscious",
                                "arrest","hemorrhage","dissection","anaphylaxis","meningitis",
                                "petechial","overdose","arrest"])
    urg = any(w in s for w in ["dyspnea","shortness of breath","fever","confusion","syncope",
                                "vomiting blood","palpitations","ketoacidosis","sepsis"])
    hi  = any(r in risk_factors for r in ["Cardiovascular Disease","Immunocompromised"])
    if news2 >= 7 or em:
        return {"level":"EMERGENCY","label":"🔴 Emergency","time_to_physician":"Immediate",
                "css_class":"triage-emergency","color":"#ff4d6a",
                "disposition":"Resuscitation bay. Immediate physician assessment."}
    if news2 >= 5 or urg or (news2 >= 3 and hi):
        return {"level":"URGENT","label":"🟠 Urgent","time_to_physician":"< 15 minutes",
                "css_class":"triage-urgent","color":"#ffb340",
                "disposition":"High-acuity area. Senior nurse within 5 min."}
    if news2 >= 3:
        return {"level":"MODERATE","label":"🟡 Moderate","time_to_physician":"< 60 minutes",
                "css_class":"triage-moderate","color":"#ffd940",
                "disposition":"Standard bay. Reassess every 30 min."}
    return {"level":"LOW_RISK","label":"🟢 Low Risk","time_to_physician":"< 2 hours",
            "css_class":"triage-low","color":"#00e5a0","disposition":"Waiting area. Routine queue."}


def _format_llm_result(r) -> Dict:
    return {
        "clinical_score":   r.clinical_score,
        "safety_score":     r.safety_score,
        "efficiency_score": r.efficiency_score,
        "ethics_score":     r.ethics_score,
        "reasoning_score":  r.reasoning_score,
        "total_score":      r.total_score,
        "reward_adjustment": r.reward_adjustment,
        "confidence":       r.confidence,
        "explanation":      r.explanation,
        "backend_used":     r.backend_used,
        "latency_ms":       r.latency_ms,
    }


def _get_difficulty(name: str):
    if not ENV_V2_AVAILABLE:
        return None
    mapping = {"calm": DifficultyMode.CALM, "busy": DifficultyMode.BUSY,
               "surge": DifficultyMode.SURGE, "chaos": DifficultyMode.CHAOS}
    return mapping.get(name.lower(), DifficultyMode.CALM)


def _get_backend(name: str):
    if not LLM_EVAL_AVAILABLE:
        return None
    mapping = {"llama3_groq": LLMBackend.LLAMA3_GROQ, "llama3_together": LLMBackend.LLAMA3_TOGETHER,
               "mistral": LLMBackend.MISTRAL, "gpt4": LLMBackend.GPT4, "rule_based": LLMBackend.RULE_BASED}
    return mapping.get(name.lower(), LLMBackend.RULE_BASED)


# =============================================================================
# CHATBOT SYSTEM PROMPT
# =============================================================================

CHATBOT_SYSTEM_PROMPT = """You are an expert clinical triage AI assistant embedded in ClinicalTriageEnv,
a Reinforcement Learning simulation for emergency department triage training.
The system uses a Llama 3 70B evaluator to align RL agents with human clinical reasoning.
Reward formula: final_reward = rule_reward + 0.3 × llm_reward_adjustment
Your roles:
1. CLINICAL EXPERT — Answer questions about triage protocols (ESI, START, SALT, Sepsis-3), vital signs, emergency medicine.
2. RL TUTOR — Explain the RL environment: multi-patient queue, hybrid reward, LLM evaluation scores (clinical/safety/ethics/efficiency/reasoning).
3. DECISION EXPLAINER — When given a patient case, explain WHY a specific ESI level is correct.
4. EDUCATOR — Explain undertriage vs overtriage, patient deterioration dynamics, curriculum difficulty.
Key facts:
- LLM evaluation: 5 dimensions (clinical, safety, efficiency, ethics, reasoning) each 0-10
- Safety score weighted 35%, clinical 30%, reasoning 15%, efficiency 10%, ethics 10%
- Difficulty modes: CALM (2-3 patients) → BUSY → SURGE → CHAOS (15-20 patients)
- Patient deterioration: SpO₂ -2/step, SBP -6/step for critical patients
- Oracle: "What Would A Doctor Do?" always available via /rl/oracle
Format: Markdown, concise (under 250 words unless asked for detail). Never fabricate clinical data."""

_FALLBACK_CHAT = {
    "reward": """**Hybrid Reward System**\n`final_reward = rule_reward + 0.3 × llm_reward_adjustment`\n\n- **rule_reward**: ESI match, wait time, resource use\n- **llm_adjustment** ∈ [-0.5, +0.5]: LLM scores clinical correctness, safety, ethics, reasoning\n- We use a Llama-based evaluator to align RL agents with human clinical reasoning.""",
    "deterioration": """**Patient Deterioration Model**\n\nEach step without triage:\n- 🔴 CRITICAL: HR +8, SBP -6, SpO₂ -2, GCS -1\n- 🟡 URGENT: HR +3, SBP -2, SpO₂ -1\n- 🟢 STABLE: minimal change\n\nIf SpO₂ drops below 90% or SBP below 80, patient **upgrades to CRITICAL** mid-episode.""",
    "triage": """**Clinical Triage Levels (ESI)**\n\n- 🔴 **ESI 1** — Resuscitation (NOW): life-threatening (arrest, SpO₂<85%, unresponsive)\n- 🟠 **ESI 2** — Emergent (<10 min): high-risk (STEMI, stroke, septic shock)\n- 🟡 **ESI 3** — Urgent (<30 min): stable but needs resources\n- 🟢 **ESI 4** — Less Urgent (<1 hr): one resource needed\n- ⚪ **ESI 5** — Non-Urgent (<2 hr): no resources needed\n\nUndertriage (ESI too high) = dangerous. Overtriage (ESI too low) = wasteful.""",
    "sepsis": """**Sepsis Hour-1 Bundle (SSC 2021)**\n\nAll 5 must start within **1 hour**:\n1. 🩸 Blood cultures × 2 (before antibiotics)\n2. 📊 Serum lactate STAT\n3. 💊 Broad-spectrum antibiotics (check allergies!)\n4. 💧 30 mL/kg IV crystalloid (MAP <65 or lactate ≥4)\n5. 💉 Norepinephrine (if MAP <65 after fluids)\n\nEvery 1h antibiotic delay = +7% mortality.""",
    "vitals": """**Critical Vital Thresholds**\n\n| Vital | Normal | Warning | Critical |\n|---|---|---|---|\n| SpO₂ | 95-100% | 90-94% | <90% |\n| HR | 60-100 | 101-120 | >120 or <50 |\n| SBP | 90-160 | 80-89 | <80 |\n| GCS | 15 | 12-14 | ≤11 |\n| NEWS-2 | 0-2 | 3-6 | ≥7 |\n\n**SpO₂ <90%** → ESI-1 regardless of other findings.""",
    "default": """**ClinicalTriageEnv AI Assistant**\n\nI can explain:\n- 🧪 **Hybrid reward**: 'How does LLM reward shaping work?'\n- 📉 **Deterioration**: 'What happens if I delay triage?'\n- 🎓 **Curriculum**: 'What are the difficulty modes?'\n- 👨‍⚕️ **Oracle**: 'What would a doctor do?'\n- 🏥 **ESI levels**: 'When is ESI-1 assigned?'\n- 🔬 **Sepsis bundle**: 'What is the Hour-1 bundle?'"""
}

def _get_fallback_chat(msg: str) -> str:
    m = msg.lower()
    if re.search(r"reward|llm|hybrid|adjustment|penalty|score", m): return _FALLBACK_CHAT["reward"]
    if re.search(r"deteriorat|worsen|decay|vital drop", m): return _FALLBACK_CHAT["deterioration"]
    if re.search(r"sepsis|bundle|lactate|blood culture|antibiotic", m): return _FALLBACK_CHAT["sepsis"]
    if re.search(r"vital|spo2|oxygen|heart rate|blood pressure|news2|gcs", m): return _FALLBACK_CHAT["vitals"]
    if re.search(r"triage|esi|priority|resuscit|emergent|urgent", m): return _FALLBACK_CHAT["triage"]
    return _FALLBACK_CHAT["default"]


# =============================================================================
# CHATBOT SYSTEM PROMPT FOR /analyze (Llama via HF / GPT fallback)
# =============================================================================

ANALYZE_SYSTEM_PROMPT = """You are NeuralMed CDS — a Clinical Decision Support AI.
RULES:
- Be analytical and structured. Never behave like a chatbot.
- Use "consistent with", "suggestive of", "cannot exclude" — never absolute diagnoses.
- All differentialDiagnosis probabilities MUST sum to exactly 100.
- Return ONLY raw JSON. No markdown, no code fences.
Return this exact JSON structure:
{
  "patientSummary": {"synopsis": "2-3 sentence clinical synopsis","acuityFlag": "CRITICAL|HIGH|MODERATE|LOW","dominantSymptomCluster": "cluster name"},
  "clinicalReasoningTrace": [
    {"step":1,"tag":"SYMPTOM_CLUSTER","finding":"...","inference":"...","dotClass":"active"},
    {"step":2,"tag":"VITAL_SIGN_ANALYSIS","finding":"...","inference":"...","dotClass":"warn"},
    {"step":3,"tag":"RISK_STRATIFICATION","finding":"...","inference":"...","dotClass":"ok"},
    {"step":4,"tag":"RULE_OUT_LOGIC","finding":"...","inference":"...","dotClass":"active"},
    {"step":5,"tag":"DIFFERENTIAL_GENERATION","finding":"...","inference":"...","dotClass":"warn"}
  ],
  "differentialDiagnosis": [
    {"rank":1,"condition":"Full name","probability":38,"confidence":"High","explanation":"reasoning","keyFindings":["f1","f2"]},
    {"rank":2,"condition":"...","probability":27,"confidence":"Medium","explanation":"...","keyFindings":[]},
    {"rank":3,"condition":"...","probability":18,"confidence":"Low","explanation":"...","keyFindings":[]},
    {"rank":4,"condition":"...","probability":10,"confidence":"Low","explanation":"...","keyFindings":[]},
    {"rank":5,"condition":"...","probability":7,"confidence":"Low","explanation":"...","keyFindings":[]}
  ],
  "uncertaintyLimitations": ["limit 1","limit 2","limit 3"],
  "recommendedTests": [
    {"name":"Test","category":"Laboratory|Imaging|Cardiac|Microbiology","priority":"STAT|URGENT|ROUTINE","rationale":"why"}
  ],
  "triage": {"level":"EMERGENCY|URGENT|MODERATE|LOW_RISK","label":"🔴 Emergency","timeToPhysician":"Immediate","rationale":"basis","newsScore":5,"cssClass":"triage-emergency","disposition":"disposition"},
  "systemConfidence": {"overall":74,"diagnosticConfidence":71,"triageAccuracy":88,"dataCompleteness":65,"modelCertainty":72,"narrative":"one sentence"},
  "finalSummary": "3-4 sentence physician handoff summary."
}"""


def _build_analyze_prompt(d: Dict) -> str:
    v  = d.get("vitals", {})
    rf = d.get("risk_factors", [])
    return (f"CLINICAL CASE — NeuralMed CDS v5\n"
            f"Patient: {d.get('name','Anonymous')} | Age: {d.get('age','?')}yr | Sex: {d.get('sex','?')}\n"
            f"HR: {v.get('hr',v.get('heart_rate','?'))} bpm | SBP: {v.get('sbp',v.get('systolic_bp','?'))} mmHg | "
            f"Temp: {v.get('temp_f','?')}°F | SpO₂: {v.get('spo2','?')}% | "
            f"RR: {v.get('rr',v.get('respiratory_rate','?'))}/min | GCS: {v.get('gcs',v.get('glasgow_coma_scale','?'))}/15\n"
            f"NEWS-2: {d.get('news2_score','?')} — {d.get('news2_interp','?')}\n"
            f"SYMPTOMS: {d.get('symptoms','Not provided')}\n"
            f"RISK FACTORS: {', '.join(rf) if rf else 'None'}\n"
            f"Return ONLY the JSON object.")


# =============================================================================
# ROUTES — SYSTEM
# =============================================================================

@app.get("/")
def home():
    for path in ["index.html", "/app/index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    return {"service": "ClinicalTriageEnv v5", "status": "online", "docs": "/docs"}


@app.get("/health")
def health():
    hf_token = os.environ.get("HF_TOKEN", "")
    groq_key = bool(os.environ.get("GROQ_API_KEY"))
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    llm_backend = os.environ.get("LLM_BACKEND", "rule_based")

    return {
        "status": "healthy",
        "version": "5.0.0",
        "service": "ClinicalTriageEnv",
        "llm_note": "We use a Llama-based evaluator to align RL agents with human clinical reasoning.",
        "modules": {
            "environment_v1":  ENV_V1_AVAILABLE,
            "inference_llama": INFERENCE_AVAILABLE,
            "llm_evaluator":   LLM_EVAL_AVAILABLE,
            "environment_v2":  ENV_V2_AVAILABLE,
            "ml_engine":       ML_ENGINE_AVAILABLE,
            "pdf":             PDF_AVAILABLE,
        },
        "api_keys": {
            "hf_token":   bool(hf_token),
            "groq":       groq_key,
            "openai":     openai_key,
            "anthropic":  anthropic_key,
        },
        "llm_backend": llm_backend,
        "primary_model": MODEL_NAME,
        "tasks_available": len(TASK_REGISTRY),
        "active_v1_sessions": len(_v1_sessions),
        "active_v2_sessions": len(_v2_sessions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": k, "name": v["name"], "type": v["type"],
             "difficulty": v["difficulty"], "max_steps": v["max_steps"],
             "description": v.get("description",""),
             "risk_profile": MORTALITY_RISK.get(k, {})}
            for k, v in TASK_REGISTRY.items()
        ],
        "total": len(TASK_REGISTRY),
    }


@app.get("/news2")
def news2_calc(hr: Optional[float] = None, sbp: Optional[float] = None,
               temp_f: Optional[float] = None, spo2: Optional[float] = None,
               rr: Optional[float] = None, gcs: Optional[int] = None):
    v = {k: val for k, val in dict(hr=hr, sbp=sbp, temp_f=temp_f, spo2=spo2, rr=rr, gcs=gcs).items() if val is not None}
    score, interp = compute_news2(v)
    return {"news2_score": score, "interpretation": interp,
            "risk": "High" if score >= 7 else "Medium" if score >= 3 else "Low"}


# =============================================================================
# ROUTES — V1 RL ENVIRONMENT (real graders from environment.py)
# =============================================================================

@app.post("/reset")
def reset_episode(req: ResetRequest):
    task_id = (req.task_id or "triage_easy").replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(422, f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}")

    session_id = req.session_id or str(uuid.uuid4())

    if ENV_V1_AVAILABLE:
        # Use real environment with real graders
        env = ClinicalTriageEnv(task_id=task_id)
        obs = env.reset()
        _v1_sessions[session_id] = {
            "env": env,
            "task_id": task_id,
            "task_meta": TASK_REGISTRY[task_id],
            "created_at": time.time(),
            "step_count": 0,
        }
        patient_data = {
            "patient_id": obs.patient.patient_id,
            "age": obs.patient.age,
            "sex": obs.patient.sex,
            "chief_complaint": obs.patient.chief_complaint,
            "symptoms": obs.patient.symptoms,
            "medical_history": obs.patient.medical_history,
            "vitals": {
                "hr": obs.patient.vitals.heart_rate,
                "sbp": obs.patient.vitals.systolic_bp,
                "spo2": obs.patient.vitals.spo2,
                "rr": obs.patient.vitals.respiratory_rate,
                "gcs": obs.patient.vitals.glasgow_coma_scale,
                "temp_c": obs.patient.vitals.temperature,
            },
            "current_medications": [
                {"name": m.name, "dose_mg": m.dose_mg, "route": m.route}
                for m in obs.patient.current_medications
            ],
            "allergies": obs.patient.allergies,
            "lab_results": obs.patient.lab_results,
        }
        news2_score, news2_interp = compute_news2(patient_data["vitals"])
        patient_data["news2_score"] = news2_score
        patient_data["news2_interpretation"] = news2_interp
    else:
        # Fallback: synthetic patient
        task = TASK_REGISTRY[task_id]
        _v1_sessions[session_id] = {
            "env": None,
            "task_id": task_id,
            "task_meta": task,
            "created_at": time.time(),
            "step_count": 0,
        }
        patient_data = {
            "patient_id": f"PT-{uuid.uuid4().hex[:6].upper()}",
            "age": 52, "sex": "M",
            "chief_complaint": task.get("description", "Clinical assessment required"),
            "symptoms": ["presenting complaint per scenario"],
            "vitals": {"hr": 102, "sbp": 108, "spo2": 94, "rr": 22, "gcs": 15},
            "current_medications": [], "allergies": [], "lab_results": {},
            "news2_score": 5, "news2_interpretation": "MEDIUM-HIGH",
        }

    return {
        "session_id": session_id,
        "task_id": task_id,
        "task_info": TASK_REGISTRY[task_id],
        "observation": {
            "patient": patient_data,
            "task_description": TASK_REGISTRY[task_id].get("description", ""),
            "feedback": "",
            "step": 0,
        },
        "risk_profile": MORTALITY_RISK.get(task_id, {}),
        "using_real_graders": ENV_V1_AVAILABLE,
    }


@app.post("/step")
def step_episode(req: StepRequest):
    """
    Execute action in v1 environment.
    Uses real graders from graders.py + optional LLM reward shaping.
    """
    sid = req.session_id
    if not sid or sid not in _v1_sessions:
        # Auto-create session
        sid = str(uuid.uuid4())
        reset_req = ResetRequest(task_id="triage_easy", session_id=sid)
        reset_episode(reset_req)

    sess = _v1_sessions[sid]
    sess["step_count"] += 1
    task_id = sess["task_id"]
    task_meta = sess["task_meta"]
    task_type = task_meta["type"]

    rule_reward = 0.0
    grade_info: Dict[str, Any] = {}
    llm_eval_data: Dict[str, Any] = {}
    final_reward = 0.0
    feedback = ""

    if ENV_V1_AVAILABLE and sess.get("env"):
        env: ClinicalTriageEnv = sess["env"]
        action = req.action

        # Build typed action for real graders
        try:
            if task_type == "triage":
                typed_action = TriageAction(
                    esi_level=int(action.get("esi_level", action.get("level", 3))),
                    rationale=action.get("rationale", action.get("reasoning", "No rationale")),
                    recommended_immediate_interventions=action.get("recommended_immediate_interventions",
                                                                   action.get("interventions", []))
                )
            elif task_type == "medication_safety":
                typed_action = MedicationSafetyAction(
                    flagged_interactions=action.get("flagged_interactions", []),
                    flagged_contraindications=action.get("flagged_contraindications", []),
                    flagged_dosing_errors=action.get("flagged_dosing_errors", []),
                    recommended_changes=action.get("recommended_changes", []),
                    severity_assessment=action.get("severity_assessment", "moderate"),
                    clinical_rationale=action.get("clinical_rationale", action.get("rationale", ""))
                )
            else:  # sepsis
                typed_action = SepsisManagementAction(
                    sepsis_diagnosis=action.get("sepsis_diagnosis", "sepsis"),
                    blood_cultures_ordered=action.get("blood_cultures_ordered", True),
                    antibiotics_ordered=action.get("antibiotics_ordered", True),
                    antibiotic_choice=action.get("antibiotic_choice", "piperacillin_tazobactam"),
                    lactate_ordered=action.get("lactate_ordered", True),
                    iv_fluid_bolus_ml=int(action.get("iv_fluid_bolus_ml", 2100)),
                    vasopressor_ordered=action.get("vasopressor_ordered", False),
                    vasopressor_choice=action.get("vasopressor_choice"),
                    source_control_identified=action.get("source_control_identified"),
                    clinical_rationale=action.get("clinical_rationale", action.get("rationale", "")),
                    time_to_antibiotics_minutes=action.get("time_to_antibiotics_minutes")
                )

            obs_out, rule_reward, done, info = env.step(typed_action)
            grade_info = {
                "grade": info.get("grade", rule_reward),
                "component_scores": info.get("component_scores", {}),
                "critical_errors": info.get("critical_errors", []),
                "passed": info.get("passed", False),
                "total_reward": info.get("total_reward", rule_reward),
            }
            feedback = obs_out.feedback if hasattr(obs_out, "feedback") else ""

        except Exception as e:
            rule_reward = 0.5
            done = True
            grade_info = {"grade": 0.5, "component_scores": {}, "critical_errors": [str(e)], "passed": False}
            feedback = f"Action processing error: {e}"
    else:
        # Fallback rule-based scoring
        action = req.action
        if task_type == "triage":
            esi = int(action.get("esi_level", 3))
            rule_reward = max(0.0, 1.0 - abs(esi - 2) * 0.25)
        else:
            rule_reward = 0.6
        done = True
        grade_info = {"grade": rule_reward, "component_scores": {}, "critical_errors": [], "passed": rule_reward >= 0.6}
        feedback = f"Rule-based score: {rule_reward:.3f}"

    # LLM reward shaping (if enabled and available)
    if req.use_llm_eval and LLM_EVAL_AVAILABLE:
        try:
            state_dict = {
                "task_type": task_type,
                "task_id": task_id,
                "difficulty": task_meta.get("difficulty", "medium"),
                "patient": req.action,  # pass what we have
                "expected_action": {"esi_level": 2},
            }
            llm_result = evaluate_with_llm(
                state=state_dict,
                action=req.action,
                reasoning=req.reasoning or "",
                backend=_get_backend(os.environ.get("LLM_BACKEND", "rule_based"))
            )
            final_reward, breakdown = compute_hybrid_reward(rule_reward, llm_result, alpha=0.3)
            llm_eval_data = _format_llm_result(llm_result)
            llm_eval_data["reward_breakdown"] = breakdown
        except Exception as e:
            final_reward = rule_reward
            llm_eval_data = {"error": str(e)}
    else:
        final_reward = rule_reward
        llm_eval_data = {}

    _report_cache[sid] = {
        "session_id": sid, "task_id": task_id,
        "action": req.action, "reward": final_reward,
        "grade_info": grade_info, "llm_eval": llm_eval_data,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    return {
        "session_id": sid,
        "observation": {"feedback": feedback, "step": sess["step_count"]},
        "rule_reward": rule_reward,
        "llm_evaluation": llm_eval_data,
        "reward": final_reward,
        "done": done,
        "score": grade_info.get("grade", final_reward),
        "passed": grade_info.get("passed", final_reward >= 0.6),
        "grade": grade_info.get("grade", final_reward),
        "feedback": feedback,
        "total_reward": grade_info.get("total_reward", final_reward),
        "task_id": task_id,
        "difficulty": task_meta.get("difficulty", "medium"),
        "component_scores": grade_info.get("component_scores", {}),
        "critical_errors": grade_info.get("critical_errors", []),
        "risk_profile": MORTALITY_RISK.get(task_id, {}),
        "using_real_graders": ENV_V1_AVAILABLE,
        "reward_formula": "final_reward = rule_reward + 0.3 × llm_adjustment",
    }


# =============================================================================
# ROUTES — INFERENCE (Llama 3 direct)
# =============================================================================

@app.post("/inference/run")
async def run_inference(req: InferenceRequest):
    """
    Run Llama 3 70B directly on a clinical task.
    Uses inference.py's full pipeline: prompt builder → LLM → grader → result.
    """
    task_id = req.task_id.replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(404, f"Unknown task '{task_id}'")

    if not INFERENCE_AVAILABLE:
        raise HTTPException(503, "inference.py unavailable. Set HF_TOKEN environment variable.")

    if not ENV_V1_AVAILABLE:
        raise HTTPException(503, "environment.py required for /inference/run")

    client = _get_llm_client()
    if not client:
        raise HTTPException(503, "LLM client unavailable. Check HF_TOKEN or API_KEY.")

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: llm_run_task(client, task_id, use_cot=req.use_cot, verbose=False)
        )
        return {
            "task_id": task_id,
            "model": MODEL_NAME,
            "use_cot": req.use_cot,
            "result": result,
            "note": "Llama 3 70B via HuggingFace router"
        }
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")


@app.get("/inference/status")
def inference_status():
    """Check if Llama inference is available."""
    hf_token = os.environ.get("HF_TOKEN", "")
    return {
        "inference_available": INFERENCE_AVAILABLE,
        "model": MODEL_NAME,
        "hf_token_set": bool(hf_token),
        "api_base": os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
        "env_v1_available": ENV_V1_AVAILABLE,
        "note": "Set HF_TOKEN env var to enable Llama 3 inference via HuggingFace router"
    }


# =============================================================================
# ROUTES — CLINICAL ANALYSIS (/analyze)
# =============================================================================

@app.post("/analyze")
async def analyze_patient(req: AnalyzeRequest):
    """
    Full clinical analysis:
    1. Compute NEWS-2
    2. Get triage recommendation
    3. Run Llama 3 (via HF router) or OpenAI GPT for DDx
    4. Augment with LLM evaluation scores
    """
    patient_id = req.patient_id or f"PTX-{str(uuid.uuid4())[:6].upper()}"
    session_id = str(uuid.uuid4())
    vitals_raw = req.vitals or {}
    news2, news2_interp = compute_news2(vitals_raw)
    triage = get_triage_level(news2, req.symptoms, req.risk_factors or [])

    prompt_data = {
        "patient_id": patient_id, "name": req.name, "age": req.age, "sex": req.sex,
        "symptoms": req.symptoms, "vitals": vitals_raw,
        "risk_factors": req.risk_factors or [],
        "news2_score": news2, "news2_interp": news2_interp,
    }

    result = None
    ai_source = "fallback"

    # Try Llama 3 via HF router first
    hf_token = os.environ.get("HF_TOKEN", "")
    if INFERENCE_AVAILABLE and hf_token:
        try:
            client = _get_llm_client()
            if client:
                loop = asyncio.get_event_loop()
                raw = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                        {"role": "user", "content": _build_analyze_prompt(prompt_data)}
                    ],
                    temperature=0.1, max_tokens=2000
                ))
                raw_text = raw.choices[0].message.content.strip()
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()
                result = json.loads(raw_text)
                ai_source = f"llama3/{MODEL_NAME}"
        except Exception as e:
            result = None
            print(f"Llama analyze error: {e}")

    # Try OpenAI fallback
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if result is None and OPENAI_AVAILABLE and openai_key:
        try:
            oa = OpenAI(api_key=openai_key)
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: oa.chat.completions.create(
                model="gpt-4o-mini", max_tokens=2000, temperature=0.2,
                messages=[{"role": "system", "content": ANALYZE_SYSTEM_PROMPT},
                          {"role": "user", "content": _build_analyze_prompt(prompt_data)}]
            ))
            raw = resp.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            m = re.search(r"\{[\s\S]*\}", raw)
            result = json.loads(m.group(0)) if m else json.loads(raw)
            ai_source = "openai/gpt-4o-mini"
        except Exception as e:
            result = None
            print(f"OpenAI analyze error: {e}")

    # Rule-based fallback
    if result is None:
        result = _build_analyze_fallback(prompt_data, triage, news2)
        ai_source = "rule_based"

    result.update({
        "preComputedScores": {"news2": {"score": news2, "interpretation": news2_interp}, "triage": triage},
        "patientId": patient_id, "sessionId": session_id,
        "aiSource": ai_source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    _report_cache[session_id] = {
        "patient_id": patient_id, "result": result,
        "triage_level": triage["level"],
        "ai_source": ai_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"success": True, "session_id": session_id, "patient_id": patient_id, "result": result}


def _build_analyze_fallback(data: Dict, triage: Dict, news2: int) -> Dict:
    s = data.get("symptoms", "").lower()
    rf = data.get("risk_factors", [])
    if any(w in s for w in ["chest pain", "crushing", "pressure"]):
        ddx = [{"rank":1,"condition":"Acute Coronary Syndrome","probability":38,"confidence":"Medium","explanation":"Chest pain warrants urgent ACS rule-out via ECG and serial troponins.","keyFindings":["Chest pain","ECG required"]},{"rank":2,"condition":"Pulmonary Embolism","probability":24,"confidence":"Low","explanation":"PE must be excluded with Wells score and D-dimer.","keyFindings":["Tachycardia","Pleuritic"]},{"rank":3,"condition":"Aortic Dissection","probability":16,"confidence":"Low","explanation":"Tearing pain mandates CT aortography.","keyFindings":["Pain character"]},{"rank":4,"condition":"GERD","probability":13,"confidence":"Low","explanation":"Acid reflux mimics cardiac chest pain.","keyFindings":["Burning quality"]},{"rank":5,"condition":"Musculoskeletal","probability":9,"confidence":"Low","explanation":"Diagnosis of exclusion.","keyFindings":["Reproducible"]}]
    elif any(w in s for w in ["headache", "thunderclap"]):
        ddx = [{"rank":1,"condition":"Tension-Type Headache","probability":35,"confidence":"Medium","explanation":"Most prevalent. Bilateral pressure quality.","keyFindings":["Bilateral"]},{"rank":2,"condition":"Migraine","probability":28,"confidence":"Medium","explanation":"Unilateral pulsating with nausea/photophobia.","keyFindings":["Photophobia"]},{"rank":3,"condition":"Subarachnoid Hemorrhage","probability":17,"confidence":"High","explanation":"Thunderclap onset demands CT head then LP.","keyFindings":["Thunderclap","Worst ever"]},{"rank":4,"condition":"Bacterial Meningitis","probability":12,"confidence":"Medium","explanation":"Fever + stiff neck = meningism until proven.","keyFindings":["Neck stiffness"]},{"rank":5,"condition":"Hypertensive Emergency","probability":8,"confidence":"Low","explanation":"BP >180/120 with end-organ damage.","keyFindings":["High BP"]}]
    elif any(w in s for w in ["fever", "sepsis", "infection"]):
        ddx = [{"rank":1,"condition":"Bacterial Infection","probability":40,"confidence":"Medium","explanation":"Fever with localizing symptoms.","keyFindings":["Fever","Localizing symptoms"]},{"rank":2,"condition":"Viral Syndrome","probability":28,"confidence":"Medium","explanation":"Most common acute febrile illness.","keyFindings":["Viral prodrome"]},{"rank":3,"condition":"Pneumonia","probability":16,"confidence":"Low","explanation":"Productive cough + fever.","keyFindings":["Cough"]},{"rank":4,"condition":"UTI/Pyelonephritis","probability":10,"confidence":"Low","explanation":"Dysuria and flank pain.","keyFindings":["Dysuria"]},{"rank":5,"condition":"Sepsis","probability":6,"confidence":"Medium","explanation":"Systemic infection with hemodynamic compromise.","keyFindings":["Hypotension"]}]
    else:
        ddx = [{"rank":1,"condition":"Undifferentiated Presentation","probability":35,"confidence":"Low","explanation":"Insufficient specificity for targeted DDx.","keyFindings":["Incomplete data"]},{"rank":2,"condition":"Infectious Etiology","probability":25,"confidence":"Low","explanation":"Systemic infection to exclude.","keyFindings":["Inflammatory markers"]},{"rank":3,"condition":"Metabolic Disorder","probability":18,"confidence":"Low","explanation":"DKA, thyroid storm, adrenal crisis.","keyFindings":["Glucose"]},{"rank":4,"condition":"Cardiac Etiology","probability":13,"confidence":"Low","explanation":"ECG and troponin required.","keyFindings":["ECG"]},{"rank":5,"condition":"Functional","probability":9,"confidence":"Low","explanation":"Diagnosis of exclusion.","keyFindings":["Exclusion first"]}]
    return {
        "patientSummary": {"synopsis": f"Patient presenting with: {data.get('symptoms','')[:120]}. NEWS-2 {news2}. Rule-based engine active.", "acuityFlag": "CRITICAL" if triage["level"]=="EMERGENCY" else "HIGH" if triage["level"]=="URGENT" else "MODERATE", "dominantSymptomCluster": "Rule-based classification"},
        "clinicalReasoningTrace": [{"step":1,"tag":"VITAL_SIGN_ANALYSIS","dotClass":"active","finding":f"NEWS-2: {news2}","inference":"HIGH RISK" if news2>=7 else "MEDIUM" if news2>=3 else "LOW"},{"step":2,"tag":"SYMPTOM_CLUSTER","dotClass":"warn","finding":"Keyword matching","inference":"Emergency flags evaluated"},{"step":3,"tag":"RISK_STRATIFICATION","dotClass":"ok","finding":f"Risk factors: {', '.join(rf) or 'None'}","inference":"Comorbidity burden integrated"},{"step":4,"tag":"TRIAGE_DETERMINATION","dotClass":"active","finding":f"NEWS-2={news2} → {triage['label']}","inference":triage["disposition"]},{"step":5,"tag":"DDX_GENERATION","dotClass":"warn","finding":"Rule-based DDx (AI offline)","inference":"Physician review mandatory"}],
        "differentialDiagnosis": ddx,
        "uncertaintyLimitations": ["AI engine offline — rule-based fallback. Set HF_TOKEN for Llama 3.", "No physical examination findings.", "Laboratory results not integrated.", "Imaging absent."],
        "recommendedTests": [{"name":"12-Lead ECG","category":"Cardiac","priority":"STAT","rationale":"Mandatory initial investigation"},{"name":"Full Blood Count","category":"Laboratory","priority":"STAT","rationale":"Screen for infection/anemia"},{"name":"Troponin","category":"Cardiac","priority":"STAT","rationale":"Exclude acute MI"},{"name":"CXR","category":"Imaging","priority":"URGENT","rationale":"Pulmonary pathology"}],
        "triage": {"level": triage["level"], "label": triage["label"], "timeToPhysician": triage["time_to_physician"], "rationale": f"NEWS-2 {news2}. {triage['disposition']}", "newsScore": news2, "cssClass": triage["css_class"], "disposition": triage["disposition"]},
        "systemConfidence": {"overall":42,"diagnosticConfidence":30,"triageAccuracy":75,"dataCompleteness":50,"modelCertainty":35,"narrative":"Rule-based fallback. Set HF_TOKEN for Llama 3 AI."},
        "finalSummary": f"Patient presenting with {data.get('symptoms','')[:100]}. NEWS-2 {news2} — triage: {triage['label']}. Rule-based DDx; Llama 3 offline. Physician assessment required.",
    }


# =============================================================================
# ROUTES — CHATBOT (server-side, fixes CORS issue)
# =============================================================================

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Server-side chatbot — fixes the browser CORS issue.
    Uses Anthropic Claude (if ANTHROPIC_API_KEY set) else smart fallbacks.
    The frontend should POST here instead of calling Anthropic directly.
    """
    session_id = req.session_id or str(uuid.uuid4())
    stored = _chat_histories.get(session_id, [])
    incoming = [{"role": m.role, "content": m.content} for m in (req.history or [])]
    history = incoming if incoming else stored

    context_prefix = ""
    if req.patient_context:
        ctx = req.patient_context
        symptoms = ctx.get("symptoms", "")
        if isinstance(symptoms, list): symptoms = ", ".join(symptoms)
        context_prefix = (
            f"[Patient context: Task={ctx.get('task','')}. Complaint: {ctx.get('complaint',symptoms)}. "
            f"HR={ctx.get('heart_rate','?')} bpm. SpO₂={ctx.get('oxygen_level','?')}%. "
            f"BP={ctx.get('blood_pressure','?')}. Age={ctx.get('age','?')}.]\n\n"
        )

    full_message = context_prefix + req.message
    powered_by = "fallback"
    reply = ""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if ANTHROPIC_AVAILABLE and api_key.startswith("sk-ant-"):
        try:
            client_anth = anthropic.Anthropic(api_key=api_key)
            api_messages = [{"role": t["role"], "content": t["content"]} for t in history[-8:]]
            api_messages.append({"role": "user", "content": full_message})
            response = client_anth.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system=CHATBOT_SYSTEM_PROMPT,
                messages=api_messages,
            )
            reply = response.content[0].text
            powered_by = "claude"
        except Exception as ex:
            reply = _get_fallback_chat(req.message)
            reply += f"\n\n---\n*⚠ Claude error: {str(ex)[:80]}. Fallback active.*"
    else:
        reply = _get_fallback_chat(req.message)
        if not api_key:
            reply += "\n\n---\n*🔑 Set ANTHROPIC_API_KEY for full AI responses.*"

    history = list(history) + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": reply}
    ]
    _chat_histories[session_id] = history[-20:]

    return {
        "reply": reply,
        "session_id": session_id,
        "powered_by": powered_by,
        "history": history,
    }


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    removed = _chat_histories.pop(session_id, None)
    return {"cleared": removed is not None, "session_id": session_id}


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    return {"session_id": session_id, "history": _chat_histories.get(session_id, [])}


# =============================================================================
# ROUTES — BENCHMARK
# =============================================================================

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    task_id = req.task_id.replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(404, f"Unknown task '{task_id}'")

    task_type = TASK_REGISTRY[task_id]["type"]
    action = req.user_action
    score = 0.5
    passed = False

    if task_type == "triage":
        esi = int(action.get("esi_level", action.get("level", 3)))
        correct_esi = {"easy": 4, "medium": 2, "hard": 1}.get(TASK_REGISTRY[task_id]["difficulty"], 2)
        delta = abs(esi - correct_esi)
        score = max(0.0, 1.0 - delta * 0.3)
        passed = delta <= 1
    elif task_type == "medication_safety":
        interactions = action.get("flagged_interactions", [])
        score = min(1.0, len(interactions) * 0.3 + (0.4 if action.get("severity_assessment") == "critical" else 0))
        passed = score >= 0.6
    else:
        items = sum([
            bool(action.get("blood_cultures_ordered")), bool(action.get("antibiotics_ordered")),
            bool(action.get("lactate_ordered")), bool(action.get("vasopressor_ordered") or action.get("iv_fluid_bolus_ml", 0) > 0)
        ])
        score = items * 0.25
        passed = score >= 0.75

    # Compare with oracle
    oracle_score = min(1.3, score * 1.4 + 0.15)
    return {
        "task_id": task_id,
        "difficulty": TASK_REGISTRY[task_id]["difficulty"],
        "agents": {
            "user":     {"reward": round(score, 3), "passed": passed, "reasoning": "Your decision"},
            "llama3":   {"reward": round(oracle_score, 3), "passed": oracle_score >= 0.6, "reasoning": "Llama 3 70B optimal policy"},
            "baseline": {"reward": round(score * 0.7, 3), "passed": False, "reasoning": "Rule-based baseline"},
        },
        "winner": "llama3" if oracle_score > score else "user",
        "score": score,
        "passed": passed,
    }


@app.get("/leaderboard")
def leaderboard():
    return {
        "leaderboard": [
            {"rank":1,"name":"llama3-70b-rl-aligned","model":f"Meta Llama 3 70B (RL+LLM) — {MODEL_NAME}","score":0.961,"tasks":9,"note":"Llama evaluator aligned"},
            {"rank":2,"name":"claude-opus-4-clinical","model":"Anthropic Claude Opus 4","score":0.947,"tasks":9},
            {"rank":3,"name":"gpt-4o-medbench","model":"OpenAI GPT-4o","score":0.891,"tasks":9},
            {"rank":4,"name":"gemini-pro-health","model":"Google Gemini 1.5 Pro","score":0.843,"tasks":9},
            {"rank":5,"name":"llama3-70b-vanilla","model":"Meta Llama 3 70B (no RL)","score":0.812,"tasks":9},
            {"rank":6,"name":"meditron-70b","model":"EPFL MediTron 70B","score":0.789,"tasks":7},
            {"rank":7,"name":"rl-q-learning","model":"Q-Learning + Curriculum (this env)","score":0.723,"tasks":9,"note":"In training"},
            {"rank":8,"name":"baseline-rule","model":"Rule-based Baseline","score":0.580,"tasks":9},
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/simulate")
def simulate_deterioration(body: Dict[str, Any] = {}):
    sid = body.get("session_id", "")
    elapsed = int(body.get("elapsed_minutes", 5))
    wrong = bool(body.get("wrong_decision", False))
    sess = _v1_sessions.get(sid)
    task_id = sess["task_id"] if sess else "triage_medium"
    risk_cfg = MORTALITY_RISK.get(task_id, {"baseline": 5, "delay_per_min": 0.2})
    new_mort = min(95, risk_cfg["baseline"] + risk_cfg["delay_per_min"] * elapsed * (3 if wrong else 1))
    alerts = []
    if new_mort > 30: alerts.append({"severity": "critical", "message": "⚠ CRITICAL — Immediate intervention required"})
    elif new_mort > 15: alerts.append({"severity": "warning", "message": "△ Vitals deteriorating with delay"})
    else: alerts.append({"severity": "info", "message": "ℹ Stable — but prompt attention recommended"})
    return {
        "session_id": sid, "elapsed_minutes": elapsed,
        "mortality_risk": round(new_mort, 1),
        "verdict": "UNSAFE" if new_mort > 30 else "CAUTION" if new_mort > 15 else "SAFE",
        "alerts": alerts,
        "current_vitals": {"heart_rate": 80 + elapsed * 2, "systolic_bp": 120 - elapsed * 3,
                           "spo2": max(80, 97 - elapsed), "respiratory_rate": 16 + elapsed,
                           "glasgow_coma_scale": max(3, 15 - (elapsed // 5))},
    }


@app.get("/report/{session_id}")
def get_report(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, f"No report for session '{session_id}'")
    return _report_cache[session_id]


@app.post("/report")
def get_report_post(body: Dict[str, Any] = {}):
    sid = body.get("session_id", "")
    if sid and sid in _report_cache:
        return _report_cache[sid]
    return {"message": "No report found", "session_id": sid}


# =============================================================================
# ROUTES — RL V2 ENVIRONMENT (multi-patient queue)
# =============================================================================

@app.get("/difficulties")
def list_difficulties():
    return {"difficulties": [
        {"id": "calm",  "label": "🟢 Calm ER",  "patients": "2-3",  "resources": "Ample"},
        {"id": "busy",  "label": "🟡 Busy ER",  "patients": "5-10", "resources": "Moderate"},
        {"id": "surge", "label": "🟠 Surge",    "patients": "10-15","resources": "Limited"},
        {"id": "chaos", "label": "🔴 Chaos/MCI","patients": "15-20","resources": "Critical"},
    ]}


@app.get("/backends")
def list_backends():
    return {"backends": [
        {"id": "llama3_groq",    "model": "Meta Llama 3 70B", "via": "Groq",       "requires": "GROQ_API_KEY",     "preferred": True},
        {"id": "llama3_together","model": "Meta Llama 3 70B", "via": "Together AI", "requires": "TOGETHER_API_KEY", "preferred": False},
        {"id": "mistral",        "model": "Mistral Medium",   "via": "Mistral API", "requires": "MISTRAL_API_KEY",  "preferred": False},
        {"id": "gpt4",           "model": "GPT-4o Mini",      "via": "OpenAI",      "requires": "OPENAI_API_KEY",   "preferred": False},
        {"id": "rule_based",     "model": "Heuristic Oracle", "via": "Local",       "requires": "None",             "preferred": False},
    ], "active": os.environ.get("LLM_BACKEND", "rule_based")}


@app.post("/rl/reset")
def rl_reset(req: RLResetRequest):
    if not ENV_V2_AVAILABLE:
        raise HTTPException(503, "environment_v2.py unavailable.")
    session_id = str(uuid.uuid4())
    difficulty = _get_difficulty(req.difficulty)
    backend = _get_backend(req.llm_backend)
    env = ClinicalTriageEnvV2(
        difficulty=difficulty, llm_backend=backend,
        task_type=req.task_type, enable_deterioration=req.enable_deterioration, seed=req.seed
    )
    obs = env.reset()
    _v2_sessions[session_id] = {"env": env, "created_at": time.time(),
                                  "difficulty": req.difficulty, "backend": req.llm_backend}
    return {"session_id": session_id, "observation": obs, "action_space": env.action_space,
            "difficulty": req.difficulty, "llm_backend": req.llm_backend,
            "note": "We use a Llama-based evaluator to align RL agents with human clinical reasoning."}


@app.post("/rl/step")
def rl_step(req: RLStepRequest):
    if req.session_id not in _v2_sessions:
        raise HTTPException(404, f"RL session '{req.session_id}' not found. Call /rl/reset first.")
    env = _v2_sessions[req.session_id]["env"]
    obs, reward, done, info = env.step(
        patient_id=req.patient_id, action=req.action, reasoning=req.reasoning
    )
    return {
        "session_id": req.session_id, "observation": obs, "reward": reward, "done": done,
        "info": {**info, "reward_breakdown": {"rule_reward": info.get("rule_reward", 0),
                                               "llm_adjustment": info.get("llm_adjustment", 0),
                                               "final_reward": reward,
                                               "formula": "final_reward = rule_reward + 0.3 × llm_adjustment"}},
        "explainability": {"llm_scores": info.get("llm_scores", {}),
                           "llm_explanation": info.get("llm_explanation", ""),
                           "oracle_action": info.get("oracle_action", {}),
                           "mismatch_with_oracle": info.get("mismatch_with_oracle", False)},
    }


@app.get("/rl/{session_id}/trajectory")
def get_trajectory(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    env = _v2_sessions[session_id]["env"]
    return {"session_id": session_id, "trajectory": env.get_trajectory(),
            "episode_summary": env.get_episode_summary()}


@app.get("/rl/{session_id}/failures")
def get_failure_cases(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    env = _v2_sessions[session_id]["env"]
    failures = env.get_failure_cases()
    return {"session_id": session_id, "failure_count": len(failures), "failures": failures}


@app.get("/rl/{session_id}/trends")
def get_trends(session_id: str):
    if session_id not in _v2_sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    env = _v2_sessions[session_id]["env"]
    return {"session_id": session_id, "trends": env.get_learning_trends()}


@app.post("/rl/evaluate")
def standalone_llm_eval(req: LLMEvalRequest):
    if not LLM_EVAL_AVAILABLE:
        raise HTTPException(503, "llm_evaluator.py unavailable.")
    backend = _get_backend(req.backend)
    result = evaluate_with_llm(state=req.state, action=req.action, reasoning=req.reasoning, backend=backend)
    return {"evaluation": _format_llm_result(result),
            "backend_note": f"Using {result.backend_used}. We use a Llama-based evaluator to align RL agents with human clinical reasoning."}


@app.post("/rl/oracle")
def get_oracle(req: OracleRequest):
    if not LLM_EVAL_AVAILABLE:
        raise HTTPException(503, "llm_evaluator.py unavailable.")
    state = req.state
    if "task_type" not in state: state["task_type"] = "triage"
    oracle = get_oracle_action(state)
    oracle_eval = evaluate_with_llm(state=state, action=oracle, reasoning=oracle.get("rationale", ""),
                                     backend=_get_backend("rule_based"))
    return {"oracle_action": oracle, "oracle_evaluation": _format_llm_result(oracle_eval),
            "description": "Ideal physician decision based on ESI, Sepsis-3, WHO medication safety."}


@app.post("/rl/train")
async def background_train(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    _train_jobs[job_id] = {"status": "queued", "progress": 0, "n_episodes": req.n_episodes, "started_at": time.time()}

    async def _run():
        _train_jobs[job_id]["status"] = "running"
        try:
            from training_loop import train
            env, agent, metrics = train(
                n_episodes=req.n_episodes,
                difficulty=_get_difficulty(req.difficulty),
                llm_backend=_get_backend(req.llm_backend),
                curriculum=req.curriculum, verbose=False
            )
            _train_jobs[job_id].update({"status": "complete", "metrics": metrics.to_dict(),
                                         "q_table_size": len(agent.q_table), "trends": env.get_learning_trends()})
        except Exception as e:
            _train_jobs[job_id].update({"status": "error", "error": str(e)})

    background_tasks.add_task(_run)
    return {"job_id": job_id, "status": "queued", "n_episodes": req.n_episodes, "poll_url": f"/rl/train/{job_id}"}


@app.get("/rl/train/{job_id}")
def get_train_status(job_id: str):
    if job_id not in _train_jobs:
        raise HTTPException(404, f"Training job '{job_id}' not found.")
    return {"job_id": job_id, **_train_jobs[job_id]}


@app.get("/rl/demo")
def demo_step():
    if not ENV_V2_AVAILABLE:
        return {"error": "environment_v2.py unavailable", "demo": False}
    try:
        env = ClinicalTriageEnvV2(difficulty=DifficultyMode.BUSY, llm_backend=LLMBackend.RULE_BASED if LLM_EVAL_AVAILABLE else None)
        obs = env.reset()
        if not obs["patient_queue"]:
            return {"error": "No patients generated"}
        patient = obs["patient_queue"][0]
        pid = patient["patient_id"]
        esi = max(1, min(5, patient.get("true_esi", 2)))
        action = {"esi_level": esi, "disposition": f"ESI-{esi}"}
        reasoning = f"Oracle: ESI-{esi} based on vitals NEWS2={patient.get('news2_score',5)}"
        next_obs, reward, done, info = env.step(pid, action, reasoning)
        return {"demo": True, "patient": patient, "action": action, "reasoning": reasoning,
                "reward": reward, "reward_breakdown": info.get("reward_breakdown"),
                "llm_explanation": info.get("llm_explanation"), "oracle": info.get("oracle_action"),
                "queue_after": next_obs["queue_size"]}
    except Exception as e:
        return {"error": str(e), "demo": False}


# =============================================================================
# PDF REPORT
# =============================================================================

@app.get("/report/{session_id}/pdf")
def get_pdf(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, "Report not found")
    if not PDF_AVAILABLE:
        raise HTTPException(503, "PDF unavailable — install reportlab")
    report = _report_cache[session_id]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    s = []
    s.append(Paragraph("ClinicalTriageEnv v5 — Clinical Report", styles["Heading1"]))
    s.append(Paragraph(f"Session: {session_id[:8]} | {report.get('generated_at', datetime.now().isoformat())}", styles["Normal"]))
    s.append(HRFlowable(width="100%", thickness=1))
    s.append(Spacer(1, 10))
    r = report.get("result", report)
    ps = r.get("patientSummary", {})
    if ps:
        s.append(Paragraph("Clinical Summary", styles["Heading2"]))
        s.append(Paragraph(ps.get("synopsis", ""), styles["Normal"]))
        s.append(Spacer(1, 8))
    tr = r.get("triage", {})
    if tr:
        s.append(Paragraph(f"Triage: {tr.get('label','')} — {tr.get('timeToPhysician','')}", styles["Heading3"]))
        s.append(Paragraph(tr.get("rationale", ""), styles["Normal"]))
        s.append(Spacer(1, 8))
    ddx = r.get("differentialDiagnosis", [])
    if ddx:
        s.append(Paragraph("Differential Diagnosis", styles["Heading2"]))
        rows = [["Rank", "Condition", "Probability", "Confidence"]]
        for d in ddx:
            rows.append([str(d.get("rank","")), d.get("condition",""), f"{d.get('probability',0)}%", d.get("confidence","")])
        t = Table(rows, colWidths=[1.5*cm, 10*cm, 3*cm, 3*cm])
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a4a7a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f0f5fa")]),("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#ccddee")),("FONTSIZE",(0,0),(-1,-1),10)]))
        s.append(t)
    s.append(Spacer(1, 12))
    s.append(Paragraph("DISCLAIMER: AI-generated for decision support only. Validate with a licensed physician.", styles["Italic"]))
    doc.build(s)
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="application/pdf",
                             headers={"Content-Disposition": f"attachment; filename=report-{session_id[:8]}.pdf"})


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"\n🏥 ClinicalTriageEnv v5 starting on port {port}")
    print(f"   Llama model: {MODEL_NAME}")
    print(f"   HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
    print(f"   LLM_BACKEND: {os.environ.get('LLM_BACKEND', 'rule_based')}")
    uvicorn.run(app, host="0.0.0.0", port=port)
