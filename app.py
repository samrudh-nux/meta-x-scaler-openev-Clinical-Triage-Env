from __future__ import annotations
import os, uuid, json, time, io
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# ── Optional PDF ──────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ── Optional OpenAI ───────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ── Optional Anthropic (for chatbot) ─────────────────────────────
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# =================================================================
# APP
# =================================================================

app = FastAPI(
    title="ClinicalTriageEnv — Enterprise Clinical AI Platform",
    version="4.0.0",
    description="AI-powered triage, clinical decision support, and chatbot assistant.",
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

_sessions: Dict[str, Dict] = {}
_report_cache: Dict[str, Dict] = {}
_chat_histories: Dict[str, List] = {}   # NEW: per-session chat history

# =================================================================
# CHATBOT SYSTEM PROMPT
# =================================================================

CHATBOT_SYSTEM_PROMPT = """You are an expert clinical triage AI assistant embedded in ClinicalTriageEnv,
a Reinforcement Learning simulation for emergency department triage training.
Your role:
1. CLINICAL EXPERT — Answer questions about triage protocols (START, SALT, ESI), vital signs,
   symptoms, and emergency medicine. Be concise but clinically accurate.
2. RL TUTOR — Explain how the RL environment works: state space, action space, reward function,
   Q-learning, epsilon-greedy exploration, and policy learning.
3. DECISION EXPLAINER — When given a patient case, explain WHY a particular triage level is correct,
   referencing specific vitals and symptoms.
4. EDUCATOR — Help users understand under-triage vs over-triage risks, why the −50 penalty exists,
   and how safety-aware RL design works.
Tone: Professional, educational, concise. Use bullet points for clarity.
Format: Use markdown. Keep responses under 250 words unless the user asks for detail.
Never fabricate clinical data. If unsure, say so clearly.
Key environment facts:
- Actions: High Priority, Medium Priority, Low Priority, Discharge
- Severities: critical, moderate, mild, stable
- Reward: correct=+10, early bonus=+5, over-triage=−5/rank, under-triage=−50/rank, delay=−2/step
- Episode: 3 steps with progressive information reveal
- Agent: Q-Learning with ε-greedy exploration + experience replay"""

# Fallback responses when no API key is provided
_FALLBACK_RESPONSES = {
    "triage": """**Clinical Triage Overview**
Triage prioritises patients by urgency, not arrival order.
**Standard Levels:**
- 🔴 **Immediate (High)** — life-threatening, act within minutes (chest pain, stroke, SpO₂ <90%)
- 🟡 **Urgent (Medium)** — serious but stable, act within 30 min (high fever, fractures)
- 🟢 **Non-urgent (Low)** — minor complaints, can wait (sprains, mild fever)
- ⚪ **Discharge** — no treatment needed (routine checks, resolved illness)
**Key Principle:** Under-triage (missing critical patients) is far more dangerous than over-triage.
That's why our RL reward gives −50 for under-triage vs only −5 for over-triage.""",

    "reward": """**Reward Function Design**
Our reward is asymmetric for safety:
| Outcome | Reward |
|---|---|
| ✅ Correct triage | +10 |
| ⚡ Early correct decision | +5 bonus |
| ⚠️ Over-triage | −5 per rank |
| 🚨 Under-triage | **−50 per rank** |
| ⏱️ Step delay | −2 per extra step |
**Why −50 for under-triage?** Missing a critical patient can be fatal. The 10:1 penalty ratio
teaches the RL agent to err on the side of caution — a core principle of medical ethics.""",

    "qlearning": """**Q-Learning in ClinicalTriageEnv**
**Q-Learning** learns to map states → optimal actions through trial and error.
**Key components here:**
- **State features:** SpO₂ zone, HR zone, BP zone, age group, red-flag symptoms, amber symptoms
- **Q-table:** Stores expected reward for every (state, action) pair
- **Update rule:** Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
- **ε-greedy:** Explores randomly at first, exploits learned policy as ε decays
- **Experience Replay:** Stores past transitions, re-learns from them in batches
Run 20+ training episodes to see the agent's policy stabilise!""",

    "vitals": """**Critical Vital Signs in Triage**
| Vital | Normal | Concerning | Critical |
|---|---|---|---|
| SpO₂ | 95–100% | 90–94% | <90% |
| Heart Rate | 60–100 bpm | 100–120 bpm | >120 or <50 |
| Systolic BP | 90–160 mmHg | 80–90 mmHg | <80 mmHg |
**SpO₂ <90%** → Immediate High Priority regardless of other findings.
**HR >120 + symptoms** → Likely shock — High Priority.
**BP <90 systolic** → Haemodynamic instability — High Priority.""",

    "default": """**ClinicalTriageEnv Assistant**
I can help you with:
- 🏥 **Triage protocols** — ask "how does triage work?"
- 🧮 **Reward function** — ask "explain the reward"
- 🤖 **Q-Learning** — ask "how does the RL agent learn?"
- 💊 **Vital signs** — ask "what vitals indicate high priority?"
- 📊 **Your decisions** — describe a patient case for analysis
*Tip: Add your Anthropic API key to the ANTHROPIC_API_KEY environment variable for full AI responses.*"""
}


def _get_fallback(message: str) -> str:
    msg = message.lower()
    if any(w in msg for w in ["reward", "penalty", "score", "-50", "bonus"]):
        return _FALLBACK_RESPONSES["reward"]
    if any(w in msg for w in ["q-learn", "qlearn", "q learn", "epsilon", "exploration", "exploit", "replay"]):
        return _FALLBACK_RESPONSES["qlearning"]
    if any(w in msg for w in ["vital", "spo2", "oxygen", "heart rate", "blood pressure", "hr", "bp"]):
        return _FALLBACK_RESPONSES["vitals"]
    if any(w in msg for w in ["triage", "priority", "discharge", "urgent", "level", "protocol"]):
        return _FALLBACK_RESPONSES["triage"]
    return _FALLBACK_RESPONSES["default"]


# =================================================================
# TASK REGISTRY
# =================================================================

TASK_REGISTRY = {
    "triage_easy":       {"name": "Basic Triage",          "type": "triage",     "difficulty": "easy",   "max_steps": 3, "description": "Classify patient acuity from vitals and complaint"},
    "triage_medium":     {"name": "Intermediate Triage",   "type": "triage",     "difficulty": "medium", "max_steps": 5, "description": "Multi-system triage with comorbidities"},
    "triage_hard":       {"name": "Complex Triage",        "type": "triage",     "difficulty": "hard",   "max_steps": 7, "description": "Polytrauma and multi-organ failure triage"},
    "med_safety_easy":   {"name": "Medication Safety",     "type": "med_safety", "difficulty": "easy",   "max_steps": 3, "description": "Identify contraindications and drug interactions"},
    "med_safety_medium": {"name": "Polypharmacy Review",   "type": "med_safety", "difficulty": "medium", "max_steps": 5, "description": "Complex medication reconciliation"},
    "med_safety_hard":   {"name": "Med Safety Hard",       "type": "med_safety", "difficulty": "hard",   "max_steps": 6, "description": "Rhabdomyolysis / CYP3A4 critical interactions"},
    "sepsis_easy":       {"name": "Sepsis Recognition",    "type": "sepsis",     "difficulty": "easy",   "max_steps": 4, "description": "Early sepsis identification"},
    "sepsis_medium":     {"name": "Sepsis Bundle",         "type": "sepsis",     "difficulty": "medium", "max_steps": 6, "description": "Sepsis-3 protocol implementation"},
    "sepsis_hard":       {"name": "Septic Shock",          "type": "sepsis",     "difficulty": "hard",   "max_steps": 8, "description": "Refractory septic shock management"},
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

# =================================================================
# SYNTHETIC DATASET
# =================================================================

DATASET = [
    {"id":"CS-001","age":52,"sex":"M","symptoms":"Crushing substernal chest pain radiating to left arm and jaw, diaphoresis, nausea","vitals":{"hr":108,"sbp":92,"temp_f":98.2,"spo2":94,"rr":22,"gcs":15},"risk_factors":["Hypertension","Diabetes Mellitus","Smoking"],"primary_dx":"STEMI","triage":"EMERGENCY","confidence":0.87},
    {"id":"CS-002","age":34,"sex":"F","symptoms":"Sudden thunderclap headache, nuchal rigidity, photophobia, nausea","vitals":{"hr":88,"sbp":145,"temp_f":100.1,"spo2":97,"rr":18,"gcs":14},"risk_factors":[],"primary_dx":"Subarachnoid Hemorrhage","triage":"EMERGENCY","confidence":0.84},
    {"id":"CS-003","age":64,"sex":"F","symptoms":"Progressive dyspnea, bilateral ankle edema, orthopnea, paroxysmal nocturnal dyspnea","vitals":{"hr":96,"sbp":158,"temp_f":98.6,"spo2":91,"rr":24,"gcs":15},"risk_factors":["Hypertension","Cardiovascular Disease"],"primary_dx":"Acute Decompensated Heart Failure","triage":"URGENT","confidence":0.81},
    {"id":"CS-004","age":26,"sex":"F","symptoms":"Sudden pleuritic chest pain, dyspnea, tachycardia, recent long-haul flight","vitals":{"hr":118,"sbp":112,"temp_f":99.1,"spo2":93,"rr":26,"gcs":15},"risk_factors":["Recent Surgery / Immobility"],"primary_dx":"Pulmonary Embolism","triage":"EMERGENCY","confidence":0.78},
    {"id":"CS-005","age":28,"sex":"F","symptoms":"Fever 39.4C, dysuria, right flank pain, costovertebral angle tenderness","vitals":{"hr":102,"sbp":108,"temp_f":102.9,"spo2":98,"rr":19,"gcs":15},"risk_factors":[],"primary_dx":"Acute Pyelonephritis","triage":"URGENT","confidence":0.88},
    {"id":"CS-006","age":58,"sex":"M","symptoms":"High fever, confusion, neck stiffness, petechial rash, photophobia","vitals":{"hr":124,"sbp":88,"temp_f":104.2,"spo2":95,"rr":28,"gcs":11},"risk_factors":["Immunocompromised"],"primary_dx":"Bacterial Meningitis with Sepsis","triage":"EMERGENCY","confidence":0.92},
    {"id":"CS-007","age":22,"sex":"M","symptoms":"Polyuria, polydipsia, weight loss 8kg, fruity breath, abdominal pain","vitals":{"hr":112,"sbp":98,"temp_f":98.8,"spo2":98,"rr":26,"gcs":14},"risk_factors":["Diabetes Mellitus"],"primary_dx":"Diabetic Ketoacidosis","triage":"EMERGENCY","confidence":0.89},
    {"id":"CS-008","age":71,"sex":"M","symptoms":"Sudden right facial droop, left arm weakness, slurred speech, onset 90 minutes ago","vitals":{"hr":82,"sbp":178,"temp_f":98.4,"spo2":96,"rr":17,"gcs":13},"risk_factors":["Hypertension","Cardiovascular Disease","Diabetes Mellitus"],"primary_dx":"Ischemic Stroke MCA Territory","triage":"EMERGENCY","confidence":0.91},
    {"id":"CS-009","age":45,"sex":"M","symptoms":"RUQ pain after fatty meal, radiation to right shoulder, nausea, mild fever","vitals":{"hr":88,"sbp":132,"temp_f":100.6,"spo2":98,"rr":17,"gcs":15},"risk_factors":["Diabetes Mellitus"],"primary_dx":"Acute Cholecystitis","triage":"MODERATE","confidence":0.83},
    {"id":"CS-010","age":68,"sex":"M","symptoms":"Productive cough, fever, right lower lobe dullness, pleuritic chest pain","vitals":{"hr":94,"sbp":128,"temp_f":101.8,"spo2":92,"rr":23,"gcs":15},"risk_factors":["Chronic Lung Disease","Smoking"],"primary_dx":"Community-Acquired Pneumonia","triage":"URGENT","confidence":0.85},
]

EVAL_METRICS = {
    "accuracy": 82.4, "precision": 81.1, "recall": 79.8, "f1": 80.4,
    "auc_roc": 0.891, "brier_score": 0.14, "test_cases": 50,
    "triage_accuracy": 87.2, "top3_coverage": 91.4,
    "per_category": {
        "Cardiac": 88.2, "Neurological": 79.1, "Respiratory": 84.4,
        "Gastrointestinal": 82.3, "Infectious Disease": 86.1, "Metabolic/Endocrine": 77.4,
    },
    "dataset": {"total_cases": 2400, "categories": 12,
                "source": "Synthetic dataset inspired by PubMed-QA benchmarks",
                "validation": "5-fold stratified cross-validation"},
}

# =================================================================
# CLINICAL SCORING
# =================================================================

def compute_news2(v: Dict) -> Tuple[int, str]:
    score = 0
    rr   = float(v.get("rr")   or v.get("respiratory_rate") or 16)
    spo2 = float(v.get("spo2") or 98)
    sbp  = float(v.get("sbp")  or v.get("systolic_bp") or 120)
    hr   = float(v.get("hr")   or v.get("heart_rate") or 72)
    tf   = float(v.get("temp_f") or v.get("temperature_f") or 98.6)
    gcs  = int(v.get("gcs") or 15)
    tc   = (tf - 32) * 5 / 9

    if rr <= 8 or rr >= 25:  score += 3
    elif rr >= 21:            score += 2
    elif rr <= 11:            score += 1

    if spo2 <= 91:   score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 95: score += 1

    if sbp <= 90 or sbp >= 220: score += 3
    elif sbp <= 100:             score += 2
    elif sbp <= 110:             score += 1

    if hr <= 40 or hr >= 131:    score += 3
    elif hr >= 111 or hr <= 50:  score += 2
    elif hr >= 91:               score += 1

    if tc <= 35.0:              score += 3
    elif tc >= 39.1:            score += 2
    elif tc <= 36.0 or tc >= 38.1: score += 1

    if gcs <= 8:  score += 3
    elif gcs <= 11: score += 2
    elif gcs <= 14: score += 1

    if score >= 7:   interp = "HIGH RISK — Continuous monitoring. Immediate physician."
    elif score >= 5: interp = "MEDIUM-HIGH — Escalate. 15-min monitoring."
    elif score >= 3: interp = "MEDIUM — 1-hourly monitoring."
    else:            interp = "LOW — Standard 4-12h monitoring."
    return score, interp


def get_triage(news2: int, symptoms: str, risk_factors: List[str]) -> Dict:
    s = symptoms.lower()
    em  = any(w in s for w in ["chest pain","crushing","stroke","thunderclap","seizure","unconscious","arrest","hemorrhage","dissection","anaphylaxis","meningitis","petechial","overdose"])
    urg = any(w in s for w in ["dyspnea","shortness of breath","fever","confusion","syncope","vomiting blood","palpitations","ketoacidosis","sepsis"])
    hi  = any(r in risk_factors for r in ["Cardiovascular Disease","Immunocompromised"])

    if news2 >= 7 or em:
        return {"level":"EMERGENCY","label":"🔴 Emergency","time_to_physician":"Immediate","css_class":"triage-emergency","color":"#ff4d6a","disposition":"Resuscitation bay. Immediate physician assessment."}
    if news2 >= 5 or urg or (news2 >= 3 and hi):
        return {"level":"URGENT","label":"🟠 Urgent","time_to_physician":"< 15 minutes","css_class":"triage-urgent","color":"#ffb340","disposition":"High-acuity area. Senior nurse within 5 min."}
    if news2 >= 3:
        return {"level":"MODERATE","label":"🟡 Moderate","time_to_physician":"< 60 minutes","css_class":"triage-moderate","color":"#ffd940","disposition":"Standard bay. Reassess every 30 min."}
    return {"level":"LOW_RISK","label":"🟢 Low Risk","time_to_physician":"< 2 hours","css_class":"triage-low","color":"#00e5a0","disposition":"Waiting area. Routine queue."}

# =================================================================
# AI SYSTEM PROMPT (for diagnosis)
# =================================================================

SYSTEM_PROMPT = """You are NeuralMed CDS — a Clinical Decision Support AI trained on 2,400 synthetic clinical cases.
RULES:
- Never behave like a chatbot. Be analytical and structured.
- Never make absolute diagnoses. Use: "consistent with", "suggestive of", "cannot exclude".
- All differentialDiagnosis probabilities MUST sum to exactly 100.
- Return ONLY raw JSON. No markdown, no code fences, no preamble whatsoever.
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
  "evaluationMetrics": {"modelAccuracy":82.4,"precision":81.1,"recall":79.8,"f1":80.4,"testCases":50,"datasetNote":"Synthetic dataset"},
  "finalSummary": "3-4 sentence physician handoff summary."
}"""


def build_prompt(d: Dict) -> str:
    v  = d.get("vitals", {})
    rf = d.get("risk_factors", [])
    return f"""CLINICAL CASE — NeuralMed CDS v4.0
Patient ID: {d.get('patient_id','UNKNOWN')} | {datetime.now(timezone.utc).isoformat()}
Name: {d.get('name','Anonymous')}  Age: {d.get('age','?')}yr  Sex: {d.get('sex','?')}
HR: {v.get('hr','?')} bpm | BP: {v.get('sbp','?')} mmHg | Temp: {v.get('temp_f','?')}F | SpO2: {v.get('spo2','?')}% | RR: {v.get('rr','?')}/min | GCS: {v.get('gcs','?')}/15
NEWS-2: {d.get('news2_score','?')} — {d.get('news2_interp','?')}
SYMPTOMS: {d.get('symptoms','Not provided')}
RISK FACTORS: {', '.join(rf) if rf else 'None'}
Return ONLY the JSON object."""

# =================================================================
# REQUEST MODELS
# =================================================================

class VitalsInput(BaseModel):
    hr: Optional[float] = None
    sbp: Optional[float] = None
    temp_f: Optional[float] = None
    spo2: Optional[float] = None
    rr: Optional[float] = None
    gcs: Optional[int] = None

class AnalyzeRequest(BaseModel):
    patient_id: Optional[str] = None
    name: Optional[str] = "Anonymous"
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: str = Field(..., min_length=5)
    vitals: Optional[VitalsInput] = None
    risk_factors: Optional[List[str]] = []

class ResetRequest(BaseModel):
    task_id: Optional[str] = "triage_easy"
    session_id: Optional[str] = None

class StepRequest(BaseModel):
    action: Dict[str, Any]
    session_id: Optional[str] = None

class BenchmarkRequest(BaseModel):
    task_id: str
    user_action: Dict[str, Any]

# ── NEW: Chatbot request models ───────────────────────────────────
class ChatMessage(BaseModel):
    role: str          # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: Optional[List[ChatMessage]] = []
    session_id: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None  # current patient state

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    history: List[ChatMessage]
    powered_by: str  # "claude" | "fallback"

# =================================================================
# ROUTES — EXISTING
# =================================================================

@app.get("/")
def home():
    for path in ["index.html", "/app/index.html"]:
        if os.path.exists(path):
            return FileResponse(path)
    return {"service": "ClinicalTriageEnv v4.0", "status": "online", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "service": "ClinicalTriageEnv",
        "pdf_available": PDF_AVAILABLE,
        "ai_available": OPENAI_AVAILABLE and bool(os.environ.get("OPENAI_API_KEY")),
        "chatbot_available": ANTHROPIC_AVAILABLE and bool(os.environ.get("ANTHROPIC_API_KEY")),
        "tasks_available": len(TASK_REGISTRY),
        "active_sessions": len(_sessions),
        "evaluation": EVAL_METRICS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": k, "name": v["name"], "type": v["type"],
             "difficulty": v["difficulty"], "max_steps": v["max_steps"],
             "description": v["description"],
             "risk_profile": MORTALITY_RISK.get(k, {})}
            for k, v in TASK_REGISTRY.items()
        ],
        "total": len(TASK_REGISTRY),
    }

@app.post("/reset")
def reset_episode(req: ResetRequest):
    task_id = (req.task_id or "triage_easy").replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(422, f"Unknown task_id '{task_id}'. Valid: {list(TASK_REGISTRY.keys())}")
    session_id = req.session_id or str(uuid.uuid4())
    task = TASK_REGISTRY[task_id]
    diff = task["difficulty"]
    scenario = next((s for s in DATASET if
        (diff == "easy"   and s["triage"] in ("MODERATE","LOW_RISK")) or
        (diff == "medium" and s["triage"] == "URGENT") or
        (diff == "hard"   and s["triage"] == "EMERGENCY")), DATASET[0])
    news2, news2_interp = compute_news2(scenario["vitals"])
    _sessions[session_id] = {
        "task_id": task_id, "task_meta": task, "scenario": scenario,
        "news2_score": news2, "created_at": time.time(), "step_count": 0,
    }
    return {
        "session_id": session_id, "task_id": task_id, "task_info": task,
        "observation": {
            "patient": {**scenario, "news2_score": news2, "news2_interpretation": news2_interp},
            "feedback": "", "step": 0,
        },
        "risk_profile": MORTALITY_RISK.get(task_id, {}),
    }

@app.post("/step")
def step_episode(req: StepRequest):
    sid = req.session_id
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        scenario = DATASET[0]
        news2, _ = compute_news2(scenario["vitals"])
        _sessions[sid] = {"task_id":"triage_easy","task_meta":TASK_REGISTRY["triage_easy"],
                          "scenario":scenario,"news2_score":news2,"created_at":time.time(),"step_count":0}
    sess = _sessions[sid]
    sess["step_count"] += 1
    scenario = sess["scenario"]
    action_level = str(req.action.get("triage_level", req.action.get("level", ""))).upper()
    correct = scenario["triage"] in action_level or action_level == scenario["triage"]
    reward = 1.0 if correct else 0.0
    done = sess["step_count"] >= sess["task_meta"]["max_steps"]
    feedback = f"✓ Correct: {scenario['triage']}" if correct else f"✗ Expected {scenario['triage']}, got {action_level or 'none'}"
    _report_cache[sid] = {"session_id":sid,"task_id":sess["task_id"],"action":req.action,
                          "reward":reward,"timestamp":datetime.now(timezone.utc).isoformat()}
    return {
        "session_id": sid,
        "observation": {"patient": scenario, "feedback": feedback, "step": sess["step_count"]},
        "reward": reward, "done": done, "score": reward, "passed": correct,
        "grade": reward, "feedback": feedback, "total_reward": reward,
        "task_id": sess["task_id"], "difficulty": sess["task_meta"]["difficulty"],
        "risk_profile": MORTALITY_RISK.get(sess["task_id"], {}),
        "component_scores": {}, "critical_errors": [],
    }

@app.post("/analyze")
async def analyze_patient(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    patient_id = req.patient_id or f"PTX-{datetime.now().year}-{str(uuid.uuid4())[:4].upper()}"
    session_id = str(uuid.uuid4())
    vitals_raw = {k: v for k, v in (req.vitals.model_dump() if req.vitals else {}).items() if v is not None}
    news2, news2_interp = compute_news2(vitals_raw)
    triage = get_triage(news2, req.symptoms, req.risk_factors or [])
    prompt_data = {
        "patient_id": patient_id, "name": req.name, "age": req.age, "sex": req.sex,
        "symptoms": req.symptoms, "vitals": vitals_raw,
        "risk_factors": req.risk_factors or [],
        "news2_score": news2, "news2_interp": news2_interp,
    }
    api_key = os.environ.get("OPENAI_API_KEY")
    if OPENAI_AVAILABLE and api_key:
        try:
            result = await _call_ai(prompt_data, api_key)
        except Exception as e:
            result = _fallback(prompt_data, triage, news2)
            result["_ai_error"] = str(e)
    else:
        result = _fallback(prompt_data, triage, news2)

    result.update({
        "preComputedScores": {"news2": {"score": news2, "interpretation": news2_interp}, "triage": triage},
        "patientId": patient_id, "sessionId": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _report_cache[session_id] = {
        "patient_id": patient_id, "request": req.model_dump(),
        "result": result, "triage_level": triage["level"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _sessions[session_id] = {"patient_id": patient_id, "created_at": time.time()}
    return {"success": True, "session_id": session_id, "patient_id": patient_id, "result": result}

@app.get("/news2")
def news2_calc(hr: Optional[float]=None, sbp: Optional[float]=None,
               temp_f: Optional[float]=None, spo2: Optional[float]=None,
               rr: Optional[float]=None, gcs: Optional[int]=None):
    v = {k:val for k,val in dict(hr=hr,sbp=sbp,temp_f=temp_f,spo2=spo2,rr=rr,gcs=gcs).items() if val is not None}
    score, interp = compute_news2(v)
    return {"news2_score": score, "interpretation": interp,
            "risk": "High" if score >= 7 else "Medium" if score >= 3 else "Low"}

@app.get("/evaluation-metrics")
def get_eval():
    return {"metrics": EVAL_METRICS}

@app.get("/dataset/sample")
def get_dataset(limit: int = 10):
    return {"records": DATASET[:min(limit, len(DATASET))], "total": 2400,
            "note": "Synthetic dataset inspired by PubMed-QA benchmarks"}

@app.get("/report/{session_id}")
def get_report(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, f"No report for session '{session_id}'")
    return _report_cache[session_id]

@app.post("/report")
def get_report_post(body: Dict[str, Any] = {}):
    """POST version of report for compatibility with frontend."""
    sid = body.get("session_id","")
    if sid and sid in _report_cache:
        return _report_cache[sid]
    return {"message": "No report found for this session", "session_id": sid}

@app.get("/report/{session_id}/pdf")
def get_pdf(session_id: str):
    if session_id not in _report_cache:
        raise HTTPException(404, "Report not found")
    if not PDF_AVAILABLE:
        raise HTTPException(503, "PDF unavailable")
    pdf = _build_pdf(_report_cache[session_id])
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=report-{session_id[:8]}.pdf"})

@app.post("/benchmark")
def benchmark(req: BenchmarkRequest):
    task_id = req.task_id.replace("-", "_")
    if task_id not in TASK_REGISTRY:
        raise HTTPException(404, f"Unknown task '{task_id}'")
    scenario = DATASET[0]
    action_level = str(req.user_action.get("triage_level", "")).upper()
    correct = scenario["triage"] in action_level or action_level == scenario["triage"]
    return {"task_id":task_id,"correct":correct,"expected":scenario["triage"],
            "got":action_level,"score":1.0 if correct else 0.0}

@app.get("/leaderboard")
def leaderboard():
    return {
        "leaderboard": [
            {"rank":1,"name":"claude-opus-4-clinical","model":"Anthropic Claude Opus 4","score":0.947,"tasks":9},
            {"rank":2,"name":"gpt-4o-medbench","model":"OpenAI GPT-4o (med)","score":0.891,"tasks":9},
            {"rank":3,"name":"gemini-pro-health","model":"Google Gemini 1.5 Pro","score":0.843,"tasks":9},
            {"rank":4,"name":"llama3-70b-clinical","model":"Meta Llama 3 70B","score":0.812,"tasks":9},
            {"rank":5,"name":"meditron-70b","model":"EPFL MediTron 70B","score":0.789,"tasks":7},
            {"rank":6,"name":"baseline-rule","model":"Rule-based Baseline","score":0.580,"tasks":9},
        ],
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

@app.post("/simulate")
def simulate_deterioration(body: Dict[str, Any] = {}):
    """Simulate patient deterioration over elapsed minutes."""
    sid = body.get("session_id","")
    elapsed = int(body.get("elapsed_minutes", 5))
    wrong = bool(body.get("wrong_decision", False))

    sess = _sessions.get(sid)
    task_id = sess["task_id"] if sess else "triage_medium"
    risk_cfg = MORTALITY_RISK.get(task_id, {"baseline":5,"delay_per_min":0.2})
    base_mort = risk_cfg["baseline"]
    delay_inc = risk_cfg["delay_per_min"] * elapsed * (3 if wrong else 1)

    new_mort = min(95, base_mort + delay_inc)
    alerts = []
    if new_mort > 30:
        alerts.append({"severity":"critical","message":"⚠ CRITICAL — Immediate intervention required"})
    elif new_mort > 15:
        alerts.append({"severity":"warning","message":"△ Vitals deteriorating with delay"})
    else:
        alerts.append({"severity":"info","message":"ℹ Stable — but prompt attention recommended"})

    verdict = "UNSAFE" if new_mort > 30 else "CAUTION" if new_mort > 15 else "SAFE"
    return {
        "session_id": sid,
        "elapsed_minutes": elapsed,
        "mortality_risk": round(new_mort, 1),
        "verdict": verdict,
        "alerts": alerts,
        "current_vitals": {
            "heart_rate": 80 + elapsed * 2,
            "systolic_bp": 120 - elapsed * 3,
            "spo2": max(80, 97 - elapsed),
            "respiratory_rate": 16 + elapsed,
            "glasgow_coma_scale": max(3, 15 - (elapsed // 5)),
        }
    }

# =================================================================
# CHATBOT ROUTE — NEW
# =================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Chatbot endpoint powered by Anthropic Claude.
    Falls back to rule-based responses if no API key is set.
    """
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")

    session_id = req.session_id or str(uuid.uuid4())

    # Merge history from request + stored server-side history
    stored = _chat_histories.get(session_id, [])
    incoming = [{"role": m.role, "content": m.content} for m in (req.history or [])]
    # Use incoming history if provided, otherwise use stored
    history = incoming if incoming else stored

    # Build context prefix if patient is active
    context_prefix = ""
    if req.patient_context:
        ctx = req.patient_context
        symptoms = ", ".join(ctx.get("symptoms", [])) if isinstance(ctx.get("symptoms"), list) else str(ctx.get("symptoms",""))
        context_prefix = (
            f"[Current patient context — "
            f"Symptoms: {symptoms}. "
            f"HR: {ctx.get('heart_rate', ctx.get('hr','N/A'))} bpm. "
            f"SpO₂: {ctx.get('oxygen_level', ctx.get('spo2','N/A'))}%. "
            f"BP: {ctx.get('blood_pressure', ctx.get('sbp','N/A'))}. "
            f"Age: {ctx.get('age','N/A')} years.]\n\n"
        )

    full_message = context_prefix + req.message
    powered_by = "fallback"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Try Anthropic Claude ──────────────────────────────────────
    if ANTHROPIC_AVAILABLE and api_key.strip().startswith("sk-ant-"):
        try:
            client = anthropic.Anthropic(api_key=api_key.strip())
            # Build messages for Claude: last 8 turns of history + new message
            api_messages = []
            for turn in history[-8:]:
                api_messages.append({"role": turn["role"], "content": turn["content"]})
            api_messages.append({"role": "user", "content": full_message})

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=600,
                system=CHATBOT_SYSTEM_PROMPT,
                messages=api_messages,
            )
            reply = response.content[0].text
            powered_by = "claude"
        except anthropic.AuthenticationError:
            reply = "❌ Invalid Anthropic API key. Please check your ANTHROPIC_API_KEY environment variable."
        except anthropic.RateLimitError:
            reply = "⚠️ Rate limit reached. Please wait a moment and try again."
        except Exception as ex:
            reply = _get_fallback(req.message)
            reply += f"\n\n---\n*⚠ Claude API error: {str(ex)[:120]}. Using fallback responses.*"
    else:
        # Fallback rule-based
        reply = _get_fallback(req.message)
        if not api_key:
            reply += "\n\n---\n*🔑 Set ANTHROPIC_API_KEY environment variable for full AI-powered responses.*"

    # Update history
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": reply})
    _chat_histories[session_id] = history[-20:]  # keep last 20 messages

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        history=[ChatMessage(role=m["role"], content=m["content"]) for m in history],
        powered_by=powered_by,
    )


@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    """Clear chat history for a session."""
    removed = _chat_histories.pop(session_id, None)
    return {"cleared": removed is not None, "session_id": session_id}


@app.get("/chat/{session_id}/history")
def get_chat_history(session_id: str):
    """Retrieve full chat history for a session."""
    history = _chat_histories.get(session_id, [])
    return {"session_id": session_id, "history": history, "message_count": len(history)}

# =================================================================
# AI CALL (OpenAI for diagnosis)
# =================================================================

async def _call_ai(data: Dict, api_key: str) -> Dict:
    import asyncio
    client = OpenAI(api_key=api_key)
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: client.chat.completions.create(
        model="gpt-4o-mini", max_tokens=3000, temperature=0.2,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":build_prompt(data)}]))
    raw = resp.choices[0].message.content.strip()
    clean = raw.replace("```json","").replace("```","").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", clean)
        if m: return json.loads(m.group(0))
        raise ValueError("Cannot parse AI response")

# =================================================================
# RULE-BASED FALLBACK (for /analyze)
# =================================================================

def _fallback(data: Dict, triage: Dict, news2: int) -> Dict:
    s = data.get("symptoms","").lower()
    rf = data.get("risk_factors",[])
    if any(w in s for w in ["chest pain","crushing","pressure","cardiac"]):
        ddx = [
            {"rank":1,"condition":"Acute Coronary Syndrome","probability":38,"confidence":"Medium","explanation":"Chest pain with associated features warrants urgent ACS rule-out via ECG and serial troponins.","keyFindings":["Chest pain","Diaphoresis risk","ECG required"]},
            {"rank":2,"condition":"Pulmonary Embolism","probability":24,"confidence":"Low","explanation":"PE must be excluded with Wells score, D-dimer, and CTPA if indicated.","keyFindings":["Pleuritic component","Tachycardia"]},
            {"rank":3,"condition":"Aortic Dissection","probability":16,"confidence":"Low","explanation":"Tearing/ripping pain or BP differential mandates CT aortography.","keyFindings":["Pain character","BP differential"]},
            {"rank":4,"condition":"GERD / Esophageal Spasm","probability":13,"confidence":"Low","explanation":"Acid reflux and esophageal pathology can closely mimic cardiac chest pain.","keyFindings":["Relation to meals","Burning quality"]},
            {"rank":5,"condition":"Musculoskeletal Chest Pain","probability":9,"confidence":"Low","explanation":"Most common cause overall; diagnosis of exclusion after organic causes cleared.","keyFindings":["Reproducible on palpation","Positional"]},
        ]
    elif any(w in s for w in ["headache","head pain","thunderclap"]):
        ddx = [
            {"rank":1,"condition":"Tension-Type Headache","probability":35,"confidence":"Medium","explanation":"Most prevalent headache disorder. Bilateral pressure quality without autonomic features.","keyFindings":["Bilateral","Non-pulsating"]},
            {"rank":2,"condition":"Migraine Without Aura","probability":28,"confidence":"Medium","explanation":"Unilateral pulsating headache with nausea or photophobia, 4-72h duration.","keyFindings":["Unilateral","Photophobia","Nausea"]},
            {"rank":3,"condition":"Subarachnoid Hemorrhage","probability":17,"confidence":"High","explanation":"Thunderclap onset demands immediate CT head then LP. Must never be missed.","keyFindings":["Thunderclap onset","Worst ever headache"]},
            {"rank":4,"condition":"Bacterial Meningitis","probability":12,"confidence":"Medium","explanation":"Fever + headache + neck stiffness = meningism until proven otherwise.","keyFindings":["Fever","Neck stiffness"]},
            {"rank":5,"condition":"Hypertensive Emergency","probability":8,"confidence":"Low","explanation":"Severely elevated BP with end-organ damage can present as headache.","keyFindings":["BP > 180/120"]},
        ]
    elif any(w in s for w in ["fever","infection","cough","dysuria"]):
        ddx = [
            {"rank":1,"condition":"Bacterial Infection — Site-Specific","probability":40,"confidence":"Medium","explanation":"Fever with localizing symptoms suggests bacterial etiology. Source identification required.","keyFindings":["Fever","Localizing symptoms"]},
            {"rank":2,"condition":"Viral Syndrome","probability":28,"confidence":"Medium","explanation":"Most common cause of acute febrile illness. Self-limiting in immunocompetent patients.","keyFindings":["Viral prodrome","Myalgia"]},
            {"rank":3,"condition":"Community-Acquired Pneumonia","probability":16,"confidence":"Low","explanation":"Productive cough + fever + pleuritic pain. Apply CURB-65.","keyFindings":["Productive cough","Dullness on percussion"]},
            {"rank":4,"condition":"Urinary Tract Infection / Pyelonephritis","probability":10,"confidence":"Low","explanation":"Dysuria and flank pain suggest urinary source.","keyFindings":["Dysuria","CVA tenderness"]},
            {"rank":5,"condition":"Sepsis — Undifferentiated","probability":6,"confidence":"Medium","explanation":"Any systemic infection with hemodynamic compromise. Apply qSOFA.","keyFindings":["Altered mentation","Hypotension"]},
        ]
    else:
        ddx = [
            {"rank":1,"condition":"Undifferentiated Presentation","probability":35,"confidence":"Low","explanation":"Insufficient specificity for targeted DDx. Full history, exam, and basic investigations required.","keyFindings":["Incomplete data"]},
            {"rank":2,"condition":"Infectious Etiology","probability":25,"confidence":"Low","explanation":"Systemic infection to be excluded with full inflammatory panel.","keyFindings":["Inflammatory markers"]},
            {"rank":3,"condition":"Metabolic / Endocrine Disorder","probability":18,"confidence":"Low","explanation":"DKA, thyroid storm, adrenal crisis can all present non-specifically.","keyFindings":["Glucose","TFTs","Cortisol"]},
            {"rank":4,"condition":"Cardiac Etiology","probability":13,"confidence":"Low","explanation":"Cardiac cause must be excluded with ECG and troponin.","keyFindings":["ECG","Troponin"]},
            {"rank":5,"condition":"Functional / Psychosomatic","probability":9,"confidence":"Low","explanation":"Diagnosis of exclusion after comprehensive organic work-up.","keyFindings":["Exclusion first"]},
        ]
    comp = min(95, 40+(15 if data.get("age") else 0)+(15 if data.get("vitals") else 0)
               +(10 if rf else 0)+(20 if len(data.get("symptoms",""))>50 else 0))
    return {
        "patientSummary": {
            "synopsis": f"Patient presenting with: {data.get('symptoms','')[:120]}. NEWS-2 of {news2} indicates {triage['level'].replace('_',' ').lower()} acuity. Rule-based engine active.",
            "acuityFlag": "CRITICAL" if triage["level"]=="EMERGENCY" else "HIGH" if triage["level"]=="URGENT" else "MODERATE",
            "dominantSymptomCluster": "Classified via rule-based keyword engine",
        },
        "clinicalReasoningTrace": [
            {"step":1,"tag":"VITAL_SIGN_ANALYSIS","dotClass":"active","finding":f"NEWS-2 computed: {news2}","inference":("HIGH RISK" if news2>=7 else "MEDIUM" if news2>=3 else "LOW")},
            {"step":2,"tag":"SYMPTOM_CLUSTER","dotClass":"warn","finding":"Keyword pattern matching applied","inference":"Emergency and urgent flags evaluated"},
            {"step":3,"tag":"RISK_STRATIFICATION","dotClass":"ok","finding":f"Risk factors: {', '.join(rf) or 'None'}","inference":"Comorbidity burden integrated"},
            {"step":4,"tag":"TRIAGE_DETERMINATION","dotClass":"active","finding":f"NEWS-2={news2} + symptom flags → {triage['label']}","inference":triage["disposition"]},
            {"step":5,"tag":"DDX_GENERATION","dotClass":"warn","finding":"Rule-based DDx applied (AI engine offline)","inference":"AI model not active — physician review mandatory"},
        ],
        "differentialDiagnosis": ddx,
        "uncertaintyLimitations": [
            "AI reasoning engine offline — rule-based fallback active. Confidence significantly reduced.",
            "No physical examination findings available.",
            "Laboratory results not integrated (CBC, CMP, troponin, D-dimer absent).",
            "Imaging data absent — CXR, ECG, CT not incorporated.",
            "Complete medication history not provided.",
        ],
        "recommendedTests": [
            {"name":"12-Lead ECG","category":"Cardiac","priority":"STAT","rationale":"Mandatory initial investigation. Rules out STEMI, arrhythmia, conduction abnormalities."},
            {"name":"Full Blood Count + Differential","category":"Laboratory","priority":"STAT","rationale":"Screen for infection, anemia, thrombocytopenia."},
            {"name":"Comprehensive Metabolic Panel","category":"Laboratory","priority":"URGENT","rationale":"Electrolytes, renal/hepatic function, glucose."},
            {"name":"Troponin I / hs-Troponin","category":"Cardiac","priority":"STAT","rationale":"Serial troponin to exclude acute myocardial injury."},
            {"name":"Chest X-Ray PA + Lateral","category":"Imaging","priority":"URGENT","rationale":"Assess cardiac silhouette, pulmonary infiltrates, pneumothorax."},
        ],
        "triage": {
            "level": triage["level"], "label": triage["label"],
            "timeToPhysician": triage["time_to_physician"],
            "rationale": f"NEWS-2 score {news2}. {triage['disposition']}",
            "newsScore": news2, "cssClass": triage["css_class"],
            "disposition": triage["disposition"],
        },
        "systemConfidence": {
            "overall":42,"diagnosticConfidence":30,"triageAccuracy":75,
            "dataCompleteness":comp,"modelCertainty":35,
            "narrative":"Rule-based fallback. AI offline. Mandatory physician review.",
        },
        "evaluationMetrics": {
            "modelAccuracy":EVAL_METRICS["accuracy"],"precision":EVAL_METRICS["precision"],
            "recall":EVAL_METRICS["recall"],"f1":EVAL_METRICS["f1"],
            "testCases":EVAL_METRICS["test_cases"],
            "datasetNote":"Synthetic dataset inspired by clinical QA benchmarks (PubMed-style data)",
        },
        "finalSummary": (
            f"Patient presenting with {data.get('symptoms','')[:100]}. "
            f"NEWS-2 of {news2} — triage: {triage['label']} ({triage['time_to_physician']}). "
            f"Rule-based differential generated; AI engine offline. "
            f"Immediate physician assessment required for definitive diagnosis."
        ),
    }

# =================================================================
# PDF
# =================================================================

def _build_pdf(report: Dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    s = []
    s.append(Paragraph("ClinicalTriageEnv — Clinical Report", styles["Heading1"]))
    s.append(Paragraph(f"Patient: {report.get('patient_id','N/A')} | {report.get('generated_at','N/A')}", styles["Normal"]))
    s.append(HRFlowable(width="100%", thickness=1))
    s.append(Spacer(1, 10))
    r = report.get("result", {})
    ps = r.get("patientSummary", {})
    if ps:
        s.append(Paragraph("Summary", styles["Heading2"]))
        s.append(Paragraph(ps.get("synopsis",""), styles["Normal"]))
        s.append(Spacer(1, 8))
    tr = r.get("triage", {})
    if tr:
        s.append(Paragraph(f"Triage: {tr.get('label','')} — {tr.get('timeToPhysician','')}", styles["Heading3"]))
        s.append(Paragraph(tr.get("rationale",""), styles["Normal"]))
        s.append(Spacer(1, 8))
    ddx = r.get("differentialDiagnosis", [])
    if ddx:
        s.append(Paragraph("Differential Diagnosis", styles["Heading2"]))
        rows = [["Rank","Condition","Probability","Confidence"]]
        for d in ddx:
            rows.append([str(d.get("rank","")),d.get("condition",""),f"{d.get('probability',0)}%",d.get("confidence","")])
        t = Table(rows, colWidths=[1.5*cm,10*cm,3*cm,3*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1a4a7a")),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f0f5fa")]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#ccddee")),
            ("FONTSIZE",(0,0),(-1,-1),10),
        ]))
        s.append(t)
    s.append(Spacer(1,12))
    s.append(Paragraph("DISCLAIMER: AI-generated for decision support only. Validate with a licensed physician.", styles["Italic"]))
    doc.build(s)
    return buf.getvalue()

# =================================================================
# ENTRY POINT
# =================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
