from __future__ import annotations
import os
import sys
import json
import time
import traceback
from typing import Any, Dict, List, Optional

from openai import OpenAI

# =============================================================================
# READ ENV VARS — read at module level, but DO NOT exit or create client here.
# app.py imports this module at startup; env vars may not be set yet then.
# =============================================================================
def _env_base_url() -> str:
    return os.environ.get("API_BASE_URL", "").strip()

def _env_api_key() -> str:
    return os.environ.get("API_KEY", "").strip()

def _env_model() -> str:
    return os.environ.get("MODEL_NAME", "gpt-4o-mini").strip()

MODEL_NAME = _env_model()

# =============================================================================
# LAZY CLIENT — created fresh each call so env vars are always current
# =============================================================================
_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    """Return a cached OpenAI client pointed at the validator's proxy."""
    global _client
    base_url = _env_base_url()
    api_key  = _env_api_key()
    if _client is None:
        if not base_url:
            raise RuntimeError("API_BASE_URL environment variable is not set.")
        if not api_key:
            raise RuntimeError("API_KEY environment variable is not set.")
        _client = OpenAI(base_url=base_url, api_key=api_key)
        print(f"[CLIENT] Initialized → base_url={base_url} key={'*'*8}", flush=True)
    return _client

# =============================================================================
# TASKS — fully self-contained, zero dependency on environment.py / models.py
# =============================================================================
TASKS: List[Dict[str, Any]] = [
    {
        "task_id":   "triage_easy",
        "task_type": "triage",
        "difficulty": "easy",
        "system": (
            "You are a board-certified emergency physician. Follow ESI v4 triage levels. "
            "ESI-1=immediate life threat, ESI-2=high risk/cannot wait, "
            "ESI-3=stable but needs resources, ESI-4=stable one resource, ESI-5=stable no resources. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Patient: 45-year-old male. Chief complaint: crushing chest pain radiating to left arm, "
            "diaphoresis, shortness of breath x 30 minutes. Vitals: BP 90/60, HR 112, SpO2 94%.\n\n"
            'Respond with JSON: {"esi_level": int, "primary_diagnosis": str, "immediate_actions": [str], "rationale": str}'
        ),
        "keywords": ["esi_level", "1", "stemi", "acs", "ecg", "cardiac", "immediate", "aspirin"],
    },
    {
        "task_id":   "triage_medium",
        "task_type": "triage",
        "difficulty": "medium",
        "system": (
            "You are a board-certified emergency physician. Follow ESI v4 triage levels. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Patient: 7-year-old child. Chief complaint: sudden stridor, drooling, tripod positioning, "
            "fever 39.8C. Child is anxious, leaning forward, refusing to lie down.\n\n"
            'Respond with JSON: {"esi_level": int, "primary_diagnosis": str, '
            '"immediate_actions": [str], "do_not_do": [str], "rationale": str}'
        ),
        "keywords": ["esi_level", "1", "epiglottitis", "airway", "ent", "do not", "agitate", "calm"],
    },
    {
        "task_id":   "triage_hard",
        "task_type": "triage",
        "difficulty": "hard",
        "system": (
            "You are a board-certified emergency physician. Follow ESI v4 triage levels. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Patient: 67-year-old female. Sudden onset severe headache, "
            "'worst headache of my life', 10/10 pain, photophobia, neck stiffness, nausea. "
            "Vitals: BP 178/102, HR 88, Temp 37.8C, GCS 14.\n\n"
            'Respond with JSON: {"esi_level": int, "primary_diagnosis": str, "must_rule_out": str, '
            '"immediate_workup": [str], "rationale": str}'
        ),
        "keywords": ["esi_level", "1", "subarachnoid", "sah", "thunderclap", "ct", "lumbar", "aneurysm"],
    },
    {
        "task_id":   "med_safety_easy",
        "task_type": "medication_safety",
        "difficulty": "easy",
        "system": (
            "You are a clinical pharmacist expert in drug interactions, CYP450 metabolism, "
            "renal/hepatic dosing adjustments, and medication safety. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Patient on warfarin (INR therapeutic) is prescribed fluconazole 150mg x1 for vaginal candidiasis.\n\n"
            'Respond with JSON: {"interaction_severity": str, "mechanism": str, '
            '"recommendation": str, "monitoring": str, "alternative": str}'
        ),
        "keywords": ["cyp", "2c9", "warfarin", "inr", "interaction", "monitor", "bleeding", "dose"],
    },
    {
        "task_id":   "med_safety_medium",
        "task_type": "medication_safety",
        "difficulty": "medium",
        "system": (
            "You are a clinical pharmacist expert in drug interactions, CYP450 metabolism, "
            "renal/hepatic dosing adjustments, and medication safety. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "HIV patient on ritonavir-boosted regimen is prescribed simvastatin 40mg for hyperlipidemia. "
            "Patient also has CKD stage 3 (eGFR 38 mL/min).\n\n"
            'Respond with JSON: {"interaction_severity": str, "mechanism": str, "recommendation": str, '
            '"safe_alternative": str, "renal_consideration": str}'
        ),
        "keywords": ["cyp", "3a4", "statin", "myopathy", "rhabdomyolysis", "contraindicated", "pravastatin", "renal"],
    },
    {
        "task_id":   "med_safety_hard",
        "task_type": "medication_safety",
        "difficulty": "hard",
        "system": (
            "You are a clinical pharmacist expert in drug interactions, CYP450 metabolism, "
            "renal/hepatic dosing adjustments, and medication safety. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Post-MI patient on aspirin 81mg + clopidogrel 75mg receives a drug-eluting stent. "
            "Also has atrial fibrillation requiring anticoagulation. "
            "Physician wants to add rivaroxaban 20mg daily.\n\n"
            'Respond with JSON: {"bleeding_risk": str, "recommendation": str, '
            '"duration_dapt": str, "monitoring_parameters": [str], "rationale": str}'
        ),
        "keywords": ["triple", "antiplatelet", "anticoagul", "bleeding", "dapt", "duration", "ppi", "monitor"],
    },
    {
        "task_id":   "sepsis_easy",
        "task_type": "sepsis",
        "difficulty": "easy",
        "system": (
            "You are a sepsis specialist following Surviving Sepsis Campaign 2021 guidelines. "
            "Hour-1 bundle: blood cultures -> antibiotics -> lactate -> 30mL/kg crystalloid -> vasopressors if MAP<65. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Patient: 55-year-old diabetic male. Fever 39.2C, confusion (new), HR 118, RR 24, BP 88/56, SpO2 96%. "
            "Appears lethargic. No known allergies.\n\n"
            'Respond with JSON: {"sepsis_criteria_met": bool, "probable_source": str, '
            '"hour1_bundle": [str], "antibiotic_choice": str, "vasopressor": str}'
        ),
        "keywords": ["sepsis", "bundle", "culture", "antibiotic", "fluid", "lactate", "norepinephrine", "vasopressor"],
    },
    {
        "task_id":   "sepsis_medium",
        "task_type": "sepsis",
        "difficulty": "medium",
        "system": (
            "You are a sepsis specialist following Surviving Sepsis Campaign 2021 guidelines. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Patient: 72-year-old female, nursing home resident. Altered mental status x 2 days, "
            "fever 38.9C, HR 104, BP 96/58, RR 22, urine cloudy/foul-smelling. "
            "History: penicillin allergy (anaphylaxis), CKD stage 4 (eGFR 22).\n\n"
            'Respond with JSON: {"sepsis_source": str, "antibiotic_choice": str, '
            '"dose_adjustment_reason": str, "fluid_strategy": str, "monitoring": [str]}'
        ),
        "keywords": ["uti", "urosepsis", "penicillin", "allergy", "renal", "dose", "fluid", "culture", "monitor"],
    },
    {
        "task_id":   "sepsis_hard",
        "task_type": "sepsis",
        "difficulty": "hard",
        "system": (
            "You are a sepsis specialist following Surviving Sepsis Campaign 2021 guidelines. "
            "Respond ONLY with valid JSON and nothing else."
        ),
        "user": (
            "Patient: 38-year-old immunocompromised (post-bone marrow transplant, on tacrolimus). "
            "Neutropenic fever (ANC 0.1), HR 130, BP 78/44, Temp 40.1C, lactate 4.8 mmol/L. "
            "Central line present x 14 days. Allergy: sulfa drugs.\n\n"
            'Respond with JSON: {"neutropenic_sepsis": bool, "crbsi_risk": str, '
            '"empiric_antibiotics": [str], "antifungal_needed": bool, '
            '"additional_interventions": [str], "vasopressor_choice": str}'
        ),
        "keywords": ["neutropenic", "crbsi", "antifungal", "broad", "piperacillin", "meropenem", "norepinephrine", "line"],
    },
]

# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

def emit_start(task_id: str) -> None:
    print(f"[START] task={task_id} model={_env_model()}", flush=True)

def emit_step(task_id: str, step: int, reward: float, info: str = "") -> None:
    print(f"[STEP] task={task_id} step={step} reward={reward:.4f} info={info}", flush=True)

def emit_end(task_id: str, score: float, steps: int, elapsed: float, passed: bool) -> None:
    print(
        f"[END] task={task_id} score={score:.4f} steps={steps} "
        f"passed={str(passed).lower()} elapsed={elapsed:.2f}s",
        flush=True,
    )

# =============================================================================
# LLM CALL — uses lazy client; safe to call only after env vars are set
# =============================================================================

def call_llm(system: str, user: str, task_id: str) -> str:
    for attempt in range(3):
        try:
            c = get_client()
            response = c.chat.completions.create(
                model=_env_model(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.05,
                max_tokens=800,
            )
            content = response.choices[0].message.content.strip()
            print(
                f"[STEP] task={task_id} llm_call=ok attempt={attempt + 1} "
                f"tokens={response.usage.total_tokens}",
                flush=True,
            )
            return content
        except Exception as exc:
            wait = 2 ** attempt
            print(f"[STEP] task={task_id} llm_call=error attempt={attempt + 1} err={exc} wait={wait}s", flush=True)
            if attempt < 2:
                time.sleep(wait)
    return ""

# =============================================================================
# GRADER
# =============================================================================

def grade(task: Dict[str, Any], response_text: str) -> float:
    text = response_text.lower()
    keywords = task["keywords"]
    hits = sum(1 for kw in keywords if kw.lower() in text)
    raw = hits / max(1, len(keywords))
    score = raw / 0.40
    # Score must be strictly between 0 and 1 (exclusive) per validator rules
    score = max(0.0001, min(0.9999, score))
    return round(score, 4)

# =============================================================================
# SINGLE TASK RUNNER
# =============================================================================

def run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    task_id = task["task_id"]
    start_time = time.time()
    emit_start(task_id)
    response_text = call_llm(task["system"], task["user"], task_id)
    score = grade(task, response_text)
    elapsed = time.time() - start_time
    passed = score >= 0.5
    emit_step(task_id, step=1, reward=score, info=f"len={len(response_text)}")
    emit_end(task_id, score=score, steps=1, elapsed=elapsed, passed=passed)
    return {
        "task_id":    task_id,
        "task_type":  task["task_type"],
        "difficulty": task["difficulty"],
        "score":      score,
        "passed":     passed,
        "elapsed":    round(elapsed, 2),
    }

# =============================================================================
# MAIN — sys.exit lives HERE only, never at module level
# =============================================================================

def main() -> None:
    base_url = _env_base_url()
    api_key  = _env_api_key()
    model    = _env_model()

    if not base_url:
        print("[FATAL] API_BASE_URL is not set.", flush=True)
        sys.exit(1)
    if not api_key:
        print("[FATAL] API_KEY is not set.", flush=True)
        sys.exit(1)

    print("=" * 60, flush=True)
    print("ClinicalTriageEnv Benchmark", flush=True)
    print(f"  model       : {model}", flush=True)
    print(f"  api_base    : {base_url}", flush=True)
    print(f"  api_key_set : True", flush=True)
    print(f"  num_tasks   : {len(TASKS)}", flush=True)
    print("=" * 60, flush=True)

    results = []
    total_score = 0.0

    for task in TASKS:
        try:
            result = run_task(task)
            results.append(result)
            total_score += result["score"]
        except Exception as exc:
            task_id = task.get("task_id", "unknown")
            print(f"[ERROR] task={task_id} error={exc}", flush=True)
            traceback.print_exc(file=sys.stderr)
            results.append({
                "task_id":    task_id,
                "task_type":  task.get("task_type", "unknown"),
                "difficulty": task.get("difficulty", "unknown"),
                "score":      0.0,
                "passed":     False,
                "elapsed":    0.0,
                "error":      str(exc),
            })

    avg_score = total_score / len(TASKS) if TASKS else 0.0
    passed_count = sum(1 for r in results if r.get("passed"))

    print("=" * 60, flush=True)
    print(f"RESULTS  tasks={len(results)}  passed={passed_count}  avg_score={avg_score:.4f}", flush=True)
    for r in results:
        flag = "PASS" if r.get("passed") else "FAIL"
        print(f"  {flag}  {r['task_id']:30s}  score={r['score']:.4f}", flush=True)
    print("=" * 60, flush=True)

    print(json.dumps({
        "model":        model,
        "total_tasks":  len(results),
        "passed_tasks": passed_count,
        "avg_score":    round(avg_score, 4),
        "results":      results,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
