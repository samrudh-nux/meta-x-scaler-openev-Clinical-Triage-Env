from __future__ import annotations
import os
import sys
import json
import time
import argparse
import csv
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import ClinicalTriageEnv, TASK_REGISTRY
from models import TriageAction, MedicationSafetyAction, SepsisManagementAction


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
USE_COT      = os.getenv("USE_COT", "true").lower() == "true"
MAX_RETRIES  = 3
MAX_TOKENS   = 1400  # increased for CoT

TEMPERATURE_BY_DIFFICULTY = {
    "easy":   0.05,
    "medium": 0.08,
    "hard":   0.10,
}

ALL_TASKS = [
    "triage_easy", "triage_medium", "triage_hard",
    "med_safety_easy", "med_safety_medium", "med_safety_hard",
    "sepsis_easy", "sepsis_medium", "sepsis_hard",
]


# ─────────────────────────────────────────────────────────────────────────────
# System prompts (domain-specific)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "triage": """You are a board-certified emergency physician with 15+ years of triage experience.
You follow the Emergency Severity Index (ESI) v4 algorithm precisely.
KEY RULES:
  - ESI-1: Any immediate life-saving intervention needed (airway, resuscitation)
  - ESI-2: High-risk situation OR severe pain/distress OR vital sign abnormality
  - Never assign ESI ≥ 3 to a patient with suspected STEMI, stroke, or septic shock
  - Undertriage (assigning too-high ESI number) is more dangerous than overtriage
When responding, first THINK through the case, then output ONLY a valid JSON object.""",

    "medication_safety": """You are a clinical pharmacist and medication safety specialist.
You are expert in:
  - CYP450 drug metabolism (especially 3A4, 2C9, 2C19, 2D6)
  - Anticoagulation management and bleeding risks
  - Renal/hepatic dose adjustments
  - HIV antiretroviral drug interactions (ritonavir, cobicistat boosting)
  - Statin safety (myopathy, rhabdomyolysis)
  - Triple antithrombotic therapy risks
When responding, first THINK through each drug pair, then output ONLY a valid JSON object.""",

    "sepsis": """You are an intensivist and sepsis specialist following SSC 2021 guidelines.
KEY RULES:
  - Hour-1 bundle: cultures → antibiotics → lactate → fluids → vasopressors if MAP<65
  - Septic shock = sepsis + vasopressors needed + lactate >2 mmol/L despite fluids
  - ALWAYS check allergy history before selecting antibiotics
  - Norepinephrine is FIRST-LINE vasopressor
  - Blood cultures MUST be drawn BEFORE antibiotics (if <45 min delay)
  - 30 mL/kg crystalloid for MAP<65 or lactate ≥ 4 mmol/L
When responding, first THINK through the bundle checklist, then output ONLY a valid JSON object.""",
}


# ─────────────────────────────────────────────────────────────────────────────
# LLM client
# ─────────────────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy-key")


def call_llm(
    client: OpenAI,
    messages: List[Dict[str, str]],
    temperature: float = 0.05,
) -> str:
    """Call LLM with retry and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            if attempt < MAX_RETRIES - 1:
                print(f"    ⟳ Retry {attempt+1}/{MAX_RETRIES} after {wait}s ({e})")
                time.sleep(wait)
            else:
                print(f"    ✗ LLM call failed after {MAX_RETRIES} attempts: {e}")
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _vitals_block(v) -> str:
    map_val = int((v.systolic_bp + 2 * v.diastolic_bp) / 3)
    return (
        f"HR {v.heart_rate} bpm | BP {v.systolic_bp}/{v.diastolic_bp} mmHg "
        f"(MAP {map_val}) | SpO2 {v.spo2}% | Temp {v.temperature}°C | "
        f"RR {v.respiratory_rate}/min | GCS {v.glasgow_coma_scale}/15"
    )


def build_triage_prompt(obs, use_cot: bool) -> str:
    p, v = obs.patient, obs.patient.vitals
    meds = ", ".join(f"{m.name} {m.dose_mg}mg" for m in p.current_medications) or "None"
    cot_instruction = (
        "\n\nSTEP 1 — Think through the case (3-5 sentences covering vital signs, "
        "chief complaint, and ESI decision). Label this section 'REASONING:'.\n"
        "STEP 2 — Output ONLY a JSON object. Label it 'ACTION:'.\n"
        "Example format:\nREASONING: The patient presents with...\nACTION: {\"esi_level\": ...}"
        if use_cot else
        "\n\nRespond ONLY with a valid JSON object."
    )
    return f"""## EMERGENCY TRIAGE TASK — {obs.task_description[:80]}
**Patient**: {p.age}yo {p.sex} | ID: {p.patient_id}
**Chief Complaint**: {p.chief_complaint}
**Vitals**: {_vitals_block(v)}
**Symptoms**: {", ".join(p.symptoms)}
**Medical History**: {", ".join(p.medical_history)}
**Medications**: {meds}
**Allergies**: {", ".join(p.allergies) or "NKDA"}
**Labs**: {json.dumps(p.lab_results) if p.lab_results else "Pending"}
ESI Reference: 1=Resuscitation | 2=Emergent | 3=Urgent | 4=Less Urgent | 5=Non-Urgent
{cot_instruction}
JSON schema:
{{
  "esi_level": <int 1-5>,
  "rationale": "<clinical reasoning, min 30 words>",
  "recommended_immediate_interventions": ["<action>", ...]
}}"""


def build_med_safety_prompt(obs, use_cot: bool) -> str:
    p = obs.patient
    meds = "\n".join(
        f"  • {m.name} {m.dose_mg}mg {m.frequency} ({m.route})"
        for m in p.current_medications
    )
    drug_info = (
        f"\n**Drug Reference**:\n{json.dumps(obs.available_drug_info, indent=2)}"
        if obs.available_drug_info else ""
    )
    cot_instruction = (
        "\n\nSTEP 1 — Systematically review each drug pair for interactions (label 'REASONING:').\n"
        "STEP 2 — Output ONLY a JSON object (label 'ACTION:')."
        if use_cot else
        "\n\nRespond ONLY with a valid JSON object."
    )
    return f"""## MEDICATION SAFETY REVIEW
**Patient**: {p.age}yo {p.sex} | History: {", ".join(p.medical_history)}
**Allergies**: {", ".join(p.allergies) or "NKDA"}
**Chief Complaint**: {p.chief_complaint}
**Current Medications**:
{meds}
**Labs**: {json.dumps(p.lab_results, indent=2) if p.lab_results else "None"}
{drug_info}
Severity scale: safe < minor < moderate < major < critical
{cot_instruction}
JSON schema:
{{
  "flagged_interactions": ["<drug+drug: mechanism/risk>", ...],
  "flagged_contraindications": ["<drug_condition: reason>", ...],
  "flagged_dosing_errors": ["<drug: error and correction>", ...],
  "recommended_changes": ["<specific actionable change>", ...],
  "severity_assessment": "<safe|minor|moderate|major|critical>",
  "clinical_rationale": "<detailed explanation, min 50 words>"
}}"""


def build_sepsis_prompt(obs, use_cot: bool) -> str:
    p, v = obs.patient, obs.patient.vitals
    map_val = int((v.systolic_bp + 2 * v.diastolic_bp) / 3)
    meds = ", ".join(f"{m.name} {m.dose_mg}mg" for m in p.current_medications) or "None"
    cot_instruction = (
        "\n\nSTEP 1 — Check each bundle element and allergy profile (label 'REASONING:').\n"
        "STEP 2 — Output ONLY a JSON object (label 'ACTION:')."
        if use_cot else
        "\n\nRespond ONLY with a valid JSON object."
    )
    return f"""## SEPSIS RECOGNITION & MANAGEMENT
**Patient**: {p.age}yo {p.sex} | Arrived: {obs.time_elapsed_minutes}min ago
**Chief Complaint**: {p.chief_complaint}
**Allergies**: {", ".join(p.allergies) or "NKDA"}  ← CHECK BEFORE PRESCRIBING
**History**: {", ".join(p.medical_history)}
**Medications**: {meds}
**Vitals**: {_vitals_block(v)}
**MAP**: {map_val} mmHg  |  **qSOFA**: {obs.qsofa_score}/3
**Labs**: {json.dumps(p.lab_results, indent=2) if p.lab_results else "Pending"}
Hour-1 SSC Bundle checklist:
  ☐ Blood cultures × 2 BEFORE antibiotics
  ☐ Broad-spectrum antibiotics (check allergies!)
  ☐ Serum lactate (repeat if >2)
  ☐ 30 mL/kg IV crystalloid if MAP<65 or lactate≥4
  ☐ Vasopressors (NE first-line) if MAP<65 despite fluids
{cot_instruction}
JSON schema:
{{
  "sepsis_diagnosis": "<sepsis|septic_shock|SIRS_only|no_sepsis>",
  "blood_cultures_ordered": <bool>,
  "antibiotics_ordered": <bool>,
  "antibiotic_choice": "<drug_name|null>",
  "lactate_ordered": <bool>,
  "iv_fluid_bolus_ml": <int>,
  "vasopressor_ordered": <bool>,
  "vasopressor_choice": "<drug_name|null>",
  "source_control_identified": "<source|null>",
  "clinical_rationale": "<detailed reasoning, min 50 words>",
  "time_to_antibiotics_minutes": <int|null>
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction (multi-strategy)
# ─────────────────────────────────────────────────────────────────────────────

def extract_json(response: str) -> Optional[Dict]:
    """Try multiple strategies to extract JSON from LLM response."""
    if not response:
        return None

    # Strategy 1: Direct parse
    try:
        return json.loads(response.strip())
    except Exception:
        pass

    # Strategy 2: Extract after "ACTION:" label
    if "ACTION:" in response:
        after = response.split("ACTION:")[-1].strip()
        try:
            return json.loads(after)
        except Exception:
            pass

    # Strategy 3: Find JSON block with braces
    depth = 0
    start = -1
    for i, ch in enumerate(response):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(response[start:i+1])
                except Exception:
                    pass

    # Strategy 4: Strip markdown fences
    cleaned = response.strip()
    for fence in ["```json", "```JSON", "```"]:
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            try:
                return json.loads(cleaned.strip())
            except Exception:
                pass

    return None


def get_fallback_action(task_type: str) -> Dict:
    """Minimal valid fallback action when LLM fails."""
    if task_type == "triage":
        return {"esi_level": 3, "rationale": "Unable to parse response — default ESI-3 assigned.",
                "recommended_immediate_interventions": []}
    elif task_type == "medication_safety":
        return {"flagged_interactions": [], "flagged_contraindications": [],
                "flagged_dosing_errors": [], "recommended_changes": [],
                "severity_assessment": "moderate",
                "clinical_rationale": "Unable to parse response."}
    else:
        return {"sepsis_diagnosis": "sepsis", "blood_cultures_ordered": True,
                "antibiotics_ordered": True, "antibiotic_choice": "piperacillin_tazobactam",
                "lactate_ordered": True, "iv_fluid_bolus_ml": 2100,
                "vasopressor_ordered": False, "vasopressor_choice": None,
                "source_control_identified": None,
                "clinical_rationale": "Unable to parse response — default SSC bundle applied.",
                "time_to_antibiotics_minutes": 60}


def build_action(data: Dict, task_type: str):
    """Build typed Pydantic action from raw dict."""
    if task_type == "triage":
        # Normalise field aliases
        if "reasoning" in data and "rationale" not in data:
            data["rationale"] = data.pop("reasoning")
        if "immediate_actions" in data and "recommended_immediate_interventions" not in data:
            data["recommended_immediate_interventions"] = data.pop("immediate_actions")
        data.setdefault("rationale", "No rationale provided.")
        data.setdefault("recommended_immediate_interventions", [])
        return TriageAction(**{k: v for k, v in data.items()
                               if k in TriageAction.model_fields})
    elif task_type == "medication_safety":
        data.setdefault("clinical_rationale", "No rationale.")
        data.setdefault("severity_assessment", "moderate")
        return MedicationSafetyAction(**{k: v for k, v in data.items()
                                         if k in MedicationSafetyAction.model_fields})
    else:
        data.setdefault("clinical_rationale", "No rationale.")
        data.setdefault("sepsis_diagnosis", "sepsis")
        return SepsisManagementAction(**{k: v for k, v in data.items()
                                         if k in SepsisManagementAction.model_fields})


# ─────────────────────────────────────────────────────────────────────────────
# Single task runner
# ─────────────────────────────────────────────────────────────────────────────

def run_task(
    client: OpenAI,
    task_id: str,
    use_cot: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single task end-to-end and return full result dict."""
    start_time = time.time()
    meta       = TASK_REGISTRY[task_id]
    task_type  = meta["type"]
    difficulty = meta["difficulty"]
    temperature = TEMPERATURE_BY_DIFFICULTY.get(difficulty, 0.05)

    env = ClinicalTriageEnv(task_id=task_id)
    obs = env.reset()

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Task: {task_id}  ({difficulty})")
        print(f"  Patient: {obs.patient.age}yo {obs.patient.sex} — "
              f"{obs.patient.chief_complaint[:60]}")

    # Build prompt
    sys_prompt = SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["triage"])
    if task_type == "triage":
        user_prompt = build_triage_prompt(obs, use_cot)
    elif task_type == "medication_safety":
        user_prompt = build_med_safety_prompt(obs, use_cot)
    else:
        user_prompt = build_sepsis_prompt(obs, use_cot)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # LLM call
    raw_response = call_llm(client, messages, temperature=temperature)

    # Extract reasoning (CoT) and JSON
    reasoning = ""
    if use_cot and "REASONING:" in raw_response:
        parts = raw_response.split("ACTION:")
        reasoning = parts[0].replace("REASONING:", "").strip()
        action_part = parts[1].strip() if len(parts) > 1 else raw_response
    else:
        action_part = raw_response

    data = extract_json(action_part) or get_fallback_action(task_type)

    # Build and step
    try:
        action = build_action(data, task_type)
    except Exception as e:
        if verbose:
            print(f"    ✗ Action build error: {e}. Using fallback.")
        data   = get_fallback_action(task_type)
        action = build_action(data, task_type)

    obs_out, reward, done, info = env.step(action)
    elapsed = round(time.time() - start_time, 2)

    if verbose:
        passed_icon = "✅" if info.get("passed") else "❌"
        print(f"  Reward: {reward:.3f}  Grade: {info.get('grade', reward):.3f}  "
              f"{passed_icon}  ({elapsed}s)")
        if info.get("critical_errors"):
            for err in info["critical_errors"]:
                print(f"    ⚠  {err[:90]}")
        if info.get("component_scores"):
            for comp, score in info["component_scores"].items():
                bar = "█" * int(score * 8) + "░" * (8 - int(score * 8))
                print(f"    {comp:35s} {bar} {score:.3f}")

    return {
        "task_id":          task_id,
        "task_type":        task_type,
        "difficulty":       difficulty,
        "reward":           reward,
        "grade":            info.get("grade", reward),
        "passed":           info.get("passed", False),
        "total_reward":     info.get("total_reward", reward),
        "component_scores": info.get("component_scores", {}),
        "critical_errors":  info.get("critical_errors", []),
        "reasoning":        reasoning,
        "raw_action":       data,
        "feedback":         obs_out.feedback if hasattr(obs_out, "feedback") else "",
        "elapsed_seconds":  elapsed,
        "model":            MODEL_NAME,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    tasks: List[str],
    use_cot: bool = True,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    client  = get_client()
    results = []

    print(f"\n{'═'*60}")
    print(f"  ClinicalTriageEnv Benchmark")
    print(f"  Model:  {MODEL_NAME}")
    print(f"  Tasks:  {len(tasks)}")
    print(f"  CoT:    {'enabled' if use_cot else 'disabled'}")
    print(f"  Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}")

    for task_id in tasks:
        if task_id not in TASK_REGISTRY:
            print(f"  ⚠  Unknown task: {task_id} — skipped")
            continue
        result = run_task(client, task_id, use_cot=use_cot, verbose=verbose)
        results.append(result)
        time.sleep(0.5)  # brief rate-limit buffer

    # ── Summary ─────────────────────────────────────────────────────────────
    if results:
        rewards   = [r["reward"] for r in results]
        passed    = [r for r in results if r["passed"]]
        avg_r     = sum(rewards) / len(rewards)
        max_r     = max(rewards)
        min_r     = min(rewards)

        # Per-domain breakdown
        domains: Dict[str, List[float]] = {}
        for r in results:
            domains.setdefault(r["task_type"], []).append(r["reward"])

        print(f"\n{'═'*60}")
        print(f"  BENCHMARK RESULTS")
        print(f"{'─'*60}")
        print(f"  Tasks completed:   {len(results)}/{len(tasks)}")
        print(f"  Passed (≥0.60):    {len(passed)}/{len(results)}")
        print(f"  Average reward:    {avg_r:.3f}")
        print(f"  Best / Worst:      {max_r:.3f} / {min_r:.3f}")
        print(f"\n  Per-Domain Averages:")
        for domain, scores in domains.items():
            d_avg = sum(scores) / len(scores)
            bar   = "█" * int(d_avg * 20) + "░" * (20 - int(d_avg * 20))
            print(f"    {domain:20s} {bar} {d_avg:.3f}")
        print(f"\n  Per-Task Summary:")
        for r in results:
            icon = "✅" if r["passed"] else "❌"
            print(f"    {icon} {r['task_id']:25s} {r['reward']:.3f}  "
                  f"({r['difficulty']})")
        print(f"{'═'*60}")

    summary = {
        "model":        MODEL_NAME,
        "cot_enabled":  use_cot,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "tasks_run":    len(results),
        "tasks_passed": len(passed) if results else 0,
        "avg_reward":   round(avg_r, 4) if results else 0.0,
        "max_reward":   round(max_r, 4) if results else 0.0,
        "min_reward":   round(min_r, 4) if results else 0.0,
        "per_domain":   {d: round(sum(s)/len(s), 4) for d, s in domains.items()} if results else {},
        "results":      results,
    }

    # ── Export ───────────────────────────────────────────────────────────────
    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  💾  Results saved → {output_path}")

        # Also write CSV for easy analysis
        csv_path = output_path.replace(".json", ".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "task_id", "task_type", "difficulty",
                "reward", "grade", "passed", "elapsed_seconds",
            ])
            writer.writeheader()
            for r in results:
                writer.writerow({k: r[k] for k in writer.fieldnames})
        print(f"  💾  CSV saved      → {csv_path}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ClinicalTriageEnv Inference Benchmark v2")
    parser.add_argument("--tasks",  nargs="+", default=ALL_TASKS, help="Task IDs to run")
    parser.add_argument("--output", default=None,  help="Output JSON path")
    parser.add_argument("--model",  default=None,  help="Override MODEL_NAME")
    parser.add_argument("--no-cot", action="store_true", help="Disable chain-of-thought")
    parser.add_argument("--quiet",  action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.model:
        global MODEL_NAME
        MODEL_NAME = args.model

    run_benchmark(
        tasks=args.tasks,
        use_cot=not args.no_cot,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
