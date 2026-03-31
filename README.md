------
#Creator: Samrudh 
-https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u 
---
# 🏥 ClinicalTriageEnv — Clinical Decision Intelligence Engine

<div align="center">

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange?style=for-the-badge)](https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-green?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN3YxMGwxMCA1IDEwLTV2LTEweiIvPjwvc3ZnPg==)](https://openenv.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**A real-world reinforcement learning environment where AI agents learn life-or-death clinical decisions.**

*Emergency triage · Drug safety · Sepsis management · Partial scoring · Real undertriage penalties*

[**Live Demo**](https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u) · [**API Docs**](https://samrudh-nux-my-healthcare-ev4u.hf.space/docs) · [**OpenEnv Spec**](openenv.yaml)

</div>

---

## Overview

ClinicalTriageEnv is an **OpenEnv-compatible reinforcement learning environment** built on real clinical medicine. It challenges AI agents to make decisions that matter — triage classification, drug interaction detection, and sepsis bundle execution — with medically accurate patient scenarios, programmatic grading, and partial reward shaping.

Unlike toy environments, every scenario is grounded in actual clinical guidelines:

- **ESI triage** follows the Emergency Severity Index 5-tier system used in real EDs
- **Medication safety** tests CYP450 pharmacokinetics, contraindications, and polypharmacy risks
- **Sepsis management** enforces the Surviving Sepsis Campaign 2021 Hour-1 Bundle

The environment is designed for the **OpenEnv Hackathon 2025** and serves as both a training ground for clinical AI agents and a live demonstration dashboard for human learners.

---

## Features

### 🎯 9 Clinical Tasks across 3 Domains

| Domain | Easy | Medium | Hard |
|---|---|---|---|
| **ED Triage** | Ankle sprain (ESI 5) | ACS — chest pain (ESI 2) | Acute stroke on warfarin (ESI 1) |
| **Medication Safety** | Safe medication review | Triple antithrombotic therapy | HIV/Ritonavir + Simvastatin rhabdomyolysis |
| **Sepsis Management** | UTI sepsis, elderly | MRSA septic shock + PCN allergy | Post-op anastomotic leak + DIC + vancomycin allergy |

### ⚡ Key Capabilities

- **Partial scoring** — reward shaped across components, not just pass/fail
- **Undertriage penalty** — assigning a lower ESI than required carries safety penalties
- **Difficulty multiplier** — hard tasks award up to 1.3× reward (vs 0.8× for easy)
- **Session management** — each episode has a unique `session_id` for multi-agent use
- **OpenEnv compatible** — implements `reset()` / `step()` / `state()` interface
- **Multi-patient ICU dashboard** — 9-patient board with live ECG, timers, priority scores
- **Streaming AI analysis** — real Claude API integration for live clinical reasoning

---

## Quick Start

### 1. Use the Live API

The environment is hosted on HuggingFace Spaces. No setup required.

```python
import requests

BASE = "https://samrudh-nux-my-healthcare-ev4u.hf.space"

# Check health
print(requests.get(f"{BASE}/health").json())
# → {"status": "healthy", "tasks_available": 9}

# List all tasks
tasks = requests.get(f"{BASE}/tasks").json()
```

### 2. Run a Full Episode

```python
import requests

BASE = "https://samrudh-nux-my-healthcare-ev4u.hf.space"

# Step 1: Start an episode
reset = requests.post(f"{BASE}/reset", json={
    "task_id": "triage_hard"
}).json()

session_id = reset["session_id"]
patient    = reset["observation"]["patient"]

print(f"Patient: {patient['age']}yo {patient['sex']}")
print(f"Complaint: {patient['chief_complaint']}")
print(f"Vitals: BP {patient['vitals']['systolic_bp']}/{patient['vitals']['diastolic_bp']}")

# Step 2: Submit a clinical decision
result = requests.post(f"{BASE}/step", json={
    "session_id": session_id,
    "action": {
        "esi_level": 1,
        "rationale": "Acute ischaemic stroke — FAST positive, LKW <2h, on warfarin. BP 188/108. GCS 13. ESI-1: immediate life threat.",
        "recommended_immediate_interventions": [
            "stroke_alert",
            "CT_head",
            "INR_stat",
            "neurology_stat",
            "glucose_check"
        ]
    }
}).json()

print(f"\nReward:   {result['reward']:.3f}")
print(f"Passed:   {result['passed']}")
print(f"Grade:    {result['grade']:.3f}")
print(f"Scores:   {result['component_scores']}")
print(f"Feedback: {result['feedback'][:200]}")
```

---

## API Reference

Base URL: `https://samrudh-nux-my-healthcare-ev4u.hf.space`

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Ping — returns service status |
| `GET` | `/tasks` | List all 9 tasks with metadata |
| `GET` | `/tasks/{task_id}` | Get single task metadata |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit a clinical action |
| `GET` | `/state` | Get current episode state |
| `GET` | `/leaderboard` | Agent rankings |
| `DELETE` | `/session/{id}` | Clean up a session |
| `GET` | `/docs` | Swagger UI |

### `POST /reset`

```json
{
  "task_id": "triage_hard",
  "session_id": "optional-client-id"
}
```

**Valid task IDs** (use underscores):

```
triage_easy    triage_medium    triage_hard
med_safety_easy    med_safety_medium    med_safety_hard
sepsis_easy    sepsis_medium    sepsis_hard
```

**Returns:**

```json
{
  "session_id": "uuid",
  "task_id": "triage_hard",
  "observation": {
    "patient": {
      "patient_id": "T-005",
      "age": 72,
      "sex": "female",
      "chief_complaint": "Confusion and weakness...",
      "vitals": {
        "heart_rate": 88,
        "systolic_bp": 188,
        "diastolic_bp": 108,
        "spo2": 95,
        "temperature": 36.9,
        "respiratory_rate": 18,
        "glasgow_coma_scale": 13
      },
      "symptoms": ["acute confusion", "right arm weakness", "facial droop"],
      "medical_history": ["atrial fibrillation", "hypertension"],
      "current_medications": [{"name": "warfarin", "dose_mg": 5.0, ...}],
      "allergies": [],
      "lab_results": {"INR_pending": "unknown"}
    },
    "task_description": "...",
    "done": false
  },
  "task_info": {
    "name": "Emergency Triage - Hard",
    "type": "triage",
    "difficulty": "hard",
    "description": "..."
  }
}
```

### `POST /step` — Action Schemas

#### Triage Tasks

```json
{
  "session_id": "your-session-id",
  "action": {
    "esi_level": 1,
    "rationale": "Clinical reasoning — minimum 10 characters",
    "recommended_immediate_interventions": [
      "stroke_alert", "CT_head", "INR_stat", "neurology_stat"
    ]
  }
}
```

| ESI | Label | Clinical Meaning |
|---|---|---|
| 1 | Resuscitation | Immediate life threat — requires immediate physician |
| 2 | Emergent | High risk situation — should be seen within 15 min |
| 3 | Urgent | Stable but needs multiple resources |
| 4 | Less Urgent | Stable, needs one resource |
| 5 | Non-Urgent | Stable, no resources needed |

#### Medication Safety Tasks

```json
{
  "session_id": "your-session-id",
  "action": {
    "flagged_interactions": ["simvastatin+clarithromycin", "ritonavir+simvastatin"],
    "flagged_contraindications": ["simvastatin_CYP3A4_inhibitor"],
    "flagged_dosing_errors": ["simvastatin_80mg_with_CYP3A4_inhibitor"],
    "recommended_changes": [
      "discontinue_simvastatin",
      "switch_to_pravastatin",
      "monitor_CK_levels"
    ],
    "severity_assessment": "critical",
    "clinical_rationale": "Ritonavir is a potent CYP3A4 inhibitor. Co-administration with simvastatin increases statin AUC by up to 3000%, causing rhabdomyolysis..."
  }
}
```

Valid `severity_assessment` values: `"critical"` · `"major"` · `"moderate"` · `"minor"` · `"safe"`

#### Sepsis Management Tasks

```json
{
  "session_id": "your-session-id",
  "action": {
    "sepsis_diagnosis": "septic_shock",
    "blood_cultures_ordered": true,
    "antibiotics_ordered": true,
    "antibiotic_choice": "piperacillin_tazobactam",
    "lactate_ordered": true,
    "iv_fluid_bolus_ml": 2100,
    "vasopressor_ordered": true,
    "vasopressor_choice": "norepinephrine",
    "source_control_identified": "UTI",
    "clinical_rationale": "qSOFA 3. MAP 58 — below target. Lactate 4.8 = tissue hypoperfusion. Hour-1 bundle initiated...",
    "time_to_antibiotics_minutes": 35
  }
}
```

Valid `sepsis_diagnosis` values: `"sepsis"` · `"septic_shock"` · `"SIRS_only"` · `"no_sepsis"`

### Step Response

```json
{
  "session_id": "uuid",
  "reward": 0.847,
  "done": true,
  "score": 0.847,
  "passed": true,
  "grade": 0.651,
  "component_scores": {
    "esi_accuracy": 1.0,
    "interventions": 0.8,
    "rationale_quality": 0.75
  },
  "critical_errors": [],
  "feedback": "✅ PASSED — ESI 1 correct. Stroke protocol activated. INR check critical given warfarin use...",
  "total_reward": 0.847,
  "task_id": "triage_hard",
  "difficulty": "hard"
}
```

---

## Reward Function

```
reward = (base_score - safety_penalty + efficiency_bonus) × difficulty_multiplier
```

| Component | Value | Notes |
|---|---|---|
| Base score | 0.0 – 1.0 | From clinical grader |
| Safety penalty | −0.3 per critical error | Patient safety violations |
| Efficiency bonus | +0.05 per unused step | Solved faster = slight bonus |
| Difficulty multiplier | ×0.8 / ×1.0 / ×1.3 | Easy / Medium / Hard |
| Passing threshold | ≥ 0.60 | `passed: true` |
| Max possible reward | 1.5 | Hard task, solved in 1 step, no errors |

### Undertriage Penalty

Assigning an ESI level **higher than required** (i.e. lower priority) is the most dangerous error in real triage. The grader applies a dedicated penalty on top of the accuracy deduction for any undertriage:

```
undertriage_penalty = −0.15 (applied per ESI level above correct)
```

---

## Environment Architecture

```
┌─────────────┐    action     ┌──────────────┐    grade     ┌─────────────────┐
│             │ ──────────→   │              │ ──────────→  │                 │
│  AI Agent   │               │   step()     │              │  Grader Engine  │
│             │ ←──────────   │  FastAPI     │ ←──────────  │  (per task)     │
│             │  observation  │              │   reward     │                 │
└─────────────┘               └──────────────┘              └─────────────────┘
      ↑                               │
      └──────────── state ────────────┘
```

### Components

```
ClinicalTriageEnv/
├── app.py              ← FastAPI server — all routes
├── environment.py      ← OpenEnv core: reset(), step(), state()
├── models.py           ← Pydantic action/observation schemas
├── scenarios.py        ← 9 medically accurate patient scenarios
├── graders.py          ← Programmatic clinical scoring engines
├── inference.py        ← LLM inference utilities
├── openenv.yaml        ← OpenEnv spec declaration
├── index.html          ← ICU multi-patient dashboard (frontend)
├── requirements.txt    ← Python dependencies
└── Dockerfile          ← HuggingFace Spaces deployment
```

---

## Grading Criteria

### Triage Tasks

| Criterion | Weight | Notes |
|---|---|---|
| ESI level — exact match | High | Full credit |
| ESI level — off by ±1 | Medium | Partial credit |
| ESI level — off by ≥2 | None | Zero base + undertriage penalty if higher |
| Critical interventions | Medium | Must identify time-critical actions (ECG, CT head, INR, etc.) |
| Clinical rationale | Low | Quality of reasoning narrative |

### Medication Safety Tasks

| Criterion | Weight | Notes |
|---|---|---|
| Critical interaction identified | High | CYP3A4, contraindications |
| Mechanism explained | Medium | Pharmacokinetic reasoning |
| Safe alternative named | Medium | Correct substitute (e.g. pravastatin) |
| Severity correctly assessed | Low | critical / major / moderate |

### Sepsis Tasks

| Criterion | Weight | Notes |
|---|---|---|
| Correct diagnosis | High | sepsis vs septic_shock distinction |
| Hour-1 bundle completeness | High | All 5 elements: cultures, abx, lactate, fluid, vasopressors |
| Allergy-safe antibiotic | High | Hard tasks penalise contraindicated agents |
| Time to antibiotics | Medium | Faster = better (target ≤60 min) |
| Source control identified | Low | UTI / pneumonia / abdominal |

---

## Running Locally

### Prerequisites

- Python 3.10+
- Docker (optional, for full environment parity)

### Installation

```bash
# Clone
git clone https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u
cd my-healthcare-ev4u

# Install dependencies
pip install -r requirements.txt

# Run server
python app.py
# → Server starts on http://localhost:7860
```

### With Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

The ICU dashboard will be available at `http://localhost:7860`.

---

## Python Client Example — Full Multi-Task Benchmark

```python
import requests

BASE = "https://samrudh-nux-my-healthcare-ev4u.hf.space"

TASK_ACTIONS = {
    "triage_easy": {
        "esi_level": 5,
        "rationale": "Non-urgent ankle sprain. Normal vitals. ESI-5: no resources needed.",
        "recommended_immediate_interventions": []
    },
    "triage_medium": {
        "esi_level": 2,
        "rationale": "High-risk ACS presentation. Crushing chest pain + radiation + diaphoresis. ESI-2 emergent.",
        "recommended_immediate_interventions": ["ECG", "aspirin_325mg", "IV_access", "troponin"]
    },
    "triage_hard": {
        "esi_level": 1,
        "rationale": "Acute stroke — FAST positive, LKW <2h, INR unknown on warfarin. ESI-1 immediate.",
        "recommended_immediate_interventions": ["stroke_alert", "CT_head", "INR_stat", "neurology_stat"]
    },
    "sepsis_easy": {
        "sepsis_diagnosis": "septic_shock",
        "blood_cultures_ordered": True,
        "antibiotics_ordered": True,
        "antibiotic_choice": "piperacillin_tazobactam",
        "lactate_ordered": True,
        "iv_fluid_bolus_ml": 1950,
        "vasopressor_ordered": True,
        "vasopressor_choice": "norepinephrine",
        "source_control_identified": "UTI",
        "clinical_rationale": "qSOFA 3. MAP <65. SSC Hour-1 bundle initiated immediately.",
        "time_to_antibiotics_minutes": 40
    },
}

results = {}
for task_id, action in TASK_ACTIONS.items():
    # Reset
    r = requests.post(f"{BASE}/reset", json={"task_id": task_id}).json()
    session_id = r["session_id"]

    # Step
    result = requests.post(f"{BASE}/step", json={
        "session_id": session_id,
        "action": action
    }).json()

    results[task_id] = {
        "reward": result["reward"],
        "passed": result["passed"],
        "grade":  result["grade"],
    }
    print(f"{task_id:25s}  reward={result['reward']:.3f}  passed={result['passed']}")

avg = sum(r["reward"] for r in results.values()) / len(results)
print(f"\nAverage reward: {avg:.3f}")
```

---

## ICU Dashboard

The environment ships with a full **multi-patient ICU Command Board** as `index.html`.

### Features

- **9-patient simultaneous board** — all scenarios displayed at once
- **Status indicators** — 🔴 Critical / 🟡 At Risk / 🟢 Stable with real-time colour coding
- **Priority scores** — 1–5 scale, colour-coded red → green
- **Countdown timers** — Hour-1 bundle, door-to-ECG, medication review deadlines
- **Live ECG waveforms** — animated per-patient, A-fib shows correctly as irregularly irregular
- **Drug interaction scanner** — table view with CYP450 mechanism and severity rating
- **Streaming AI analysis** — real Claude API integration for live clinical reasoning per patient
- **Sepsis bundle checklist** — interactive SSC 2021 Hour-1 checklist with progress bar
- **ACS/STEMI protocol checklist** — door-to-balloon tracking
- **Episode log** — timestamped history of all decisions and rewards
- **Live leaderboard** — your session score ranked against published agent benchmarks
- **Reward graph** — Chart.js plot of reward trajectory over episodes
- **API Reference** — full endpoint documentation with copyable Python examples

---

## Scenario Reference

### Triage Scenarios

| ID | Patient | Age/Sex | Key Finding | Correct ESI | Difficulty |
|---|---|---|---|---|---|
| `triage_easy` | Ankle sprain | 45M | Normal vitals, non-urgent | 5 | Easy |
| `triage_medium` | ACS / chest pain | 67M | HR 102, BP 148/92, diaphoresis | 2 | Medium |
| `triage_hard` | Acute stroke | 72F | BP 188/108, GCS 13, A-fib, warfarin | 1 | Hard |

### Medication Safety Scenarios

| ID | Patient | Key Interaction | Severity | Difficulty |
|---|---|---|---|---|
| `med_safety_easy` | Routine HTN review | None — safe combination | Safe | Easy |
| `med_safety_medium` | Post-PCI triple therapy | Rivaroxaban + Warfarin (duplication) | Critical | Medium |
| `med_safety_hard` | HIV + rhabdomyolysis | Ritonavir × Simvastatin (CYP3A4) | Critical | Hard |

### Sepsis Scenarios

| ID | Patient | Source | Key Complication | Difficulty |
|---|---|---|---|---|
| `sepsis_easy` | Elderly nursing home | UTI | MAP 88/54, PCN allergy (rash only) | Easy |
| `sepsis_medium` | 83F, MRSA bacteraemia | Blood | PCN allergy — anaphylaxis | Medium |
| `sepsis_hard` | 58M, post-op day 2 | Anastomotic leak | DIC + AKI + ARDS + vancomycin allergy | Hard |

---

## Baseline Performance

| Agent | Avg Reward | Tasks Passed | Notes |
|---|---|---|---|
| Claude Opus 4 (clinical prompt) | 0.947 | 9/9 | Best published result |
| GPT-4o (med tuned) | 0.891 | 9/9 | Strong on medication tasks |
| Gemini 1.5 Pro | 0.843 | 9/9 | |
| Llama 3 70B | 0.812 | 9/9 | |
| MediTron 70B | 0.789 | 7/9 | Fails complex sepsis |
| **Rule-based baseline** | **0.580** | **5/9** | Threshold for comparison |

---

## Technical Details

### Stack

| Component | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| Data validation | Pydantic v2 |
| Hosting | HuggingFace Spaces (Docker) |
| Frontend | Vanilla HTML/CSS/JS + Chart.js |
| AI integration | Anthropic Claude API (streaming) |
| ECG simulation | Canvas API (real-time animation) |

### CORS

All origins are allowed (`*`). The environment is designed for open agent access.

### Rate Limits

HuggingFace Spaces imposes standard rate limits. For high-volume training runs (>1000 episodes), consider deploying locally with Docker.

---

## Citation

If you use ClinicalTriageEnv in your research or competition submission, please cite:

```bibtex
@software{clinicaltriagenenv2025,
  author    = {samrudh-nux},
  title     = {ClinicalTriageEnv: A Clinical Decision Intelligence Environment for RL},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u},
  note      = {OpenEnv Hackathon 2025}
}
```

---

## Medical Disclaimer

> This environment is designed for **AI research and education only**. The clinical scenarios, scoring criteria, and grading logic are based on published guidelines (ESI v4, SSC 2021, etc.) but are simplified for machine learning purposes. Do **not** use this system for real clinical decision-making.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for **OpenEnv Hackathon 2025** · MIT License · [HuggingFace Space](https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u)

</div>
