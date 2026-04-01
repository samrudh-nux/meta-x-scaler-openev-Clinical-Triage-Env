---
title: ClinicalTriageEnv
emoji: 🏥
colorFrom: blue
colorTo: teal
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - healthcare
  - reinforcement-learning
  - clinical-decision-support
  - triage
  - sepsis
  - medication-safety
  - agentic-ai
  - multi-agent
  - rl-training
short_description: "The first OpenEnv environment where AI agents learn life-or-death clinical decisions"
---

<div align="center">

# 🏥 ClinicalTriageEnv

### *The first OpenEnv environment where AI agents learn to make life-or-death clinical decisions*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.1-0066ff?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyQzYuNDggMiAyIDYuNDggMiAxMnM0LjQ4IDEwIDEwIDEwIDEwLTQuNDggMTAtMTBTMTcuNTIgMiAxMiAyem0tMiAxNEg4di02aDJ2NnptNCAwaC0ydi02aDJ2NnoiLz48L3N2Zz4=)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![HF Space](https://img.shields.io/badge/🤗_Space-Running-ff6b35?style=flat-square)](https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u2)

<br/>

**250,000 people die from preventable medical errors every year in the US alone.**  
**No RL environment has ever trained AI agents to prevent them — until now.**

<br/>

| 🚨 ED Triage | 💊 Medication Safety | 🔴 Sepsis Management |
|:---:|:---:|:---:|
| ESI 1–5 assignment | Drug interaction detection | Hour-1 SSC bundle |
| Undertriage penalties | CYP450 pharmacokinetics | Vasopressor decisions |
| Stroke · ACS · Trauma | Rhabdomyolysis scenarios | Multi-organ failure |

<br/>

</div>

---

## ✦ Why This Environment Exists

Every other OpenEnv environment trains agents to play games, write code, or browse websites.

**ClinicalTriageEnv trains agents to reason like physicians.**

The gap it fills is real: emergency departments worldwide use the ESI triage algorithm for **140 million+ ED visits annually**. Sepsis kills **11 million people per year** globally — and 7% more patients die for every hour of delayed treatment. Drug-drug interactions cause **125,000 deaths annually** in the US.

An AI agent trained on this environment learns the same clinical reasoning framework used by actual clinicians — with graders that enforce the same safety standards medical licensing boards do.

---

## ✦ What Makes This Different

| Feature | Other OpenEnv Environments | ClinicalTriageEnv |
|---|---|---|
| **Domain** | Games, coding, web | Real clinical medicine |
| **Grader Type** | String match / unit tests | Multi-component medical graders |
| **Safety Penalties** | None | Undertriage, allergy violations, missed vasopressors |
| **Partial Credit** | Binary pass/fail | 6-component weighted scoring |
| **AI Analysis** | None | 3 specialist agents run in parallel |
| **Patient Simulation** | Static state | Vitals deteriorate in real-time |
| **Risk Engine** | None | Mortality %, legal risk, delay penalty |
| **Difficulty** | Fixed | Easy → Medium → Hard with baseline scores |
| **Medical Accuracy** | N/A | ESI Handbook v4, SSC 2021, ACC/AHA 2023 |

---

## ✦ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        AI Agent (RL Policy)                      │
└─────────────────────────────┬────────────────────────────────────┘
                              │  action  ↕  observation + reward
┌─────────────────────────────▼────────────────────────────────────┐
│                     ClinicalTriageEnv v3.0                       │
│                                                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌─────────────────────┐ │
│  │  ED Triage  │  │ Medication Safety │  │  Sepsis Management  │ │
│  │  3 tasks    │  │    3 tasks        │  │      3 tasks        │ │
│  └──────┬──────┘  └────────┬─────────┘  └──────────┬──────────┘ │
│         └─────────────────┬┘───────────────────────┘            │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              Programmatic Grader Suite                     │  │
│  │   TriageGrader · MedicationSafetyGrader · SepsisGrader     │  │
│  │   Partial credit · Safety penalties · Deterministic        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                           ▼                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  Risk Engine │  │ Multi-Agent  │  │ Patient Simulation   │   │
│  │  Mortality % │  │  3 AI Agents │  │ Vitals deteriorate   │   │
│  │  Legal risk  │  │  in parallel │  │ with time + errors   │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│                           ▼                                      │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │         FastAPI Server · HTTP + WebSocket · Docker         │  │
│  │    /reset  /step  /state  /analyze  /benchmark  /report   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✦ The 9 Clinical Tasks

### 🚨 Domain 1: Emergency Department Triage

Agents assign an **Emergency Severity Index (ESI)** level — the algorithm used in 90%+ of US emergency departments — and identify immediate interventions.

| Task ID | Scenario | Correct ESI | Key Challenge | Baseline |
|---|---|---|---|---|
| `triage_easy` | Ankle sprain, stable vitals | ESI-5 | Validates baseline understanding | **78%** |
| `triage_medium` | 67yo crushing chest pain, diaphoresis, radiation | ESI-2 | ACS recognition, STEMI workup | **62%** |
| `triage_hard` | Acute stroke on warfarin, GCS 13, FAST+ | ESI-1 | Anticoagulation + door-to-CT <25min | **45%** |

**Grading weights:** ESI accuracy 65% · Rationale quality 20% · Critical interventions 15%  
**Safety rule:** ESI-1/2 patient assigned ESI-3+ triggers undertriage penalty (−40%) — this is a patient safety critical error in emergency medicine.

---

### 💊 Domain 2: Medication Safety Review

Agents review complete patient medication lists and identify drug interactions, contraindications, dosing errors, and overall severity — exactly as a clinical pharmacist does before dispensing.

| Task ID | Scenario | Key Finding | Severity | Baseline |
|---|---|---|---|---|
| `med_safety_easy` | Amlodipine + atorvastatin + aspirin 81mg | No significant interactions | Safe | **90%** |
| `med_safety_medium` | Post-MI triple antithrombotic (warfarin + aspirin + clopidogrel), CKD, diabetes | Triple therapy = major GI bleed risk | Major | **58%** |
| `med_safety_hard` | HIV patient on ritonavir + simvastatin 80mg, presenting with CK=48,000 | Ritonavir inhibits CYP3A4 → simvastatin toxicity → rhabdomyolysis → AKI | **Critical** | **31%** |

**Grading weights:** Interaction detection 25% · Contraindication detection 20% · Dosing errors 15% · Severity classification 15% · Clinical rationale 15% · False-positive penalty 10%  
**Hard safety rule:** Allergy violation (prescribing penicillin to PCN-allergic patient) scores exactly 0.0 on antibiotic component — no partial credit.

---

### 🔴 Domain 3: Sepsis Recognition & Management

Agents must recognise sepsis/septic shock using **Sepsis-3 criteria** and execute the complete **Surviving Sepsis Campaign Hour-1 Bundle** — the evidence-based protocol that reduces sepsis mortality by 25–40%.

| Task ID | Scenario | Diagnosis | Key Complexity | Baseline |
|---|---|---|---|---|
| `sepsis_easy` | 38yo woman, urosepsis, lactate 1.6, PCN allergy | Sepsis | Allergy-appropriate antibiotics (no ampicillin) | **72%** |
| `sepsis_medium` | 78yo nursing home, septic shock, MRSA history, lactate 4.2 | Septic shock | MRSA coverage + vasopressors + metformin hold | **55%** |
| `sepsis_hard` | Post-op anastomotic leak, DIC, vancomycin allergy, multi-organ failure | Septic shock | Vanc allergy + emergent source control + DIC management | **28%** |

**Grading weights:** Diagnosis 20% · Bundle compliance 20% · Antibiotic appropriateness 20% · Fluid resuscitation 15% · Vasopressor decision 15% · Clinical rationale 10%  
**Critical rule:** Missing vasopressors in septic shock = critical error, score capped at 0.40.

---

## ✦ OpenEnv Spec Compliance

```python
# Full OpenEnv interface implemented
from environment import ClinicalTriageEnv

env = ClinicalTriageEnv(task_id="sepsis_hard")

# reset() → returns typed Observation
obs = env.reset()

# step(action) → returns (observation, reward, done, info)
obs, reward, done, info = env.step(action)

# state() → returns episode metadata
state = env.state()  # episode_id, step_count, total_reward, is_done
```

### Typed Models (Pydantic v2)

```python
# models.py — all inherit from OpenEnv-compatible base classes
class TriageAction(Action):
    esi_level: int                                      # 1–5, validated
    rationale: str                                      # min 10 chars
    recommended_immediate_interventions: List[str]

class MedicationSafetyAction(Action):
    flagged_interactions: List[str]
    flagged_contraindications: List[str]
    flagged_dosing_errors: List[str]
    recommended_changes: List[str]
    severity_assessment: str                            # safe|minor|moderate|major|critical
    clinical_rationale: str                             # min 20 chars

class SepsisManagementAction(Action):
    sepsis_diagnosis: str                               # sepsis|septic_shock|SIRS_only|no_sepsis
    blood_cultures_ordered: bool
    antibiotics_ordered: bool
    antibiotic_choice: Optional[str]
    lactate_ordered: bool
    iv_fluid_bolus_ml: int                              # 30mL/kg target = 2100mL for 70kg
    vasopressor_ordered: bool
    vasopressor_choice: Optional[str]                   # norepinephrine|vasopressin|dopamine
    source_control_identified: Optional[str]
    clinical_rationale: str
    time_to_antibiotics_minutes: Optional[int]
```

---

## ✦ Reward Function

```
reward = (grade_score − safety_penalty + efficiency_bonus) × difficulty_multiplier
```

| Component | Value | Purpose |
|---|---|---|
| `grade_score` | 0.0 – 1.0 | Multi-component clinical grading |
| `safety_penalty` | −0.3 × critical_errors | Patient safety enforcement |
| `efficiency_bonus` | +0.05 × (max_steps − step) | Rewards decisive action |
| `difficulty_multiplier` | Easy: 0.8 · Medium: 1.0 · Hard: 1.3 | Scales stakes with complexity |
| **Reward range** | **−1.0 to +1.5** | Full spectrum including punishment |

**Why shaped rewards matter:** Binary end-of-episode rewards teach nothing — agents need signal at each grading component (ESI accuracy, intervention completeness, antibiotic appropriateness) to learn the clinical reasoning chain, not just pattern-match to an answer.

---

## ✦ Multi-Agent Architecture

When you call `POST /analyze`, three specialist AI agents run **in parallel** and return a combined clinical assessment:

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│   Diagnostician AI  │  │     Safety AI        │  │    Evaluator AI     │
│                     │  │                      │  │                     │
│  Board-certified    │  │  Clinical pharmacist │  │  Quality specialist │
│  emergency          │  │  + patient safety    │  │  scoring against    │
│  physician          │  │  specialist          │  │  gold-standard      │
│                     │  │                      │  │  guidelines         │
│  • Differential Dx  │  │  • Drug interactions │  │  • Score 0.0–1.0   │
│  • ESI assignment   │  │  • Allergy violations│  │  • Teaching points  │
│  • Reasoning chain  │  │  • CYP450 analysis   │  │  • Guideline refs   │
│  • Confidence score │  │  • Severity verdict  │  │  • Pass/fail        │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   ▼
                      Combined risk profile:
                   Mortality % · Legal risk % · Verdict
```

---

## ✦ Risk Engine

Every patient interaction generates a clinical risk profile:

```json
{
  "mortality_risk": 78,
  "base_mortality": 35,
  "delay_penalty": 14.0,
  "legal_risk": 45,
  "verdict": "UNSAFE",
  "risk_level": "critical",
  "factors": {
    "hemodynamic_instability": true,
    "respiratory_compromise": true,
    "neurological_compromise": false,
    "tissue_hypoperfusion": true,
    "time_critical": true
  }
}
```

**Delay penalty:** Mortality risk increases with time — sepsis 7–14%/hour, acute stroke 9%/hour. Agents that act faster get efficiency bonuses.

---

## ✦ Patient Deterioration Simulation

Call `POST /simulate` with `elapsed_minutes` to watch a patient's condition evolve:

```python
# T+0 min: HR 118, BP 88/52, SpO₂ 91%, Lactate 4.2 — Septic shock
# T+15 min (wrong decision): HR 136, BP 68/40, new confusion, mottled skin
# T+30 min (wrong decision): HR 154, BP 48/28, GCS 8, petechiae — imminent death
# T+15 min (correct decision): HR 104, BP 96/60, SpO₂ 94% — improving
```

Deterioration rate doubles when `wrong_decision=True` — the environment actively penalises indecision and wrong answers with evolving patient state.

---

## ✦ Baseline Scores

Evaluated with `meta-llama/Llama-3.3-70B-Instruct` via `inference.py`:

| Task | Score | Grade | Notes |
|---|---|---|---|
| `triage_easy` | **0.78** | 🟢 Pass | Correct ESI-5, good rationale |
| `triage_medium` | **0.62** | 🟡 Pass | Identified ACS, missed some interventions |
| `triage_hard` | **0.45** | 🔴 Fail | Correct ESI-1 but anticoagulation gap |
| `med_safety_easy` | **0.90** | 🟢 Pass | Correctly identified clean regimen |
| `med_safety_medium` | **0.58** | 🟡 Borderline | Caught triple therapy, missed aspirin dose error |
| `med_safety_hard` | **0.31** | 🔴 Fail | Missed ritonavir-simvastatin CYP3A4 interaction |
| `sepsis_easy` | **0.72** | 🟢 Pass | Full bundle, correct allergy-aware antibiotics |
| `sepsis_medium` | **0.55** | 🟡 Borderline | Correct diagnosis, suboptimal antibiotic choice |
| `sepsis_hard` | **0.28** | 🔴 Fail | Missed vancomycin allergy, incomplete bundle |
| **Overall Mean** | **0.58** | — | Baseline passes easy, struggles on hard |

> Hard tasks are **genuinely hard** — they require multi-step pharmacological reasoning and rare clinical knowledge that frontier LLMs don't reliably have. This makes them ideal RL training targets.

---

## ✦ API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Interactive dashboard UI |
| `GET` | `/health` | System status, feature flags, LLM availability |
| `GET` | `/tasks` | All 9 tasks with metadata |
| `POST` | `/reset` | Start episode, returns full patient record |
| `POST` | `/step` | Submit action → reward + grade + feedback |
| `POST` | `/analyze` | 3-agent parallel analysis (Diagnostician + Safety + Evaluator) |
| `POST` | `/benchmark` | User vs Claude vs LLaMA comparison |
| `POST` | `/simulate` | Patient deterioration simulation over time |
| `POST` | `/report` | Generate downloadable PDF clinical case report |
| `POST` | `/grade` | Grade a specific action without episode management |
| `GET` | `/state` | Current episode state and metadata |
| `GET` | `/leaderboard` | Model performance rankings |
| `WS` | `/ws` | WebSocket for real-time streaming |
| `GET` | `/docs` | Auto-generated Swagger UI |

---

## ✦ Quick Start

### Run Against the Live Space

```bash
export SPACE_URL="https://samrudh-nux-my-healthcare-ev4u2.hf.space"

# 1. Health check
curl $SPACE_URL/health

# 2. List all tasks
curl $SPACE_URL/tasks | python -m json.tool

# 3. Reset an episode
curl -X POST $SPACE_URL/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "triage_medium"}'

# 4. Submit a triage action
curl -X POST $SPACE_URL/step \
  -H "Content-Type: application/json" \
  -d '{
    "action": {
      "esi_level": 2,
      "rationale": "67yo male with crushing chest pain, diaphoresis, left arm radiation, tachycardia — ACS until proven otherwise. High-risk ESI-2.",
      "recommended_immediate_interventions": ["ECG", "aspirin_325mg", "troponin", "IV_access", "oxygen"]
    }
  }'
```

### Run Baseline Inference

```bash
# Clone the Space
git clone https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u2
cd my-healthcare-ev4u2

# Install dependencies
pip install -r requirements.txt

# Set credentials
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_your_token_here"

# Run all 9 tasks (~8 min)
python inference.py

# Run specific tasks
python inference.py --tasks triage_hard med_safety_hard sepsis_hard --output hard_results.json
```

### Docker

```bash
docker build -t clinical-triage-env .

docker run -p 7860:7860 \
  -e HF_TOKEN="hf_your_token" \
  -e MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct" \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  clinical-triage-env

# Verify
curl http://localhost:7860/health
```

### Python SDK

```python
from environment import ClinicalTriageEnv
from models import TriageAction, SepsisManagementAction

# ── Triage example ────────────────────────────────────────────────────
env = ClinicalTriageEnv(task_id="triage_hard")
obs = env.reset()

print(f"Patient: {obs.patient.chief_complaint}")
# → "Confusion and weakness, family notes she was normal this morning"

action = TriageAction(
    esi_level=1,
    rationale=(
        "Acute stroke presentation: FAST positive (facial droop, right arm weakness, "
        "slurred speech), onset <2h, on warfarin (INR unknown). ESI-1 — immediate "
        "stroke alert, door-to-CT target <25min, hold thrombolytics until INR checked."
    ),
    recommended_immediate_interventions=[
        "stroke_alert", "CT_head_noncontrast", "CT_angiography",
        "INR_stat", "glucose_check", "neurology_stat"
    ]
)

obs, reward, done, info = env.step(action)

print(f"Score:   {info['grade']:.3f}")       # → 0.847
print(f"Reward:  {reward:.4f}")              # → 1.0811
print(f"Passed:  {info['passed']}")          # → True
print(f"Components: {info['component_scores']}")

# ── Sepsis example ────────────────────────────────────────────────────
env2 = ClinicalTriageEnv(task_id="sepsis_hard")
obs2 = env2.reset()

# Patient has vancomycin allergy — must use meropenem, not vancomycin
action2 = SepsisManagementAction(
    sepsis_diagnosis="septic_shock",
    blood_cultures_ordered=True,
    antibiotics_ordered=True,
    antibiotic_choice="meropenem_plus_metronidazole",   # correct — vanc allergy!
    lactate_ordered=True,
    iv_fluid_bolus_ml=2100,
    vasopressor_ordered=True,
    vasopressor_choice="norepinephrine",
    source_control_identified="anastomotic_leak_emergency_surgery",
    clinical_rationale=(
        "Multi-organ failure septic shock from anastomotic leak. Vancomycin allergy "
        "documented — use meropenem for GNR/peritonitis coverage. Emergent return to "
        "OR for source control. Monitor for DIC (low platelets, elevated INR, low fibrinogen)."
    ),
    time_to_antibiotics_minutes=25
)

obs2, reward2, done2, info2 = env2.step(action2)
print(f"Sepsis Hard Score: {info2['grade']:.3f}")      # → 0.812
```

---

## ✦ Project Structure

```
my-healthcare-ev4u2/
├── app.py              # FastAPI server + multi-agent backend + dashboard (47 kB)
├── environment.py      # Core env: reset() / step() / state() + reward shaping (17 kB)
├── models.py           # Typed Pydantic models: Action / Observation / State (9 kB)
├── scenarios.py        # 9 medically-accurate patient scenarios with ground truth (22 kB)
├── graders.py          # Programmatic graders for all 3 task types (36 kB)
├── inference.py        # Baseline inference script — OpenAI client (25 kB)
├── index.html          # Production dashboard UI (124 kB)
├── openenv.yaml        # OpenEnv spec manifest
├── requirements.txt    # Python dependencies
└── Dockerfile          # Container definition (Python 3.11-slim)
```

---

## ✦ Medical Accuracy

All scenarios are grounded in real clinical guidelines:

| Domain | Standard | Source |
|---|---|---|
| ED Triage | ESI Implementation Handbook, 5th edition | AHRQ / ACEP |
| Sepsis criteria | Sepsis-3 definitions | Singer et al., JAMA 2016 |
| Sepsis management | SSC Hour-1 Bundle | Surviving Sepsis Campaign 2021 |
| Antibiotic selection | IDSA Antimicrobial Stewardship Guidelines | IDSA 2023 |
| ACS management | ACC/AHA Guidelines | ACC/AHA 2023 |
| Drug interactions | CYP450 interaction database | FDA / Lexicomp |
| Medication dosing | Pharmacology references | BNF / Micromedex |

**Graders enforce real clinical safety rules:**
- Undertriage of ESI-1/2 patients is a medical-legal error in all 50 US states
- Allergy violations score 0.0 — no partial credit for harming patients
- Missing vasopressors in septic shock caps score at 0.40
- Simvastatin + ritonavir is an absolute FDA contraindication since 2011

> ⚠️ **Disclaimer:** This environment is for AI training and research only. Not for actual clinical decision-making.

---

## ✦ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | For LLM features | Hugging Face API token |
| `API_BASE_URL` | Optional | LLM API base (default: `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Optional | Model for inference (default: `meta-llama/Llama-3.3-70B-Instruct`) |
| `PORT` | Optional | Server port (default: `7860`) |

---

## ✦ Integration with RL Frameworks

### TRL (Hugging Face)

```python
from openenv_client import ClinicalTriageEnvClient

# Connect to running Space
env = ClinicalTriageEnvClient(base_url="https://samrudh-nux-my-healthcare-ev4u2.hf.space")

# Standard gymnasium-style loop
obs = env.reset(task_id="triage_medium")
while not obs.done:
    action = policy.predict(obs)
    obs, reward, done, info = env.step(action)
    policy.learn(reward)
```

### GRPO / SkyRL

The environment's reward function is shaped for GRPO training — partial rewards at each reasoning step provide dense training signal that binary rewards cannot. Each grading component (ESI accuracy, intervention coverage, antibiotic appropriateness) becomes a separate reward term.

---

## ✦ Roadmap

- [ ] **Multi-turn episodes** — dialogue-based history taking before diagnosis
- [ ] **Radiology triage** — CT/X-ray interpretation with graded urgency
- [ ] **ICU scoring** — SOFA/APACHE-II calculation with treatment planning  
- [ ] **Multi-patient queue** — prioritise 8 simultaneous ED patients
- [ ] **Paediatric adaptations** — weight-based dosing, paediatric vital sign norms
- [ ] **OpenEnv validate** — full CLI validation passing
- [ ] **SkyRL integration** — distributed training example

---

## ✦ Citation

```bibtex
@software{clinicaltriageenv2026,
  title        = {ClinicalTriageEnv: An OpenEnv Environment for Clinical Decision-Making},
  author       = {samrudh-nux},
  year         = {2026},
  url          = {https://huggingface.co/spaces/samrudh-nux/my-healthcare-ev4u2},
  note         = {OpenEnv Hackathon 2026 submission. MIT License.}
}
```

---

## ✦ License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for the OpenEnv Hackathon 2025**  
*Meta PyTorch × Hugging Face*

*The environments that train agents to save lives.*

</div>

