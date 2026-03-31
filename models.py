from __future__ import annotations
  from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# 
# Core OpenEnv-compatible base types
# 

class Action(BaseModel):
    """Base action class (OpenEnv-compatible)."""
    pass


class Observation(BaseModel):
    """Base observation class (OpenEnv-compatible)."""
            done: bool = Field(default=False, description="Whether the episode is finished")
    reward: Optional[float] = Field(default=None, description="Reward for this step")


# 
# Domain Models
# 

class VitalSigns(BaseModel):
    heart_rate: int = Field(..., description="Heart rate in bpm (normal: 60-100)")
                        systolic_bp: int = Field(..., description="Systolic blood pressure mmHg (normal: 90-140)")
                                        diastolic_bp: int = Field(..., description="Diastolic blood pressure mmHg (normal: 60-90)")
                        temperature: float = Field(..., description="Body temperature in Celsius (normal: 36.1-37.2)")
    spo2: int = Field(..., description="Blood oxygen saturation % (normal: 95-100)")
                respiratory_rate: int = Field(..., description="Breaths per minute (normal: 12-20)")
    glasgow_coma_scale: int = Field(..., description="GCS score 3-15 (15=fully alert)")


class Medication(BaseModel):
    name: str
    dose_mg: float
      frequency: str  # e.g. "twice_daily", "once_daily", "every_8h"
     route: str      # e.g. "oral", "IV", "subcutaneous"


class PatientRecord(BaseModel):
    patient_id: str
    age: int
      sex: str
    chief_complaint: str
      vitals: VitalSigns
    symptoms: List[str]
       medical_history: List[str]
      current_medications: List[]
    lab_results: Dict[str, Any]
        arrival_time_minutes: int = Field(description="Minutes since patient arrived")
 allergies: List[str] = Field(default_factory=list)


# 
# Task 1: ED Triage (ESI Level Assignment)
#

class TriageAction(Action):
    """
    Action for Task 1: Emergency Department Triage.
    Agent assigns an ESI (Emergency Severity Index) level and
    provides a brief clinical rationale.
    """
    esi_level: int = Field(
        ...,
        ge=1, le=5,
        description=(
            "ESI triage level 1-5. "
                "1=Resuscitation (immediate life threat), "
                "2=Emergent (high risk), "
            "3=Urgent (stable but needs resources), "
            "4=Less Urgent, "
            "5=Non-urgent"
        )
    )
    rationale: str = Field(
        ...,
        min_length=10,
        description="Clinical reasoning for triage decision (min 10 chars)"
    )
    recommended_immediate_interventions: List[str] = Field(
        default_factory=list,
        description="List of immediate actions needed (e.g. ['oxygen', 'IV_access', 'ECG'])"
    )


class TriageObservation(Observation):
    """Observation returned after a triage action."""
                       patient: PatientRecord
                                  task_description: str
                                         current_step: int
    max_steps: int
                                 feedback: str = Field(default="", description="Feedback on last action")
    score_so_far: float = Field(default=0.0)
                # Additional clinical context revealed progressively
    additional_info: Optional[Dict[str, Any]] = None


# 
# Task 2: Medication Safety Review
# 

class MedicationSafetyAction(Action):
    """
    Action for Task 2: Medication Safety Check.
    Agent reviews a patient's medication list and identifies
    dangerous interactions, contraindications, and dosing errors.
    """
    flagged_interactions: List[str] = Field(
        default_factory=list,
        description="Drug-drug interaction pairs found, e.g. ['warfarin+aspirin', 'metformin+contrast']"
    )
    flagged_contraindications: List[str] = Field(
          default_factory=list,
                            description="Contraindications found given patient history, e.g. ['metformin_renal_failure']"
    )
    flagged_dosing_errors: List[str] = Field(
                  default_factory=list,
        description="Dose errors, e.g. ['metformin_500mg_renal_gfr<30']"
    )
    recommended_changes: List[str] = Field(
        default_factory=list,
                    description="Specific recommended changes, e.g. ['discontinue_metformin', 'reduce_warfarin_dose']"
    )
    severity_assessment: str = Field(
        ...,
                   description="Overall severity: 'critical', 'major', 'moderate', 'minor', or 'safe'"
    )
    clinical_rationale: str = Field(
        ...,
        min_length=20,
                       description="Detailed explanation of findings"
    )


class MedicationSafetyObservation(Observation):
    """Observation returned during medication safety review."""
    patient: PatientRecord
    task_description: str
    current_step: int
    max_steps: int
    feedback: str = Field(default="")
    score_so_far: float = Field(default=0.0)
    available_drug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Drug reference info available to agent (simulates formulary lookup)"
    )


#
# Task 3: Sepsis Early Warning & Management
#

class SepsisManagementAction(Action):
    """
    Action for Task 3: Sepsis Recognition & Hour-1 Bundle.
    Agent must recognise sepsis criteria AND execute the correct
    time-critical management bundle within simulated time.
    """
    sepsis_diagnosis: str = Field(
        ...,
        description="Diagnosis: 'sepsis', 'septic_shock', 'SIRS_only', 'no_sepsis'"
    )
    # Hour-1 Surviving Sepsis Campaign bundle items
                  blood_cultures_ordered: bool = Field(default=False, description="Blood cultures before antibiotics")
               antibiotics_ordered: bool = Field(default=False, description="Broad-spectrum antibiotics started")
              antibiotic_choice: Optional[str] = Field(default=None, description="e.g. 'piperacillin_tazobactam', 'meropenem'")
    lactate_ordered: bool = Field(default=False, description="Serum lactate measured")
    iv_fluid_bolus_ml: int = Field(default=0, description="IV fluid bolus in mL (30mL/kg for hypotension/lactate>4)")
               vasopressor_ordered: bool = Field(default=False, description="Norepinephrine for MAP<65 despite fluids")
            vasopressor_choice: Optional[str] = Field(default=None, description="e.g. 'norepinephrine', 'vasopressin'")
          source_control_identified: Optional[str] = Field(default=None, description="Infection source, e.g. 'pneumonia', 'UTI', 'abdominal'")
    clinical_rationale: str = Field(..., min_length=20, description="Clinical reasoning")
           time_to_antibiotics_minutes: Optional[int] = Field(default=None, description="Estimated time to antibiotic administration")


class SepsisManagementObservation(Observation):
    """Observation returned during sepsis management."""
           patient: PatientRecord
    task_description: str
             current_step: int
    max_steps: int
              feedback: str = Field(default="")
    score_so_far: float = Field(default=0.0)
         time_elapsed_minutes: int = Field(default=0, description="Simulated time elapsed in episode")
    sofa_score: Optional[int] = Field(default=None, description="SOFA score if calculated")
         qsofa_score: Optional[int] = Field(default=None, description="qSOFA score")


# 
# Universal Episode State
# 

class ClinicalState(BaseModel):
    """OpenEnv-compatible State object tracking episode metadata."""
    episode_id: str
    step_count: int = 0
    task_id: str = ""
    task_name: str = ""
      total_reward: float = 0.0
    is_done: bool = False
      patient_id: str = ""
    partial_scores: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
