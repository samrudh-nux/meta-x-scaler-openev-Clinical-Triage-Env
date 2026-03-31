from models import PatientRecord, VitalSigns, Medication

# 
# TASK 1: ED TRIAGE SCENARIOS (ESI Level Assignment)
# Difficulty progression: Easy → Medium → Hard
# 

TRIAGE_SCENARIOS = {
    "triage_easy_01": {
        "patient": PatientRecord(
            patient_id="T-001",
            age=45,
               sex="male",
                 chief_complaint="Mild ankle sprain after walking, no weight-bearing difficulty",
                    vitals=VitalSigns(
        heart_rate=84,
   systolic_hp=122,
        diastolic_bp=78,
          temperature=27,8,
                spo2=99,
         respiratory_rate=14,
                glasgow_coma_scale=15
            ),
            symptoms=["right ankle pain", "mild swelling", "no bruising", "can walk with discomfort"],
            medical_history=["none"],
            current_medications=[],
            lab_results={},
            arrival_time_minutes=15,
            allergies=[]
        ),
        "ground_truth_esi": 5,
        "acceptable_esi_range": [4, 5],
        "critical_interventions": [],
        "difficulty": "easy",
        "teaching_point": "ESI-5: No resources needed. Vital signs normal. Non-urgent complaint."
    },

    "triage_easy_02": {
                     "patient": PatientRecord(
                           patient_id="T-002",
            age=32,
            sex="female",
                            chief_complaint="Sore throat and low-grade fever for 2 days",
                     vitals=VitalSigns(
                heart_rate=84,
                systolic_bp=118,
                     diastolic_bp=74,
                       temperature=37.9,
           spo2=98,
                respiratory_rate=16,
                      glasgow_coma_scale=15
            ),
            symptoms=["sore throat", "low-grade fever", "mild fatigue", "difficulty swallowing"],
                         medical_history=["no significant history"],
                     current_medications=[],
            lab_results={},
        arrival_time_minutes=30,
              allergies=["penicillin"]
        ),
        "ground_truth_esi": 4,
                "acceptable_esi_range": [3, 4],
                  "critical_interventions": [],
        "difficulty": "easy",
    "teaching_point": "ESI-4: One resource needed (throat swab). Stable vitals."
    },

    "triage_medium_01": {
        "patient": PatientRecord(
            patient_id="T-003",
                    age=67,
                          sex="male",
                  chief_complaint="Crushing chest pain radiating to left arm, started 45 minutes ago",
            vitals=VitalSigns(
                          heart_rate=102,
                systolic_bp=148,
                          diastolic_bp=92,
                         temperature=36.8,
                spo2=96,
         respiratory_rate=20,
                glasgow_coma_scale=15
            ),
            symptoms=["chest pain 8/10", "diaphoresis", "nausea", "left arm radiation", "dyspnea"],
            medical_history=["hypertension", "type 2 diabetes", "smoker 30 pack-years"],
            current_medications=[
                       Medication(name="metformin", dose_mg=500, frequency="twice_daily", route="oral"),
                            Medication(name="lisinopril", dose_mg=10, frequency="once_daily", route="oral"),
                Medication(name="aspirin", dose_mg=81, frequency="once_daily", route="oral")
            ],
            lab_results={},
            arrival_time_minutes=5,
            allergies=[]
        ),
        "ground_truth_esi": 2,
                 "acceptable_esi_range": [1, 2],
        "critical_interventions": ["ECG", "aspirin_325mg", "IV_access", "troponin", "oxygen"],
                      "difficulty": "medium",
        "teaching_point": "ESI-2: High-risk presentation. ACS until proven otherwise. Needs immediate ECG and STEMI workup."
    },

    "triage_medium_02": {
        "patient": PatientRecord(
            patient_id="T-004",
            age=28,
            sex="female",
            chief_complaint="Severe headache 10/10, worst headache of life, sudden onset",
            vitals=VitalSigns(
                heart_rate=88,
                systolic_bp=162,
                    diastolic_bp=94,
                temperature=37.1,
                            spo2=99,
                respiratory_rate=18,
                glasgow_coma_scale=14
            ),
            symptoms=["thunderclap headache", "neck stiffness", "photophobia", "nausea", "vomiting"],
            medical_history=["oral contraceptive use", "migraines (but states this is different)"],
            current_medications=[
                Medication(name="ethinyl_estradiol_norethindrone", dose_mg=0.035, frequency="once_daily", route="oral")
            ],
            lab_results={},
            arrival_time_minutes=10,
            allergies=[]
        ),
        "ground_truth_esi": 2,
             "acceptable_esi_range": [1, 2],
                     "critical_interventions": ["CT_head_noncontrast", "lumbar_puncture_if_CT_negative", "neurology_consult"],
        "difficulty": "medium",
    "teaching_point": "ESI-2: Thunderclap headache = subarachnoid hemorrhage until proven otherwise. Time-critical."
    },

    "triage_hard_01": {
        "patient": PatientRecord(
              patient_id="T-005",
            age=72,
                      sex="female",
            chief_complaint="Confusion and weakness, family notes she was normal this morning",
            vitals=VitalSigns(
                heart_rate=88,
                         systolic_bp=188,
                    diastolic_bp=108,
                      temperature=36.9,
                spo2=95,
                respiratory_rate=18,
                glasgow_coma_scale=13
            ),
            symptoms=["acute confusion", "right arm weakness", "facial droop (right)", "slurred speech", "onset <2 hours ago"],
            medical_history=["atrial fibrillation", "hypertension", "previous TIA 3 years ago"],
            current_medications=[
                         Medication(name="warfarin", dose_mg=5, frequency="once_daily", route="oral"),
                      Medication(name="metoprolol", dose_mg=25, frequency="twice_daily", route="oral"),
                Medication(name="atorvastatin", dose_mg=40, frequency="once_daily", route="oral")
            ],
            lab_results={"INR_pending": "unknown", "glucose": 6.2},
            arrival_time_minutes=8,
                  allergies=[]
        ),
        "ground_truth_esi": 1,
        "acceptable_esi_range": [1, 2],
                  "critical_interventions": ["stroke_alert", "CT_head", "CT_angiography", "INR_stat", "glucose_check", "neurology_stat"],
                   "difficulty": "hard",
        "teaching_point": "ESI-1: Acute stroke (FAST positive). On anticoagulation. tPA contraindicated if INR>1.7. Time is brain - door-to-CT <25min."
    }
}


# 
# TASK 2: MEDICATION SAFETY SCENARIOS
# 

MEDICATION_SCENARIOS = {
    "med_easy_01": {
                     "patient": PatientRecord(
            patient_id="M-001",
            age=55,
                    sex="male",
            chief_complaint="Routine medication review",
                    vitals=VitalSigns(
                        heart_rate=72, systolic_bp=130, diastolic_bp=82,
                temperature=36.7, spo2=98, respiratory_rate=14, glasgow_coma_scale=15
            ),
            symptoms=["well-controlled hypertension"],
            medical_history=["hypertension", "hyperlipidemia"],
            current_medications=[
                              Medication(name="amlodipine", dose_mg=5, frequency="once_daily", route="oral"),
                         Medication(name="atorvastatin", dose_mg=40, frequency="once_daily", route="oral"),
                Medication(name="aspirin", dose_mg=81, frequency="once_daily", route="oral"),
            ],
            lab_results={"creatinine": 0.9, "eGFR": 85, "K": 4.1, "Na": 138},
                    arrival_time_minutes=60,
            allergies=[]
        ),
        "ground_truth": {
                 "interactions": [],
            "contraindications": [],
                         "dosing_errors": [],
            "severity": "safe",
            "key_findings": "No significant interactions. Safe combination."
        },
        "difficulty": "easy"
    },

    "med_medium_01": {
        "patient": PatientRecord(
            patient_id="M-002",
            age=68,
                     sex="female",
            chief_complaint="Post-cardiac cath, started on anticoagulation",
                  vitals=VitalSigns(
                heart_rate=78, systolic_bp=134, diastolic_bp=80,
                temperature=36.8, spo2=97, respiratory_rate=15, glasgow_coma_scale=15
            ),
            symptoms=["recent MI", "stable now"],
            medical_history=["STEMI 5 days ago", "type 2 diabetes", "hypertension", "CKD stage 3a (eGFR 48)"],
            current_medications=[
                Medication(name="warfarin", dose_mg=5, frequency="once_daily", route="oral"),
                        Medication(name="aspirin", dose_mg=325, frequency="once_daily", route="oral"),
                              Medication(name="clopidogrel", dose_mg=75, frequency="once_daily", route="oral"),
                          Medication(name="metformin", dose_mg=1000, frequency="twice_daily", route="oral"),
                Medication(name="lisinopril", dose_mg=5, frequency="once_daily", route="oral"),
           Medication(name="metoprolol", dose_mg=25, frequency="twice_daily", route="oral"),
            ],
            lab_results={"creatinine": 1.6, "eGFR": 48, "INR": 2.1, "HbA1c": 7.2, "K": 4.8},
            arrival_time_minutes=45,
            allergies=[]
        ),
        "ground_truth": {
            "interactions": ["warfarin+aspirin+clopidogrel (triple therapy - major bleeding risk)"],
            "contraindications": ["metformin_with_eGFR_48 (caution but borderline - monitor; consider dose reduction)"],
                "dosing_errors": ["aspirin_325mg_post_MI_should_be_96mg_for_long_term"],
                 "severity": "major",
            "key_findings": "Triple antithrombotic therapy (warfarin+aspirin+clopidogrel) massively increases GI bleed risk. Post-MI guideline: reduce aspirin to 81mg. Evaluate if warfarin truly needed vs NOAC. Metformin caution with eGFR 48 - monitor."
        },
        "difficulty": "medium"
    },

    "med_hard_01": {
        "patient": PatientRecord(
            patient_id="M-003",
            age=52,
            sex="male",
            chief_complaint="Fatigue, muscle pain, dark urine for 3 days",
            vitals=VitalSigns(
                heart_rate=96, systolic_bp=142, diastolic_bp=88,
                temperature=36.9, spo2=97, respiratory_rate=17, glasgow_coma_scale=15
            ),
            symptoms=["myalgia", "weakness", "dark/cola-coloured urine", "fatigue", "decreased urine output"],
            medical_history=["HIV on antiretrovirals", "hyperlipidemia", "recent fungal infection treated"],
            current_medications=[
                           Medication(name="simvastatin", dose_mg=80, frequency="once_daily", route="oral"),
                Medication(name="ritonavir", dose_mg=100, frequency="twice_daily", route="oral"),  # HIV PI
                              Medication(name="atazanavir", dose_mg=300, frequency="once_daily", route="oral"),  # HIV PI
                Medication(name="fluconazole", dose_mg=200, frequency="once_daily", route="oral"),  # antifungal
          Medication(name="omeprazole", dose_mg=20, frequency="once_daily", route="oral"),
            ],
            lab_results={
                "creatinine": 2.8, "eGFR": 24, "CK": 48000,  # massively elevated (normal <200)
                "AST": 320, "ALT": 280, "K": 5.6,
                "myoglobin_urine": "positive", "BUN": 42
            },
            arrival_time_minutes=20,
            allergies=["sulfa"]
        ),
        "ground_truth": {
            "interactions": [
                "simvastatin+ritonavir (ritonavir is potent CYP3A4 inhibitor → 3000% increase in simvastatin levels → rhabdomyolysis)",
                         "simvastatin+fluconazole (fluconazole inhibits CYP3A4 → increased simvastatin levels)",
                                 "atazanavir+omeprazole (PPIs reduce atazanavir absorption significantly)"
            ],
            "contraindications": [
                "simvastatin_absolutely_contraindicated_with_HIV_protease_inhibitors",
                           "simvastatin_80mg_FDA_dose_restriction_2011 (max 20mg unless established)"
            ],
            "dosing_errors": [
                "simvastatin_80mg_contraindicated_with_ritonavir",
                "atazanavir_should_not_be_taken_with_PPI"
            ],
            "severity": "critical",
            "key_findings": "This is rhabdomyolysis caused by statin-protease inhibitor interaction. Ritonavir inhibits CYP3A4 by >90%, causing massive simvastatin accumulation → muscle destruction → myoglobinuria → acute kidney injury (eGFR 24, CK 48,000). IMMEDIATE: discontinue simvastatin, aggressive IV hydration, monitor K+ (hyperkalemia risk), consider dialysis. Simvastatin is absolutely contraindicated with HIV PIs - switch to pravastatin or rosuvastatin (not CYP3A4 metabolized)."
        },
        "difficulty": "hard"
    }
}


# 
# TASK 3: SEPSIS MANAGEMENT SCENARIOS
# 

SEPSIS_SCENARIOS = {
    "sepsis_easy_01": {
        "patient": PatientRecord(
            patient_id="S-001",
            age=38,
            sex="female",
            chief_complaint="Fever, chills, painful urination for 2 days",
            vitals=VitalSigns(
                heart_rate=104,
                systolic_bp=112,
                diastolic_bp=68,
                temperature=38.9,
                spo2=97,
                respiratory_rate=20,
                glasgow_coma_scale=15
            ),
            symptoms=["fever", "rigors", "dysuria", "flank pain", "nausea"],
            medical_history=["recurrent UTIs", "otherwise healthy"],
            current_medications=[],
            lab_results={
                "WBC": 14.2, "bands": 18,
                "lactate": 1.6,
                "creatinine": 1.1, "eGFR": 72,
                "UA": "positive nitrites, positive leukocyte esterase, >100 WBC/hpf",
                "urine_culture": "pending",
                "procalcitonin": 2.8
            },
            arrival_time_minutes=25,
            allergies=["penicillin"]
        ),
        "ground_truth": {
            "diagnosis": "sepsis",
            "sepsis_criteria_met": ["temperature>38", "HR>100", "WBC>12", "suspected_infection"],
            "qsofa": 1,
            "sofa_estimated": 2,
            "bundle": {
                "blood_cultures": True,
                "antibiotics": True,
                "antibiotic_choice": "ceftriaxone_or_ciprofloxacin",  # PCN allergy, so no ampicillin
                "lactate": True,
                "fluids_ml": 1800,  # 30mL/kg x 60kg
                "vasopressors": False
            },
            "key_note": "Urosepsis. PCN allergy - avoid ampicillin. Ceftriaxone or fluoroquinolone appropriate. Lactate 1.6 = not elevated, moderate severity."
        },
        "difficulty": "easy"
    },

    "sepsis_medium_01": {
        "patient": PatientRecord(
            patient_id="S-002",
            age=78,
            sex="male",
            chief_complaint="Nursing home patient, altered mental status, found unresponsive this morning",
            vitals=VitalSigns(
                heart_rate=118,
                systolic_bp=88,
                diastolic_bp=52,
                temperature=38.6,
                spo2=91,
                respiratory_rate=26,
                glasgow_coma_scale=11
            ),
            symptoms=["altered mental status", "hypotension", "tachycardia", "tachypnea", "decreased urine output x24h", "productive cough"],
            medical_history=["COPD", "type 2 diabetes", "CKD stage 3", "previous MRSA infection", "dementia"],
            current_medications=[
                Medication(name="metformin", dose_mg=500, frequency="twice_daily", route="oral"),
                Medication(name="tiotropium", dose_mg=18, frequency="once_daily", route="inhaled"),
                Medication(name="amlodipine", dose_mg=5, frequency="once_daily", route="oral"),
            ],
            lab_results={
                "WBC": 22.4, "bands": 32,
                "lactate": 4.2,  # >4 = high mortality
                "creatinine": 2.4, "eGFR": 28,
                "glucose": 18.2,
                "procalcitonin": 42,
                "CXR": "bilateral infiltrates, right lower lobe consolidation",
                "blood_cultures": "pending",
                "pH": 7.28, "pCO2": 32, "HCO3": 14  # metabolic acidosis
            },
            arrival_time_minutes=12,
            allergies=[]
        ),
        "ground_truth": {
            "diagnosis": "septic_shock",
            "sepsis_criteria_met": ["HR>100", "RR>20", "temp>38", "AMS", "suspected_pneumonia", "MAP<65", "lactate>4"],
            "qsofa": 3,
            "sofa_estimated": 8,
            "bundle": {
                "blood_cultures": True,
                "antibiotics": True,
                "antibiotic_choice": "vancomycin_plus_piperacillin_tazobactam",  # MRSA coverage needed
                "lactate": True,
                "fluids_ml": 2100,  # 30mL/kg x 70kg
                "vasopressors": True,
                "vasopressor_choice": "norepinephrine"
            },
            "key_note": "Septic shock (MAP<65, lactate>4). MRSA history mandates vancomycin. Caution: metformin hold (lactate acidosis risk, AKI). Aggressive fluids + vasopressors. ICU."
        },
        "difficulty": "medium"
    },

    "sepsis_hard_01": {
        "patient": PatientRecord(
            patient_id="S-003",
            age=61,
            sex="female",
            chief_complaint="Post-op day 3 after bowel resection, fever and abdominal pain",
            vitals=VitalSigns(
                heart_rate=126,
                systolic_bp=82,
                diastolic_bp=46,
                temperature=39.4,
                spo2=89,
                respiratory_rate=28,
                glasgow_coma_scale=13
            ),
            symptoms=["fever", "hypotension", "tachycardia", "worsening abdominal pain", "wound erythema", "purulent discharge", "decreased urine output", "mottled skin"],
            medical_history=["colorectal cancer (sigmoid)", "diabetes", "immunosuppression (prednisolone 20mg/day)", "recent hospitalization"],
            current_medications=[
                Medication(name="prednisolone", dose_mg=20, frequency="once_daily", route="oral"),
                Medication(name="metformin", dose_mg=500, frequency="twice_daily", route="oral"),
                Medication(name="enoxaparin", dose_mg=40, frequency="once_daily", route="subcutaneous"),
                Medication(name="morphine_PCA", dose_mg=2, frequency="PRN", route="IV"),
            ],
            lab_results={
                "WBC": 28.6, "bands": 42,
                "lactate": 6.8,  # severely elevated
                            "creatinine": 3.2, "eGFR": 18,
                "glucose": 22.4,
                "platelets": 68,  # thrombocytopenia
                            "INR": 2.1,  # coagulopathy
                          "fibrinogen": 1.2,  # low - DIC?
                "ALT": 180, "AST": 220,  # hepatic dysfunction
                                "procalcitonin": 88,
                "blood_cultures": "gram_negative_rods_preliminary",
                "CT_abdomen": "anastomotic_leak_with_free_air_and_peritoneal_contamination"
            },
            arrival_time_minutes=5,
            allergies=["vancomycin (red man syndrome)"]
        ),
        "ground_truth": {
            "diagnosis": "septic_shock",
            "sepsis_criteria_met": ["all_sofa_criteria", "MAP<65_despite_fluids", "lactate>6", "multi_organ_failure"],
            "qsofa": 3,
            "sofa_estimated": 14,
            "bundle": {
                "blood_cultures": True,
                "antibiotics": True,
                "antibiotic_choice": "meropenem_plus_metronidazole",  # GNR, peritonitis, vanc allergy
                "lactate": True,
                "fluids_ml": 2100,
                "vasopressors": True,
                "vasopressor_choice": "norepinephrine",
                "source_control": "emergency_surgical_return_washout"
            },
            "key_note": "Multi-organ failure septic shock from anastomotic leak. CRITICAL: vancomycin ALLERGY (use meropenem for GNR coverage). DIC developing (low platelets, INR>2, low fibrinogen). Immunosuppressed (steroids). Source control = emergent return to OR. Consider stress-dose steroids for adrenal insufficiency risk. Stop metformin immediately."
        },
        "difficulty": "hard"
    }
}
