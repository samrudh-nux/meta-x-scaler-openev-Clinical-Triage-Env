from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import TriageAction, MedicationSafetyAction, SepsisManagementAction


@dataclass
class GradeResult:
    """Detailed grading result returned by every grader."""
    score: float                          # 0.0 – 1.0 base (before difficulty multiplier)
    component_scores: Dict[str, float]   # per-axis breakdown
    feedback: str                         # human-readable explanation
    critical_errors: List[str] = field(default_factory=list)  # patient-safety failures
    passed: bool = False                  # score >= 0.60 AND no critical errors
    confidence: str = "high"             # "high" | "medium" | "low"
    teaching_point: str = ""             # one sentence educational note



def _tokenise(text: str) -> List[str]:
    """Lowercase word tokens, strip punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())

def _token_overlap(a: str, b: str) -> float:
    """Jaccard-like token overlap between two strings."""
    ta, tb = set(_tokenise(a)), set(_tokenise(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def _keyword_score(text: str, keywords: List[str], threshold: int = 3) -> float:
    """Fraction of expected keywords found in text (case-insensitive)."""
    if not keywords or not text:
        return 0.0
    text_l = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_l)
    return min(1.0, found / max(threshold, len(keywords) * 0.4))

def _fuzzy_list_recall(proposed: List[str], ground_truth: List[str]) -> float:
    """Recall: what fraction of GT items were detected (fuzzy match)."""
    if not ground_truth:
        return 1.0
    if not proposed:
        return 0.0
    proposed_blob = " ".join(proposed).lower()
    found = 0
    for gt_item in ground_truth:
        words = [w for w in _tokenise(gt_item) if len(w) > 3]
        if not words:
            continue
        hits = sum(1 for w in words if w in proposed_blob)
        if hits >= max(1, int(len(words) * 0.4)):
            found += 1
    return round(found / len(ground_truth), 3)

def _false_positive_rate(proposed: List[str], ground_truth: List[str]) -> float:
    """Fraction of proposed items that appear to be hallucinations."""
    if not proposed:
        return 0.0
    if not ground_truth:
        return min(1.0, len(proposed) * 0.15)
    gt_blob = " ".join(ground_truth).lower()
    fps = 0
    for item in proposed:
        words = [w for w in _tokenise(item) if len(w) > 3]
        if words and not any(w in gt_blob for w in words):
            fps += 1
    return fps / len(proposed)



class TriageGrader:
    """
    8-component triage grader.
    Components: ESI accuracy, acceptable-range bonus, undertriage safety,
    rationale keyword density, clinical flag recognition, intervention recall,
    intervention precision, time-sensitivity awareness.
    """

    # Weights must sum to 1.0
    WEIGHTS = {
        "esi_accuracy":           0.40,
        "acceptable_range_bonus": 0.05,
        "rationale_keywords":     0.15,
        "clinical_flags":         0.10,
        "intervention_recall":    0.15,
        "intervention_precision": 0.05,
        "time_sensitivity":       0.10,
    }

    # ESI accuracy lookup
    _ESI_SCORES = {0: 1.0, 1: 0.55, 2: 0.20, 3: 0.05}

    # Undertriage penalty (GT ≤ 2, assigned ≥ 3 → life-threatening)
    UNDERTRIAGE_PENALTY_MAJOR = 0.45   # assigned ≥ 3 when GT ≤ 2
    UNDERTRIAGE_PENALTY_MINOR = 0.20   # assigned GT+1 when GT == 1

    def grade(self, action: TriageAction, scenario: Dict[str, Any]) -> GradeResult:
        gt_esi      = scenario["ground_truth_esi"]
        acceptable  = scenario["acceptable_esi_range"]
        critical_iv = scenario.get("critical_interventions", [])
        teaching    = scenario.get("teaching_point", "")
        patient     = scenario["patient"]

        cs: Dict[str, float] = {}
        errors: List[str] = []

        # ── 1. ESI accuracy ─────────────────────────────────────────────────
        diff = abs(action.esi_level - gt_esi)
        esi_score = self._ESI_SCORES.get(diff, 0.0)

        # Undertriage: assigned higher number = lower priority = dangerous
        if gt_esi <= 2 and action.esi_level >= 3:
            esi_score = max(0.0, esi_score - self.UNDERTRIAGE_PENALTY_MAJOR)
            errors.append(
                f"UNDERTRIAGE ALERT: Patient requires ESI-{gt_esi} "
                f"(immediate/emergent) but received ESI-{action.esi_level}. "
                f"This represents a life-threatening triage error."
            )
        elif gt_esi == 1 and action.esi_level == 2:
            esi_score = max(0.0, esi_score - self.UNDERTRIAGE_PENALTY_MINOR)
            errors.append(
                "MINOR UNDERTRIAGE: ESI-1 resuscitation patient assigned ESI-2. "
                "Immediate physician presence is mandatory for ESI-1."
            )

        # Hard cap: ESI-1 patient assigned ESI-4/5 → near-zero
        if gt_esi == 1 and action.esi_level >= 4:
            esi_score = 0.0
            errors.append(
                "CRITICAL PATIENT SAFETY FAILURE: ESI-1 resuscitation patient "
                f"assigned to non-urgent category (ESI-{action.esi_level})."
            )

        cs["esi_accuracy"] = round(esi_score, 3)

        # ── 2. Acceptable range bonus ────────────────────────────────────────
        cs["acceptable_range_bonus"] = 1.0 if action.esi_level in acceptable else 0.0

        # ── 3. Rationale keyword density ────────────────────────────────────
        cs["rationale_keywords"] = self._score_rationale_keywords(
            action.rationale, patient, gt_esi
        )

        # ── 4. Clinical flag recognition ─────────────────────────────────────
        cs["clinical_flags"] = self._score_clinical_flags(
            action.rationale, patient
        )

        # ── 5 & 6. Intervention recall + precision ───────────────────────────
        if critical_iv:
            recall    = _fuzzy_list_recall(action.recommended_immediate_interventions, critical_iv)
            fp_rate   = _false_positive_rate(action.recommended_immediate_interventions, critical_iv)
            precision = max(0.0, 1.0 - fp_rate * 0.5)  # soft precision
        else:
            recall = precision = 1.0  # no interventions expected → full credit
        cs["intervention_recall"]    = round(recall, 3)
        cs["intervention_precision"] = round(precision, 3)

        # ── 7. Time-sensitivity awareness ────────────────────────────────────
        cs["time_sensitivity"] = self._score_time_awareness(
            action.rationale, action.recommended_immediate_interventions, gt_esi, patient
        )

        # ── Weighted final ───────────────────────────────────────────────────
        final = sum(self.WEIGHTS[k] * cs[k] for k in self.WEIGHTS)
        final = round(max(0.0, min(1.0, final)), 4)

        confidence = "high" if diff == 0 else ("medium" if diff == 1 else "low")

        return GradeResult(
            score=final,
            component_scores=cs,
            feedback=self._build_feedback(action, gt_esi, acceptable, cs, errors, scenario),
            critical_errors=errors,
            passed=(final >= 0.60 and not errors),
            confidence=confidence,
            teaching_point=teaching,
        )

    def _score_rationale_keywords(self, rationale: str, patient, gt_esi: int) -> float:
        if not rationale or len(rationale.strip()) < 10:
            return 0.0
        v = patient.vitals
        kws: List[str] = []

        if gt_esi <= 2:
            kws += ["urgent", "immediate", "emergent", "critical", "high risk",
                    "life threat", "priority", "time-sensitive"]
        if v.heart_rate > 100:  kws += ["tachycardia", "heart rate", "hr"]
        if v.systolic_bp < 90:  kws += ["hypotension", "blood pressure", "shock"]
        if v.spo2 < 94:         kws += ["hypoxia", "oxygen", "spo2", "saturation"]
        if v.temperature > 38.3: kws += ["fever", "febrile", "temperature"]
        if v.glasgow_coma_scale < 14: kws += ["altered", "confusion", "gcs", "consciousness"]

        cc = patient.chief_complaint.lower()
        if "chest" in cc:     kws += ["chest", "cardiac", "acs", "mi", "stemi", "ecg", "troponin"]
        if any(x in cc for x in ("weakness", "confusion", "droop", "stroke")): 
            kws += ["stroke", "neuro", "focal", "fast", "deficit", "tpa", "ct head"]
        if "headache" in cc:  kws += ["headache", "thunderclap", "subarachnoid", "sah", "lumbar"]
        if "sepsis" in cc or "fever" in cc: kws += ["sepsis", "infection", "antibiotics", "cultures"]

        return _keyword_score(rationale, kws, threshold=3)

    def _score_clinical_flags(self, rationale: str, patient) -> float:
        """Bonus for identifying specific vital sign abnormalities."""
        if not rationale:
            return 0.0
        v = patient.vitals
        flags_expected = []
        flags_found = 0

        if v.heart_rate > 100 or v.heart_rate < 50:
            flags_expected.append(str(v.heart_rate))
        if v.systolic_bp < 90 or v.systolic_bp > 180:
            flags_expected.append(str(v.systolic_bp))
        if v.spo2 < 95:
            flags_expected.append(str(v.spo2))
        if v.glasgow_coma_scale < 15:
            flags_expected.append(str(v.glasgow_coma_scale))
        if v.temperature > 38.5 or v.temperature < 36.0:
            flags_expected.append(str(v.temperature))

        if not flags_expected:
            return 1.0  # all normal — full credit for any coherent rationale

        for flag in flags_expected:
            if flag in rationale:
                flags_found += 1

        return min(1.0, flags_found / len(flags_expected))

    def _score_time_awareness(self, rationale: str, interventions: List[str],
                               gt_esi: int, patient) -> float:
        if gt_esi >= 4:
            return 1.0  # not time-critical
        all_text = (rationale + " " + " ".join(interventions)).lower()
        time_words = ["stat", "immediate", "now", "urgent", "asap", "within",
                      "minute", "door-to", "activate", "alert", "consult"]
        found = sum(1 for w in time_words if w in all_text)
        return min(1.0, found / 2)  # need at least 2 time-awareness signals

    def _build_feedback(self, action, gt_esi, acceptable, cs, errors, scenario) -> str:
        lines = [
            "=== TRIAGE GRADER FEEDBACK (v2) ===",
            f"Patient: {scenario['patient'].chief_complaint}",
            f"Assigned ESI: {action.esi_level}  |  Correct ESI: {gt_esi}  "
            f"|  Acceptable Range: {acceptable}",
            "",
            "Component Scores:",
        ]
        for k, v in cs.items():
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"  {k:35s} {bar}  {v:.3f}")
        if errors:
            lines.append("\n⚠️  PATIENT SAFETY ALERTS:")
            for e in errors:
                lines.append(f"  ✗ {e}")
        if scenario.get("teaching_point"):
            lines.append(f"\n📚 Teaching Point: {scenario['teaching_point']}")
        return "\n".join(lines)


class MedicationSafetyGrader:
    """
    7-component medication safety grader.
    Components: interaction recall, contraindication recall, dosing error recall,
    severity classification, recommended changes quality, clinical rationale depth,
    false positive penalty.
    """

    WEIGHTS = {
        "interaction_recall":      0.28,
        "contraindication_recall": 0.18,
        "dosing_error_recall":     0.10,
        "severity_accuracy":       0.15,
        "recommended_changes":     0.12,
        "rationale_depth":         0.12,
        "fp_penalty":              0.05,
    }

    _SEVERITY_MAP = {"safe": 0, "minor": 1, "moderate": 2, "major": 3, "critical": 4}

    def grade(self, action: MedicationSafetyAction, scenario: Dict[str, Any]) -> GradeResult:
        gt      = scenario["ground_truth"]
        patient = scenario["patient"]
        cs: Dict[str, float] = {}
        errors: List[str] = []

        # ── Recall components ────────────────────────────────────────────────
        cs["interaction_recall"]      = _fuzzy_list_recall(action.flagged_interactions,
                                                            gt["interactions"])
        cs["contraindication_recall"] = _fuzzy_list_recall(action.flagged_contraindications,
                                                            gt["contraindications"])
        cs["dosing_error_recall"]     = _fuzzy_list_recall(action.flagged_dosing_errors,
                                                            gt["dosing_errors"])

        # ── Severity ─────────────────────────────────────────────────────────
        cs["severity_accuracy"] = self._score_severity(
            action.severity_assessment, gt["severity"]
        )

        # ── Recommended changes ──────────────────────────────────────────────
        cs["recommended_changes"] = self._score_recommendations(
            action.recommended_changes, gt, patient
        )

        # ── Rationale depth ──────────────────────────────────────────────────
        cs["rationale_depth"] = self._score_rationale(
            action.clinical_rationale, gt, patient
        )

        # ── False positive penalty ───────────────────────────────────────────
        all_gt = gt["interactions"] + gt["contraindications"] + gt["dosing_errors"]
        all_proposed = (action.flagged_interactions
                        + action.flagged_contraindications
                        + action.flagged_dosing_errors)
        fp_rate = _false_positive_rate(all_proposed, all_gt)
        cs["fp_penalty"] = max(0.0, 1.0 - fp_rate * 1.5)

        # ── Critical error rules ────────────
        gt_sev = gt["severity"]
        prop_sev = action.severity_assessment.lower()

        if gt_sev == "critical" and prop_sev == "safe":
            errors.append("CRITICAL: Life-threatening drug interaction classified as 'safe'.")
        elif gt_sev == "critical" and prop_sev == "minor":
            errors.append("SEVERE UNDERESTIMATE: Critical severity rated as 'minor'.")
        elif gt_sev == "major" and prop_sev == "safe":
            errors.append("SEVERITY ERROR: Major drug interaction classified as 'safe'.")

        # ── Weighted final ───────────────────────────────────────────────────
        final = sum(self.WEIGHTS[k] * cs[k] for k in self.WEIGHTS)
        if errors:
            final = min(final, 0.20 if "CRITICAL" in errors[0] else 0.35)
        final = round(max(0.0, min(1.0, final)), 4)

        return GradeResult(
            score=final,
            component_scores=cs,
            feedback=self._build_feedback(action, gt, cs, errors, scenario),
            critical_errors=errors,
            passed=(final >= 0.60 and not errors),
            confidence="high" if cs["interaction_recall"] > 0.8 else "medium",
            teaching_point=gt.get("key_findings", ""),
        )

    def _score_severity(self, proposed: str, ground_truth: str) -> float:
        p = self._SEVERITY_MAP.get(proposed.lower().strip(), -1)
        g = self._SEVERITY_MAP.get(ground_truth.lower().strip(), -1)
        if p < 0 or g < 0:
            return 0.3
        diff = abs(p - g)
        return {0: 1.0, 1: 0.65, 2: 0.30, 3: 0.10, 4: 0.0}.get(diff, 0.0)

    def _score_recommendations(self, recommended: List[str],
                                gt: Dict, patient) -> float:
        if not recommended:
            return 0.0
        rec_blob = " ".join(recommended).lower()
        # Check for actionable verbs (discontinue, reduce, switch, monitor, hold)
        action_words = ["discontinue", "stop", "hold", "reduce", "switch",
                        "change", "monitor", "avoid", "replace", "add"]
        action_score = min(1.0, sum(1 for w in action_words if w in rec_blob) / 3)

        # Check if GT drugs are mentioned
        gt_drugs = []
        for item in gt["interactions"] + gt["contraindications"]:
            gt_drugs.extend(_tokenise(item))
        drug_score = _keyword_score(rec_blob, [d for d in gt_drugs if len(d) > 4], threshold=2)

        return round((action_score * 0.5 + drug_score * 0.5), 3)

    def _score_rationale(self, rationale: str, gt: Dict, patient) -> float:
        if not rationale or len(rationale.strip()) < 20:
            return 0.0
        score = 0.2  # base for any meaningful text

        # Length bonus (deeper explanation)
        length = len(rationale.split())
        if length >= 50:  score += 0.2
        if length >= 100: score += 0.15
        if length >= 150: score += 0.10

        # Clinical mechanism keywords
        mech_words = ["cyp", "cyp3a4", "inhibit", "substrate", "clearance",
                      "metabolism", "auc", "halflife", "bioavailability",
                      "interaction", "contraindication", "renal", "hepatic",
                      "rhabdomyolysis", "myopathy", "bleed", "coagulation"]
        score += _keyword_score(rationale, mech_words, threshold=3) * 0.35

        return round(min(1.0, score), 3)

    def _build_feedback(self, action, gt, cs, errors, scenario) -> str:
        lines = [
            "=== MEDICATION SAFETY GRADER FEEDBACK (v2) ===",
            f"Patient: {scenario['patient'].chief_complaint}",
            f"GT Severity: {gt['severity']}  |  Proposed: {action.severity_assessment}",
            "",
            "Component Scores:",
        ]
        for k, v in cs.items():
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"  {k:35s} {bar}  {v:.3f}")
        if errors:
            lines.append("\n⚠️  ERRORS:")
            for e in errors:
                lines.append(f"  ✗ {e}")
        if gt.get("key_findings"):
            lines.append(f"\n📚 Key Finding: {gt['key_findings']}")
        return "\n".join(lines)


class SepsisGrader:
    """
    9-component sepsis grader implementing SSC 2021 Hour-1 Bundle.
    Components: diagnosis accuracy, bundle completeness (5 elements),
    antibiotic allergy safety, vasopressor appropriateness,
    fluid volume accuracy, rationale clinical depth.
    """

    WEIGHTS = {
        "diagnosis_accuracy":     0.15,
        "blood_cultures":         0.08,
        "antibiotics":            0.15,
        "antibiotic_safety":      0.12,   # allergy cross-check
        "lactate":                0.08,
        "fluid_volume":           0.10,
        "vasopressor":            0.12,
        "source_control":         0.05,
        "rationale_depth":        0.15,
    }

    # qSOFA thresholds for vasopressor indication
    MAP_THRESHOLD = 65

    def grade(self, action: SepsisManagementAction, scenario: Dict[str, Any]) -> GradeResult:
        patient    = scenario["patient"]
        gt         = scenario.get("ground_truth", {})
        vitals     = patient.vitals
        allergies  = [a.lower() for a in patient.allergies]
        cs: Dict[str, float] = {}
        errors: List[str] = []

        gt_diagnosis = gt.get("diagnosis", "septic_shock")
        gt_antibiotic = gt.get("antibiotic", "piperacillin_tazobactam")

        # ── 1. Diagnosis accuracy ────────────────────────────────────────────
        cs["diagnosis_accuracy"] = self._score_diagnosis(
            action.sepsis_diagnosis, gt_diagnosis, vitals
        )

        # ── 2. Blood cultures ────────────────────────────────────────────────
        cs["blood_cultures"] = 1.0 if action.blood_cultures_ordered else 0.0
        if not action.blood_cultures_ordered:
            errors.append(
                "BUNDLE INCOMPLETE: Blood cultures must be drawn BEFORE antibiotics "
                "to guide de-escalation therapy."
            )

        # ── 3. Antibiotics ordered ───────────────────────────────────────────
        cs["antibiotics"] = 1.0 if action.antibiotics_ordered else 0.0
        if not action.antibiotics_ordered:
            errors.append("BUNDLE INCOMPLETE: Broad-spectrum antibiotics must be given within 1 hour of sepsis recognition.")

        # ── 4. Antibiotic allergy safety ─────────────────────────────────────
        cs["antibiotic_safety"], allergy_error = self._score_antibiotic_safety(
            action.antibiotic_choice, allergies, gt_antibiotic
        )
        if allergy_error:
            errors.append(allergy_error)

        # ── 5. Lactate ordered ───────────────────────────────────────────────
        cs["lactate"] = 1.0 if action.lactate_ordered else 0.0
        if not action.lactate_ordered:
            errors.append("BUNDLE INCOMPLETE: Serum lactate is required to stratify sepsis severity (lactate ≥4 indicates tissue hypoperfusion).")

        # ── 6. Fluid volume ──────────────────────────────────────────────────
        cs["fluid_volume"] = self._score_fluid_volume(
            action.iv_fluid_bolus_ml, vitals, gt
        )

        # ── 7. Vasopressor decision ──────────────────────────────────────────
        map_mmhg = int((vitals.systolic_bp + 2 * vitals.diastolic_bp) / 3)
        requires_vasopressor = map_mmhg < self.MAP_THRESHOLD
        cs["vasopressor"] = self._score_vasopressor(
            action.vasopressor_ordered,
            action.vasopressor_choice,
            requires_vasopressor,
            allergies
        )
        if requires_vasopressor and not action.vasopressor_ordered:
            errors.append(
                f"BUNDLE INCOMPLETE: MAP={map_mmhg} mmHg (below 65 threshold). "
                "Vasopressors (norepinephrine first-line) should be initiated."
            )

        # ── 8. Source control ────────────────────────────────────────────────
        gt_source = gt.get("source", "")
        if gt_source and action.source_control_identified:
            sim = _token_overlap(action.source_control_identified, gt_source)
            cs["source_control"] = min(1.0, sim * 2 + 0.3)
        elif not gt_source:
            cs["source_control"] = 1.0  # source unclear — full credit
        else:
            cs["source_control"] = 0.0

        # ── 9. Rationale depth ───────────────────────────────────────────────
        cs["rationale_depth"] = self._score_sepsis_rationale(
            action.clinical_rationale, vitals, action
        )

        # ── Time-to-antibiotics bonus ────────────────────────────────────────
        tta_bonus = 0.0
        if action.time_to_antibiotics_minutes is not None:
            if action.time_to_antibiotics_minutes <= 30:   tta_bonus = 0.03
            elif action.time_to_antibiotics_minutes <= 60: tta_bonus = 0.01
            elif action.time_to_antibiotics_minutes > 120: tta_bonus = -0.02

        # ── Weighted final ───────────────────────────────────────────────────
        final = sum(self.WEIGHTS[k] * cs[k] for k in self.WEIGHTS) + tta_bonus
        final = round(max(0.0, min(1.0, final)), 4)

        confidence = (
            "high"   if len(errors) == 0 and final >= 0.75 else
            "medium" if len(errors) <= 1 and final >= 0.50 else
            "low"
        )

        return GradeResult(
            score=final,
            component_scores=cs,
            feedback=self._build_feedback(action, cs, errors, scenario, map_mmhg),
            critical_errors=errors,
            passed=(final >= 0.60 and not [e for e in errors if "BUNDLE INCOMPLETE" in e]),
            confidence=confidence,
            teaching_point=gt.get("teaching_point", ""),
        )

    def _score_diagnosis(self, proposed: str, gt: str, vitals) -> float:
        """Partial credit for diagnosis — spectrum: SIRS → sepsis → septic_shock."""
        diagnosis_order = ["no_sepsis", "SIRS_only", "sepsis", "septic_shock"]
        p_idx = next((i for i, d in enumerate(diagnosis_order)
                      if d.lower() == proposed.lower()), -1)
        g_idx = next((i for i, d in enumerate(diagnosis_order)
                      if d.lower() == gt.lower()), -1)
        if p_idx < 0 or g_idx < 0:
            return 0.3
        diff = abs(p_idx - g_idx)
        return {0: 1.0, 1: 0.55, 2: 0.20, 3: 0.0}.get(diff, 0.0)

    def _score_antibiotic_safety(
        self, choice: Optional[str], allergies: List[str], gt_antibiotic: str
    ) -> Tuple[float, str]:
        """
        Cross-checks antibiotic choice against documented allergies.
        Returns (score, error_message_or_empty).
        """
        if not choice:
            return 0.0, ""

        choice_l = choice.lower().replace("_", " ").replace("-", " ")

        # Define dangerous cross-reactivities
        CONTRAINDICATED = {
            "penicillin":   ["penicillin", "amoxicillin", "ampicillin",
                             "piperacillin tazobactam", "piperacillin"],
            "vancomycin":   ["vancomycin"],
            "cephalosporin":["ceftriaxone", "cefazolin", "cefepime",
                             "cefuroxime", "cephalosporin"],
            "sulfa":        ["trimethoprim sulfamethoxazole", "sulfamethoxazole"],
        }

        for allergy in allergies:
            for key, drugs in CONTRAINDICATED.items():
                if key in allergy:
                    for drug in drugs:
                        if drug in choice_l:
                            return 0.0, (
                                f"ALLERGY VIOLATION: Patient has documented {allergy} allergy. "
                                f"Prescribed '{choice}' is contraindicated. "
                                f"Use a safe alternative (e.g., vancomycin for MRSA if no vanco allergy, "
                                f"meropenem or daptomycin for pen-allergic patients)."
                            )

        # Score the choice quality vs ground truth
        gt_score = _token_overlap(choice_l, gt_antibiotic.lower().replace("_", " "))
        return round(0.4 + gt_score * 0.6, 3), ""

    def _score_fluid_volume(self, bolus_ml: int, vitals, gt: Dict) -> float:
        """30ml/kg is the SSC standard. Penalise under- and over-resuscitation."""
        # Estimate expected volume (assume 70kg average)
        expected_ml = 2100  # 30 ml/kg × 70 kg
        gt_ml = gt.get("expected_fluid_ml", expected_ml)

        map_mmhg = int((vitals.systolic_bp + 2 * vitals.diastolic_bp) / 3)
        requires_fluids = (map_mmhg < 65 or
                           getattr(vitals, "lactate", 0) >= 4)

        if not requires_fluids:
            # Still reasonable to give some fluids, but not mandatory
            if bolus_ml == 0:
                return 1.0
            if bolus_ml <= 1000:
                return 0.8
            return 0.6

        if bolus_ml == 0:
            return 0.0  # failed to give fluids despite indication

        # Partial credit based on % of target
        ratio = bolus_ml / gt_ml
        if   0.80 <= ratio <= 1.20: return 1.0
        elif 0.60 <= ratio <  0.80: return 0.75
        elif 0.40 <= ratio <  0.60: return 0.50
        elif 1.20 <  ratio <= 1.60: return 0.70  # mild over-resuscitation
        elif ratio > 1.60:          return 0.40  # over-resuscitation risk
        else:                       return 0.20  # severely inadequate

    def _score_vasopressor(self, ordered: bool, choice: Optional[str],
                            required: bool, allergies: List[str]) -> float:
        if not required:
            return 1.0 if not ordered else 0.6  # not indicated, but mild penalty if given anyway

        if not ordered:
            return 0.0  # required but not given — covered by error above

        if not choice:
            return 0.5  # ordered but no specific choice — partial credit

        choice_l = choice.lower().replace("_", " ")
        # Norepinephrine is first-line per SSC
        if "norepinephrine" in choice_l or "noradrenaline" in choice_l:
            return 1.0
        # Vasopressin is acceptable as adjunct
        if "vasopressin" in choice_l:
            return 0.75
        # Epinephrine / dopamine — acceptable if NE contraindicated
        if any(x in choice_l for x in ["epinephrine", "adrenaline", "dopamine"]):
            return 0.60
        # Phenylephrine — weak, not recommended for septic shock
        if "phenylephrine" in choice_l:
            return 0.30
        return 0.40

    def _score_sepsis_rationale(
        self, rationale: str, vitals, action: SepsisManagementAction
    ) -> float:
        if not rationale or len(rationale.strip()) < 20:
            return 0.0
        score = 0.2

        length = len(rationale.split())
        if length >= 50:  score += 0.2
        if length >= 100: score += 0.15

        clinical_kws = [
            "sepsis", "bundle", "ssc", "sofa", "qsofa", "lactate", "map",
            "crystalloid", "fluid", "antibiotic", "cultures", "vasopressor",
            "norepinephrine", "hypotension", "organ", "perfusion", "infection",
            "source", "control", "hour", "mortality"
        ]
        score += _keyword_score(rationale, clinical_kws, threshold=4) * 0.45

        return round(min(1.0, score), 3)

    def _build_feedback(self, action, cs, errors, scenario, map_mmhg: int) -> str:
        p = scenario["patient"]
        lines = [
            "=== SEPSIS GRADER FEEDBACK (v2) ===",
            f"Patient: {p.chief_complaint}",
            f"MAP: {map_mmhg} mmHg  |  Diagnosis: {action.sepsis_diagnosis}",
            f"Antibiotics: {action.antibiotic_choice or 'none'}  "
            f"|  Fluids: {action.iv_fluid_bolus_ml}mL  "
            f"|  Vasopressors: {'YES' if action.vasopressor_ordered else 'NO'}",
            "",
            "SSC Hour-1 Bundle Completion:",
            f"  {'✅' if action.blood_cultures_ordered else '❌'} Blood cultures before antibiotics",
            f"  {'✅' if action.antibiotics_ordered      else '❌'} Broad-spectrum antibiotics",
            f"  {'✅' if action.lactate_ordered           else '❌'} Serum lactate",
            f"  {'✅' if action.iv_fluid_bolus_ml > 0    else '❌'} IV fluid bolus",
            f"  {'✅' if action.vasopressor_ordered       else ('N/A' if map_mmhg >= 65 else '❌')} Vasopressors (MAP={map_mmhg})",
            "",
            "Component Scores:",
        ]
        for k, v in cs.items():
            bar = "█" * int(v * 10) + "░" * (10 - int(v * 10))
            lines.append(f"  {k:35s} {bar}  {v:.3f}")
        if errors:
            lines.append("\n⚠️  ERRORS:")
            for e in errors:
                lines.append(f"  ✗ {e}")
        tp = scenario.get("ground_truth", {}).get("teaching_point", "")
        if tp:
            lines.append(f"\n📚 Teaching Point: {tp}")
        return "\n".join(lines)
