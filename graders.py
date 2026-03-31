from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re

from models import TriageAction, MedicationSafetyAction, SepsisManagementAction


@dataclass
class GradeResult:
    """Detailed grading result."""
    score: float                        # 0.0 – 1.0
                    component_scores: Dict[str, float]  # breakdown by criterion
    feedback: str                        # human-readable explanation
                critical_errors: List[str] = field(default_factory=list)  # safety-critical failures
    passed: bool = False


# 
# GRADER 1: Emergency Triage (ESI Level Assignment)
# 

class TriageGrader:
    """
    Grades ESI level assignment. Partial credit for being 1 level off.
    Critical penalties for undertriage (assigning lower acuity than needed).
    Undertriage is MORE dangerous than overtriage in emergency medicine.
    """

    # Penalty weights
                 EXACT_MATCH_SCORE = 1.0
                         ONE_OFF_SCORE = 0.6
                     TWO_OFF_SCORE = 0.2
             UNDERTRIAGE_PENALTY = 0.4   # Extra penalty: ESI assigned HIGHER number = lower priority
    RATIONALE_WEIGHT = 0.2
    INTERVENTION_WEIGHT = 0.15

    def grade(
        self,
                action: TriageAction,
        scenario: Dict[str, Any]
    ) -> GradeResult:

        ground_truth_esi = scenario["ground_truth_esi"]
        acceptable_range = scenario["acceptable_esi_range"]
  critical_interventions = scenario.get("critical_interventions", [])
        difficulty = scenario.get("difficulty", "medium")

        component_scores: Dict[str, float] = {}
                      critical_errors: List[str] = []

        # ── Component 1: ESI accuracy (0–1) ──────────────────────
        esi_diff = abs(action.esi_level - ground_truth_esi)
        if esi_diff == 0:
            esi_score = 1.0
          
        elif action.esi_level in acceptable_range:
              esi_score = 0.75
        elif esi_diff == 1:
          
            esi_score = 0.5
          
        elif esi_diff == 2:
            esi_score = 0.2
        else:
            esi_score = 0.0

        # Undertriage: assigning HIGHER ESI number (lower priority) when patient is critical
        if ground_truth_esi <= 2 and action.esi_level >= 3:
            esi_score = max(0.0, esi_score - self.UNDERTRIAGE_PENALTY)
            critical_errors.append(
                f"UNDERTRIAGE: Patient requires ESI-{ground_truth_esi} but assigned ESI-{action.esi_level}. "
                f"This is a life-threatening safety error in emergency medicine."
            )

        component_scores["esi_accuracy"] = esi_score

        # ── Component 2: Rationale quality
        rationale_score = self._score_rationale(
            action.rationale, scenario, ground_truth_esi
        )
        component_scores["rationale_quality"] = rationale_score

        # ── Component 3: Critical interventions 
        if critical_interventions:
            intervention_score = self._score_interventions(
                action.recommended_immediate_interventions,
                critical_interventions
            )
        else:
            intervention_score = 1.0  # No interventions expected, full credit
        component_scores["critical_interventions"] = intervention_score

        # ── Weighted final score 
        # ESI accuracy is most important (65%), rationale 20%, interventions 15%
        final = (
            0.65 * esi_score +
            0.20 * rationale_score +
            0.15 * intervention_score
        )
        # Clamp
        final = max(0.0, min(1.0, final))

        # Hard fail: ESI-1 patient assigned ESI-4 or 5
        if ground_truth_esi == 1 and action.esi_level >= 4:
            final = min(final, 0.1)
            critical_errors.append("CRITICAL: ESI-1 resuscitation patient assigned to non-urgent category.")

        feedback = self._build_feedback(
            action, ground_truth_esi, acceptable_range,
            component_scores, critical_errors, scenario
        )

        return GradeResult(
            score=round(final, 3),
            component_scores=component_scores,
            feedback=feedback,
            critical_errors=critical_errors,
            passed=(final >= 0.6 and not critical_errors)
        )

    def _score_rationale(self, rationale: str, scenario: Dict, gt_esi: int) -> float:
        """Score rationale based on clinical keywords and completeness."""
        if not rationale or len(rationale.strip()) < 10:
            return 0.0

        rationale_lower = rationale.lower()
        score = 0.3  # Base for any rationale

        # Check for relevant clinical terms
        vitals = scenario["patient"].vitals
        keywords_found = 0
        expected_keywords = []

        # Build expected keywords from scenario
        if gt_esi <= 2:
            expected_keywords.extend(["urgent", "immediate", "critical", "emergent", "high risk",
                                       "life threat", "time", "priority"])
        if vitals.heart_rate > 100:
                  expected_keywords.extend(["tachycardia", "heart rate", "hr"])
        if vitals.systolic_bp < 90:
            expected_keywords.extend(["hypotension", "blood pressure", "bp", "shock"])
          
                if vitals.spo2 < 94:
            expected_keywords.extend(["hypoxia", "oxygen", "spo2", "saturation"])

      
        if vitals.temperature > 38.5:
            expected_keywords.extend(["fever", "temperature", "febrile"])
          

        chief = scenario["patient"].chief_complaint.lower()
        if "chest" in chief:
            expected_keywords.extend(["chest", "cardiac", "acs", "mi", "stemi", "ecg"])
        if "stroke" in chief or "weakness" in chief or "confusion" in chief:
            expected_keywords.extend(["stroke", "neuro", "focal", "fast", "deficit"])
        if "headache" in chief:
            expected_keywords.extend(["headache", "sah", "subarachnoid", "thunderclap"])

        for kw in expected_keywords:
            if kw in rationale_lower:
                keywords_found += 1

        if expected_keywords:
            keyword_score = min(1.0, keywords_found / max(3, len(expected_keywords) * 0.4))
            score += 0.7 * keyword_score

        return min(1.0, score)

    def _score_interventions(self, proposed: List[str], expected: List[str]) -> float:
        """Score intervention completeness."""
        if not expected:
            return 1.0
        if not proposed:
            return 0.0

        proposed_str = " ".join(proposed).lower()
        found = 0
        for intervention in expected:
            # Flexible matching
            intervention_words = intervention.lower().replace("_", " ").split()
            if any(word in proposed_str for word in intervention_words):
                found += 1

        return min(1.0, found / len(expected))

    def _build_feedback(self, action, gt_esi, acceptable, components, errors, scenario):
        lines = [
            f"=== TRIAGE GRADER FEEDBACK ===",
                     f"Patient: {scenario['patient'].chief_complaint}",
                        f"Your ESI: {action.esi_level} | Correct ESI: {gt_esi} (acceptable: {acceptable})",
                       f"",
            f"Component Scores:",
                     f"  ESI Accuracy:            {components['esi_accuracy']:.2f}",
  f"  Rationale Quality:       {components['rationale_quality']:.2f}",
            f"  Critical Interventions:  {components['critical_interventions']:.2f}",
        ]
        if errors:
            lines.append(f"\n⚠️  CRITICAL ERRORS:")
            for e in errors:
                lines.append(f"  - {e}")
        lines.append(f"\nTeaching Point: {scenario.get('teaching_point', 'N/A')}")
          return "\n".join(lines)


#
# GRADER 2: Medication Safety Review
# 

class MedicationSafetyGrader:
    """
    Grades medication safety review. Rewards catching interactions,
    contraindications, and dosing errors. Penalizes false positives
    (hallucinating interactions that don't exist).
    """

    def grade(
        self,
        action: MedicationSafetyAction,
        scenario: Dict[str, Any]
    ) -> GradeResult:

        gt = scenario["ground_truth"]
        component_scores: Dict[str, float] = {}
        critical_errors: List[str] = []

        # ── Component 1: Interaction detection ──────────────────
        interaction_score = self._score_list_detection(
            proposed=action.flagged_interactions,
            ground_truth=gt["interactions"],
            allow_partial_match=True
        )
        component_scores["interaction_detection"] = interaction_score

        # ── Component 2: Contraindication detection ─────────────
        contra_score = self._score_list_detection(
            proposed=action.flagged_contraindications,
                     ground_truth=gt["contraindications"],
            allow_partial_match=True
        )
        component_scores["contraindication_detection"] = contra_score

        # ── Component 3: Dosing error detection ─────────────────
        dose_score = self._score_list_detection(
                             proposed=action.flagged_dosing_errors,
            ground_truth=gt["dosing_errors"],
            allow_partial_match=True
        )
        component_scores["dosing_error_detection"] = dose_score

        # ── Component 4: Severity classification ─────────────────
        severity_score = self._score_severity(action.severity_assessment, gt["severity"])
                   component_scores["severity_classification"] = severity_score

        # ── Component 5: Clinical rationale ──────────────────────
        rationale_score = self._score_med_rationale(
            action.clinical_rationale, gt, scenario
        )
        component_scores["clinical_rationale"] = rationale_score

        # ── Component 6: False positive penalty ──────────────────
        fp_penalty = self._false_positive_penalty(
            action.flagged_interactions,
            action.flagged_contraindications,
            gt
        )
        component_scores["false_positive_penalty"] = fp_penalty

        # Weighted final
        final = (
            0.25 * interaction_score +
                      0.20 * contra_score +
                     0.15 * dose_score +
            0.15 * severity_score +
                    0.15 * rationale_score +
            0.10 * fp_penalty
        )
        final = max(0.0, min(1.0, final))

        # Critical error: missed critical severity and called it "safe"
        if gt["severity"] == "critical" and action.severity_assessment == "safe":
            final = min(final, 0.15)
            critical_errors.append("CRITICAL: Classified a critical/life-threatening medication situation as 'safe'.")

        if gt["severity"] == "critical" and action.severity_assessment == "minor":
              final = min(final, 0.25)
                      critical_errors.append("Severely underestimated severity of critical medication interaction.")

        feedback = self._build_feedback(action, gt, component_scores, critical_errors, scenario)

        return GradeResult(
            score=round(final, 3),
            component_scores=component_scores,
                       feedback=feedback,
                     critical_errors=critical_errors,
            passed=(final >= 0.6 and not critical_errors)
        )

    def _score_list_detection(self, proposed: List[str], ground_truth: List[str],
                               allow_partial_match: bool = True) -> float:
        if not ground_truth:
            # No issues expected: reward if agent also found nothing
            if not proposed:
                return 1.0
            else:
                return max(0.0, 1.0 - 0.1 * len(proposed))  # Minor FP penalty

        if not proposed:
            return 0.0

        proposed_str = " ".join(proposed).lower()
        found = 0
        for gt_item in ground_truth:
            gt_words = re.split(r'[\s\+\-\_\(\)]', gt_item.lower())
            gt_words = [w for w in gt_words if len(w) > 3]
            matches = sum(1 for w in gt_words if w in proposed_str)
            if allow_partial_match:
                if matches >= max(1, len(gt_words) * 0.4):
                    found += 1
            else:
                if matches == len(gt_words):
                    found += 1

        recall = found / len(ground_truth)
        return round(recall, 3)

    def _score_severity(self, proposed: str, ground_truth: str) -> float:
        severity_map = {"safe": 0, "minor": 1, "moderate": 2, "major": 3, "critical": 4}
        p = severity_map.get(proposed.lower(), -1)
        g = severity_map.get(ground_truth.lower(), -1)
        if p == -1 or g == -1:
            return 0.3
        diff = abs(p - g)
        return {0: 1.0, 1: 0.6, 2: 0.3, 3: 0.1, 4: 0.0}.get(diff, 0.0)

    def _score_med_rationale(self, rationale: str, gt: Dict, scenario: Dict) -> float:
        if not rationale or len(rationale.strip()) < 20:
            return 0.0

        score = 0.2
        rationale_lower = rationale.lower()

        # Check for drug names mentioned
        meds = [m.name.lower() for m in scenario["patient"].current_medications]
                 mentioned = sum(1 for m in meds if m.split("_")[0] in rationale_lower)
        score += 0.3 * min(1.0, mentioned / max(1, len(meds)))

        # Key finding keywords
        key_findings = gt.get("key_findings", "").lower()
        key_words = [w for w in key_findings.split() if len(w) > 5]
        found = sum(1 for w in key_words[:10] if w in rationale_lower)
        score += 0.5 * min(1.0, found / max(1, min(10, len(key_words))) * 2)

        return min(1.0, score)

    def _false_positive_penalty(self, proposed_interactions, proposed_contras, gt):
        """Penalize hallucinated interactions."""
        if not proposed_interactions and not proposed_contras:
            return 1.0
        # Simple heuristic: if claimed many interactions but GT has few, penalize
        total_proposed = len(proposed_interactions) + len(proposed_contras)
        total_gt = len(gt["interactions"]) + len(gt["contraindications"])
        if total_gt == 0 and total_proposed > 3:
            return 0.4
        ratio = total_proposed / max(1, total_gt)
        if ratio > 4:
            return 0.5
        elif ratio > 2.5:
            return 0.75
        return 1.0

    def _build_feedback(self, action, gt, components, errors, scenario):
        lines = [
            "=== MEDICATION SAFETY GRADER FEEDBACK ===",
            f"Patient: {scenario['patient'].chief_complaint}",
            f"Expected severity: {gt['severity']} | Your severity: {action.severity_assessment}",
            "",
            "Component Scores:",
            f"  Interaction Detection:    {components['interaction_detection']:.2f}",
                         f"  Contraindication Detect:  {components['contraindication_detection']:.2f}",
                      f"  Dosing Error Detection:   {components['dosing_error_detection']:.2f}",
                                    f"  Severity Classification:  {components['severity_classification']:.2f}",
            f"  Clinical Rationale:       {components['clinical_rationale']:.2f}",
    f"  False Positive Penalty:   {components['false_positive_penalty']:.2f}",
        ]
        if errors:
            lines.append("\n⚠️  CRITICAL ERRORS:")
            for e in errors:
                lines.append(f"  - {e}")
        lines.append(f"\nKey Findings: {gt['key_findings']}")
        return "\n".join(lines)


# 
# GRADER 3: Sepsis Management (Hour-1 Bundle)
# 

class SepsisGrader:
    """
    Grades sepsis recognition and Hour-1 Surviving Sepsis Campaign bundle execution.
    Time-sensitive elements have higher weight. Missing vasopressors in septic shock
    is a critical error.
    """

    def grade(
        self,
        action: SepsisManagementAction,
        scenario: Dict[str, Any]
    ) -> GradeResult:

        gt = scenario["ground_truth"]
        component_scores: Dict[str, float] = {}
        critical_errors: List[str] = []

        # ── Component 1: Correct diagnosis ─────────────────────
        diagnosis_score = self._score_diagnosis(action.sepsis_diagnosis, gt["diagnosis"])
        component_scores["diagnosis"] = diagnosis_score

        # ── Component 2: Blood cultures before antibiotics ──────
        # Cultures before antibiotics is critical (don't start abx before cultures)
        bundle_score, bundle_errors = self._score_bundle(action, gt, scenario)
        component_scores["bundle_compliance"] = bundle_score
        critical_errors.extend(bundle_errors)

        # ── Component 3: Antibiotic appropriateness ─────────────
        abx_score = self._score_antibiotics(action, gt, scenario)
        component_scores["antibiotic_appropriateness"] = abx_score

        # ── Component 4: Fluid resuscitation ────────────────────
        fluid_score = self._score_fluids(action, gt, scenario)
        component_scores["fluid_resuscitation"] = fluid_score

        # ── Component 5: Vasopressor decision ───────────────────
        vaso_score, vaso_errors = self._score_vasopressors(action, gt, scenario)
        component_scores["vasopressor_decision"] = vaso_score
        critical_errors.extend(vaso_errors)

        # ── Component 6: Clinical rationale ─────────────────────
        rationale_score = self._score_sepsis_rationale(action.clinical_rationale, gt, scenario)
        component_scores["clinical_rationale"] = rationale_score

        # Weighted final
        final = (
            0.20 * diagnosis_score +
                  0.20 * bundle_score +
            0.20 * abx_score +
                    0.15 * fluid_score +
            0.15 * vaso_score +
                     0.10 * rationale_score
        )
        final = max(0.0, min(1.0, final))

        # Hard penalties for critical errors
        if critical_errors:
            final = min(final, 0.4)

        feedback = self._build_feedback(action, gt, component_scores, critical_errors, scenario)

        return GradeResult(
                       score=round(final, 3),
            component_scores=component_scores,
                       feedback=feedback,
            critical_errors=critical_errors,
                     passed=(final >= 0.6 and not critical_errors)
        )

    def _score_diagnosis(self, proposed: str, gt: str) -> float:
        diagnosis_scores = {
            ("septic_shock", "septic_shock"): 1.0,
            ("sepsis", "sepsis"): 1.0,
                       ("SIRS_only", "SIRS_only"): 1.0,
                            ("no_sepsis", "no_sepsis"): 1.0,
            ("sepsis", "septic_shock"): 0.5,      # Underdiagnosed severity
            ("septic_shock", "sepsis"): 0.7,       # Overdiagnosed but safer error
            ("SIRS_only", "sepsis"): 0.3,
       ("SIRS_only", "septic_shock"): 0.1,
            ("no_sepsis", "sepsis"): 0.0,
       ("no_sepsis", "septic_shock"): 0.0,
        }
        return diagnosis_scores.get((proposed, gt), 0.2)

    def _score_bundle(self, action: SepsisManagementAction, gt: Dict, scenario: Dict):
        errors = []
        score = 0.0
        bundle_gt = gt["bundle"]

        # Blood cultures
        if action.blood_cultures_ordered:
            score += 0.3
        elif bundle_gt["blood_cultures"]:
            errors.append("CRITICAL: Blood cultures should be drawn before antibiotics.")

        # Antibiotics ordered
        if action.antibiotics_ordered:
            score += 0.4
        elif bundle_gt["antibiotics"]:
            errors.append("CRITICAL: Antibiotics not ordered. Time to antibiotics is the #1 mortality predictor in sepsis.")

        # Lactate ordered
        if action.lactate_ordered:
            score += 0.3
        return min(1.0, score), errors

    def _score_antibiotics(self, action: SepsisManagementAction, gt: Dict, scenario: Dict):
        if not action.antibiotics_ordered:
            return 0.0
        if not action.antibiotic_choice:
            return 0.3  # Ordered but no choice specified

        choice = action.antibiotic_choice.lower()
        expected = gt["bundle"]["antibiotic_choice"].lower()

        # Check allergies
        allergies = [a.lower() for a in scenario["patient"].allergies]
        allergy_violations = []

        # Map allergy keywords to drug class
        if "penicillin" in allergies:
            forbidden = ["ampicillin", "amoxicillin", "piperacillin", "oxacillin"]
            for f in forbidden:
                if f in choice:
                    allergy_violations.append(f"Used {f} in penicillin-allergic patient!")

        if "vancomycin" in allergies:
            if "vancomycin" in choice:
                allergy_violations.append("Used vancomycin in vancomycin-allergic patient (red man syndrome)!")

        if allergy_violations:
            return 0.0  # Hard fail for allergy violation

        # Score antibiotic choice
        expected_words = re.split(r'[_\+\s]', expected)
        choice_words = re.split(r'[_\+\s]', choice)
        matches = sum(1 for w in expected_words if any(w in c for c in choice_words))
        score = matches / max(1, len(expected_words))

        # Reward broad-spectrum for severe sepsis/shock
        broad_spectrum = ["meropenem", "piperacillin_tazobactam", "cefepime", "vancomycin",
                          "imipenem", "ceftriaxone", "ciprofloxacin"]
        if any(b in choice for b in broad_spectrum) and gt["diagnosis"] in ["sepsis", "septic_shock"]:
            score = max(score, 0.6)

        return min(1.0, score)

    def _score_fluids(self, action: SepsisManagementAction, gt: Dict, scenario: Dict):
        if gt["diagnosis"] == "no_sepsis":
            if action.iv_fluid_bolus_ml == 0:
                return 1.0
            return 0.8

        # Calculate expected fluid based on weight (estimated)
        patient_age = scenario["patient"].age
        # Rough weight estimate: 70kg default
        estimated_weight = 70
        expected_fluids = estimated_weight * 30  # 30mL/kg = 2100mL

        gt_vasopressors = gt["bundle"].get("vasopressors", False)
        if gt_vasopressors and action.iv_fluid_bolus_ml == 0:
            return 0.2  # Septic shock needs fluids

        if action.iv_fluid_bolus_ml <= 0:
            return 0.1

        ratio = action.iv_fluid_bolus_ml / expected_fluids
        if 0.6 <= ratio <= 1.5:
            return 1.0
        elif 0.4 <= ratio < 0.6:
            return 0.7
        elif 1.5 < ratio <= 2.0:
            return 0.8  # More is ok in shock
        elif ratio > 2.0:
            return 0.5  # Excessive fluids risk
        return 0.4

    def _score_vasopressors(self, action: SepsisManagementAction, gt: Dict, scenario: Dict):
        errors = []
        needs_vaso = gt["bundle"].get("vasopressors", False)

        if needs_vaso and not action.vasopressor_ordered:
            errors.append("CRITICAL: Septic shock (MAP<65 despite fluids) requires vasopressors. Norepinephrine is first-line.")
            return 0.1, errors

        if not needs_vaso and action.vasopressor_ordered:
            # Ordered when not needed - minor error
            return 0.6, []

        if needs_vaso and action.vasopressor_ordered:
            score = 0.7  # Correct decision
            if action.vasopressor_choice:
                choice = action.vasopressor_choice.lower()
                if "norepinephrine" in choice or "noradrenaline" in choice:
                    score = 1.0  # First-line
                elif "vasopressin" in choice or "epinephrine" in choice:
                    score = 0.85  # Acceptable second-line
                elif "dopamine" in choice:
                    score = 0.6  # No longer first-line (arrhythmia risk)
            return score, []

        return 1.0, []  # No vasopressors needed and not ordered

    def _score_sepsis_rationale(self, rationale: str, gt: Dict, scenario: Dict):
        if not rationale or len(rationale.strip()) < 20:
            return 0.0

        score = 0.2
        r = rationale.lower()

        # Check for sepsis-specific terminology
        sepsis_terms = ["sepsis", "sirs", "qsofa", "sofa", "lactate", "fluid", "antibiotic",
                        "blood culture", "organ", "shock", "map", "vasopressor"]
        found = sum(1 for t in sepsis_terms if t in r)
        score += 0.4 * min(1.0, found / 4)

        # Key finding keywords
        key_words = gt.get("key_note", "").lower().split()
        key_words = [w for w in key_words if len(w) > 4][:15]
        found_key = sum(1 for w in key_words if w in r)
        score += 0.4 * min(1.0, found_key / max(1, min(8, len(key_words))))

        return min(1.0, score)

    def _build_feedback(self, action, gt, components, errors, scenario):
        lines = [
            "=== SEPSIS MANAGEMENT GRADER FEEDBACK ===",
            f"Patient: {scenario['patient'].chief_complaint}",
            f"Your diagnosis: {action.sepsis_diagnosis} | Correct: {gt['diagnosis']}",
            "",
            "Component Scores:",
            f"  Diagnosis:               {components['diagnosis']:.2f}",
            f"  Bundle Compliance:       {components['bundle_compliance']:.2f}",
            f"  Antibiotic Choice:       {components['antibiotic_appropriateness']:.2f}",
            f"  Fluid Resuscitation:     {components['fluid_resuscitation']:.2f}",
            f"  Vasopressor Decision:    {components['vasopressor_decision']:.2f}",
            f"  Clinical Rationale:      {components['clinical_rationale']:.2f}",
        ]
        if errors:
            lines.append("\n⚠️  CRITICAL ERRORS:")
            for e in errors:
                lines.append(f"  - {e}")
        lines.append(f"\nKey Clinical Note: {gt.get('key_note', 'N/A')}")
        return "\n".join(lines)
