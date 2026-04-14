from __future__ import annotations

import os
import json
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

ACTIONS = [
    "Assign ESI-1 (Resuscitation)",    # Immediate life threat
    "Assign ESI-2 (Emergent)",         # High-risk, < 10 min
    "Assign ESI-3 (Urgent)",           # < 30 min, ≥2 resources
    "Assign ESI-4 (Less Urgent)",      # < 1 hour, 1 resource
    "Assign ESI-5 (Non-Urgent)",       # < 2 hours, no resources
]

ESI_FROM_ACTION = {a: i + 1 for i, a in enumerate(ACTIONS)}
ACTION_FROM_ESI = {i + 1: a for i, a in enumerate(ACTIONS)}

ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_ACTION = {i: a for i, a in enumerate(ACTIONS)}

# Clinical safety: under-triaging ESI-1 to ESI-3+ is lethal
SAFETY_MATRIX = {
    # (true_esi, agent_esi): safety_score
    (1, 1): 1.0, (1, 2): 0.5, (1, 3): -0.5, (1, 4): -1.0, (1, 5): -1.0,
    (2, 1): 0.9, (2, 2): 1.0, (2, 3): 0.3,  (2, 4): -0.5, (2, 5): -0.8,
    (3, 1): 0.6, (3, 2): 0.8, (3, 3): 1.0,  (3, 4): 0.4,  (3, 5): -0.2,
    (4, 1): 0.3, (4, 2): 0.5, (4, 3): 0.8,  (4, 4): 1.0,  (4, 5): 0.6,
    (5, 1): 0.1, (5, 2): 0.3, (5, 3): 0.6,  (5, 4): 0.9,  (5, 5): 1.0,
}


def featurise(state: Dict[str, Any]) -> Tuple:
    """
    Convert raw clinical state into a discrete 7-tuple Q-table key.
    Dimensions:
      spo2_zone:    0=crisis(<90%), 1=low(90-94%), 2=normal(95-100%)
      hr_zone:      0=brady(<60),   1=normal(60-100), 2=tachy(>100)
      bp_zone:      0=hypo(<90),    1=normal(90-160), 2=hyper(>160)
      gcs_zone:     0=severe(≤8),   1=moderate(9-13), 2=normal(14-15)
      age_zone:     0=child(<18),   1=adult(18-64), 2=elderly(65+)
      red_flag:     0/1 (immediate life threat keywords)
      amber_flag:   0/1 (urgent presentation keywords)
    """
    spo2 = float(state.get("oxygen_level", state.get("spo2", 98)))
    hr   = float(state.get("heart_rate",   state.get("hr", 75)))
    age  = int(state.get("age", 35))
    gcs  = int(state.get("glasgow_coma_scale", state.get("gcs", 15)))
    news2 = int(state.get("news2_score", 0))

    # Parse BP
    bp_raw = state.get("blood_pressure", state.get("sbp", "120/80"))
    try:
        sys_bp = int(str(bp_raw).split("/")[0]) if "/" in str(bp_raw) else int(bp_raw)
    except (ValueError, TypeError):
        sys_bp = 120

    # Zones
    spo2_zone = 0 if spo2 < 90 else (1 if spo2 < 95 else 2)
    hr_zone   = 0 if hr < 60  else (2 if hr > 100 else 1)
    bp_zone   = 0 if sys_bp < 90 else (2 if sys_bp > 160 else 1)
    gcs_zone  = 0 if gcs <= 8 else (1 if gcs <= 13 else 2)
    age_zone  = 0 if age < 18 else (2 if age > 64 else 1)

    # Red/amber flag keywords
    syms = " ".join(str(s).lower() for s in state.get("symptoms", state.get("chief_complaint", "")))
    red_kw = [
        "chest pain", "crushing", "loss of consciousness", "unresponsive", "stroke",
        "anaphylaxis", "massive bleeding", "cyanosis", "respiratory distress",
        "throat swelling", "facial droop", "slurred speech", "arrest", "seizure",
        "shortness of breath", "no light perception", "dissection", "haemorrhage",
    ]
    amber_kw = [
        "fever", "stiff neck", "abdominal pain", "vomiting", "confusion",
        "headache", "wheezing", "broken bone", "back pain", "dyspnea",
        "palpitations", "syncope", "dizziness",
    ]

    red_flag   = int(any(k in syms for k in red_kw) or spo2 < 88 or sys_bp < 80 or gcs <= 8)
    amber_flag = int(any(k in syms for k in amber_kw) or news2 >= 4)

    return (spo2_zone, hr_zone, bp_zone, gcs_zone, age_zone, red_flag, amber_flag)


def _esi_from_feat(feat: Tuple) -> int:
    """Heuristic ESI from feature vector (for policy initialisation)."""
    spo2_zone, hr_zone, bp_zone, gcs_zone, _, red_flag, amber_flag = feat
    if red_flag or gcs_zone == 0 or spo2_zone == 0 or bp_zone == 0:
        return 1 if (gcs_zone == 0 or spo2_zone == 0) else 2
    if amber_flag or hr_zone == 2 or spo2_zone == 1:
        return 2
    return 3



@dataclass
class Experience:
    state_feat:  Tuple
    action_idx:  int
    reward:      float
    next_feat:   Tuple
    done:        bool
    td_error:    float = 1.0  # Initialised high for new experiences
    timestamp:   float = field(default_factory=time.time)


class PrioritisedReplayBuffer:
    """
    Prioritised Experience Replay buffer.
    Samples experiences proportional to |TD error|^α.
    """

    def __init__(self, capacity: int = 2000, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha   # Priority exponent
        self.beta  = beta    # Importance sampling exponent
        self._buf: deque[Experience] = deque(maxlen=capacity)

    def push(self, exp: Experience):
        self._buf.append(exp)

    def sample(self, n: int) -> List[Experience]:
        if len(self._buf) == 0:
            return []
        n = min(n, len(self._buf))
        priorities = [abs(e.td_error) ** self.alpha + 1e-6 for e in self._buf]
        total = sum(priorities)
        probs = [p / total for p in priorities]
        indices = random.choices(range(len(self._buf)), weights=probs, k=n)
        return [self._buf[i] for i in indices]

    def update_td(self, idx: int, td_error: float):
        if 0 <= idx < len(self._buf):
            self._buf[idx].td_error = td_error

    def __len__(self) -> int:
        return len(self._buf)



@dataclass
class EpisodeMetrics:
    episode:          int
    total_reward:     float
    mean_reward:      float
    esi_accuracy:     float   # % exact ESI match
    undertriage_rate: float   # % under-triage events
    safety_score:     float   # Mean safety score from SAFETY_MATRIX
    epsilon:          float
    q_table_size:     int
    duration_s:       float
    grade:            str

    def to_dict(self) -> Dict:
        return asdict(self)



class QLearningAgent:
    """
    Double Q-Learning agent for ClinicalTriageEnv v5.
    Uses Double Q-Learning to reduce overestimation bias:
      - Q_A selects the greedy action
      - Q_B evaluates it
      - Update alternates between the two tables
    Curriculum:
      - ε decays with cosine annealing over warm_up episodes
      - Then switches to exponential decay for exploitation
    Safety:
      - Tracks SAFETY_MATRIX scores per step
      - Penalises undertriage of critical patients in Q-updates
    """

    def __init__(
        self,
        lr:             float = 0.12,
        gamma:          float = 0.92,
        epsilon:        float = 1.0,
        epsilon_min:    float = 0.05,
        epsilon_decay:  float = 0.975,
        replay_batch:   int   = 32,
        warm_up_eps:    int   = 20,
        double_q:       bool  = True,
    ):
        self.lr             = lr
        self.gamma          = gamma
        self.epsilon        = epsilon
        self.epsilon_min    = epsilon_min
        self.epsilon_decay  = epsilon_decay
        self.replay_batch   = replay_batch
        self.warm_up_eps    = warm_up_eps
        self.double_q       = double_q

        # Double Q-tables
        self.q_a: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0] * len(ACTIONS))
        self.q_b: Dict[Tuple, List[float]] = defaultdict(lambda: [0.0] * len(ACTIONS))
        self._use_a = True  # Alternates each update

        # PER buffer
        self.buffer = PrioritisedReplayBuffer(capacity=2000)

        # Visit counts for confidence estimation
        self.visit_counts: Dict[Tuple, List[int]] = defaultdict(lambda: [0] * len(ACTIONS))

        # Analytics
        self.episode_metrics:   List[EpisodeMetrics] = []
        self.episode_rewards:   List[float]           = []
        self.episode_accuracy:  List[float]           = []
        self.episode_safety:    List[float]           = []
        self.epsilon_history:   List[float]           = []
        self.total_episodes:    int                   = 0
        self.total_steps:       int                   = 0

        # Episode tracking
        self._ep_start       = time.time()
        self._ep_rewards:     List[float] = []
        self._ep_esi_hits:    List[bool]  = []
        self._ep_safety:      List[float] = []
        self._ep_undertriage: List[bool]  = []


    def select_action(
        self,
        state: Dict[str, Any],
        true_esi: Optional[int] = None,
    ) -> Tuple[str, str, float]:
        """
        ε-greedy action selection with safety override.
        Returns:
            (action_str, mode_label, confidence_pct)
        """
        feat = featurise(state)

        # Safety override: if obvious life-threat, never choose ESI 4-5
        _, _, _, gcs_zone, _, red_flag, _ = feat
        if red_flag or gcs_zone == 0:
            safe_actions = [0, 1]  # ESI 1 or 2 only
        else:
            safe_actions = list(range(len(ACTIONS)))

        if random.random() < self.epsilon:
            idx  = random.choice(safe_actions)
            mode = f"🎲 Exploring (ε={self.epsilon:.3f})"
        else:
            # Double Q: use average of both tables for selection
            if self.double_q:
                q_avg = [
                    (self.q_a[feat][i] + self.q_b[feat][i]) / 2
                    for i in range(len(ACTIONS))
                ]
            else:
                q_avg = self.q_a[feat]

            # Mask unsafe actions
            masked = [q_avg[i] if i in safe_actions else -1e9 for i in range(len(ACTIONS))]
            idx  = masked.index(max(masked))
            mode = f"🧠 Exploiting (ε={self.epsilon:.3f})"

        self.visit_counts[feat][idx] += 1
        conf = self._confidence(feat, idx)
        return ACTIONS[idx], mode, conf

    def update(
        self,
        state:       Dict[str, Any],
        action:      str,
        reward:      float,
        next_state:  Dict[str, Any],
        done:        bool,
        true_esi:    Optional[int] = None,
        agent_esi:   Optional[int] = None,
    ):
        """Store transition, compute TD error, batch-update Q-tables."""
        feat      = featurise(state)
        next_feat = featurise(next_state)
        act_idx   = ACTION_IDX.get(action, 2)

        # Safety-augmented reward
        if true_esi and agent_esi:
            safety = SAFETY_MATRIX.get((true_esi, agent_esi), 0.5)
            aug_reward = reward + 0.1 * (safety - 0.5)
        else:
            aug_reward = reward
            safety = 0.5

        # TD error for PER priority
        q_curr = self.q_a[feat][act_idx]
        q_next = max(self.q_a[next_feat]) if not done else 0.0
        td_error = aug_reward + self.gamma * q_next - q_curr

        exp = Experience(
            state_feat=feat, action_idx=act_idx,
            reward=aug_reward, next_feat=next_feat,
            done=done, td_error=td_error,
        )
        self.buffer.push(exp)

        # Episode tracking
        self._ep_rewards.append(reward)
        self._ep_safety.append(safety)
        self.total_steps += 1

        if true_esi and agent_esi:
            self._ep_esi_hits.append(agent_esi == true_esi)
            self._ep_undertriage.append(agent_esi > true_esi)

        # Batch PER update
        batch = self.buffer.sample(self.replay_batch)
        for exp in batch:
            self._q_update(exp)

        if done:
            self._end_episode()

    def _q_update(self, exp: Experience):
        """Double Q-Learning update step."""
        sf, ai, r, nf, done = exp.state_feat, exp.action_idx, exp.reward, exp.next_feat, exp.done

        if self.double_q and self._use_a:
            # Q_A selects best action, Q_B evaluates
            best_idx  = self.q_a[nf].index(max(self.q_a[nf]))
            q_next    = self.q_b[nf][best_idx] if not done else 0.0
            target    = r + self.gamma * q_next
            old_q     = self.q_a[sf][ai]
            self.q_a[sf][ai] += self.lr * (target - old_q)
        else:
            # Q_B selects, Q_A evaluates
            best_idx  = self.q_b[nf].index(max(self.q_b[nf])) if self.double_q else self.q_a[nf].index(max(self.q_a[nf]))
            q_next    = self.q_a[nf][best_idx] if not done else 0.0
            target    = r + self.gamma * q_next
            old_q     = self.q_b[sf][ai] if self.double_q else self.q_a[sf][ai]
            if self.double_q:
                self.q_b[sf][ai] += self.lr * (target - old_q)
            else:
                self.q_a[sf][ai] += self.lr * (target - old_q)

        self._use_a = not self._use_a

    def _end_episode(self):
        n = len(self._ep_rewards)
        if n == 0:
            return

        total_r   = sum(self._ep_rewards)
        mean_r    = total_r / n
        accuracy  = sum(self._ep_esi_hits) / max(1, len(self._ep_esi_hits))
        safety    = sum(self._ep_safety) / max(1, len(self._ep_safety))
        under_r   = sum(self._ep_undertriage) / max(1, len(self._ep_undertriage))

        grade = (
            "S — Expert" if mean_r >= 0.9 else
            "A — Proficient" if mean_r >= 0.75 else
            "B — Competent" if mean_r >= 0.60 else
            "C — Developing" if mean_r >= 0.45 else
            "F — Critical Review Required"
        )

        metrics = EpisodeMetrics(
            episode=self.total_episodes,
            total_reward=round(total_r, 4),
            mean_reward=round(mean_r, 4),
            esi_accuracy=round(accuracy, 3),
            undertriage_rate=round(under_r, 3),
            safety_score=round(safety, 3),
            epsilon=round(self.epsilon, 4),
            q_table_size=len(self.q_a),
            duration_s=round(time.time() - self._ep_start, 2),
            grade=grade,
        )
        self.episode_metrics.append(metrics)
        self.episode_rewards.append(mean_r)
        self.episode_accuracy.append(accuracy)
        self.episode_safety.append(safety)
        self.epsilon_history.append(round(self.epsilon, 4))
        self.total_episodes += 1

        # Cosine annealing during warm-up, then exponential decay
        if self.total_episodes <= self.warm_up_eps:
            t = self.total_episodes / self.warm_up_eps
            self.epsilon = self.epsilon_min + 0.5 * (1.0 - self.epsilon_min) * (1 + math.cos(math.pi * t))
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Reset episode buffers
        self._ep_rewards.clear()
        self._ep_esi_hits.clear()
        self._ep_safety.clear()
        self._ep_undertriage.clear()
        self._ep_start = time.time()


    def get_q_values(self, state: Dict[str, Any]) -> Dict[str, float]:
        feat = featurise(state)
        if self.double_q:
            q = [(self.q_a[feat][i] + self.q_b[feat][i]) / 2 for i in range(len(ACTIONS))]
        else:
            q = self.q_a[feat]
        return {a: round(q[i], 3) for i, a in enumerate(ACTIONS)}

    def get_value_estimate(self, state: Dict[str, Any]) -> float:
        return max(self.get_q_values(state).values())

    def _confidence(self, feat: Tuple, idx: int) -> float:
        visits = self.visit_counts[feat]
        total  = sum(visits) + 1e-9
        return round(visits[idx] / total * 100, 1)

    def get_analytics(self) -> Dict[str, Any]:
        rewards = self.episode_rewards
        n = len(rewards)
        if n == 0:
            return {
                "total_episodes": 0, "total_steps": 0, "mean_reward": 0,
                "best_reward": 0, "worst_reward": 0, "recent_mean": 0,
                "q_table_size": 0, "epsilon": round(self.epsilon, 4),
                "trend": "No data yet", "rewards_history": [],
                "epsilon_history": [], "esi_accuracy_history": [],
                "safety_history": [], "double_q": self.double_q,
                "buffer_size": len(self.buffer),
            }

        recent = rewards[-min(10, n):]
        trend = (
            "📈 Improving" if len(recent) > 2 and recent[-1] > recent[0] + 0.05 else
            "📉 Declining" if len(recent) > 2 and recent[-1] < recent[0] - 0.05 else
            "➡️ Stable"
        )

        return {
            "total_episodes":   self.total_episodes,
            "total_steps":      self.total_steps,
            "mean_reward":      round(sum(rewards) / n, 4),
            "best_reward":      round(max(rewards), 4),
            "worst_reward":     round(min(rewards), 4),
            "recent_mean":      round(sum(recent) / len(recent), 4),
            "q_table_size":     len(self.q_a),
            "buffer_size":      len(self.buffer),
            "epsilon":          round(self.epsilon, 4),
            "trend":            trend,
            "double_q":         self.double_q,
            "rewards_history":  [round(r, 4) for r in rewards[-50:]],
            "epsilon_history":  self.epsilon_history[-50:],
            "esi_accuracy_history": [round(a, 3) for a in self.episode_accuracy[-50:]],
            "safety_history":   [round(s, 3) for s in self.episode_safety[-50:]],
            "latest_metrics":   self.episode_metrics[-1].to_dict() if self.episode_metrics else {},
        }

    def get_policy_heatmap_data(self) -> List[Dict[str, Any]]:
        """Return top policy states for heatmap visualisation."""
        spo2_labels = ["🔴 Crisis (<90%)", "🟡 Low (90–94%)", "🟢 Normal (95%+)"]
        hr_labels   = ["⬇️ Brady (<60)", "✅ Normal (60–100)", "⬆️ Tachy (>100)"]
        gcs_labels  = ["⚠️ Severe (≤8)", "⚡ Moderate (9–13)", "✅ Normal (14–15)"]
        esi_labels  = {1: "🔴 ESI-1", 2: "🟠 ESI-2", 3: "🟡 ESI-3", 4: "🟢 ESI-4", 5: "⚪ ESI-5"}

        rows = []
        seen = set()

        for feat, q_vals_a in self.q_a.items():
            if len(feat) < 7:
                continue
            key = feat[:4]
            if key in seen:
                continue
            seen.add(key)

            if self.double_q:
                q_avg = [(q_vals_a[i] + self.q_b[feat][i]) / 2 for i in range(len(ACTIONS))]
            else:
                q_avg = q_vals_a

            best_idx = q_avg.index(max(q_avg))
            best_esi = best_idx + 1
            visits   = sum(self.visit_counts[feat])
            conf     = self._confidence(feat, best_idx)
            safety   = SAFETY_MATRIX.get((feat[0] + 1, best_esi), 0.5)  # Approx true ESI from feat

            rows.append({
                "SpO₂ Zone":    spo2_labels[feat[0]] if feat[0] < 3 else "Unknown",
                "HR Zone":      hr_labels[feat[1]]   if feat[1] < 3 else "Unknown",
                "GCS Zone":     gcs_labels[feat[3]]  if feat[3] < 3 else "Unknown",
                "Red Flag":     "⚠️ YES" if feat[5] else "✅ No",
                "Policy Action": esi_labels.get(best_esi, f"ESI-{best_esi}"),
                "Q-Value":      round(max(q_avg), 3),
                "Confidence":   f"{conf:.1f}%",
                "Safety Score": f"{safety:.2f}",
                "Visits":       visits,
            })

        rows.sort(key=lambda r: -r["Visits"])
        return rows[:25]

    def save(self, path: str = "q_table.json"):
        """Serialise Q-tables and analytics to JSON."""
        data = {
            "version": "2.0",
            "double_q": self.double_q,
            "epsilon": self.epsilon,
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "q_a": {str(k): v for k, v in self.q_a.items()},
            "q_b": {str(k): v for k, v in self.q_b.items()} if self.double_q else {},
            "visit_counts": {str(k): v for k, v in self.visit_counts.items()},
            "episode_rewards": self.episode_rewards[-200:],
            "episode_accuracy": self.episode_accuracy[-200:],
            "saved_at": time.time(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def load(self, path: str = "q_table.json") -> bool:
        """Load Q-tables from JSON."""
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            import ast
            self.epsilon        = float(data.get("epsilon", self.epsilon))
            self.total_episodes = int(data.get("total_episodes", 0))
            self.total_steps    = int(data.get("total_steps", 0))
            self.episode_rewards  = data.get("episode_rewards", [])
            self.episode_accuracy = data.get("episode_accuracy", [])

            for k_str, v in data.get("q_a", {}).items():
                k = ast.literal_eval(k_str)
                self.q_a[k] = v
            for k_str, v in data.get("q_b", {}).items():
                k = ast.literal_eval(k_str)
                self.q_b[k] = v
            for k_str, v in data.get("visit_counts", {}).items():
                k = ast.literal_eval(k_str)
                self.visit_counts[k] = v
            return True
        except Exception as e:
            print(f"Q-table load error: {e}")
            return False


    def run_training_episode(self, env) -> Dict[str, Any]:
        """
        Run one full autonomous training episode using environment_v2.
        Args:
            env: ClinicalTriageEnvV2 instance
        Returns:
            Episode summary dict
        """
        obs = env.reset()
        ep_reward = 0.0
        steps = 0

        while obs["queue_size"] > 0 and steps < 50:
            queue = obs.get("patient_queue", [])
            if not queue:
                break

            # Select most critical patient
            patient = queue[0]
            pid = patient["patient_id"]

            # Build state for agent
            state = {
                "spo2":            patient["vitals"].get("spo2", 98),
                "hr":              patient["vitals"].get("hr", 80),
                "sbp":             patient["vitals"].get("sbp", 120),
                "gcs":             patient["vitals"].get("gcs", 15),
                "age":             patient.get("age", 45),
                "chief_complaint": patient.get("chief_complaint", ""),
                "news2_score":     patient.get("news2_score", 0),
            }

            action_str, mode, conf = self.select_action(state)
            esi_level = ESI_FROM_ACTION.get(action_str, 3)

            action_dict = {
                "esi_level": esi_level,
                "rationale": f"Agent policy: {action_str}. Confidence: {conf:.1f}%.",
                "interventions": [],
            }

            next_obs, reward, done, info = env.step(pid, action_dict, action_dict["rationale"])
            ep_reward += reward

            # Next state = next patient if available
            next_queue = next_obs.get("patient_queue", [])
            next_state = state if not next_queue else {
                "spo2":            next_queue[0]["vitals"].get("spo2", 98),
                "hr":              next_queue[0]["vitals"].get("hr", 80),
                "sbp":             next_queue[0]["vitals"].get("sbp", 120),
                "gcs":             next_queue[0]["vitals"].get("gcs", 15),
                "age":             next_queue[0].get("age", 45),
                "chief_complaint": next_queue[0].get("chief_complaint", ""),
                "news2_score":     next_queue[0].get("news2_score", 0),
            }

            self.update(
                state=state, action=action_str, reward=reward,
                next_state=next_state, done=done,
                true_esi=patient.get("true_esi"),
                agent_esi=esi_level,
            )

            obs = next_obs
            steps += 1
            if done:
                break

        summary = env.get_episode_summary()
        summary["agent_epsilon"] = round(self.epsilon, 4)
        summary["agent_q_table_size"] = len(self.q_a)
        return summary
