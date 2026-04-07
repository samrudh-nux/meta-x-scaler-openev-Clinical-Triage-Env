from __future__ import annotations

import time
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

try:
    from environment_v2 import ClinicalTriageEnvV2, DifficultyMode
    ENV_V2_OK = True
except ImportError:
    ENV_V2_OK = False

try:
    from rl_engine import QLearningAgent
    RL_OK = True
except ImportError:
    RL_OK = False

try:
    from llm_evaluator import LLMBackend
    LLM_OK = True
except ImportError:
    LLM_OK = False


@dataclass
class TrainingMetrics:
    n_episodes:       int
    total_steps:      int
    mean_reward:      float
    best_reward:      float
    worst_reward:     float
    final_epsilon:    float
    esi_accuracy:     float
    undertriage_rate: float
    safety_score:     float
    q_table_size:     int
    duration_s:       float
    rewards_history:  List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)
    safety_history:   List[float] = field(default_factory=list)
    difficulty_progression: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def train(
    n_episodes:   int = 20,
    difficulty:   Any = None,
    llm_backend:  Any = None,
    curriculum:   bool = True,
    verbose:      bool = True,
    save_path:    Optional[str] = None,
) -> tuple:
    """
    Run RL training loop.
    Args:
        n_episodes:  Number of training episodes.
        difficulty:  DifficultyMode (default: BUSY).
        llm_backend: LLMBackend (default: RULE_BASED).
        curriculum:  Enable automatic difficulty ramp.
        verbose:     Print per-episode stats.
        save_path:   Optional path to save Q-table.
    Returns:
        (env, agent, metrics)
    """
    if not ENV_V2_OK or not RL_OK:
        raise ImportError("environment_v2.py and rl_engine.py required.")

    if difficulty is None:
        difficulty = DifficultyMode.BUSY
    if llm_backend is None and LLM_OK:
        llm_backend = LLMBackend.RULE_BASED

    t0 = time.time()

    env = ClinicalTriageEnvV2(
        difficulty=difficulty,
        llm_backend=llm_backend,
        enable_deterioration=True,
        curriculum=curriculum,
    )
    agent = QLearningAgent(
        lr=0.12, gamma=0.92,
        epsilon=1.0, epsilon_min=0.05,
        epsilon_decay=0.975,
        replay_batch=32, warm_up_eps=max(5, n_episodes // 5),
        double_q=True,
    )

    rewards_history    = []
    accuracy_history   = []
    safety_history     = []
    difficulty_prog    = []
    total_steps        = 0

    for ep in range(n_episodes):
        summary = agent.run_training_episode(env)
        ep_reward = summary.get("mean_reward", 0.0)
        ep_acc    = summary.get("esi_exact_accuracy", 0.0)
        ep_under  = summary.get("undertriage_rate", 0.0)
        ep_grade  = summary.get("performance_grade", "?")
        ep_diff   = summary.get("difficulty", difficulty.value if hasattr(difficulty, 'value') else str(difficulty))

        rewards_history.append(ep_reward)
        accuracy_history.append(ep_acc)
        difficulty_prog.append(ep_diff)

        # Safety from agent's own tracking
        if agent.episode_safety:
            safety_history.append(agent.episode_safety[-1])

        total_steps += summary.get("steps", 0)

        if verbose:
            print(
                f"  Ep {ep+1:3d}/{n_episodes} | "
                f"R={ep_reward:+.3f} | "
                f"ESI Acc={ep_acc:.1%} | "
                f"Undertriage={ep_under:.1%} | "
                f"ε={agent.epsilon:.3f} | "
                f"Diff={ep_diff} | "
                f"Grade={ep_grade}"
            )

    duration = time.time() - t0

    if save_path:
        agent.save(save_path)
        if verbose:
            print(f"  ✅ Q-table saved → {save_path}")

    analytics = agent.get_analytics()

    metrics = TrainingMetrics(
        n_episodes=n_episodes,
        total_steps=total_steps,
        mean_reward=round(sum(rewards_history) / max(1, len(rewards_history)), 4),
        best_reward=round(max(rewards_history) if rewards_history else 0, 4),
        worst_reward=round(min(rewards_history) if rewards_history else 0, 4),
        final_epsilon=round(agent.epsilon, 4),
        esi_accuracy=round(sum(accuracy_history) / max(1, len(accuracy_history)), 3),
        undertriage_rate=round(
            sum(agent.episode_metrics[-n_episodes:][i].undertriage_rate
                for i in range(min(n_episodes, len(agent.episode_metrics)))) /
            max(1, min(n_episodes, len(agent.episode_metrics))), 3
        ) if agent.episode_metrics else 0.0,
        safety_score=round(sum(safety_history) / max(1, len(safety_history)), 3),
        q_table_size=len(agent.q_a),
        duration_s=round(duration, 2),
        rewards_history=rewards_history,
        accuracy_history=accuracy_history,
        safety_history=safety_history,
        difficulty_progression=difficulty_prog,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training Complete — {n_episodes} episodes in {duration:.1f}s")
        print(f"  Mean Reward:    {metrics.mean_reward:+.4f}")
        print(f"  ESI Accuracy:   {metrics.esi_accuracy:.1%}")
        print(f"  Undertriage:    {metrics.undertriage_rate:.1%}")
        print(f"  Safety Score:   {metrics.safety_score:.3f}")
        print(f"  Q-Table Size:   {metrics.q_table_size} states")
        print(f"  Final Epsilon:  {metrics.final_epsilon:.4f}")
        print(f"{'='*60}\n")

    return env, agent, metrics
