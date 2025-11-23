"""
Bayesian optimization for A/B testing curriculum variants.

Uses Thompson sampling with Beta distributions for arm selection.
"""

import json
import logging
from typing import Dict, List, Optional
import random
from dataclasses import dataclass
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    """Statistics for a single variant arm."""
    name: str
    alpha: float = 1.0  # Beta distribution shape parameter
    beta: float = 1.0   # Beta distribution shape parameter
    successes: int = 0
    failures: int = 0
    total_objective: float = 0.0
    sample_count: int = 0

    def update_metric(self, objective_value: float) -> None:
        """Update arm with a new objective value."""
        self.total_objective += objective_value
        self.sample_count += 1

    def get_mean_objective(self) -> float:
        """Get mean objective value for this arm."""
        if self.sample_count == 0:
            return 0.0
        return self.total_objective / self.sample_count

    def sample_posterior(self) -> float:
        """Sample from posterior distribution using Beta distribution."""
        try:
            return stats.beta.rvs(self.alpha, self.beta)
        except Exception:
            return 0.5


class AxOptimizer:
    """Thompson Sampling Bayesian optimizer for curriculum variants."""

    def __init__(
        self,
        db_logger=None,
        metrics_weights: Optional[Dict[str, float]] = None,
        exploration_rate: float = 0.2,
    ):
        """
        Initialize Bayesian optimizer for A/B testing.

        Args:
            db_logger: ABTestLogger instance for accessing trial data
            metrics_weights: Weights for multi-objective optimization
                {engagement: 0.4, retention: 0.4, satisfaction: 0.2}
            exploration_rate: Probability of random exploration (0.0-1.0)
        """
        self.db_logger = db_logger
        self.metrics_weights = metrics_weights or {
            "engagement": 0.4,
            "retention": 0.4,
            "satisfaction": 0.2,
        }
        self.exploration_rate = exploration_rate
        self.arms: Dict[str, ArmStats] = {}
        self.arm_probabilities = {}

        # Define parameter space
        self.parameter_space = {
            "phrase_ordering": ["random", "difficulty_asc", "difficulty_desc"],
            "category_focus": ["mixed", "specialized"],
            "repetition_strategy": ["immediate", "spaced_1h", "spaced_24h"],
            "difficulty_balance": ["easy_first", "varied_mix"],
        }

        # Initialize all possible arms
        self._initialize_arms()
        logger.info("Initialized Bayesian optimizer with Thompson sampling")

    def _initialize_arms(self) -> None:
        """Initialize all possible arm combinations."""
        arm_count = 0
        for ordering in self.parameter_space["phrase_ordering"]:
            for category in self.parameter_space["category_focus"]:
                for repetition in self.parameter_space["repetition_strategy"]:
                    for difficulty in self.parameter_space["difficulty_balance"]:
                        variant = {
                            "phrase_ordering": ordering,
                            "category_focus": category,
                            "repetition_strategy": repetition,
                            "difficulty_balance": difficulty,
                        }
                        arm_id = json.dumps(variant)
                        self.arms[arm_id] = ArmStats(name=arm_id)
                        arm_count += 1

        logger.info(f"Initialized {arm_count} arms for optimization")

    def get_next_arm(self) -> Dict[str, str]:
        """
        Select next curriculum variant using Thompson sampling.

        Uses posterior sampling: sample from each arm's distribution,
        select the arm with highest sample, with epsilon probability of random exploration.

        Returns:
            Dictionary of variant parameters
        """
        # Epsilon-greedy: explore with probability
        if random.random() < self.exploration_rate:
            logger.debug("Exploration phase: selecting random arm")
            return self._get_random_arm()

        # Thompson sampling: sample from posterior for each arm
        best_sample = -float('inf')
        best_arm_id = None

        for arm_id, arm in self.arms.items():
            sample = arm.sample_posterior()
            if sample > best_sample:
                best_sample = sample
                best_arm_id = arm_id

        if best_arm_id:
            return json.loads(best_arm_id)

        # Fallback to random
        return self._get_random_arm()

    def _get_random_arm(self) -> Dict[str, str]:
        """Return a random curriculum variant."""
        return {
            "phrase_ordering": random.choice(
                self.parameter_space["phrase_ordering"]
            ),
            "category_focus": random.choice(self.parameter_space["category_focus"]),
            "repetition_strategy": random.choice(
                self.parameter_space["repetition_strategy"]
            ),
            "difficulty_balance": random.choice(
                self.parameter_space["difficulty_balance"]
            ),
        }

    def update_posterior(self, metrics_data: List[Dict]) -> None:
        """
        Update Bayesian posterior with new trial data.

        Args:
            metrics_data: List of trial outcomes from ABTestLogger
                Each entry: {
                    variant_id: str,
                    engagement_score: float,
                    retention_score: float,
                    satisfaction_score: float,
                    count: int
                }
        """
        if not metrics_data:
            return

        try:
            # Aggregate metrics for each arm
            for trial in metrics_data:
                arm_id = trial.get("variant_id")
                if arm_id not in self.arms:
                    continue

                # Scalarized objective (weighted sum of metrics)
                objective = (
                    self.metrics_weights["engagement"]
                    * trial.get("engagement_score", 0)
                    + self.metrics_weights["retention"]
                    * trial.get("retention_score", 0)
                    + self.metrics_weights["satisfaction"]
                    * trial.get("satisfaction_score", 0)
                )

                self.arms[arm_id].update_metric(objective)

            # Update arm selection probabilities
            self._update_arm_probabilities()
            logger.info(f"Updated posterior with {len(metrics_data)} arms")

        except Exception as e:
            logger.error(f"Failed to update posterior: {e}")

    def _update_arm_probabilities(self) -> None:
        """
        Update selection probabilities based on posterior means.

        Uses softmax-like weighting.
        """
        objectives = {
            arm_id: arm.get_mean_objective() for arm_id, arm in self.arms.items()
        }

        if not objectives:
            return

        max_obj = max(objectives.values())
        min_obj = min(objectives.values())

        # Normalize and convert to probabilities
        if max_obj == min_obj:
            # All arms equal
            prob = 1.0 / len(self.arms)
            self.arm_probabilities = {arm_id: prob for arm_id in self.arms}
        else:
            # Weight by normalized objective
            total = 0
            normalized = {}
            for arm_id, obj in objectives.items():
                norm_val = (obj - min_obj) / (max_obj - min_obj)
                normalized[arm_id] = norm_val
                total += norm_val

            self.arm_probabilities = {
                arm_id: val / total for arm_id, val in normalized.items()
            }

    def get_arm_probabilities(self) -> Dict[str, float]:
        """Return current selection probability for each variant (top 10)."""
        if not self.arm_probabilities:
            return {}

        # Return top 10 arms by probability
        sorted_arms = sorted(
            self.arm_probabilities.items(), key=lambda x: x[1], reverse=True
        )
        return dict(sorted_arms[:10])

    def get_best_arm(self) -> Optional[Dict[str, str]]:
        """Return highest probability variant."""
        if not self.arm_probabilities:
            return self._get_random_arm()

        best_arm_id = max(self.arm_probabilities, key=self.arm_probabilities.get)
        try:
            return json.loads(best_arm_id)
        except Exception:
            return self._get_random_arm()

    def get_experiment_summary(self) -> Dict:
        """Return summary of current optimization state."""
        return {
            "num_arms": len([a for a in self.arms.values() if a.sample_count > 0]),
            "num_parameters": len(self.parameter_space),
            "best_arm": self.get_best_arm(),
            "arm_probabilities": self.get_arm_probabilities(),
            "metrics_weights": self.metrics_weights,
        }
