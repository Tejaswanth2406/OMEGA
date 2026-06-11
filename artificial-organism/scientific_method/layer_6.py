"""
Layer 6: Continuous Scientific Method
=======================================
Every output is a hypothesis. Every interaction is an experiment.
The organism applies the scientific method to its own cognition.
"""
from __future__ import annotations
import time
import uuid
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class ExperimentStatus(str, Enum):
    PROPOSED = "proposed"
    RUNNING = "running"
    CONCLUDED = "concluded"
    FAILED = "failed"
    REPLICATED = "replicated"


@dataclass
class Observation:
    obs_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5
    source: str = "environment"


@dataclass
class Hypothesis:
    hyp_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str = ""
    domain: str = "general"
    predicted_outcome: Any = None
    confidence: float = 0.5     # prior confidence
    supporting_obs: List[str] = field(default_factory=list)
    contradicting_obs: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def update_confidence(self, evidence_supports: bool, strength: float = 0.1) -> None:
        """Bayesian-style update."""
        if evidence_supports:
            self.confidence = min(1.0, self.confidence + strength * (1 - self.confidence))
        else:
            self.confidence = max(0.0, self.confidence - strength * self.confidence)


@dataclass
class Prediction:
    pred_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = ""
    predicted_value: Any = None
    actual_value: Any = None
    error: Optional[float] = None
    verified: bool = False
    timestamp: float = field(default_factory=time.time)

    def verify(self, actual: Any, scorer: Optional[Callable] = None) -> float:
        self.actual_value = actual
        self.verified = True
        if scorer:
            self.error = scorer(self.predicted_value, actual)
        elif isinstance(self.predicted_value, (int, float)) and isinstance(actual, (int, float)):
            self.error = abs(self.predicted_value - actual)
        else:
            self.error = 0.0 if str(self.predicted_value) == str(actual) else 1.0
        return self.error


@dataclass
class Experiment:
    exp_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    hypothesis: Optional[Hypothesis] = None
    predictions: List[Prediction] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.PROPOSED
    result: Optional[Dict] = None
    started_at: Optional[float] = None
    concluded_at: Optional[float] = None
    iterations: int = 0

    def start(self) -> None:
        self.status = ExperimentStatus.RUNNING
        self.started_at = time.time()

    def add_observation(self, obs: Observation) -> None:
        self.observations.append(obs)
        self.iterations += 1

    def conclude(self) -> Dict:
        self.concluded_at = time.time()
        self.status = ExperimentStatus.CONCLUDED

        verified = [p for p in self.predictions if p.verified]
        errors = [p.error for p in verified if p.error is not None]
        mean_error = statistics.mean(errors) if errors else None
        success_rate = sum(1 for e in errors if e < 0.5) / len(errors) if errors else 0.0

        self.result = {
            "exp_id": self.exp_id,
            "hypothesis": self.hypothesis.statement if self.hypothesis else "",
            "iterations": self.iterations,
            "predictions_made": len(self.predictions),
            "predictions_verified": len(verified),
            "mean_error": mean_error,
            "success_rate": success_rate,
            "hypothesis_supported": success_rate > 0.6 if errors else None,
            "duration_seconds": (self.concluded_at - self.started_at) if self.started_at else 0,
        }

        # Update hypothesis confidence
        if self.hypothesis and self.result["hypothesis_supported"] is not None:
            self.hypothesis.update_confidence(
                self.result["hypothesis_supported"],
                strength=0.2
            )

        return self.result


class ScientificMethodEngine:
    """
    The organism's research engine.
    Manages the full Observe → Hypothesize → Predict → Test → Measure → Update cycle.
    """

    def __init__(self):
        self.engine_id = str(uuid.uuid4())[:8]
        self.observations: List[Observation] = []
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.knowledge_updates: List[Dict] = []
        self.active_experiment: Optional[str] = None

    # ─── Step 1: Observe ─────────────────────────────────────────────────

    def observe(self, description: str, data: Any = None,
                confidence: float = 0.7, source: str = "environment") -> Observation:
        obs = Observation(description=description, data=data,
                          confidence=confidence, source=source)
        self.observations.append(obs)
        return obs

    def recent_observations(self, n: int = 10) -> List[Observation]:
        return self.observations[-n:]

    # ─── Step 2: Hypothesize ─────────────────────────────────────────────

    def hypothesize(self, statement: str, domain: str = "general",
                    prior_confidence: float = 0.5,
                    predicted_outcome: Any = None) -> Hypothesis:
        hyp = Hypothesis(statement=statement, domain=domain,
                         confidence=prior_confidence,
                         predicted_outcome=predicted_outcome)
        self.hypotheses[hyp.hyp_id] = hyp
        return hyp

    def get_active_hypotheses(self) -> List[Hypothesis]:
        return [h for h in self.hypotheses.values() if 0.2 < h.confidence < 0.95]

    def strongest_hypotheses(self, n: int = 5) -> List[Hypothesis]:
        return sorted(self.hypotheses.values(),
                      key=lambda h: h.confidence, reverse=True)[:n]

    # ─── Step 3: Predict ─────────────────────────────────────────────────

    def predict(self, hypothesis: Hypothesis, predicted_value: Any) -> Prediction:
        pred = Prediction(hypothesis_id=hypothesis.hyp_id,
                          predicted_value=predicted_value)
        # Attach to active experiment if any
        if self.active_experiment and self.active_experiment in self.experiments:
            self.experiments[self.active_experiment].predictions.append(pred)
        return pred

    # ─── Step 4: Test ────────────────────────────────────────────────────

    def start_experiment(self, name: str, hypothesis: Hypothesis) -> Experiment:
        exp = Experiment(name=name, hypothesis=hypothesis)
        exp.start()
        self.experiments[exp.exp_id] = exp
        self.active_experiment = exp.exp_id
        return exp

    def run_trial(self, exp_id: str, observation: Observation,
                  prediction: Optional[Prediction] = None,
                  actual_result: Any = None) -> None:
        if exp_id not in self.experiments:
            return
        exp = self.experiments[exp_id]
        exp.add_observation(observation)

        if prediction and actual_result is not None:
            prediction.verify(actual_result)
            exp.predictions.append(prediction)
            exp.hypothesis.supporting_obs.append(observation.obs_id) \
                if prediction.error < 0.5 else \
                exp.hypothesis.contradicting_obs.append(observation.obs_id)

    # ─── Step 5: Measure ─────────────────────────────────────────────────

    def conclude_experiment(self, exp_id: str) -> Optional[Dict]:
        if exp_id not in self.experiments:
            return None
        result = self.experiments[exp_id].conclude()
        if self.active_experiment == exp_id:
            self.active_experiment = None
        return result

    # ─── Step 6: Update Model ────────────────────────────────────────────

    def update_model(self, exp_result: Dict) -> Dict:
        """
        Integrate experiment result into organism's knowledge.
        Returns a model update record.
        """
        update = {
            "update_id": str(uuid.uuid4())[:8],
            "based_on": exp_result.get("exp_id"),
            "hypothesis": exp_result.get("hypothesis"),
            "hypothesis_supported": exp_result.get("hypothesis_supported"),
            "confidence_shift": 0.0,
            "new_knowledge": [],
            "timestamp": time.time(),
        }

        if exp_result.get("hypothesis_supported"):
            update["new_knowledge"].append(
                f"CONFIRMED: {exp_result['hypothesis']}"
            )
            update["confidence_shift"] = +exp_result.get("success_rate", 0.1)
        else:
            update["new_knowledge"].append(
                f"REFUTED: {exp_result['hypothesis']}"
            )
            update["confidence_shift"] = -exp_result.get("success_rate", 0.1)

        self.knowledge_updates.append(update)
        return update

    # ─── Full Cycle Convenience ──────────────────────────────────────────

    def run_full_cycle(self, statement: str, test_fn: Callable,
                       domain: str = "general") -> Dict:
        """
        Run a complete scientific cycle in one call.
        test_fn(hypothesis) -> (actual_result, predicted_value)
        """
        obs = self.observe(f"Initiating experiment on: {statement}")
        hyp = self.hypothesize(statement, domain=domain)
        exp = self.start_experiment(f"Auto-{hyp.hyp_id}", hyp)

        try:
            actual, predicted = test_fn(hyp)
            pred = self.predict(hyp, predicted)
            self.run_trial(exp.exp_id, obs, pred, actual)
        except Exception as e:
            obs2 = self.observe(f"Experiment error: {e}", confidence=0.0, source="system")
            exp.add_observation(obs2)

        result = self.conclude_experiment(exp.exp_id)
        update = self.update_model(result)
        return {"experiment": result, "model_update": update, "hypothesis": hyp.hyp_id}

    # ─── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        concluded = [e for e in self.experiments.values()
                     if e.status == ExperimentStatus.CONCLUDED]
        success_rates = [e.result["success_rate"] for e in concluded if e.result]
        return {
            "total_observations": len(self.observations),
            "total_hypotheses": len(self.hypotheses),
            "total_experiments": len(self.experiments),
            "concluded_experiments": len(concluded),
            "mean_success_rate": statistics.mean(success_rates) if success_rates else 0.0,
            "knowledge_updates": len(self.knowledge_updates),
        }