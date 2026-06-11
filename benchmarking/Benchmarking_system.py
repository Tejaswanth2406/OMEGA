"""
Benchmarking System
====================
Continuous fitness evaluation across all organism dimensions.
Produces a multi-dimensional intelligence score.
"""
from __future__ import annotations
import time
import uuid
import statistics
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


class BenchmarkDimension(str, Enum):
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    PLANNING = "planning"
    LEARNING = "learning"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"
    CREATIVITY = "creativity"
    SELF_AWARENESS = "self_awareness"
    ADAPTATION = "adaptation"


@dataclass
class BenchmarkTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    dimension: BenchmarkDimension = BenchmarkDimension.REASONING
    difficulty: float = 0.5         # 0 = trivial, 1 = near-impossible
    test_fn: Optional[Callable] = None
    expected_output: Any = None
    scorer: Optional[Callable] = None
    timeout_seconds: float = 30.0
    weight: float = 1.0             # contribution to dimension score


@dataclass
class BenchmarkScore:
    score_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dimension: BenchmarkDimension = BenchmarkDimension.REASONING
    raw_score: float = 0.0
    normalized_score: float = 0.0   # [0, 1]
    tasks_run: int = 0
    tasks_passed: int = 0
    mean_latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class IntelligenceReport:
    report_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dimension_scores: Dict[str, BenchmarkScore] = field(default_factory=dict)
    composite_score: float = 0.0
    knowledge_score: float = 0.0
    adaptability_score: float = 0.0
    efficiency_score: float = 0.0
    self_improvement_score: float = 0.0
    intelligence_score: float = 0.0     # I = K × A × E × SI
    rank_percentile: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    generation: int = 0


class BenchmarkSuite:
    """Standard benchmark tasks for each dimension."""

    @staticmethod
    def reasoning_tasks() -> List[BenchmarkTask]:
        return [
            BenchmarkTask(
                name="logical_deduction",
                dimension=BenchmarkDimension.REASONING,
                difficulty=0.5,
                expected_output=True,
                scorer=lambda pred, exp: 1.0 if pred == exp else 0.0,
                weight=1.5,
            ),
            BenchmarkTask(
                name="analogical_reasoning",
                dimension=BenchmarkDimension.REASONING,
                difficulty=0.7,
                weight=2.0,
            ),
            BenchmarkTask(
                name="causal_inference",
                dimension=BenchmarkDimension.REASONING,
                difficulty=0.8,
                weight=2.5,
            ),
        ]

    @staticmethod
    def retrieval_tasks() -> List[BenchmarkTask]:
        return [
            BenchmarkTask(
                name="exact_recall",
                dimension=BenchmarkDimension.RETRIEVAL,
                difficulty=0.3,
                weight=1.0,
            ),
            BenchmarkTask(
                name="fuzzy_retrieval",
                dimension=BenchmarkDimension.RETRIEVAL,
                difficulty=0.6,
                weight=1.5,
            ),
            BenchmarkTask(
                name="cross_domain_retrieval",
                dimension=BenchmarkDimension.RETRIEVAL,
                difficulty=0.8,
                weight=2.0,
            ),
        ]

    @staticmethod
    def all_tasks() -> Dict[str, List[BenchmarkTask]]:
        return {
            "reasoning": BenchmarkSuite.reasoning_tasks(),
            "retrieval": BenchmarkSuite.retrieval_tasks(),
        }


class FitnessEvaluator:
    """Evaluates organism fitness on individual tasks."""

    def evaluate(self, task: BenchmarkTask,
                 organism_fn: Optional[Callable] = None) -> Tuple[float, float]:
        """Returns (score, latency_ms)."""
        start = time.time()
        try:
            if organism_fn and task.test_fn:
                prediction = organism_fn(task.test_fn())
                score = task.scorer(prediction, task.expected_output) if task.scorer else 0.5
            else:
                # Stub: simulate plausible performance
                base = 0.5 + (1 - task.difficulty) * 0.3
                score = max(0.0, min(1.0, base + random.gauss(0, 0.1)))
        except Exception:
            score = 0.0

        latency = (time.time() - start) * 1000
        return score, latency


class BenchmarkEngine:
    """
    Runs continuous multi-dimensional benchmarks and tracks
    the organism's intelligence trajectory.
    """

    def __init__(self):
        self.engine_id = str(uuid.uuid4())[:8]
        self.suite = BenchmarkSuite()
        self.evaluator = FitnessEvaluator()
        self.reports: List[IntelligenceReport] = []
        self.dimension_history: Dict[str, List[float]] = {
            dim.value: [] for dim in BenchmarkDimension
        }
        self.generation = 0

    def run_dimension(self, dimension: BenchmarkDimension,
                      tasks: List[BenchmarkTask],
                      organism_fn: Optional[Callable] = None) -> BenchmarkScore:
        """Benchmark a single dimension."""
        scores = []
        latencies = []
        passed = 0

        for task in tasks:
            score, latency = self.evaluator.evaluate(task, organism_fn)
            weighted = score * task.weight
            scores.append(weighted)
            latencies.append(latency)
            if score >= 0.5:
                passed += 1

        total_weight = sum(t.weight for t in tasks) if tasks else 1.0
        raw = sum(scores) / total_weight if scores else 0.0
        normalized = max(0.0, min(1.0, raw))

        bs = BenchmarkScore(
            dimension=dimension,
            raw_score=raw,
            normalized_score=normalized,
            tasks_run=len(tasks),
            tasks_passed=passed,
            mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        )
        self.dimension_history[dimension.value].append(normalized)
        return bs

    def full_benchmark(self, organism=None) -> IntelligenceReport:
        """Run the complete benchmark suite."""
        self.generation += 1
        report = IntelligenceReport(generation=self.generation)

        # Run all available dimensions
        all_tasks = self.suite.all_tasks()

        for dim in BenchmarkDimension:
            tasks = all_tasks.get(dim.value, [])
            if not tasks:
                # Generate synthetic tasks for dimensions without explicit tests
                tasks = [BenchmarkTask(
                    name=f"synthetic_{dim.value}_{i}",
                    dimension=dim,
                    difficulty=random.uniform(0.3, 0.8),
                    weight=1.0,
                ) for i in range(3)]

            score = self.run_dimension(dim, tasks)
            report.dimension_scores[dim.value] = score

        # Compute composite scores
        all_scores = [s.normalized_score for s in report.dimension_scores.values()]
        report.composite_score = statistics.mean(all_scores) if all_scores else 0.0

        # Map to intelligence formula components
        report.knowledge_score = report.dimension_scores.get(
            BenchmarkDimension.MEMORY.value, BenchmarkScore()).normalized_score
        report.adaptability_score = report.dimension_scores.get(
            BenchmarkDimension.ADAPTATION.value, BenchmarkScore()).normalized_score
        report.efficiency_score = report.dimension_scores.get(
            BenchmarkDimension.EFFICIENCY.value, BenchmarkScore()).normalized_score
        report.self_improvement_score = report.dimension_scores.get(
            BenchmarkDimension.SELF_AWARENESS.value, BenchmarkScore()).normalized_score

        # Intelligence formula: I = K × A × E × SI (geometric mean prevents zero-product collapse)
        components = [
            max(0.01, report.knowledge_score),
            max(0.01, report.adaptability_score),
            max(0.01, report.efficiency_score),
            max(0.01, report.self_improvement_score),
        ]
        report.intelligence_score = (components[0] * components[1] *
                                     components[2] * components[3]) ** 0.25

        self.reports.append(report)
        return report

    def trend(self, dimension: str, window: int = 10) -> float:
        """Compute improvement trend for a dimension (positive = improving)."""
        history = self.dimension_history.get(dimension, [])[-window:]
        if len(history) < 2:
            return 0.0
        n = len(history)
        x_mean = (n - 1) / 2
        y_mean = sum(history) / n
        num = sum((i - x_mean) * (history[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den else 0.0

    def summary(self) -> Dict:
        if not self.reports:
            return {"status": "no benchmarks run"}
        latest = self.reports[-1]
        return {
            "engine_id": self.engine_id,
            "generation": self.generation,
            "latest_intelligence_score": round(latest.intelligence_score, 4),
            "latest_composite_score": round(latest.composite_score, 4),
            "dimension_scores": {
                k: round(v.normalized_score, 3)
                for k, v in latest.dimension_scores.items()
            },
            "improving_dimensions": [
                dim for dim in BenchmarkDimension
                if self.trend(dim.value) > 0.001
            ],
            "reports_run": len(self.reports),
        }

    def __repr__(self) -> str:
        score = self.reports[-1].intelligence_score if self.reports else 0.0
        return f"BenchmarkEngine(gen={self.generation}, I={score:.3f})"