"""
Layer 12: Recursive Self-Improvement
======================================
The highest layer. The organism analyzes itself, discovers weaknesses,
generates mutations, sandboxes them, benchmarks them, and deploys improvements.

Intelligence = Knowledge × Adaptability × Efficiency × Self-Improvement
"""
from __future__ import annotations
import uuid
import time
import random
import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Tuple
from enum import Enum


class ImprovementStatus(str, Enum):
    PROPOSED = "proposed"
    SANDBOXED = "sandboxed"
    BENCHMARKED = "benchmarked"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class Weakness:
    weakness_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    component: str = ""
    description: str = ""
    severity: float = 0.5           # 0 = trivial, 1 = critical
    evidence: List[str] = field(default_factory=list)
    detected_at: float = field(default_factory=time.time)
    improvement_attempts: int = 0


@dataclass
class ImprovementProposal:
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    weakness_id: str = ""
    target_component: str = ""
    description: str = ""
    mutation_type: str = "parameter_tweak"      # parameter / structural / algorithmic
    proposed_changes: Dict[str, Any] = field(default_factory=dict)
    expected_gain: float = 0.1
    risk_level: float = 0.3
    status: ImprovementStatus = ImprovementStatus.PROPOSED
    created_at: float = field(default_factory=time.time)
    benchmark_results: Optional[Dict] = None
    deployed_at: Optional[float] = None


@dataclass
class SandboxResult:
    sandbox_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    proposal_id: str = ""
    passed: bool = False
    safety_score: float = 0.0
    performance_score: float = 0.0
    side_effects: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    benchmark_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    proposal_id: str = ""
    baseline_score: float = 0.0
    improved_score: float = 0.0
    delta: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    statistically_significant: bool = False
    sample_size: int = 0
    timestamp: float = field(default_factory=time.time)


class WeaknessDetector:
    """Analyzes organism state to identify weaknesses requiring improvement."""

    WEAKNESS_THRESHOLDS = {
        "fitness": 0.4,
        "reliability": 0.5,
        "efficiency": 0.3,
        "energy_waste": 0.6,
        "error_rate": 0.2,
    }

    def detect(self, organism_state: Dict) -> List[Weakness]:
        weaknesses = []

        # Check global fitness
        fitness = organism_state.get("global_fitness", 0.5)
        if fitness < self.WEAKNESS_THRESHOLDS["fitness"]:
            weaknesses.append(Weakness(
                component="genome",
                description=f"Global fitness critically low: {fitness:.3f}",
                severity=1.0 - fitness,
                evidence=[f"fitness={fitness}"],
            ))

        # Check memory utilization
        memory_stats = organism_state.get("memory_stats", {})
        if isinstance(memory_stats, dict):
            working_mem = memory_stats.get("working", {})
            used = working_mem.get("used_slots", 0)
            total = working_mem.get("total_slots", 9)
            if total > 0 and used / total > 0.9:
                weaknesses.append(Weakness(
                    component="memory",
                    description="Working memory near capacity — consolidation needed",
                    severity=0.6,
                    evidence=[f"utilization={used/total:.0%}"],
                ))

        # Check energy efficiency
        energy_status = organism_state.get("energy_status", {})
        utilization = energy_status.get("utilization", 0)
        if utilization > self.WEAKNESS_THRESHOLDS["energy_waste"]:
            weaknesses.append(Weakness(
                component="cognition",
                description=f"High energy consumption: {utilization:.0%} utilized",
                severity=utilization - 0.5,
                evidence=[f"utilization={utilization}"],
            ))

        # Check meta-cognitive gaps
        gaps = organism_state.get("knowledge_gaps", [])
        for gap in gaps:
            if isinstance(gap, dict) and gap.get("severity", 0) > 0.7:
                weaknesses.append(Weakness(
                    component=f"knowledge_{gap.get('domain', 'unknown')}",
                    description=gap.get("description", "Critical knowledge gap"),
                    severity=gap.get("severity", 0.7),
                    evidence=["metacognition_report"],
                ))

        return sorted(weaknesses, key=lambda w: w.severity, reverse=True)


class MutationGenerator:
    """Generates improvement proposals from detected weaknesses."""

    MUTATION_STRATEGIES = {
        "genome": ["increase_elite_fraction", "reduce_mutation_rate", "crossover_boost"],
        "memory": ["expand_working_memory", "increase_consolidation_rate", "prune_episodic"],
        "cognition": ["reduce_deep_reasoning_cost", "cache_common_lookups", "lazy_evaluation"],
        "knowledge": ["targeted_exploration", "expert_consultation", "gap_filling_experiment"],
    }

    def generate(self, weakness: Weakness) -> ImprovementProposal:
        component_type = weakness.component.split("_")[0]
        strategies = self.MUTATION_STRATEGIES.get(component_type, ["parameter_tweak"])
        strategy = random.choice(strategies)

        changes = self._build_changes(component_type, strategy)
        expected_gain = min(0.3, weakness.severity * 0.4)
        risk = max(0.1, 0.3 - weakness.severity * 0.2)

        return ImprovementProposal(
            weakness_id=weakness.weakness_id,
            target_component=weakness.component,
            description=f"Apply '{strategy}' to improve {weakness.component}",
            mutation_type=self._classify_mutation(strategy),
            proposed_changes=changes,
            expected_gain=expected_gain,
            risk_level=risk,
        )

    def _build_changes(self, component: str, strategy: str) -> Dict:
        base = {"strategy": strategy, "component": component}
        if "reduce" in strategy:
            base["adjustment"] = -random.uniform(0.05, 0.15)
        elif "increase" in strategy or "expand" in strategy:
            base["adjustment"] = +random.uniform(0.05, 0.2)
        elif "boost" in strategy:
            base["multiplier"] = random.uniform(1.1, 1.5)
        else:
            base["mode"] = strategy
        return base

    def _classify_mutation(self, strategy: str) -> str:
        if any(x in strategy for x in ["rate", "cost", "threshold", "fraction"]):
            return "parameter_tweak"
        elif any(x in strategy for x in ["expand", "prune", "cache"]):
            return "structural"
        else:
            return "algorithmic"


class Sandbox:
    """
    Isolated test environment for safely evaluating proposals.
    Real implementation would use subprocess isolation; this is the interface.
    """

    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout = timeout_seconds
        self.run_count = 0

    async def test(self, proposal: ImprovementProposal,
                   test_suite: Optional[List[Callable]] = None) -> SandboxResult:
        self.run_count += 1
        start = time.time()
        result = SandboxResult(proposal_id=proposal.proposal_id)

        try:
            # Simulate sandbox execution
            await self._simulate_execution(proposal, result)
            result.runtime_seconds = time.time() - start
        except Exception as e:
            result.passed = False
            result.error = str(e)
            result.safety_score = 0.0

        return result

    async def _simulate_execution(self, proposal: ImprovementProposal,
                                   result: SandboxResult) -> None:
        import asyncio
        await asyncio.sleep(0.01)   # Simulate async test

        risk = proposal.risk_level
        result.safety_score = max(0.0, 1.0 - risk * random.uniform(0.5, 1.5))
        result.performance_score = min(1.0, 0.5 + proposal.expected_gain * random.uniform(0.5, 1.5))

        # Side effects detection
        if random.random() < risk * 0.3:
            result.side_effects.append("minor_performance_regression_in_auxiliary_component")

        result.passed = (result.safety_score > 0.6 and
                         len(result.side_effects) == 0 or
                         all("minor" in s for s in result.side_effects))


class RecursiveSelfImprover:
    """
    The organism's self-improvement executive.

    Full cycle:
    analyze → detect_weakness → generate_proposal → sandbox →
    benchmark → immune_check → deploy → monitor → repeat
    """

    def __init__(self,
                 immune_system=None,
                 benchmarker=None):
        self.improver_id = str(uuid.uuid4())[:8]
        self.detector = WeaknessDetector()
        self.generator = MutationGenerator()
        self.sandbox = Sandbox()
        self.immune_system = immune_system
        self.benchmarker = benchmarker

        self.detected_weaknesses: List[Weakness] = []
        self.proposals: Dict[str, ImprovementProposal] = {}
        self.deployed_improvements: List[str] = []
        self.rejected_proposals: List[str] = []
        self.improvement_history: List[Dict] = []
        self.cycle_count = 0

        # Running intelligence score
        self.intelligence_score = 0.5

    # ─── Full Improvement Cycle ──────────────────────────────────────────

    async def run_cycle(self, organism_state: Dict) -> Dict:
        """Execute one complete self-improvement cycle."""
        self.cycle_count += 1
        cycle_log = {
            "cycle": self.cycle_count,
            "timestamp": time.time(),
            "weaknesses_found": 0,
            "proposals_generated": 0,
            "proposals_deployed": 0,
            "proposals_rejected": 0,
            "intelligence_delta": 0.0,
        }

        # Step 1: Analyze & detect
        weaknesses = self.detector.detect(organism_state)
        self.detected_weaknesses.extend(weaknesses)
        cycle_log["weaknesses_found"] = len(weaknesses)

        # Step 2: Generate proposals (max 3 per cycle)
        new_proposals = []
        for weakness in weaknesses[:3]:
            proposal = self.generator.generate(weakness)
            weakness.improvement_attempts += 1
            self.proposals[proposal.proposal_id] = proposal
            new_proposals.append(proposal)
        cycle_log["proposals_generated"] = len(new_proposals)

        # Steps 3-6: Sandbox → Benchmark → Immune → Deploy
        deployed = 0
        rejected = 0
        fitness_gains = []

        for proposal in new_proposals:
            # Sandbox
            proposal.status = ImprovementStatus.SANDBOXED
            sandbox_result = await self.sandbox.test(proposal)

            if not sandbox_result.passed:
                proposal.status = ImprovementStatus.REJECTED
                self.rejected_proposals.append(proposal.proposal_id)
                rejected += 1
                continue

            # Benchmark
            proposal.status = ImprovementStatus.BENCHMARKED
            bench = self._benchmark(proposal, organism_state)
            proposal.benchmark_results = bench.__dict__

            if bench.delta < 0:
                proposal.status = ImprovementStatus.REJECTED
                self.rejected_proposals.append(proposal.proposal_id)
                rejected += 1
                continue

            # Immune check
            if self.immune_system:
                approved, scan = self.immune_system.scan_mutation(proposal.proposed_changes)
                if not approved:
                    proposal.status = ImprovementStatus.REJECTED
                    self.rejected_proposals.append(proposal.proposal_id)
                    rejected += 1
                    continue

            # Deploy
            proposal.status = ImprovementStatus.DEPLOYED
            proposal.deployed_at = time.time()
            self.deployed_improvements.append(proposal.proposal_id)
            fitness_gains.append(bench.delta)
            deployed += 1

        cycle_log["proposals_deployed"] = deployed
        cycle_log["proposals_rejected"] = rejected

        # Update intelligence score
        if fitness_gains:
            mean_gain = sum(fitness_gains) / len(fitness_gains)
            self.intelligence_score = min(1.0, self.intelligence_score + mean_gain * 0.1)
            cycle_log["intelligence_delta"] = mean_gain

        self.improvement_history.append(cycle_log)
        return cycle_log

    def _benchmark(self, proposal: ImprovementProposal,
                   state: Dict) -> BenchmarkResult:
        """Compare baseline vs proposed change."""
        baseline = state.get("global_fitness", 0.5)
        # Simulate: improved score = baseline + expected_gain ± noise
        improved = baseline + proposal.expected_gain * random.uniform(0.3, 1.2)
        improved = max(0.0, min(1.0, improved))
        delta = improved - baseline

        return BenchmarkResult(
            proposal_id=proposal.proposal_id,
            baseline_score=baseline,
            improved_score=improved,
            delta=delta,
            metrics={
                "accuracy": improved,
                "efficiency": improved * 0.9,
                "reliability": improved * 0.95,
            },
            statistically_significant=abs(delta) > 0.02,
            sample_size=30,
        )

    # ─── Continuous Monitoring ───────────────────────────────────────────

    def rollback(self, proposal_id: str) -> bool:
        """Rollback a deployed improvement if it causes regression."""
        if proposal_id in self.deployed_improvements:
            self.deployed_improvements.remove(proposal_id)
            if proposal_id in self.proposals:
                self.proposals[proposal_id].status = ImprovementStatus.ROLLED_BACK
            return True
        return False

    def compute_intelligence(self, knowledge: float, adaptability: float,
                              efficiency: float) -> float:
        """I = K × A × E × SI"""
        si = self.intelligence_score
        self.intelligence_score = (knowledge * adaptability * efficiency * si) ** 0.25
        return self.intelligence_score

    # ─── Reporting ───────────────────────────────────────────────────────

    def improvement_report(self) -> Dict:
        return {
            "improver_id": self.improver_id,
            "cycles_run": self.cycle_count,
            "weaknesses_detected": len(self.detected_weaknesses),
            "proposals_total": len(self.proposals),
            "deployed": len(self.deployed_improvements),
            "rejected": len(self.rejected_proposals),
            "deployment_rate": len(self.deployed_improvements) / max(1, len(self.proposals)),
            "intelligence_score": round(self.intelligence_score, 4),
            "recent_cycles": self.improvement_history[-5:],
        }

    def __repr__(self) -> str:
        return (f"RecursiveSelfImprover(id={self.improver_id}, "
                f"cycles={self.cycle_count}, "
                f"deployed={len(self.deployed_improvements)}, "
                f"intelligence={self.intelligence_score:.3f})")