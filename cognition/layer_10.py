"""
Layer 10: Energy-Based Cognition
==================================
Every cognitive operation has a cost.
The organism learns efficiency naturally through energy constraints.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


class OperationType(str, Enum):
    SIMPLE_LOOKUP = "simple_lookup"
    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    REASONING_SHALLOW = "reasoning_shallow"
    REASONING_DEEP = "reasoning_deep"
    PLANNING = "planning"
    SELF_REFLECTION = "self_reflection"
    ARCHITECTURE_REDESIGN = "architecture_redesign"
    EVOLUTION_STEP = "evolution_step"
    IMMUNE_SCAN = "immune_scan"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    GRAPH_TRAVERSAL = "graph_traversal"


# Default energy costs (calibrated empirically)
DEFAULT_COSTS: Dict[OperationType, float] = {
    OperationType.SIMPLE_LOOKUP:        1.0,
    OperationType.RETRIEVAL:            3.0,
    OperationType.TOOL_CALL:            5.0,
    OperationType.REASONING_SHALLOW:    8.0,
    OperationType.REASONING_DEEP:      20.0,
    OperationType.PLANNING:            15.0,
    OperationType.SELF_REFLECTION:     12.0,
    OperationType.ARCHITECTURE_REDESIGN: 50.0,
    OperationType.EVOLUTION_STEP:      25.0,
    OperationType.IMMUNE_SCAN:          4.0,
    OperationType.MEMORY_CONSOLIDATION: 6.0,
    OperationType.GRAPH_TRAVERSAL:      2.0,
}


@dataclass
class EnergyTransaction:
    txn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    operation: OperationType = OperationType.SIMPLE_LOOKUP
    cost: float = 0.0
    benefit: float = 0.0        # measured outcome value
    efficiency: float = 0.0     # benefit / cost
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    metadata: Dict = field(default_factory=dict)

    def compute_efficiency(self) -> float:
        self.efficiency = self.benefit / max(self.cost, 0.001)
        return self.efficiency


@dataclass
class EnergyBudget:
    total: float = 1000.0
    current: float = 1000.0
    reserved: float = 0.0       # emergency reserve
    recharge_rate: float = 10.0 # energy per tick

    @property
    def available(self) -> float:
        return max(0.0, self.current - self.reserved)

    @property
    def utilization(self) -> float:
        return 1.0 - (self.current / self.total)

    def can_afford(self, cost: float) -> bool:
        return self.available >= cost

    def spend(self, cost: float) -> bool:
        if not self.can_afford(cost):
            return False
        self.current -= cost
        return True

    def recharge(self, ticks: int = 1) -> float:
        gained = self.recharge_rate * ticks
        before = self.current
        self.current = min(self.total, self.current + gained)
        return self.current - before

    def set_reserve(self, fraction: float = 0.1) -> None:
        self.reserved = self.total * fraction


class EnergyPolicy:
    """
    Decision policy for energy allocation.
    Implements different strategies depending on energy level.
    """

    @staticmethod
    def can_run(operation: OperationType, budget: EnergyBudget,
                costs: Dict[OperationType, float]) -> Tuple[bool, str]:
        cost = costs.get(operation, 10.0)
        if not budget.can_afford(cost):
            return False, f"Insufficient energy: need {cost:.1f}, have {budget.available:.1f}"

        # Conservation mode: block expensive ops when energy < 30%
        if budget.utilization > 0.7 and cost > 15.0:
            return False, f"Conservation mode: blocking high-cost op (cost={cost})"

        # Critical mode: only basic ops when energy < 10%
        if budget.utilization > 0.9 and cost > 5.0:
            return False, f"Critical mode: energy at {budget.utilization:.0%}"

        return True, "ok"

    @staticmethod
    def select_cheapest_alternative(operation: OperationType,
                                     costs: Dict[OperationType, float]) -> Optional[OperationType]:
        """Suggest a cheaper alternative operation."""
        fallbacks = {
            OperationType.REASONING_DEEP: OperationType.REASONING_SHALLOW,
            OperationType.ARCHITECTURE_REDESIGN: OperationType.SELF_REFLECTION,
            OperationType.EVOLUTION_STEP: OperationType.IMMUNE_SCAN,
            OperationType.PLANNING: OperationType.REASONING_SHALLOW,
        }
        return fallbacks.get(operation)


class EnergyCognitionEngine:
    """
    Central energy management system for the organism.
    All cognitive operations must request energy here.
    """

    def __init__(self, initial_energy: float = 1000.0,
                 recharge_rate: float = 10.0):
        self.engine_id = str(uuid.uuid4())[:8]
        self.budget = EnergyBudget(
            total=initial_energy,
            current=initial_energy,
            recharge_rate=recharge_rate,
        )
        self.budget.set_reserve(0.05)
        self.costs: Dict[OperationType, float] = dict(DEFAULT_COSTS)
        self.transactions: List[EnergyTransaction] = []
        self.tick_count = 0
        self.total_benefit: float = 0.0

        # Learned cost adjustments (updated via feedback)
        self._cost_adjustments: Dict[str, float] = {}

    # ─── Core Operation ──────────────────────────────────────────────────

    def request(self, operation: OperationType,
                metadata: Optional[Dict] = None) -> Tuple[bool, Optional[EnergyTransaction]]:
        """
        Request energy to run an operation.
        Returns (approved, transaction) — transaction is None if denied.
        """
        allowed, reason = EnergyPolicy.can_run(operation, self.budget, self.costs)

        if not allowed:
            # Try a fallback
            alt = EnergyPolicy.select_cheapest_alternative(operation, self.costs)
            if alt and EnergyPolicy.can_run(alt, self.budget, self.costs)[0]:
                operation = alt
            else:
                return False, None

        cost = self.costs[operation]
        self.budget.spend(cost)

        txn = EnergyTransaction(
            operation=operation,
            cost=cost,
            metadata=metadata or {},
        )
        self.transactions.append(txn)
        return True, txn

    def complete(self, txn: EnergyTransaction, benefit: float, success: bool = True) -> None:
        """Record outcome of a completed operation."""
        txn.benefit = benefit
        txn.success = success
        txn.compute_efficiency()
        self.total_benefit += benefit

        # Adapt costs: if operation consistently underperforms, make it cheaper to retry
        key = txn.operation.value
        recent_efficiency = self._recent_efficiency(txn.operation, window=10)
        if recent_efficiency < 0.5:
            self._cost_adjustments[key] = self.costs[txn.operation] * 0.95
        elif recent_efficiency > 2.0:
            self._cost_adjustments[key] = self.costs[txn.operation] * 1.05

        if key in self._cost_adjustments:
            self.costs[txn.operation] = max(0.5, self._cost_adjustments[key])

    def tick(self, ticks: int = 1) -> float:
        """Advance time — recharge energy."""
        self.tick_count += ticks
        return self.budget.recharge(ticks)

    # ─── Convenience Decorator ───────────────────────────────────────────

    def energy_guarded(self, operation: OperationType, benefit_fn: Optional[Callable] = None):
        """
        Decorator: wrap a function with energy gating.
        If energy is denied, the function is not called.
        """
        def decorator(fn: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                approved, txn = self.request(operation)
                if not approved:
                    raise RuntimeError(f"Energy denied for {operation.value}")
                try:
                    result = fn(*args, **kwargs)
                    benefit = benefit_fn(result) if benefit_fn else 1.0
                    self.complete(txn, benefit, success=True)
                    return result
                except Exception as e:
                    self.complete(txn, 0.0, success=False)
                    raise
            return wrapper
        return decorator

    # ─── Analytics ───────────────────────────────────────────────────────

    def _recent_efficiency(self, operation: OperationType, window: int = 10) -> float:
        recent = [t for t in self.transactions[-window * 5:]
                  if t.operation == operation and t.efficiency > 0][-window:]
        if not recent:
            return 1.0
        return sum(t.efficiency for t in recent) / len(recent)

    def operation_report(self) -> Dict[str, Dict]:
        report = {}
        for op in OperationType:
            txns = [t for t in self.transactions if t.operation == op]
            if not txns:
                continue
            report[op.value] = {
                "count": len(txns),
                "total_cost": sum(t.cost for t in txns),
                "total_benefit": sum(t.benefit for t in txns),
                "mean_efficiency": sum(t.efficiency for t in txns) / len(txns),
                "success_rate": sum(1 for t in txns if t.success) / len(txns),
                "current_cost": self.costs[op],
            }
        return report

    def status(self) -> Dict:
        return {
            "engine_id": self.engine_id,
            "energy_level": round(self.budget.current, 2),
            "energy_total": self.budget.total,
            "utilization": round(self.budget.utilization, 3),
            "available": round(self.budget.available, 2),
            "total_transactions": len(self.transactions),
            "total_benefit": round(self.total_benefit, 2),
            "global_efficiency": round(
                self.total_benefit / max(1, sum(t.cost for t in self.transactions)), 3
            ),
            "tick": self.tick_count,
        }

    def __repr__(self) -> str:
        return (f"EnergyCognitionEngine("
                f"energy={self.budget.current:.1f}/{self.budget.total}, "
                f"utilization={self.budget.utilization:.1%})")