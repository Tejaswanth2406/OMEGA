"""
Layer 2: Artificial Cell System
================================
Each cognitive subsystem is modeled as a living cell.
Cells have state, energy, fitness, memory, and communicate via signals.
"""
from __future__ import annotations
import uuid
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from collections import deque


class CellState(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    STRESSED = "stressed"      # low energy
    REPAIRING = "repairing"
    DIVIDING = "dividing"      # spawning child cell
    APOPTOSIS = "apoptosis"    # programmed death


@dataclass
class Signal:
    """Inter-cell communication message."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    sender: str = ""
    receiver: str = ""          # "" = broadcast
    signal_type: str = "data"
    payload: Any = None
    priority: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CellMemory:
    """Local cell memory — short-term episodic buffer."""
    capacity: int = 64
    _buffer: deque = field(default_factory=lambda: deque(maxlen=64))

    def store(self, item: Any) -> None:
        self._buffer.append({"item": item, "ts": time.time()})

    def recall(self, n: int = 5) -> List[Any]:
        items = list(self._buffer)
        return [x["item"] for x in items[-n:]]

    def clear(self) -> None:
        self._buffer.clear()


class BaseCell(ABC):
    """Abstract base for all cognitive cells."""

    ENERGY_CAPACITY = 100.0
    IDLE_DRAIN = 0.1       # energy/tick at rest
    ACTIVE_DRAIN = 1.0     # energy/tick when processing

    def __init__(self, cell_id: Optional[str] = None, name: str = "cell"):
        self.cell_id = cell_id or str(uuid.uuid4())[:8]
        self.name = name
        self.state = CellState.IDLE
        self.energy = self.ENERGY_CAPACITY
        self.fitness = 0.5
        self.memory = CellMemory()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: List[Signal] = []
        self.tick_count = 0
        self.birth_time = time.time()
        self.metrics: Dict[str, float] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_energy_consumed": 0.0,
        }

    @abstractmethod
    async def process(self, signal: Signal) -> Optional[Signal]:
        """Core cell logic — override in each cell type."""
        ...

    async def tick(self) -> None:
        """One lifecycle tick: drain energy, process signals."""
        self.tick_count += 1
        self._drain_energy()

        if self.energy <= 0:
            self.state = CellState.STRESSED
            return

        if not self.inbox.empty():
            self.state = CellState.ACTIVE
            signal = await self.inbox.get()
            try:
                response = await self.process(signal)
                if response:
                    self.outbox.append(response)
                self.metrics["tasks_completed"] += 1
                self._update_fitness(success=True)
            except Exception as e:
                self.metrics["tasks_failed"] += 1
                self._update_fitness(success=False)
        else:
            self.state = CellState.IDLE

    def _drain_energy(self) -> None:
        drain = self.ACTIVE_DRAIN if self.state == CellState.ACTIVE else self.IDLE_DRAIN
        self.energy = max(0.0, self.energy - drain)
        self.metrics["total_energy_consumed"] += drain

    def refuel(self, amount: float = 20.0) -> None:
        self.energy = min(self.ENERGY_CAPACITY, self.energy + amount)

    def _update_fitness(self, success: bool, alpha: float = 0.05) -> None:
        score = 1.0 if success else 0.0
        self.fitness = (1 - alpha) * self.fitness + alpha * score

    def send(self, receiver: str, payload: Any, signal_type: str = "data") -> Signal:
        sig = Signal(sender=self.cell_id, receiver=receiver,
                     signal_type=signal_type, payload=payload)
        self.outbox.append(sig)
        return sig

    def status(self) -> Dict:
        return {
            "cell_id": self.cell_id,
            "name": self.name,
            "state": self.state.value,
            "energy": round(self.energy, 2),
            "fitness": round(self.fitness, 3),
            "tick": self.tick_count,
            "metrics": self.metrics,
        }


class MemoryCell(BaseCell):
    """Manages storage and retrieval across memory tiers."""

    def __init__(self):
        super().__init__(name="memory_cell")
        self.store: Dict[str, Any] = {}

    async def process(self, signal: Signal) -> Optional[Signal]:
        if signal.signal_type == "store":
            key, value = signal.payload["key"], signal.payload["value"]
            self.store[key] = value
            self.memory.store({"op": "store", "key": key})
            return Signal(sender=self.cell_id, receiver=signal.sender,
                          signal_type="ack", payload={"key": key})
        elif signal.signal_type == "retrieve":
            key = signal.payload["key"]
            value = self.store.get(key)
            return Signal(sender=self.cell_id, receiver=signal.sender,
                          signal_type="data", payload={"key": key, "value": value})
        return None


class ReasoningCell(BaseCell):
    """Handles logical inference and reasoning tasks."""

    def __init__(self):
        super().__init__(name="reasoning_cell")
        self.reasoning_depth = 3

    async def process(self, signal: Signal) -> Optional[Signal]:
        if signal.signal_type == "reason":
            query = signal.payload.get("query", "")
            context = signal.payload.get("context", [])
            result = await self._chain_of_thought(query, context)
            return Signal(sender=self.cell_id, receiver=signal.sender,
                          signal_type="conclusion", payload=result)
        return None

    async def _chain_of_thought(self, query: str, context: List) -> Dict:
        """Stub — replaced by LLM call in production."""
        steps = [f"Step {i+1}: analyzing '{query}'" for i in range(self.reasoning_depth)]
        return {
            "query": query,
            "reasoning_steps": steps,
            "conclusion": f"Reasoned conclusion for: {query}",
            "confidence": self.fitness,
        }


class PlanningCell(BaseCell):
    """Decomposes goals into executable plans."""

    def __init__(self):
        super().__init__(name="planning_cell")
        self.active_plans: Dict[str, List] = {}

    async def process(self, signal: Signal) -> Optional[Signal]:
        if signal.signal_type == "plan":
            goal = signal.payload.get("goal", "")
            plan = self._decompose(goal)
            self.active_plans[signal.signal_id] = plan
            return Signal(sender=self.cell_id, receiver=signal.sender,
                          signal_type="plan_ready", payload={"plan": plan, "goal": goal})
        return None

    def _decompose(self, goal: str) -> List[Dict]:
        """Hierarchical task network decomposition (stub)."""
        return [
            {"step": 1, "action": "gather_information", "goal": goal},
            {"step": 2, "action": "reason_about_options", "goal": goal},
            {"step": 3, "action": "select_best_option", "goal": goal},
            {"step": 4, "action": "execute_and_monitor", "goal": goal},
            {"step": 5, "action": "reflect_and_learn", "goal": goal},
        ]


class ReflectionCell(BaseCell):
    """Performs introspective analysis of other cells' performance."""

    def __init__(self):
        super().__init__(name="reflection_cell")
        self.observations: List[Dict] = []

    async def process(self, signal: Signal) -> Optional[Signal]:
        if signal.signal_type == "reflect":
            obs = signal.payload
            self.observations.append(obs)
            insight = self._analyze(obs)
            return Signal(sender=self.cell_id, receiver=signal.sender,
                          signal_type="insight", payload=insight)
        return None

    def _analyze(self, observation: Dict) -> Dict:
        return {
            "observation": observation,
            "pattern": "detected_pattern",
            "recommendation": "optimization_action",
            "confidence": 0.7,
        }


class RepairCell(BaseCell):
    """Detects and repairs dysfunctional cells."""

    def __init__(self):
        super().__init__(name="repair_cell")
        self.repair_log: List[Dict] = []

    async def process(self, signal: Signal) -> Optional[Signal]:
        if signal.signal_type == "diagnose":
            cell_status = signal.payload
            diagnosis = self._diagnose(cell_status)
            if diagnosis["needs_repair"]:
                self.repair_log.append(diagnosis)
            return Signal(sender=self.cell_id, receiver=signal.sender,
                          signal_type="diagnosis", payload=diagnosis)
        return None

    def _diagnose(self, status: Dict) -> Dict:
        needs_repair = status.get("energy", 100) < 20 or status.get("fitness", 1.0) < 0.3
        return {
            "cell_id": status.get("cell_id"),
            "needs_repair": needs_repair,
            "issues": ["low_energy"] if status.get("energy", 100) < 20 else [],
            "action": "refuel" if needs_repair else "none",
        }


class CellularOrchanism:
    """Orchestrates all cells — the organism's tissue layer."""

    def __init__(self):
        self.cells: Dict[str, BaseCell] = {}
        self._register_default_cells()

    def _register_default_cells(self) -> None:
        defaults = [MemoryCell(), ReasoningCell(), PlanningCell(),
                    ReflectionCell(), RepairCell()]
        for cell in defaults:
            self.cells[cell.name] = cell

    def register(self, cell: BaseCell) -> None:
        self.cells[cell.name] = cell

    async def route(self, signal: Signal) -> None:
        """Route a signal to the appropriate cell."""
        if signal.receiver and signal.receiver in self.cells:
            await self.cells[signal.receiver].inbox.put(signal)
        elif not signal.receiver:
            # Broadcast
            for cell in self.cells.values():
                await cell.inbox.put(signal)

    async def tick_all(self) -> None:
        """One global tick across all cells."""
        await asyncio.gather(*[cell.tick() for cell in self.cells.values()])

        # Collect outbox signals and route them
        for cell in self.cells.values():
            for sig in cell.outbox:
                await self.route(sig)
            cell.outbox.clear()

    def health_report(self) -> Dict:
        return {name: cell.status() for name, cell in self.cells.items()}