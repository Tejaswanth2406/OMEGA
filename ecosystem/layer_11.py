"""
Layer 11: Cognitive Ecosystem
================================
A population of competing, cooperating specialized agents.
Competition drives specialization. Diversity drives robustness.
"""
from __future__ import annotations
import uuid
import time
import random
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class AgentRole(str, Enum):
    SCIENTIST = "scientist"
    ENGINEER = "engineer"
    CRITIC = "critic"
    ARCHITECT = "architect"
    PLANNER = "planner"
    EXPLORER = "explorer"
    TEACHER = "teacher"
    AUDITOR = "auditor"
    SYNTHESIZER = "synthesizer"
    GUARDIAN = "guardian"


@dataclass
class AgentTask:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    task_type: str = "general"
    payload: Any = None
    priority: int = 0
    deadline: Optional[float] = None
    assigned_to: Optional[str] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    task_id: str = ""
    agent_id: str = ""
    output: Any = None
    confidence: float = 0.5
    quality_score: float = 0.5
    completed_at: float = field(default_factory=time.time)
    error: Optional[str] = None


class BaseAgent(ABC):
    """Abstract agent — each specialization extends this."""

    def __init__(self, role: AgentRole, name: Optional[str] = None):
        self.agent_id = str(uuid.uuid4())[:8]
        self.role = role
        self.name = name or f"{role.value}_{self.agent_id}"
        self.fitness = 0.5
        self.energy = 100.0
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.generation = 0
        self.birth_time = time.time()
        self.genome: Dict[str, Any] = self._default_genome()
        self.memory: List[Dict] = []
        self.collaborators: List[str] = []   # agent IDs

    def _default_genome(self) -> Dict:
        return {
            "specialization": self.role.value,
            "risk_tolerance": random.uniform(0.2, 0.8),
            "creativity": random.uniform(0.2, 0.8),
            "thoroughness": random.uniform(0.3, 0.9),
            "cooperation_weight": random.uniform(0.3, 0.7),
            "mutation_rate": random.uniform(0.001, 0.01),
        }

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult: ...

    def update_fitness(self, score: float, alpha: float = 0.1) -> None:
        self.fitness = (1 - alpha) * self.fitness + alpha * score

    def remember(self, task: AgentTask, result: AgentResult) -> None:
        self.memory.append({
            "task": task.description,
            "quality": result.quality_score,
            "timestamp": time.time(),
        })
        if len(self.memory) > 100:
            self.memory = self.memory[-100:]

    def status(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "fitness": round(self.fitness, 3),
            "energy": round(self.energy, 2),
            "generation": self.generation,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
        }


# ─── Specialized Agents ──────────────────────────────────────────────────────

class ScientistAgent(BaseAgent):
    """Generates hypotheses and designs experiments."""

    def __init__(self):
        super().__init__(AgentRole.SCIENTIST)
        self.hypotheses_generated = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        # Generate hypothesis about the task
        hypothesis = {
            "statement": f"If we apply systematic analysis to '{task.description}', "
                         f"then we can improve outcomes by {random.uniform(5, 30):.1f}%",
            "confidence": self.fitness * self.genome["thoroughness"],
            "experiment_design": {
                "method": "controlled_comparison",
                "variables": ["approach", "parameters"],
                "metrics": ["accuracy", "efficiency"],
            }
        }
        self.hypotheses_generated += 1
        quality = self.genome["thoroughness"] * random.uniform(0.7, 1.0)
        return AgentResult(
            task_id=task.task_id, agent_id=self.agent_id,
            output=hypothesis, confidence=hypothesis["confidence"],
            quality_score=quality,
        )


class EngineerAgent(BaseAgent):
    """Implements solutions and builds tools."""

    def __init__(self):
        super().__init__(AgentRole.ENGINEER)

    async def execute(self, task: AgentTask) -> AgentResult:
        solution = {
            "approach": "systematic_implementation",
            "steps": [
                f"Analyze requirements for: {task.description}",
                "Design data structures",
                "Implement core logic",
                "Test and validate",
                "Optimize performance",
            ],
            "estimated_complexity": random.uniform(0.3, 0.9),
            "estimated_time": random.uniform(1, 10),
        }
        quality = self.genome["thoroughness"] * (1 - self.genome["risk_tolerance"] * 0.2)
        return AgentResult(
            task_id=task.task_id, agent_id=self.agent_id,
            output=solution, quality_score=quality,
        )


class CriticAgent(BaseAgent):
    """Evaluates and critiques other agents' outputs."""

    def __init__(self):
        super().__init__(AgentRole.CRITIC)
        self.critiques_issued = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        critique = {
            "target": task.description,
            "strengths": ["Clear problem definition", "Reasonable approach"],
            "weaknesses": [f"Potential issue #{i+1}" for i in range(random.randint(1, 3))],
            "severity": random.choice(["low", "medium", "high"]),
            "recommendation": "Revise and resubmit with addressed concerns",
            "score": random.uniform(0.4, 0.9),
        }
        self.critiques_issued += 1
        quality = self.genome["thoroughness"] * 0.9
        return AgentResult(
            task_id=task.task_id, agent_id=self.agent_id,
            output=critique, quality_score=quality,
        )


class ArchitectAgent(BaseAgent):
    """Designs system architectures and structural changes."""

    def __init__(self):
        super().__init__(AgentRole.ARCHITECT)

    async def execute(self, task: AgentTask) -> AgentResult:
        design = {
            "architecture_type": random.choice(["layered", "microservices", "event-driven", "graph"]),
            "components": [f"component_{i}" for i in range(random.randint(3, 8))],
            "interfaces": ["REST", "message_bus", "shared_memory"],
            "scalability": self.genome["thoroughness"],
            "resilience_score": 1 - self.genome["risk_tolerance"] * 0.5,
            "design_rationale": f"Optimized for {task.description}",
        }
        quality = self.genome["creativity"] * self.genome["thoroughness"]
        return AgentResult(
            task_id=task.task_id, agent_id=self.agent_id,
            output=design, quality_score=quality,
        )


class ExplorerAgent(BaseAgent):
    """Explores new solution spaces and discovers novel approaches."""

    def __init__(self):
        super().__init__(AgentRole.EXPLORER)
        self.discoveries = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        exploration = {
            "novel_approaches": [
                f"Unconventional approach {i+1}: {random.choice(['bio-inspired', 'quantum-analog', 'evolutionary', 'fractal'])}"
                for i in range(random.randint(2, 5))
            ],
            "risk_level": self.genome["risk_tolerance"],
            "novelty_score": self.genome["creativity"],
            "feasibility": 1 - self.genome["creativity"] * 0.3,
        }
        self.discoveries += 1
        quality = self.genome["creativity"] * random.uniform(0.5, 1.0)
        return AgentResult(
            task_id=task.task_id, agent_id=self.agent_id,
            output=exploration, quality_score=quality,
        )


class AuditorAgent(BaseAgent):
    """Audits the organism's behavior for safety and correctness."""

    def __init__(self):
        super().__init__(AgentRole.AUDITOR)
        self.audits_completed = 0

    async def execute(self, task: AgentTask) -> AgentResult:
        audit = {
            "audit_target": task.description,
            "compliance_checks": {
                "safety": random.random() > 0.1,
                "consistency": random.random() > 0.05,
                "resource_usage": random.random() > 0.15,
                "ethics": random.random() > 0.02,
            },
            "risk_score": random.uniform(0.0, 0.3),
            "recommendations": ["Log all mutations", "Increase test coverage"],
            "passed": True,
        }
        passed = all(audit["compliance_checks"].values())
        audit["passed"] = passed
        self.audits_completed += 1
        quality = self.genome["thoroughness"] * (1.0 if passed else 0.5)
        return AgentResult(
            task_id=task.task_id, agent_id=self.agent_id,
            output=audit, quality_score=quality,
        )


# ─── Ecosystem Orchestrator ──────────────────────────────────────────────────

@dataclass
class EcosystemConfig:
    min_population: int = 2
    max_population: int = 20
    competition_interval: int = 10   # ticks between competitions
    reproduction_threshold: float = 0.7
    extinction_threshold: float = 0.2
    migration_rate: float = 0.1


class CognitiveEcosystem:
    """
    Manages the population of agents.
    Agents compete for tasks, reproduce when fit, and go extinct when unfit.
    """

    AGENT_FACTORY: Dict[AgentRole, type] = {
        AgentRole.SCIENTIST: ScientistAgent,
        AgentRole.ENGINEER: EngineerAgent,
        AgentRole.CRITIC: CriticAgent,
        AgentRole.ARCHITECT: ArchitectAgent,
        AgentRole.EXPLORER: ExplorerAgent,
        AgentRole.AUDITOR: AuditorAgent,
    }

    def __init__(self, config: Optional[EcosystemConfig] = None):
        self.ecosystem_id = str(uuid.uuid4())[:8]
        self.config = config or EcosystemConfig()
        self.agents: Dict[str, BaseAgent] = {}
        self.task_queue: List[AgentTask] = []
        self.results: List[AgentResult] = []
        self.tick_count = 0
        self.event_log: List[Dict] = []

        self._spawn_initial_population()

    def _spawn_initial_population(self) -> None:
        """Start with one agent per role."""
        for role, cls in self.AGENT_FACTORY.items():
            agent = cls()
            self.agents[agent.agent_id] = agent

    # ─── Agent Lifecycle ─────────────────────────────────────────────────

    def spawn(self, role: AgentRole) -> Optional[BaseAgent]:
        if len(self.agents) >= self.config.max_population:
            return None
        cls = self.AGENT_FACTORY.get(role)
        if not cls:
            return None
        agent = cls()
        self.agents[agent.agent_id] = agent
        self._log("spawn", {"agent_id": agent.agent_id, "role": role.value})
        return agent

    def cull(self, agent_id: str) -> bool:
        if agent_id in self.agents and len(self.agents) > self.config.min_population:
            del self.agents[agent_id]
            self._log("cull", {"agent_id": agent_id})
            return True
        return False

    def reproduce(self, parent: BaseAgent) -> Optional[BaseAgent]:
        """Create a child agent with mutated genome."""
        child_cls = self.AGENT_FACTORY.get(parent.role)
        if not child_cls or len(self.agents) >= self.config.max_population:
            return None
        child = child_cls()
        child.generation = parent.generation + 1
        # Inherit and mutate genome
        for k, v in parent.genome.items():
            if isinstance(v, float):
                child.genome[k] = max(0.0, min(1.0, v + random.gauss(0, 0.05)))
            else:
                child.genome[k] = v
        self.agents[child.agent_id] = child
        self._log("reproduce", {"parent": parent.agent_id, "child": child.agent_id})
        return child

    # ─── Task Assignment ─────────────────────────────────────────────────

    def submit_task(self, task: AgentTask) -> str:
        self.task_queue.append(task)
        return task.task_id

    def _select_agent_for_task(self, task: AgentTask) -> Optional[BaseAgent]:
        """Tournament selection weighted by fitness and energy."""
        candidates = [a for a in self.agents.values() if a.energy > 10]
        if not candidates:
            return None
        # Weight by fitness * energy
        weights = [a.fitness * (a.energy / 100) for a in candidates]
        total = sum(weights)
        if total == 0:
            return random.choice(candidates)
        pick = random.uniform(0, total)
        current = 0.0
        for agent, w in zip(candidates, weights):
            current += w
            if current >= pick:
                return agent
        return candidates[-1]

    async def process_tasks(self, max_tasks: int = 10) -> List[AgentResult]:
        """Process up to max_tasks from the queue."""
        results = []
        tasks_to_process = self.task_queue[:max_tasks]
        self.task_queue = self.task_queue[max_tasks:]

        for task in tasks_to_process:
            agent = self._select_agent_for_task(task)
            if not agent:
                continue
            task.assigned_to = agent.agent_id
            try:
                result = await agent.execute(task)
                agent.tasks_completed += 1
                agent.energy -= 5.0
                agent.update_fitness(result.quality_score)
                agent.remember(task, result)
                results.append(result)
                self.results.append(result)
            except Exception as e:
                agent.tasks_failed += 1
                agent.update_fitness(0.0)

        return results

    # ─── Ecosystem Tick ──────────────────────────────────────────────────

    async def tick(self) -> Dict:
        self.tick_count += 1

        # Recharge agents
        for agent in self.agents.values():
            agent.energy = min(100.0, agent.energy + 2.0)

        # Process available tasks
        results = await self.process_tasks()

        # Periodic competition & selection
        if self.tick_count % self.config.competition_interval == 0:
            self._compete()

        return {
            "tick": self.tick_count,
            "agents": len(self.agents),
            "tasks_processed": len(results),
            "queue_remaining": len(self.task_queue),
        }

    def _compete(self) -> None:
        """Natural selection — reward fit, cull unfit, reproduce fit."""
        agents = list(self.agents.values())

        for agent in agents:
            if agent.fitness >= self.config.reproduction_threshold:
                self.reproduce(agent)
            elif agent.fitness < self.config.extinction_threshold:
                self.cull(agent.agent_id)

    # ─── Observability ───────────────────────────────────────────────────

    def _log(self, event: str, data: Dict) -> None:
        self.event_log.append({
            "event": event, "data": data, "tick": self.tick_count,
            "timestamp": time.time(),
        })

    def population_stats(self) -> Dict:
        agents = list(self.agents.values())
        if not agents:
            return {}
        role_counts = {}
        for a in agents:
            role_counts[a.role.value] = role_counts.get(a.role.value, 0) + 1
        return {
            "ecosystem_id": self.ecosystem_id,
            "total_agents": len(agents),
            "role_distribution": role_counts,
            "mean_fitness": sum(a.fitness for a in agents) / len(agents),
            "mean_energy": sum(a.energy for a in agents) / len(agents),
            "mean_generation": sum(a.generation for a in agents) / len(agents),
            "total_tasks_processed": len(self.results),
            "tick": self.tick_count,
        }

    def leaderboard(self, n: int = 5) -> List[Dict]:
        return [a.status() for a in
                sorted(self.agents.values(), key=lambda a: a.fitness, reverse=True)[:n]]

    def __repr__(self) -> str:
        return (f"CognitiveEcosystem(id={self.ecosystem_id}, "
                f"agents={len(self.agents)}, tick={self.tick_count})")