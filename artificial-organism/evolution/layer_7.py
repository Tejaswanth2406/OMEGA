"""
Layer 7: Multi-Level Evolution Engine
=======================================
Evolution at every scale:
  parameter → prompt → tool → memory → workflow → architecture → agent
"""
from __future__ import annotations
import uuid
import random
import time
import copy
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum


class EvolutionLevel(str, Enum):
    PARAMETER = "parameter"
    PROMPT = "prompt"
    TOOL = "tool"
    MEMORY = "memory"
    WORKFLOW = "workflow"
    ARCHITECTURE = "architecture"
    AGENT = "agent"


@dataclass
class Individual:
    """A single candidate solution in an evolutionary population."""
    ind_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: EvolutionLevel = EvolutionLevel.PARAMETER
    genotype: Dict[str, Any] = field(default_factory=dict)
    phenotype: Optional[Any] = None
    fitness: float = 0.0
    age: int = 0
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    metadata: Dict = field(default_factory=dict)

    def clone(self) -> "Individual":
        child = copy.deepcopy(self)
        child.ind_id = str(uuid.uuid4())[:8]
        child.parent_ids = [self.ind_id]
        child.fitness = 0.0
        child.age = 0
        return child


@dataclass
class Population:
    """A pool of individuals at a given evolution level."""
    pop_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: EvolutionLevel = EvolutionLevel.PARAMETER
    individuals: List[Individual] = field(default_factory=list)
    generation: int = 0
    history: List[Dict] = field(default_factory=list)   # fitness snapshots

    def add(self, ind: Individual) -> None:
        self.individuals.append(ind)

    def best(self, n: int = 1) -> List[Individual]:
        return sorted(self.individuals, key=lambda i: i.fitness, reverse=True)[:n]

    def worst(self, n: int = 1) -> List[Individual]:
        return sorted(self.individuals, key=lambda i: i.fitness)[:n]

    def mean_fitness(self) -> float:
        if not self.individuals:
            return 0.0
        return statistics.mean(i.fitness for i in self.individuals)

    def diversity(self) -> float:
        """Genotypic diversity — std of fitness as proxy."""
        if len(self.individuals) < 2:
            return 0.0
        return statistics.stdev(i.fitness for i in self.individuals)

    def snapshot(self) -> Dict:
        snap = {
            "generation": self.generation,
            "size": len(self.individuals),
            "best_fitness": self.best(1)[0].fitness if self.individuals else 0.0,
            "mean_fitness": self.mean_fitness(),
            "diversity": self.diversity(),
            "timestamp": time.time(),
        }
        self.history.append(snap)
        return snap


class MutationStrategy:
    """Collection of mutation operators."""

    @staticmethod
    def gaussian_perturb(value: float, sigma: float = 0.05) -> float:
        import random
        return max(0.0, min(1.0, value + random.gauss(0, sigma)))

    @staticmethod
    def swap_keys(d: Dict) -> Dict:
        keys = list(d.keys())
        if len(keys) < 2:
            return d
        k1, k2 = random.sample(keys, 2)
        d = dict(d)
        d[k1], d[k2] = d[k2], d[k1]
        return d

    @staticmethod
    def random_reset(d: Dict, rate: float = 0.1) -> Dict:
        d = dict(d)
        for k in d:
            if random.random() < rate and isinstance(d[k], float):
                d[k] = random.random()
        return d

    @staticmethod
    def insert_delete(lst: List, item_factory: Callable, rate: float = 0.1) -> List:
        lst = list(lst)
        if random.random() < rate and lst:
            lst.pop(random.randrange(len(lst)))
        if random.random() < rate:
            lst.append(item_factory())
        return lst


class SelectionStrategy:
    """Selection operators for evolutionary algorithms."""

    @staticmethod
    def tournament(population: List[Individual], k: int = 3) -> Individual:
        candidates = random.sample(population, min(k, len(population)))
        return max(candidates, key=lambda i: i.fitness)

    @staticmethod
    def roulette(population: List[Individual]) -> Individual:
        total = sum(max(0, i.fitness) for i in population)
        if total == 0:
            return random.choice(population)
        pick = random.uniform(0, total)
        current = 0.0
        for ind in population:
            current += max(0, ind.fitness)
            if current >= pick:
                return ind
        return population[-1]

    @staticmethod
    def rank_selection(population: List[Individual]) -> Individual:
        ranked = sorted(population, key=lambda i: i.fitness)
        weights = [i + 1 for i in range(len(ranked))]
        total = sum(weights)
        pick = random.uniform(0, total)
        current = 0.0
        for ind, w in zip(ranked, weights):
            current += w
            if current >= pick:
                return ind
        return ranked[-1]


class EvolutionEngine:
    """
    Multi-level evolutionary system.
    Each level evolves independently with shared fitness signals.
    """

    def __init__(self,
                 population_size: int = 50,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.7,
                 elitism: float = 0.1):
        self.engine_id = str(uuid.uuid4())[:8]
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        self.populations: Dict[str, Population] = {}
        self.global_generation = 0
        self.fitness_evaluators: Dict[str, Callable] = {}
        self.evolution_log: List[Dict] = []

        # Initialize populations for each level
        for level in EvolutionLevel:
            pop = Population(level=level)
            self._seed_population(pop)
            self.populations[level.value] = pop

    # ─── Population Seeding ──────────────────────────────────────────────

    def _seed_population(self, pop: Population) -> None:
        for _ in range(self.population_size):
            ind = Individual(
                level=pop.level,
                generation=0,
                genotype=self._default_genotype(pop.level),
            )
            pop.add(ind)

    def _default_genotype(self, level: EvolutionLevel) -> Dict:
        if level == EvolutionLevel.PARAMETER:
            return {
                "temperature": random.uniform(0.1, 1.5),
                "top_p": random.uniform(0.7, 1.0),
                "max_tokens": random.choice([256, 512, 1024, 2048]),
                "reasoning_depth": random.randint(1, 5),
            }
        elif level == EvolutionLevel.PROMPT:
            return {
                "system_prefix": random.choice(["You are", "Act as", "Simulate"]),
                "reasoning_style": random.choice(["step_by_step", "direct", "socratic"]),
                "output_format": random.choice(["json", "markdown", "plain"]),
                "few_shot_count": random.randint(0, 5),
            }
        elif level == EvolutionLevel.WORKFLOW:
            return {
                "steps": ["observe", "reason", "act", "reflect"],
                "parallel": random.random() > 0.5,
                "retry_limit": random.randint(1, 5),
                "confidence_threshold": random.uniform(0.5, 0.95),
            }
        elif level == EvolutionLevel.ARCHITECTURE:
            return {
                "memory_tiers": random.randint(3, 7),
                "reasoning_layers": random.randint(1, 4),
                "attention_heads": random.choice([4, 8, 16]),
                "use_graph_memory": random.random() > 0.5,
                "use_immune_system": random.random() > 0.3,
            }
        else:
            return {"fitness_score": random.random()}

    # ─── Fitness Registration ────────────────────────────────────────────

    def register_evaluator(self, level: str, fn: Callable[[Individual], float]) -> None:
        """Register a fitness function for a given evolution level."""
        self.fitness_evaluators[level] = fn

    def evaluate_population(self, level: str) -> None:
        pop = self.populations.get(level)
        if not pop:
            return
        evaluator = self.fitness_evaluators.get(level)
        for ind in pop.individuals:
            if evaluator:
                ind.fitness = evaluator(ind)
            else:
                ind.fitness = random.random()  # stub

    # ─── Core Evolution Operators ────────────────────────────────────────

    def _mutate(self, ind: Individual) -> Individual:
        child = ind.clone()
        g = child.genotype

        for key, val in g.items():
            if random.random() < self.mutation_rate:
                if isinstance(val, float):
                    g[key] = MutationStrategy.gaussian_perturb(val)
                elif isinstance(val, int):
                    g[key] = max(1, val + random.choice([-1, 0, 1]))
                elif isinstance(val, bool):
                    g[key] = not val
                elif isinstance(val, str):
                    pass  # domain-specific string mutation would go here
                elif isinstance(val, list):
                    g[key] = MutationStrategy.insert_delete(val, lambda: "new_step")

        child.generation = ind.generation + 1
        return child

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        child = parent1.clone()
        child.parent_ids = [parent1.ind_id, parent2.ind_id]

        for key in child.genotype:
            if key in parent2.genotype and random.random() < 0.5:
                child.genotype[key] = parent2.genotype[key]

        return child

    # ─── Generation Step ─────────────────────────────────────────────────

    def step(self, level: str) -> Dict:
        """Execute one generation of evolution for a given level."""
        pop = self.populations.get(level)
        if not pop or not pop.individuals:
            return {}

        self.evaluate_population(level)
        snap_before = pop.snapshot()

        n_elites = max(1, int(len(pop.individuals) * self.elitism))
        elites = pop.best(n_elites)

        new_individuals = list(elites)

        while len(new_individuals) < self.population_size:
            if random.random() < self.crossover_rate and len(pop.individuals) >= 2:
                p1 = SelectionStrategy.tournament(pop.individuals)
                p2 = SelectionStrategy.tournament(pop.individuals)
                child = self._crossover(p1, p2)
            else:
                parent = SelectionStrategy.tournament(pop.individuals)
                child = self._mutate(parent)
            new_individuals.append(child)

        pop.individuals = new_individuals
        pop.generation += 1

        snap_after = pop.snapshot()
        log_entry = {
            "level": level,
            "generation": pop.generation,
            "fitness_before": snap_before["mean_fitness"],
            "fitness_after": snap_after["mean_fitness"],
            "best_fitness": snap_after["best_fitness"],
            "diversity": snap_after["diversity"],
            "timestamp": time.time(),
        }
        self.evolution_log.append(log_entry)
        return log_entry

    def step_all(self) -> Dict[str, Dict]:
        """Evolve all levels simultaneously."""
        self.global_generation += 1
        return {level: self.step(level) for level in self.populations}

    # ─── Multi-Objective Pareto ──────────────────────────────────────────

    def pareto_front(self, level: str, objectives: List[str]) -> List[Individual]:
        """
        Find the Pareto-optimal front for multi-objective optimization.
        objectives: list of genotype keys to maximize.
        """
        pop = self.populations.get(level)
        if not pop:
            return []

        def dominates(a: Individual, b: Individual) -> bool:
            a_vals = [a.genotype.get(o, 0) for o in objectives]
            b_vals = [b.genotype.get(o, 0) for o in objectives]
            return all(av >= bv for av, bv in zip(a_vals, b_vals)) and \
                   any(av > bv for av, bv in zip(a_vals, b_vals))

        front = []
        for ind in pop.individuals:
            dominated = any(dominates(other, ind) for other in pop.individuals if other is not ind)
            if not dominated:
                front.append(ind)
        return front

    # ─── Stats & Reporting ───────────────────────────────────────────────

    def global_stats(self) -> Dict:
        stats = {}
        for level, pop in self.populations.items():
            stats[level] = {
                "generation": pop.generation,
                "size": len(pop.individuals),
                "mean_fitness": pop.mean_fitness(),
                "best_fitness": pop.best(1)[0].fitness if pop.individuals else 0.0,
                "diversity": pop.diversity(),
            }
        return {
            "engine_id": self.engine_id,
            "global_generation": self.global_generation,
            "levels": stats,
        }

    def fitness_trend(self, level: str) -> List[float]:
        pop = self.populations.get(level)
        if not pop:
            return []
        return [h["mean_fitness"] for h in pop.history]

    def __repr__(self) -> str:
        return (f"EvolutionEngine(id={self.engine_id}, "
                f"gen={self.global_generation}, "
                f"levels={len(self.populations)})")