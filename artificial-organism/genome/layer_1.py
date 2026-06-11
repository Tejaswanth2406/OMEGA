"""
Layer 1: Cognitive Genome
=========================
Every cognitive component is encoded as evolvable DNA.
Genes define function, weight, mutation rate, and fitness.
"""
from __future__ import annotations
import uuid
import random
import json
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class GeneFunction(str, Enum):
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    MEMORY = "memory"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    REFLECTION = "reflection"
    IMMUNE = "immune"
    ONTOLOGY = "ontology"


@dataclass
class Gene:
    """A single cognitive gene — the atomic unit of cognitive DNA."""
    gene_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    function: GeneFunction = GeneFunction.REASONING
    weight: float = 0.5           # influence strength [0, 1]
    mutation_rate: float = 0.001  # probability of mutation per generation
    fitness: float = 0.5          # measured performance [0, 1]
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)  # ancestor gene IDs

    def mutate(self) -> "Gene":
        """Produce a mutated copy of this gene."""
        child = copy.deepcopy(self)
        child.gene_id = str(uuid.uuid4())[:8]
        child.lineage = self.lineage + [self.gene_id]

        if random.random() < self.mutation_rate:
            child.weight = max(0.0, min(1.0, self.weight + random.gauss(0, 0.05)))

        if random.random() < self.mutation_rate * 0.1:
            child.mutation_rate = max(1e-6, min(0.1, self.mutation_rate * random.uniform(0.5, 2.0)))

        return child

    def crossover(self, other: "Gene") -> "Gene":
        """Produce offspring by crossing two genes."""
        child = copy.deepcopy(self)
        child.gene_id = str(uuid.uuid4())[:8]
        child.lineage = [self.gene_id, other.gene_id]
        child.weight = (self.weight + other.weight) / 2 + random.gauss(0, 0.02)
        child.weight = max(0.0, min(1.0, child.weight))
        child.mutation_rate = (self.mutation_rate + other.mutation_rate) / 2
        child.fitness = 0.0  # reset — must be re-evaluated
        return child

    def update_fitness(self, score: float, alpha: float = 0.1) -> None:
        """Exponential moving average fitness update."""
        self.fitness = (1 - alpha) * self.fitness + alpha * score

    def to_dict(self) -> Dict:
        return {
            "gene_id": self.gene_id,
            "function": self.function.value,
            "weight": self.weight,
            "mutation_rate": self.mutation_rate,
            "fitness": self.fitness,
            "active": self.active,
            "metadata": self.metadata,
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Gene":
        d = d.copy()
        d["function"] = GeneFunction(d["function"])
        return cls(**d)


@dataclass
class Chromosome:
    """A collection of genes for a specific cognitive domain."""
    domain: str
    genes: List[Gene] = field(default_factory=list)
    generation: int = 0

    def add_gene(self, gene: Gene) -> None:
        self.genes.append(gene)

    def get_active_genes(self) -> List[Gene]:
        return [g for g in self.genes if g.active]

    def total_weight(self) -> float:
        return sum(g.weight for g in self.get_active_genes())

    def fittest_gene(self) -> Optional[Gene]:
        active = self.get_active_genes()
        return max(active, key=lambda g: g.fitness) if active else None

    def evolve(self, selection_pressure: float = 0.5) -> "Chromosome":
        """Selection + mutation to produce next generation."""
        active = self.get_active_genes()
        if not active:
            return self

        # Tournament selection — keep top half
        sorted_genes = sorted(active, key=lambda g: g.fitness, reverse=True)
        survivors = sorted_genes[:max(1, int(len(sorted_genes) * selection_pressure))]

        new_genes = list(survivors)  # elites survive

        # Fill back to original size via mutation and crossover
        while len(new_genes) < len(self.genes):
            if len(survivors) >= 2 and random.random() < 0.3:
                p1, p2 = random.sample(survivors, 2)
                new_genes.append(p1.crossover(p2))
            else:
                parent = random.choice(survivors)
                new_genes.append(parent.mutate())

        child_chrom = Chromosome(domain=self.domain, genes=new_genes, generation=self.generation + 1)
        return child_chrom

    def to_dict(self) -> Dict:
        return {
            "domain": self.domain,
            "generation": self.generation,
            "genes": [g.to_dict() for g in self.genes],
        }


class CognitiveGenome:
    """
    The complete cognitive genome — the blueprint for the entire organism.
    All layers are defined through this genome.
    """
    def __init__(self):
        self.genome_id = str(uuid.uuid4())
        self.chromosomes: Dict[str, Chromosome] = {}
        self.generation = 0
        self._initialize_default_genome()

    def _initialize_default_genome(self) -> None:
        """Bootstrap a default genome covering all cognitive functions."""
        for func in GeneFunction:
            chrom = Chromosome(domain=func.value)
            # Start each domain with 5 diverse genes
            for i in range(5):
                gene = Gene(
                    function=func,
                    weight=random.uniform(0.3, 0.9),
                    mutation_rate=random.uniform(0.0005, 0.005),
                    fitness=0.5,
                    metadata={"init_rank": i},
                )
                chrom.add_gene(gene)
            self.chromosomes[func.value] = chrom

    def get_chromosome(self, domain: str) -> Optional[Chromosome]:
        return self.chromosomes.get(domain)

    def express(self, domain: str) -> List[Gene]:
        """Gene expression — return active genes for a domain."""
        chrom = self.chromosomes.get(domain)
        return chrom.get_active_genes() if chrom else []

    def mutate_domain(self, domain: str, selection_pressure: float = 0.5) -> None:
        """Evolve a single chromosome."""
        if domain in self.chromosomes:
            self.chromosomes[domain] = self.chromosomes[domain].evolve(selection_pressure)

    def global_fitness(self) -> float:
        """Mean fitness across all domains."""
        all_genes = [g for c in self.chromosomes.values() for g in c.get_active_genes()]
        return sum(g.fitness for g in all_genes) / len(all_genes) if all_genes else 0.0

    def save(self, path: str) -> None:
        data = {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "chromosomes": {k: v.to_dict() for k, v in self.chromosomes.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "CognitiveGenome":
        with open(path) as f:
            data = json.load(f)
        genome = cls.__new__(cls)
        genome.genome_id = data["genome_id"]
        genome.generation = data["generation"]
        genome.chromosomes = {}
        for k, v in data["chromosomes"].items():
            chrom = Chromosome(domain=v["domain"], generation=v["generation"])
            chrom.genes = [Gene.from_dict(g) for g in v["genes"]]
            genome.chromosomes[k] = chrom
        return genome

    def __repr__(self) -> str:
        return (
            f"CognitiveGenome(id={self.genome_id[:8]}, "
            f"gen={self.generation}, "
            f"domains={list(self.chromosomes.keys())}, "
            f"fitness={self.global_fitness():.3f})"
        )