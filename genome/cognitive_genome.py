import uuid
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

@dataclass
class Gene:
    """
    Represents a single cognitive gene in the organism's genome.
    Every cognitive component is evolvable.
    """
    function: str
    weight: float = 1.0
    mutation_rate: float = 0.01
    fitness: float = 0.5
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def mutate(self) -> 'Gene':
        """Applies mutation to the gene based on its mutation_rate."""
        # Mutation logic: slightly adjust the weight
        new_weight = self.weight + random.uniform(-self.mutation_rate, self.mutation_rate)
        new_weight = max(0.0, min(1.0, new_weight))  # Bound between 0 and 1
        
        # Meta-mutation: slightly alter the mutation rate itself
        new_mutation_rate = self.mutation_rate * random.uniform(0.9, 1.1)
        
        return Gene(
            function=self.function,
            weight=new_weight,
            mutation_rate=new_mutation_rate,
            fitness=self.fitness,  # Inherit current fitness until re-evaluated
            parameters=self.parameters.copy()
        )

class CognitiveGenome:
    """
    The genome stores everything that defines the system.
    It models evolvable traits for reasoning, retrieval, memory, planning, tools, etc.
    """
    def __init__(self, genome_id: Optional[str] = None):
        self.genome_id = genome_id or str(uuid.uuid4())
        self.chromosomes: Dict[str, List[Gene]] = {
            "reasoning_dna": [],
            "retrieval_dna": [],
            "memory_dna": [],
            "planning_dna": [],
            "tool_dna": [],
            "communication_dna": [],
            "learning_dna": []
        }
        self.mutation_rules: Dict[str, Any] = {
            "global_mutation_rate": 0.05,
            "crossover_rate": 0.2
        }

    def initialize_default_genome(self):
        """Populates the genome with a baseline set of genes."""
        self.chromosomes["reasoning_dna"] = [
            Gene(function="deductive_reasoning", weight=0.8, mutation_rate=0.005, fitness=0.9),
            Gene(function="inductive_reasoning", weight=0.6, mutation_rate=0.01, fitness=0.7)
        ]
        self.chromosomes["retrieval_dna"] = [
            Gene(function="dense_vector_search", weight=0.9, mutation_rate=0.002, fitness=0.95),
            Gene(function="keyword_search", weight=0.3, mutation_rate=0.05, fitness=0.6)
        ]
        self.chromosomes["memory_dna"] = [
            Gene(function="episodic_consolidation", weight=0.7, mutation_rate=0.01, fitness=0.8)
        ]
        # Extensible to other dna layers

    def add_gene(self, category: str, gene: Gene):
        """Adds a new gene to a specific chromosome."""
        if category in self.chromosomes:
            self.chromosomes[category].append(gene)
        else:
            raise ValueError(f"Unknown chromosome category: {category}")

    def mutate_genome(self) -> 'CognitiveGenome':
        """
        Creates a mutated offspring of the current genome.
        Applies gene-level mutations based on global and local mutation rules.
        """
        offspring = CognitiveGenome()
        offspring.mutation_rules = self.mutation_rules.copy()
        
        for category, genes in self.chromosomes.items():
            mutated_genes = []
            for gene in genes:
                # Decide whether to mutate this gene based on global mutation rules
                if random.random() < self.mutation_rules["global_mutation_rate"]:
                    mutated_genes.append(gene.mutate())
                else:
                    # Inherit gene without mutation
                    mutated_genes.append(gene)
            offspring.chromosomes[category] = mutated_genes
            
        return offspring

    def evaluate_fitness(self, feedback_signals: Dict[str, float]):
        """
        Updates the fitness of genes based on system performance (feedback signals).
        """
        for category, genes in self.chromosomes.items():
            for gene in genes:
                # Look for specific feedback matching the gene's function
                if gene.function in feedback_signals:
                    # Moving average for fitness update
                    learning_rate = 0.1
                    gene.fitness = (1 - learning_rate) * gene.fitness + learning_rate * feedback_signals[gene.function]

    def serialize(self) -> Dict[str, Any]:
        """Serializes the genome for storage, observability, or transmission."""
        return {
            "genome_id": self.genome_id,
            "mutation_rules": self.mutation_rules,
            "chromosomes": {
                cat: [{"function": g.function, "weight": g.weight, "mutation_rate": g.mutation_rate, "fitness": g.fitness, "parameters": g.parameters} for g in genes]
                for cat, genes in self.chromosomes.items()
            }
        }

if __name__ == "__main__":
    # Example Usage:
    parent_genome = CognitiveGenome()
    parent_genome.initialize_default_genome()
    
    print(f"Parent Genome ID: {parent_genome.genome_id}")
    print("Parent Reasoning Weight:", parent_genome.chromosomes["reasoning_dna"][0].weight)
    
    # Mutate to create an offspring
    child_genome = parent_genome.mutate_genome()
    print(f"Child Genome ID: {child_genome.genome_id}")
    print("Child Reasoning Weight:", child_genome.chromosomes["reasoning_dna"][0].weight)
