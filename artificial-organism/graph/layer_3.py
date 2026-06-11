"""
Layer 3: Neural Knowledge Fabric
==================================
A dynamic knowledge graph — the organism's nervous system.
Every node carries confidence, evidence, usage frequency, and fitness.
"""
from __future__ import annotations
import uuid
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
import networkx as nx


class NodeType(str, Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    TOOL = "tool"
    ACTION = "action"
    OBSERVATION = "observation"
    HYPOTHESIS = "hypothesis"
    THEORY = "theory"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    EVENT = "event"
    RELATION = "relation"


class EdgeType(str, Enum):
    IS_A = "is_a"
    HAS_PART = "has_part"
    CAUSES = "causes"
    ENABLES = "enables"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    USES = "uses"
    SIMILAR_TO = "similar_to"
    PRECEDES = "precedes"
    DERIVED_FROM = "derived_from"


@dataclass
class KnowledgeNode:
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    label: str = ""
    node_type: NodeType = NodeType.CONCEPT
    content: Any = None
    confidence: float = 0.5     # epistemic certainty [0, 1]
    evidence: List[str] = field(default_factory=list)
    source: str = "unknown"
    usage_frequency: int = 0
    fitness: float = 0.5
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = None  # semantic vector

    def access(self) -> None:
        self.usage_frequency += 1
        self.last_accessed = time.time()

    def add_evidence(self, evidence_ref: str, confidence_boost: float = 0.05) -> None:
        self.evidence.append(evidence_ref)
        self.confidence = min(1.0, self.confidence + confidence_boost)

    def decay(self, decay_rate: float = 0.001) -> None:
        """Hebbian-style forgetting — unused nodes lose confidence."""
        age = time.time() - self.last_accessed
        self.confidence = max(0.01, self.confidence * math.exp(-decay_rate * age / 3600))

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type.value,
            "confidence": self.confidence,
            "usage_frequency": self.usage_frequency,
            "fitness": self.fitness,
            "source": self.source,
        }


@dataclass
class KnowledgeEdge:
    edge_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_id: str = ""
    target_id: str = ""
    edge_type: EdgeType = EdgeType.IS_A
    weight: float = 1.0
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)


class NeuralKnowledgeFabric:
    """
    Dynamic knowledge graph with Hebbian learning, spreading activation,
    and structural self-reorganization.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.fabric_id = str(uuid.uuid4())[:8]
        self._activation: Dict[str, float] = {}

    # ─── Node Management ─────────────────────────────────────────────────

    def add_node(self, node: KnowledgeNode) -> str:
        self.nodes[node.node_id] = node
        self.graph.add_node(
            node.node_id,
            label=node.label,
            node_type=node.node_type.value,
            confidence=node.confidence,
        )
        return node.node_id

    def add_concept(self, label: str, content: Any = None,
                    node_type: NodeType = NodeType.CONCEPT,
                    confidence: float = 0.5, source: str = "unknown") -> KnowledgeNode:
        node = KnowledgeNode(label=label, node_type=node_type,
                             content=content, confidence=confidence, source=source)
        self.add_node(node)
        return node

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        node = self.nodes.get(node_id)
        if node:
            node.access()
        return node

    def find_by_label(self, label: str) -> List[KnowledgeNode]:
        return [n for n in self.nodes.values() if label.lower() in n.label.lower()]

    # ─── Edge Management ─────────────────────────────────────────────────

    def add_edge(self, source_id: str, target_id: str,
                 edge_type: EdgeType = EdgeType.IS_A,
                 weight: float = 1.0, confidence: float = 0.5) -> Optional[KnowledgeEdge]:
        if source_id not in self.nodes or target_id not in self.nodes:
            return None
        edge = KnowledgeEdge(source_id=source_id, target_id=target_id,
                              edge_type=edge_type, weight=weight, confidence=confidence)
        self.graph.add_edge(source_id, target_id,
                            edge_type=edge_type.value,
                            weight=weight, confidence=confidence,
                            edge_id=edge.edge_id)
        return edge

    def relate(self, label_a: str, label_b: str, edge_type: EdgeType = EdgeType.IS_A,
               weight: float = 1.0) -> bool:
        """Convenience: relate two nodes by label (creates if missing)."""
        nodes_a = self.find_by_label(label_a)
        nodes_b = self.find_by_label(label_b)
        if not nodes_a:
            a = self.add_concept(label_a)
            nodes_a = [a]
        if not nodes_b:
            b = self.add_concept(label_b)
            nodes_b = [b]
        self.add_edge(nodes_a[0].node_id, nodes_b[0].node_id, edge_type, weight)
        return True

    # ─── Spreading Activation ────────────────────────────────────────────

    def activate(self, seed_ids: List[str], spread_depth: int = 3,
                 decay: float = 0.5) -> Dict[str, float]:
        """
        Spreading activation from seed nodes.
        Returns activation levels across the graph.
        """
        activation: Dict[str, float] = {sid: 1.0 for sid in seed_ids if sid in self.nodes}

        for _ in range(spread_depth):
            new_activation: Dict[str, float] = {}
            for node_id, act_level in activation.items():
                for neighbor in self.graph.successors(node_id):
                    edge_data = self.graph[node_id][neighbor]
                    edge_weight = edge_data.get("weight", 1.0)
                    propagated = act_level * edge_weight * decay
                    new_activation[neighbor] = max(
                        new_activation.get(neighbor, 0.0), propagated
                    )
            # merge
            for nid, val in new_activation.items():
                activation[nid] = max(activation.get(nid, 0.0), val)

        self._activation = activation
        # Update usage on activated nodes
        for nid in activation:
            if nid in self.nodes:
                self.nodes[nid].access()
        return activation

    def most_activated(self, n: int = 10) -> List[Tuple[KnowledgeNode, float]]:
        ranked = sorted(self._activation.items(), key=lambda x: x[1], reverse=True)[:n]
        return [(self.nodes[nid], act) for nid, act in ranked if nid in self.nodes]

    # ─── Structural Analysis ─────────────────────────────────────────────

    def find_path(self, source_label: str, target_label: str) -> List[str]:
        sources = self.find_by_label(source_label)
        targets = self.find_by_label(target_label)
        if not sources or not targets:
            return []
        try:
            path = nx.shortest_path(self.graph, sources[0].node_id, targets[0].node_id)
            return [self.nodes[p].label for p in path if p in self.nodes]
        except nx.NetworkXNoPath:
            return []

    def central_concepts(self, n: int = 10) -> List[KnowledgeNode]:
        """PageRank-based centrality — find the most important concepts."""
        if len(self.graph) == 0:
            return []
        pr = nx.pagerank(self.graph, alpha=0.85)
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:n]
        return [self.nodes[nid] for nid, _ in ranked if nid in self.nodes]

    def clusters(self) -> List[Set[str]]:
        """Find conceptual clusters (weakly connected components)."""
        undirected = self.graph.to_undirected()
        return [c for c in nx.connected_components(undirected)]

    def decay_all(self, rate: float = 0.001) -> None:
        """Apply memory decay to all nodes."""
        for node in self.nodes.values():
            node.decay(rate)

    # ─── Hebbian Learning ────────────────────────────────────────────────

    def hebbian_strengthen(self, node_ids: List[str], boost: float = 0.1) -> None:
        """'Neurons that fire together, wire together.'"""
        for i, a in enumerate(node_ids):
            for b in node_ids[i+1:]:
                if self.graph.has_edge(a, b):
                    self.graph[a][b]["weight"] = min(
                        2.0, self.graph[a][b]["weight"] + boost
                    )
                elif a in self.nodes and b in self.nodes:
                    self.add_edge(a, b, EdgeType.SIMILAR_TO, weight=boost)

    # ─── Introspection ───────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "fabric_id": self.fabric_id,
            "total_nodes": len(self.nodes),
            "total_edges": self.graph.number_of_edges(),
            "node_types": {t.value: sum(1 for n in self.nodes.values()
                           if n.node_type == t) for t in NodeType},
            "avg_confidence": sum(n.confidence for n in self.nodes.values()) / max(1, len(self.nodes)),
            "components": len(self.clusters()),
        }

    def __repr__(self) -> str:
        return (f"NeuralKnowledgeFabric("
                f"nodes={len(self.nodes)}, edges={self.graph.number_of_edges()})")