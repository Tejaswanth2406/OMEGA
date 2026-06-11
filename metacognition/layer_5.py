"""
Layer 5: Meta-Cognition — Machine Self-Model
=============================================
The organism tracks what it is, what it knows, what it doesn't know,
how reliable it is, and what recently changed.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class KnowledgeStatus(str, Enum):
    KNOWN = "known"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    CONTRADICTED = "contradicted"


@dataclass
class SelfBelief:
    """A belief the organism holds about itself."""
    belief_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    subject: str = ""           # "my reasoning", "my memory", etc.
    predicate: str = ""         # "is reliable", "has gaps in", etc.
    confidence: float = 0.5
    evidence: List[str] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)

    def update(self, new_confidence: float, evidence_ref: str = "") -> None:
        alpha = 0.15
        self.confidence = (1 - alpha) * self.confidence + alpha * new_confidence
        if evidence_ref:
            self.evidence.append(evidence_ref)
        self.last_updated = time.time()


@dataclass
class CapabilityMap:
    """Maps known capabilities and their current reliability."""
    capabilities: Dict[str, float] = field(default_factory=dict)  # name -> reliability

    def register(self, name: str, reliability: float = 0.5) -> None:
        self.capabilities[name] = reliability

    def update_reliability(self, name: str, score: float, alpha: float = 0.1) -> None:
        current = self.capabilities.get(name, 0.5)
        self.capabilities[name] = (1 - alpha) * current + alpha * score

    def weakest(self, n: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.capabilities.items(), key=lambda x: x[1])[:n]

    def strongest(self, n: int = 3) -> List[Tuple[str, float]]:
        return sorted(self.capabilities.items(), key=lambda x: x[1], reverse=True)[:n]


@dataclass
class KnowledgeGapRecord:
    """Tracks what the organism knows it does not know."""
    gap_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    domain: str = ""
    description: str = ""
    severity: float = 0.5       # how much this gap hurts performance
    detected_at: float = field(default_factory=time.time)
    resolution_attempts: int = 0


class MetaCognitionEngine:
    """
    The organism's self-model.

    Tracks:
    - What am I?            → identity_model
    - What do I know?       → knowledge_map
    - What do I NOT know?   → knowledge_gaps
    - What tools exist?     → tool_registry
    - How reliable am I?    → reliability_history
    - What changed recently?→ change_log
    """

    def __init__(self):
        self.engine_id = str(uuid.uuid4())[:8]

        # Core self-model components
        self.identity_model: Dict[str, Any] = {
            "name": "ArtificialOrganism",
            "version": "1.0.0",
            "purpose": "Self-evolving cognitive architecture",
            "generation": 0,
        }
        self.beliefs: Dict[str, SelfBelief] = {}
        self.capability_map = CapabilityMap()
        self.knowledge_gaps: List[KnowledgeGapRecord] = []
        self.tool_registry: Dict[str, Dict] = {}
        self.reliability_history: List[Dict] = []
        self.change_log: List[Dict] = []

        # Global reliability estimate
        self.global_reliability: float = 0.5

        self._bootstrap_beliefs()
        self._bootstrap_capabilities()

    # ─── Initialization ──────────────────────────────────────────────────

    def _bootstrap_beliefs(self) -> None:
        defaults = [
            ("reasoning", "is capable of", 0.6),
            ("memory", "may have gaps in", 0.5),
            ("planning", "is moderately reliable for", 0.55),
            ("self-knowledge", "is incomplete about", 0.4),
        ]
        for subject, predicate, conf in defaults:
            self.add_belief(subject, predicate, conf)

    def _bootstrap_capabilities(self) -> None:
        caps = ["reasoning", "retrieval", "planning", "tool_use",
                "reflection", "learning", "communication", "self_repair"]
        for cap in caps:
            self.capability_map.register(cap, reliability=0.5)

    # ─── Belief Management ───────────────────────────────────────────────

    def add_belief(self, subject: str, predicate: str, confidence: float = 0.5) -> SelfBelief:
        belief = SelfBelief(subject=subject, predicate=predicate, confidence=confidence)
        self.beliefs[belief.belief_id] = belief
        return belief

    def update_belief(self, belief_id: str, new_confidence: float, evidence: str = "") -> None:
        if belief_id in self.beliefs:
            self.beliefs[belief_id].update(new_confidence, evidence)

    def get_beliefs_about(self, subject: str) -> List[SelfBelief]:
        return [b for b in self.beliefs.values()
                if subject.lower() in b.subject.lower()]

    # ─── Knowledge Gap Tracking ──────────────────────────────────────────

    def detect_gap(self, domain: str, description: str, severity: float = 0.5) -> KnowledgeGapRecord:
        gap = KnowledgeGapRecord(domain=domain, description=description, severity=severity)
        self.knowledge_gaps.append(gap)
        self._log_change("gap_detected", {"domain": domain, "severity": severity})
        return gap

    def resolve_gap(self, gap_id: str) -> bool:
        for i, gap in enumerate(self.knowledge_gaps):
            if gap.gap_id == gap_id:
                self.knowledge_gaps.pop(i)
                self._log_change("gap_resolved", {"gap_id": gap_id})
                return True
        return False

    def critical_gaps(self) -> List[KnowledgeGapRecord]:
        return sorted(self.knowledge_gaps, key=lambda g: g.severity, reverse=True)[:5]

    # ─── Tool Registry ───────────────────────────────────────────────────

    def register_tool(self, name: str, description: str,
                      cost: float = 1.0, reliability: float = 0.8) -> None:
        self.tool_registry[name] = {
            "description": description,
            "cost": cost,
            "reliability": reliability,
            "usage_count": 0,
            "registered_at": time.time(),
        }

    def use_tool(self, name: str, success: bool) -> None:
        if name in self.tool_registry:
            self.tool_registry[name]["usage_count"] += 1
            rel = self.tool_registry[name]["reliability"]
            alpha = 0.05
            self.tool_registry[name]["reliability"] = (
                (1 - alpha) * rel + alpha * (1.0 if success else 0.0)
            )

    # ─── Reliability Tracking ────────────────────────────────────────────

    def record_performance(self, task: str, score: float, domain: str = "general") -> None:
        record = {
            "task": task,
            "score": score,
            "domain": domain,
            "timestamp": time.time(),
        }
        self.reliability_history.append(record)
        self.capability_map.update_reliability(domain, score)
        self._update_global_reliability(score)

        if score < 0.4:
            self.detect_gap(domain, f"Low performance ({score:.2f}) on task: {task}", severity=1 - score)

    def _update_global_reliability(self, score: float) -> None:
        self.global_reliability = 0.95 * self.global_reliability + 0.05 * score

    def reliability_trend(self, window: int = 20) -> float:
        """Slope of reliability over recent window (positive = improving)."""
        recent = self.reliability_history[-window:]
        if len(recent) < 2:
            return 0.0
        scores = [r["score"] for r in recent]
        n = len(scores)
        x_mean = (n - 1) / 2
        y_mean = sum(scores) / n
        numerator = sum((i - x_mean) * (scores[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator else 0.0

    # ─── Change Log ──────────────────────────────────────────────────────

    def _log_change(self, event_type: str, data: Dict) -> None:
        self.change_log.append({
            "event_id": str(uuid.uuid4())[:8],
            "event_type": event_type,
            "data": data,
            "timestamp": time.time(),
        })

    def recent_changes(self, n: int = 10) -> List[Dict]:
        return self.change_log[-n:]

    # ─── Introspective Report ────────────────────────────────────────────

    def introspect(self) -> Dict:
        return {
            "identity": self.identity_model,
            "global_reliability": round(self.global_reliability, 3),
            "reliability_trend": round(self.reliability_trend(), 4),
            "capabilities": self.capability_map.capabilities,
            "weakest_capabilities": self.capability_map.weakest(3),
            "strongest_capabilities": self.capability_map.strongest(3),
            "critical_gaps": [
                {"domain": g.domain, "severity": g.severity, "description": g.description}
                for g in self.critical_gaps()
            ],
            "known_tools": list(self.tool_registry.keys()),
            "total_beliefs": len(self.beliefs),
            "recent_changes": self.recent_changes(5),
        }

    def what_am_i(self) -> str:
        return (
            f"I am {self.identity_model['name']} v{self.identity_model['version']}, "
            f"generation {self.identity_model['generation']}. "
            f"Global reliability: {self.global_reliability:.2%}. "
            f"I have {len(self.capability_map.capabilities)} known capabilities, "
            f"{len(self.knowledge_gaps)} knowledge gaps, "
            f"and {len(self.tool_registry)} registered tools."
        )

    def __repr__(self) -> str:
        return f"MetaCognitionEngine(reliability={self.global_reliability:.3f}, gaps={len(self.knowledge_gaps)})"