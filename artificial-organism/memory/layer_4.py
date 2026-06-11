"""
Layer 4: Multi-Scale Memory
============================
Seven-tier biological memory hierarchy:
  sensory → working → episodic → semantic → procedural → collective → evolutionary
"""
from __future__ import annotations
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── Memory Record ────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: Any = None
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5       # [0, 1] — affects retention
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    source: str = "unknown"

    def access(self) -> Any:
        self.access_count += 1
        return self.content

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


# ─── Base Memory Tier ─────────────────────────────────────────────────────────

class MemoryTier(ABC):
    tier_name: str = "base"

    @abstractmethod
    def store(self, record: MemoryRecord) -> bool: ...

    @abstractmethod
    def retrieve(self, query: str, n: int = 5) -> List[MemoryRecord]: ...

    @abstractmethod
    def forget(self) -> int: ...  # returns count of forgotten items

    @abstractmethod
    def stats(self) -> Dict: ...


# ─── Tier 1: Sensory Memory ───────────────────────────────────────────────────

class SensoryMemory(MemoryTier):
    """
    Ultra-short-term buffer — current inputs only.
    Capacity: small. Decay: immediate (seconds).
    """
    tier_name = "sensory"

    def __init__(self, capacity: int = 16, ttl_seconds: float = 5.0):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._buffer: deque = deque(maxlen=capacity)

    def store(self, record: MemoryRecord) -> bool:
        self._buffer.append(record)
        return True

    def retrieve(self, query: str = "", n: int = 5) -> List[MemoryRecord]:
        self._evict_expired()
        return list(self._buffer)[-n:]

    def forget(self) -> int:
        return self._evict_expired()

    def _evict_expired(self) -> int:
        now = time.time()
        before = len(self._buffer)
        self._buffer = deque(
            (r for r in self._buffer if now - r.timestamp < self.ttl),
            maxlen=self.capacity
        )
        return before - len(self._buffer)

    def stats(self) -> Dict:
        return {"tier": self.tier_name, "size": len(self._buffer), "capacity": self.capacity}


# ─── Tier 2: Working Memory ───────────────────────────────────────────────────

class WorkingMemory(MemoryTier):
    """
    Active reasoning context — Miller's 7±2 slots.
    Items compete for slots by importance.
    """
    tier_name = "working"

    def __init__(self, slots: int = 9):
        self.slots = slots
        self._store: Dict[str, MemoryRecord] = {}

    def store(self, record: MemoryRecord) -> bool:
        if len(self._store) >= self.slots:
            # Evict least important
            least = min(self._store.values(), key=lambda r: r.importance)
            del self._store[least.record_id]
        self._store[record.record_id] = record
        return True

    def retrieve(self, query: str = "", n: int = 9) -> List[MemoryRecord]:
        records = sorted(self._store.values(), key=lambda r: r.importance, reverse=True)
        return records[:n]

    def clear(self) -> None:
        self._store.clear()

    def forget(self) -> int:
        before = len(self._store)
        self._store = {k: v for k, v in self._store.items() if v.importance > 0.2}
        return before - len(self._store)

    def stats(self) -> Dict:
        return {"tier": self.tier_name, "used_slots": len(self._store), "total_slots": self.slots}


# ─── Tier 3: Episodic Memory ──────────────────────────────────────────────────

class EpisodicMemory(MemoryTier):
    """
    Time-indexed autobiographical experiences.
    Supports temporal retrieval and consolidation to semantic.
    """
    tier_name = "episodic"

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self._episodes: List[MemoryRecord] = []

    def store(self, record: MemoryRecord) -> bool:
        self._episodes.append(record)
        if len(self._episodes) > self.capacity:
            self._episodes = sorted(self._episodes, key=lambda r: r.importance, reverse=True)
            self._episodes = self._episodes[:int(self.capacity * 0.9)]
        return True

    def retrieve(self, query: str = "", n: int = 10,
                 after: Optional[float] = None) -> List[MemoryRecord]:
        results = self._episodes
        if after:
            results = [r for r in results if r.timestamp > after]
        # Simple keyword match — replace with vector search in production
        if query:
            results = [r for r in results
                       if query.lower() in str(r.content).lower() or
                       any(query.lower() in t for t in r.tags)]
        return sorted(results, key=lambda r: r.importance, reverse=True)[:n]

    def forget(self) -> int:
        before = len(self._episodes)
        threshold = time.time() - 86400 * 7  # 1 week
        self._episodes = [r for r in self._episodes
                          if r.timestamp > threshold or r.importance > 0.8]
        return before - len(self._episodes)

    def recent(self, n: int = 20) -> List[MemoryRecord]:
        return sorted(self._episodes, key=lambda r: r.timestamp, reverse=True)[:n]

    def stats(self) -> Dict:
        return {"tier": self.tier_name, "episodes": len(self._episodes)}


# ─── Tier 4: Semantic Memory ──────────────────────────────────────────────────

class SemanticMemory(MemoryTier):
    """
    Abstract factual knowledge — concepts, facts, relationships.
    """
    tier_name = "semantic"

    def __init__(self):
        self._facts: Dict[str, MemoryRecord] = {}

    def store(self, record: MemoryRecord) -> bool:
        key = str(record.content)[:128]
        self._facts[key] = record
        return True

    def retrieve(self, query: str = "", n: int = 10) -> List[MemoryRecord]:
        if not query:
            return list(self._facts.values())[:n]
        results = [r for r in self._facts.values()
                   if query.lower() in str(r.content).lower() or
                   any(query.lower() in t for t in r.tags)]
        return sorted(results, key=lambda r: r.importance, reverse=True)[:n]

    def forget(self) -> int:
        return 0  # semantic memory is persistent

    def stats(self) -> Dict:
        return {"tier": self.tier_name, "facts": len(self._facts)}


# ─── Tier 5: Procedural Memory ───────────────────────────────────────────────

class ProceduralMemory(MemoryTier):
    """
    Skills, workflows, and how-to patterns.
    """
    tier_name = "procedural"

    def __init__(self):
        self._procedures: Dict[str, MemoryRecord] = {}

    def register_skill(self, name: str, steps: List, importance: float = 0.7) -> None:
        record = MemoryRecord(content={"name": name, "steps": steps},
                              importance=importance, tags=["skill", name])
        self._procedures[name] = record

    def get_skill(self, name: str) -> Optional[List]:
        record = self._procedures.get(name)
        if record:
            record.access()
            return record.content.get("steps")
        return None

    def store(self, record: MemoryRecord) -> bool:
        name = record.tags[0] if record.tags else record.record_id
        self._procedures[name] = record
        return True

    def retrieve(self, query: str = "", n: int = 10) -> List[MemoryRecord]:
        results = [r for r in self._procedures.values()
                   if query.lower() in str(r.content).lower()]
        return results[:n]

    def forget(self) -> int:
        return 0  # skills are retained

    def stats(self) -> Dict:
        return {"tier": self.tier_name, "procedures": len(self._procedures)}


# ─── Tier 6: Collective Memory ────────────────────────────────────────────────

class CollectiveMemory(MemoryTier):
    """
    Shared across agent population — inter-agent knowledge pool.
    """
    tier_name = "collective"

    def __init__(self):
        self._pool: List[MemoryRecord] = []
        self._contributors: Dict[str, int] = {}

    def store(self, record: MemoryRecord) -> bool:
        self._pool.append(record)
        self._contributors[record.source] = self._contributors.get(record.source, 0) + 1
        return True

    def retrieve(self, query: str = "", n: int = 10) -> List[MemoryRecord]:
        if not query:
            return sorted(self._pool, key=lambda r: r.importance, reverse=True)[:n]
        return [r for r in self._pool
                if query.lower() in str(r.content).lower()][:n]

    def forget(self) -> int:
        before = len(self._pool)
        self._pool = [r for r in self._pool if r.importance > 0.3]
        return before - len(self._pool)

    def stats(self) -> Dict:
        return {"tier": self.tier_name, "records": len(self._pool),
                "contributors": len(self._contributors)}


# ─── Tier 7: Evolutionary Memory ─────────────────────────────────────────────

class EvolutionaryMemory(MemoryTier):
    """
    Long-term record of successful adaptations — the organism's ancestral wisdom.
    """
    tier_name = "evolutionary"

    def __init__(self):
        self._adaptations: List[MemoryRecord] = []

    def record_adaptation(self, description: str, fitness_gain: float,
                          genome_snapshot: Optional[Dict] = None) -> None:
        record = MemoryRecord(
            content={
                "description": description,
                "fitness_gain": fitness_gain,
                "genome_snapshot": genome_snapshot,
            },
            importance=min(1.0, fitness_gain),
            tags=["adaptation"],
        )
        self._adaptations.append(record)

    def best_adaptations(self, n: int = 5) -> List[MemoryRecord]:
        return sorted(self._adaptations, key=lambda r: r.importance, reverse=True)[:n]

    def store(self, record: MemoryRecord) -> bool:
        self._adaptations.append(record)
        return True

    def retrieve(self, query: str = "", n: int = 5) -> List[MemoryRecord]:
        return self.best_adaptations(n)

    def forget(self) -> int:
        return 0  # evolutionary memory is permanent

    def stats(self) -> Dict:
        return {"tier": self.tier_name, "adaptations": len(self._adaptations)}


# ─── Unified Memory System ────────────────────────────────────────────────────

class MultiScaleMemory:
    """
    The complete 7-tier memory hierarchy.
    Handles automatic consolidation between tiers.
    """

    def __init__(self):
        self.sensory = SensoryMemory()
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.procedural = ProceduralMemory()
        self.collective = CollectiveMemory()
        self.evolutionary = EvolutionaryMemory()

        self._tiers: List[MemoryTier] = [
            self.sensory, self.working, self.episodic,
            self.semantic, self.procedural, self.collective, self.evolutionary
        ]

    def perceive(self, content: Any, tags: Optional[List[str]] = None,
                 importance: float = 0.5) -> MemoryRecord:
        """Store a new perception — enters at sensory level."""
        record = MemoryRecord(content=content, importance=importance,
                              tags=tags or [], source="perception")
        self.sensory.store(record)
        if importance > 0.5:
            self.working.store(record)
        return record

    def consolidate(self) -> None:
        """
        Memory consolidation: promote important working memories
        to episodic, then episodic → semantic (if abstract enough).
        """
        # Working → Episodic
        for record in self.working.retrieve(n=9):
            if record.importance > 0.6:
                self.episodic.store(record)

        # Episodic → Semantic (high-importance, frequently accessed)
        for record in self.episodic.retrieve(n=50):
            if record.access_count > 3 and record.importance > 0.75:
                self.semantic.store(record)

    def query(self, query: str, n: int = 10) -> Dict[str, List[MemoryRecord]]:
        """Cross-tier retrieval."""
        return {
            "episodic": self.episodic.retrieve(query, n),
            "semantic": self.semantic.retrieve(query, n),
            "procedural": self.procedural.retrieve(query, n),
            "collective": self.collective.retrieve(query, n),
        }

    def forget_all(self) -> Dict[str, int]:
        """Run forgetting across all tiers."""
        return {tier.tier_name: tier.forget() for tier in self._tiers}

    def full_stats(self) -> Dict:
        return {tier.tier_name: tier.stats() for tier in self._tiers}

    def __repr__(self) -> str:
        stats = self.full_stats()
        return f"MultiScaleMemory({stats})"