"""
Layer 9: Fast Ontological Recognition
=======================================
Pre-retrieval classification of inputs.
Dramatically reduces search cost by routing queries to the right memory/graph regions.
"""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class OntologyClass(str, Enum):
    ENTITY = "entity"
    PROCESS = "process"
    EVENT = "event"
    RELATIONSHIP = "relationship"
    TOOL = "tool"
    MEMORY_REF = "memory_ref"
    THEORY = "theory"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    QUESTION = "question"
    COMMAND = "command"
    DATA = "data"
    UNKNOWN = "unknown"


@dataclass
class OntologyTag:
    tag_id: str = ""
    primary_class: OntologyClass = OntologyClass.UNKNOWN
    secondary_classes: List[OntologyClass] = field(default_factory=list)
    confidence: float = 0.5
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    intent: str = ""
    complexity: float = 0.5     # 0 = trivial, 1 = highly complex
    routing: Dict[str, float] = field(default_factory=dict)  # layer -> priority


# ─── Pattern-Based Recognizers ───────────────────────────────────────────────

class OntologyRecognizer:
    """
    Fast regex + heuristic classifier.
    In production this is augmented by a fine-tuned classifier model.
    """

    PATTERNS: Dict[OntologyClass, List[str]] = {
        OntologyClass.QUESTION: [
            r"^(what|who|when|where|why|how|which|is|are|can|could|should|would)\b",
            r"\?$",
        ],
        OntologyClass.COMMAND: [
            r"^(create|make|build|generate|write|run|execute|start|stop|delete|update|find)\b",
            r"^(please|could you|can you)\s+(create|make|build|run|find|write)\b",
        ],
        OntologyClass.GOAL: [
            r"\b(goal|objective|aim|target|want to|need to|trying to|plan to)\b",
            r"\b(achieve|accomplish|complete|finish|solve)\b",
        ],
        OntologyClass.CONSTRAINT: [
            r"\b(must|must not|cannot|should not|forbidden|required|mandatory|limited to)\b",
            r"\b(constraint|restriction|limit|boundary|rule)\b",
        ],
        OntologyClass.PROCESS: [
            r"\b(process|workflow|pipeline|procedure|algorithm|method|approach)\b",
            r"\b(step[s]?|phase[s]?|stage[s]?|iteration)\b",
        ],
        OntologyClass.EVENT: [
            r"\b(event|happened|occurred|incident|trigger|fired|started|ended)\b",
            r"\b(at \d{1,2}:\d{2}|yesterday|today|last (week|month|year))\b",
        ],
        OntologyClass.TOOL: [
            r"\b(tool|api|function|call|library|plugin|module|service|endpoint)\b",
            r"\b(use|invoke|call|execute)\s+\w+\(\)",
        ],
        OntologyClass.THEORY: [
            r"\b(theory|theorem|hypothesis|principle|law|model|framework)\b",
            r"\b(according to|proposed by|as described in)\b",
        ],
        OntologyClass.DATA: [
            r"\b(data|dataset|database|table|row|column|record|file|json|csv)\b",
            r"\b(value[s]?|number[s]?|metric[s]?|statistic[s]?)\b",
        ],
        OntologyClass.ENTITY: [
            r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",   # proper nouns
            r"\b(person|company|organization|country|city|product)\b",
        ],
    }

    ENTITY_EXTRACTORS: List[Tuple[str, str]] = [
        (r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b", "named_entity"),
        (r"\b(\w+(?:_\w+)+)\b", "snake_case_identifier"),
        (r'"([^"]+)"', "quoted_string"),
        (r"`([^`]+)`", "code_identifier"),
    ]

    RELATION_PATTERNS: List[str] = [
        r"\b(\w+)\s+(?:is a|has|uses|causes|enables|requires)\s+(\w+)\b",
        r"\b(\w+)\s+→\s+(\w+)\b",
    ]

    COMPLEXITY_SIGNALS = {
        "high": [r"multi.step", r"complex", r"recursive", r"distributed", r"architecture", r"optimize"],
        "low":  [r"simple", r"basic", r"just", r"quick", r"hello", r"what is"],
    }

    # ─── Routing Table ───────────────────────────────────────────────────

    ROUTING_TABLE: Dict[OntologyClass, Dict[str, float]] = {
        OntologyClass.ENTITY:       {"graph": 0.9, "semantic_memory": 0.8, "retrieval": 0.7},
        OntologyClass.PROCESS:      {"procedural_memory": 0.9, "planning": 0.8, "graph": 0.6},
        OntologyClass.EVENT:        {"episodic_memory": 0.9, "graph": 0.6},
        OntologyClass.RELATIONSHIP: {"graph": 0.95, "semantic_memory": 0.7},
        OntologyClass.TOOL:         {"tool_registry": 0.95, "procedural_memory": 0.7},
        OntologyClass.MEMORY_REF:   {"episodic_memory": 0.9, "semantic_memory": 0.8},
        OntologyClass.THEORY:       {"semantic_memory": 0.9, "graph": 0.8},
        OntologyClass.GOAL:         {"planning": 0.95, "working_memory": 0.8},
        OntologyClass.CONSTRAINT:   {"working_memory": 0.9, "immune": 0.8},
        OntologyClass.QUESTION:     {"retrieval": 0.9, "reasoning": 0.85},
        OntologyClass.COMMAND:      {"planning": 0.9, "tool_registry": 0.8},
        OntologyClass.DATA:         {"semantic_memory": 0.85, "retrieval": 0.8},
        OntologyClass.UNKNOWN:      {"retrieval": 0.7, "reasoning": 0.6},
    }

    def recognize(self, text: str) -> OntologyTag:
        """Classify the input and return routing instructions."""
        tag = OntologyTag()
        text_lower = text.lower()

        # Score each class
        class_scores: Dict[OntologyClass, float] = {}
        for cls, patterns in self.PATTERNS.items():
            score = 0.0
            for pat in patterns:
                matches = re.findall(pat, text_lower if cls != OntologyClass.ENTITY else text,
                                     re.IGNORECASE)
                score += len(matches) * 0.3
            if score > 0:
                class_scores[cls] = min(1.0, score)

        if class_scores:
            sorted_classes = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            tag.primary_class = sorted_classes[0][0]
            tag.confidence = min(1.0, sorted_classes[0][1])
            tag.secondary_classes = [cls for cls, _ in sorted_classes[1:4]]
        else:
            tag.primary_class = OntologyClass.UNKNOWN
            tag.confidence = 0.3

        # Extract entities
        for pattern, _ in self.ENTITY_EXTRACTORS:
            tag.entities.extend(re.findall(pattern, text)[:5])
        tag.entities = list(set(tag.entities))[:10]

        # Extract relations
        for pattern in self.RELATION_PATTERNS:
            pairs = re.findall(pattern, text, re.IGNORECASE)
            tag.relations.extend([f"{a}->{b}" for a, b in pairs])

        # Infer intent
        tag.intent = self._infer_intent(text_lower)

        # Estimate complexity
        tag.complexity = self._estimate_complexity(text_lower)

        # Set routing
        tag.routing = dict(self.ROUTING_TABLE.get(tag.primary_class, {}))

        return tag

    def _infer_intent(self, text: str) -> str:
        if re.search(r"\b(what|who|when|where)\b", text): return "information_seeking"
        if re.search(r"\b(create|make|build|generate)\b", text): return "creation"
        if re.search(r"\b(fix|repair|correct|debug)\b", text): return "repair"
        if re.search(r"\b(optimize|improve|enhance)\b", text): return "optimization"
        if re.search(r"\b(explain|describe|show|tell)\b", text): return "explanation"
        if re.search(r"\b(compare|contrast|difference)\b", text): return "comparison"
        if re.search(r"\b(delete|remove|clean)\b", text): return "deletion"
        return "general"

    def _estimate_complexity(self, text: str) -> float:
        score = 0.5
        word_count = len(text.split())
        if word_count > 100: score += 0.2
        elif word_count < 10: score -= 0.2

        for pat in self.COMPLEXITY_SIGNALS["high"]:
            if re.search(pat, text): score += 0.1
        for pat in self.COMPLEXITY_SIGNALS["low"]:
            if re.search(pat, text): score -= 0.1

        # Count subordinate clauses
        clause_count = len(re.findall(r"\b(because|although|however|therefore|whereas)\b", text))
        score += clause_count * 0.05

        return max(0.0, min(1.0, score))

    def batch_recognize(self, texts: List[str]) -> List[OntologyTag]:
        return [self.recognize(t) for t in texts]

    def route_query(self, text: str) -> Dict[str, float]:
        """Quick routing decision without full tag creation."""
        tag = self.recognize(text)
        return tag.routing


class OntologyIndex:
    """
    Maintains an index of recognized ontology tags for fast lookup.
    """

    def __init__(self):
        self.recognizer = OntologyRecognizer()
        self._index: Dict[str, OntologyTag] = {}
        self._class_index: Dict[str, List[str]] = {}
        self.recognition_log: List[Dict] = []

    def index(self, text: str, doc_id: Optional[str] = None) -> OntologyTag:
        import hashlib
        doc_id = doc_id or hashlib.md5(text.encode()).hexdigest()[:8]
        tag = self.recognizer.recognize(text)
        tag.tag_id = doc_id
        self._index[doc_id] = tag

        cls_key = tag.primary_class.value
        if cls_key not in self._class_index:
            self._class_index[cls_key] = []
        self._class_index[cls_key].append(doc_id)

        self.recognition_log.append({
            "doc_id": doc_id,
            "class": cls_key,
            "confidence": tag.confidence,
            "routing": tag.routing,
            "timestamp": time.time(),
        })
        return tag

    def get_by_class(self, cls: OntologyClass) -> List[OntologyTag]:
        ids = self._class_index.get(cls.value, [])
        return [self._index[i] for i in ids if i in self._index]

    def stats(self) -> Dict:
        class_counts = {k: len(v) for k, v in self._class_index.items()}
        return {
            "total_indexed": len(self._index),
            "class_distribution": class_counts,
            "recognitions": len(self.recognition_log),
        }