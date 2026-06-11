"""
Layer 8: Cognitive Immune System
==================================
Protects the organism from bad mutations, hallucinations,
logic errors, inconsistencies, and security threats.
Every mutation passes through the immune gate before deployment.
"""
from __future__ import annotations
import re
import time
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ThreatLevel(str, Enum):
    CLEAR = "clear"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    CRITICAL = "critical"


class CheckResult(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class ImmuneCheck:
    check_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    checker_name: str = ""
    result: CheckResult = CheckResult.PASS
    threat_level: ThreatLevel = ThreatLevel.CLEAR
    score: float = 1.0          # 1.0 = completely safe, 0.0 = lethal
    details: str = ""
    flagged_items: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ImmuneScan:
    scan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_id: str = ""
    target_type: str = ""       # "mutation", "output", "tool_call", etc.
    checks: List[ImmuneCheck] = field(default_factory=list)
    overall_result: CheckResult = CheckResult.PASS
    overall_threat: ThreatLevel = ThreatLevel.CLEAR
    approved: bool = True
    scanned_at: float = field(default_factory=time.time)

    def finalize(self) -> None:
        if not self.checks:
            return
        # Aggregate: one FAIL → whole scan fails
        results = [c.result for c in self.checks]
        threats = [c.threat_level for c in self.checks]

        if CheckResult.FAIL in results:
            self.overall_result = CheckResult.FAIL
            self.approved = False
        elif CheckResult.WARN in results:
            self.overall_result = CheckResult.WARN
        else:
            self.overall_result = CheckResult.PASS

        threat_order = [ThreatLevel.CLEAR, ThreatLevel.SUSPICIOUS,
                        ThreatLevel.DANGEROUS, ThreatLevel.CRITICAL]
        max_threat = max(threats, key=lambda t: threat_order.index(t))
        self.overall_threat = max_threat

        if max_threat in (ThreatLevel.DANGEROUS, ThreatLevel.CRITICAL):
            self.approved = False

    def summary(self) -> Dict:
        return {
            "scan_id": self.scan_id,
            "target_id": self.target_id,
            "approved": self.approved,
            "overall_result": self.overall_result.value,
            "overall_threat": self.overall_threat.value,
            "checks": [{
                "name": c.checker_name,
                "result": c.result.value,
                "score": c.score,
                "details": c.details,
            } for c in self.checks],
        }


# ─── Individual Checkers ─────────────────────────────────────────────────────

class BaseChecker(ABC):
    name: str = "base_checker"

    @abstractmethod
    def check(self, target: Any, context: Optional[Dict] = None) -> ImmuneCheck: ...


class HallucinationDetector(BaseChecker):
    name = "hallucination_detector"

    # Patterns commonly associated with hallucinated outputs
    UNCERTAINTY_MARKERS = [
        r"\b(I think|I believe|probably|perhaps|might be|could be)\b",
        r"\b(as of my (knowledge|training))\b",
        r"\b(I am not sure|I cannot confirm)\b",
    ]
    FABRICATION_PATTERNS = [
        r"\b(according to [A-Z][a-z]+ et al\.)\b",
        r"\b(study in \d{4})\b",
        r"\bhttps?://[^\s]+\b",   # fabricated URLs
    ]

    def check(self, target: Any, context: Optional[Dict] = None) -> ImmuneCheck:
        text = str(target)
        flagged = []
        score = 1.0

        for pattern in self.UNCERTAINTY_MARKERS:
            if re.search(pattern, text, re.IGNORECASE):
                flagged.append(f"uncertainty_marker: {pattern}")
                score -= 0.05

        for pattern in self.FABRICATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                flagged.append(f"possible_fabrication: {matches[:3]}")
                score -= 0.15

        score = max(0.0, score)
        threat = ThreatLevel.CLEAR if score > 0.8 else \
                 ThreatLevel.SUSPICIOUS if score > 0.5 else \
                 ThreatLevel.DANGEROUS

        return ImmuneCheck(
            checker_name=self.name,
            result=CheckResult.PASS if score > 0.6 else
                   CheckResult.WARN if score > 0.3 else CheckResult.FAIL,
            threat_level=threat,
            score=score,
            details=f"Hallucination score: {score:.2f}",
            flagged_items=flagged,
        )


class LogicChecker(BaseChecker):
    name = "logic_checker"

    CONTRADICTION_PATTERNS = [
        (r"\balways\b", r"\bnever\b"),
        (r"\ball\b", r"\bnone\b"),
        (r"\bimpossible\b", r"\bcertain\b"),
    ]

    def check(self, target: Any, context: Optional[Dict] = None) -> ImmuneCheck:
        text = str(target).lower()
        flagged = []
        score = 1.0

        for p1, p2 in self.CONTRADICTION_PATTERNS:
            if re.search(p1, text) and re.search(p2, text):
                flagged.append(f"contradiction: '{p1}' and '{p2}' both present")
                score -= 0.2

        # Check for circular reasoning markers
        if re.search(r"\bbecause .{0,40} because\b", text):
            flagged.append("possible_circular_reasoning")
            score -= 0.15

        score = max(0.0, score)
        return ImmuneCheck(
            checker_name=self.name,
            result=CheckResult.PASS if score > 0.7 else
                   CheckResult.WARN if score > 0.4 else CheckResult.FAIL,
            threat_level=ThreatLevel.CLEAR if score > 0.7 else ThreatLevel.SUSPICIOUS,
            score=score,
            details=f"Logic consistency score: {score:.2f}",
            flagged_items=flagged,
        )


class ConsistencyChecker(BaseChecker):
    name = "consistency_checker"

    def __init__(self):
        self._history: List[str] = []

    def check(self, target: Any, context: Optional[Dict] = None) -> ImmuneCheck:
        text = str(target)
        flagged = []
        score = 1.0

        if context and "previous_outputs" in context:
            for prev in context["previous_outputs"][-5:]:
                prev_text = str(prev).lower()
                curr_text = text.lower()
                # Simple overlap check — replace with semantic similarity in production
                prev_words = set(prev_text.split())
                curr_words = set(curr_text.split())
                overlap = len(prev_words & curr_words) / max(len(prev_words | curr_words), 1)
                if overlap < 0.05 and len(curr_words) > 20:
                    flagged.append("low_semantic_overlap_with_context")
                    score -= 0.1

        self._history.append(text[:100])
        if len(self._history) > 50:
            self._history = self._history[-50:]

        score = max(0.0, score)
        return ImmuneCheck(
            checker_name=self.name,
            result=CheckResult.PASS if score > 0.6 else CheckResult.WARN,
            threat_level=ThreatLevel.CLEAR if score > 0.6 else ThreatLevel.SUSPICIOUS,
            score=score,
            details=f"Consistency score: {score:.2f}",
            flagged_items=flagged,
        )


class SecurityChecker(BaseChecker):
    name = "security_checker"

    INJECTION_PATTERNS = [
        r"(DROP TABLE|DELETE FROM|INSERT INTO)",
        r"(<script|javascript:|onerror=)",
        r"(\.\./|/etc/passwd|/etc/shadow)",
        r"(eval\(|exec\(|__import__)",
        r"(rm -rf|sudo rm|format c:)",
    ]

    def check(self, target: Any, context: Optional[Dict] = None) -> ImmuneCheck:
        text = str(target)
        flagged = []
        score = 1.0

        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                flagged.append(f"injection_pattern: {pattern}")
                score -= 0.4

        score = max(0.0, score)
        return ImmuneCheck(
            checker_name=self.name,
            result=CheckResult.PASS if score > 0.8 else
                   CheckResult.WARN if score > 0.4 else CheckResult.FAIL,
            threat_level=ThreatLevel.CLEAR if score > 0.8 else
                         ThreatLevel.DANGEROUS if score > 0.2 else ThreatLevel.CRITICAL,
            score=score,
            details=f"Security score: {score:.2f}",
            flagged_items=flagged,
        )


class OntologyChecker(BaseChecker):
    """Checks that mutations are semantically coherent with the organism's ontology."""
    name = "ontology_checker"

    def __init__(self, valid_domains: Optional[List[str]] = None):
        self.valid_domains = valid_domains or [
            "reasoning", "memory", "planning", "retrieval",
            "tool_use", "learning", "communication", "immune",
        ]

    def check(self, target: Any, context: Optional[Dict] = None) -> ImmuneCheck:
        score = 1.0
        flagged = []

        if isinstance(target, dict):
            domain = target.get("domain") or target.get("function", "")
            if domain and domain not in self.valid_domains:
                flagged.append(f"unknown_domain: {domain}")
                score -= 0.3

            # Check for degenerate values
            for k, v in target.items():
                if isinstance(v, float) and (v < 0 or v > 1):
                    if k in ("weight", "fitness", "confidence", "mutation_rate"):
                        flagged.append(f"out_of_range: {k}={v}")
                        score -= 0.1

        score = max(0.0, score)
        return ImmuneCheck(
            checker_name=self.name,
            result=CheckResult.PASS if score > 0.7 else CheckResult.WARN,
            threat_level=ThreatLevel.CLEAR if score > 0.7 else ThreatLevel.SUSPICIOUS,
            score=score,
            details=f"Ontological coherence: {score:.2f}",
            flagged_items=flagged,
        )


# ─── Immune System Orchestrator ──────────────────────────────────────────────

class CognitiveImmuneSystem:
    """
    Gates all mutations, outputs, and tool calls through a multi-checker pipeline.
    Nothing is deployed without passing the immune system.
    """

    def __init__(self):
        self.system_id = str(uuid.uuid4())[:8]
        self.checkers: List[BaseChecker] = [
            HallucinationDetector(),
            LogicChecker(),
            ConsistencyChecker(),
            SecurityChecker(),
            OntologyChecker(),
        ]
        self.scan_log: List[ImmuneScan] = []
        self.quarantine: List[Dict] = []    # rejected items
        self.approved_count = 0
        self.rejected_count = 0

    def register_checker(self, checker: BaseChecker) -> None:
        self.checkers.append(checker)

    def scan(self, target: Any, target_id: str = "",
             target_type: str = "output", context: Optional[Dict] = None) -> ImmuneScan:
        """Run all checkers on a target and return the aggregate scan result."""
        scan = ImmuneScan(
            target_id=target_id or str(uuid.uuid4())[:8],
            target_type=target_type,
        )

        for checker in self.checkers:
            try:
                result = checker.check(target, context)
                scan.checks.append(result)
            except Exception as e:
                scan.checks.append(ImmuneCheck(
                    checker_name=checker.name,
                    result=CheckResult.WARN,
                    details=f"Checker error: {e}",
                    score=0.5,
                ))

        scan.finalize()
        self.scan_log.append(scan)

        if scan.approved:
            self.approved_count += 1
        else:
            self.rejected_count += 1
            self.quarantine.append({
                "scan_id": scan.scan_id,
                "target_type": target_type,
                "reason": scan.overall_threat.value,
                "timestamp": time.time(),
            })

        return scan

    def scan_mutation(self, mutation: Dict, context: Optional[Dict] = None) -> Tuple[bool, ImmuneScan]:
        """Specialized scan for genome mutations."""
        scan = self.scan(mutation, target_type="mutation", context=context)
        return scan.approved, scan

    def scan_output(self, output: str, context: Optional[Dict] = None) -> Tuple[bool, ImmuneScan]:
        """Specialized scan for LLM outputs."""
        scan = self.scan(output, target_type="output", context=context)
        return scan.approved, scan

    def health_stats(self) -> Dict:
        total = self.approved_count + self.rejected_count
        return {
            "system_id": self.system_id,
            "total_scans": total,
            "approved": self.approved_count,
            "rejected": self.rejected_count,
            "approval_rate": self.approved_count / max(1, total),
            "quarantine_size": len(self.quarantine),
            "checkers": [c.name for c in self.checkers],
        }

    def clear_quarantine(self) -> int:
        n = len(self.quarantine)
        self.quarantine.clear()
        return n

    def __repr__(self) -> str:
        return (f"CognitiveImmuneSystem(id={self.system_id}, "
                f"checkers={len(self.checkers)}, "
                f"approval_rate={self.approved_count/max(1, self.approved_count+self.rejected_count):.2%})")