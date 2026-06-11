"""
Observability Layer
====================
Logging, tracing, metrics, and dashboards for the entire organism.
"""
from __future__ import annotations
import time
import uuid
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum
from collections import defaultdict, deque
from contextlib import contextmanager


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogEntry:
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    level: LogLevel = LogLevel.INFO
    component: str = ""
    message: str = ""
    data: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.entry_id,
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "data": self.data,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
        }


@dataclass
class Span:
    """A single unit of work within a distributed trace."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""
    name: str = ""
    component: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    status: str = "in_progress"
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def finish(self, status: str = "ok", error: Optional[str] = None) -> None:
        self.end_time = time.time()
        self.status = status
        self.error = error

    def to_dict(self) -> Dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "component": self.component,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "tags": self.tags,
            "error": self.error,
        }


class MetricsCollector:
    """Collects counters, gauges, and histograms."""

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._timestamps: Dict[str, float] = {}

    def increment(self, name: str, value: float = 1.0, tags: Optional[Dict] = None) -> None:
        key = self._key(name, tags)
        self._counters[key] += value

    def gauge(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        key = self._key(name, tags)
        self._gauges[key] = value
        self._timestamps[key] = time.time()

    def histogram(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        key = self._key(name, tags)
        self._histograms[key].append(value)

    def counter(self, name: str) -> float:
        return self._counters.get(name, 0.0)

    def get_gauge(self, name: str) -> float:
        return self._gauges.get(name, 0.0)

    def percentile(self, name: str, p: float = 0.95) -> Optional[float]:
        values = sorted(self._histograms.get(name, []))
        if not values:
            return None
        idx = int(len(values) * p)
        return values[min(idx, len(values) - 1)]

    def summary(self, name: str) -> Dict:
        values = list(self._histograms.get(name, []))
        if not values:
            return {}
        import statistics
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "p50": self.percentile(name, 0.5),
            "p95": self.percentile(name, 0.95),
            "p99": self.percentile(name, 0.99),
            "min": min(values),
            "max": max(values),
        }

    @staticmethod
    def _key(name: str, tags: Optional[Dict]) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"

    def all_metrics(self) -> Dict:
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
        }


class Tracer:
    """Distributed tracing for cognitive operations."""

    def __init__(self):
        self._traces: Dict[str, List[Span]] = defaultdict(list)
        self._active_spans: Dict[str, Span] = {}

    def start_trace(self, name: str) -> str:
        trace_id = str(uuid.uuid4())[:12]
        return trace_id

    def start_span(self, name: str, trace_id: str, component: str = "",
                   tags: Optional[Dict] = None) -> Span:
        span = Span(trace_id=trace_id, name=name, component=component,
                    tags=tags or {})
        self._traces[trace_id].append(span)
        self._active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span, status: str = "ok",
                    error: Optional[str] = None) -> None:
        span.finish(status, error)
        self._active_spans.pop(span.span_id, None)

    @contextmanager
    def span(self, name: str, trace_id: str, component: str = ""):
        s = self.start_span(name, trace_id, component)
        try:
            yield s
            self.finish_span(s, "ok")
        except Exception as e:
            self.finish_span(s, "error", str(e))
            raise

    def get_trace(self, trace_id: str) -> List[Dict]:
        return [s.to_dict() for s in self._traces.get(trace_id, [])]

    def slow_spans(self, threshold_ms: float = 100.0) -> List[Span]:
        all_spans = [s for spans in self._traces.values() for s in spans]
        return [s for s in all_spans
                if s.duration_ms and s.duration_ms > threshold_ms]


class OrganismLogger:
    """Structured logger for the whole organism."""

    def __init__(self, min_level: LogLevel = LogLevel.INFO,
                 max_entries: int = 10_000):
        self.logger_id = str(uuid.uuid4())[:8]
        self.min_level = min_level
        self._entries: deque = deque(maxlen=max_entries)
        self._level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARN,
                              LogLevel.ERROR, LogLevel.CRITICAL]
        self._active_trace_id: Optional[str] = None

    def _should_log(self, level: LogLevel) -> bool:
        return self._level_order.index(level) >= self._level_order.index(self.min_level)

    def _log(self, level: LogLevel, component: str, message: str,
             data: Optional[Dict] = None) -> Optional[LogEntry]:
        if not self._should_log(level):
            return None
        entry = LogEntry(level=level, component=component, message=message,
                         data=data, trace_id=self._active_trace_id)
        self._entries.append(entry)
        return entry

    def debug(self, component: str, msg: str, data: Dict = None) -> None:
        self._log(LogLevel.DEBUG, component, msg, data)

    def info(self, component: str, msg: str, data: Dict = None) -> None:
        self._log(LogLevel.INFO, component, msg, data)

    def warn(self, component: str, msg: str, data: Dict = None) -> None:
        self._log(LogLevel.WARN, component, msg, data)

    def error(self, component: str, msg: str, data: Dict = None) -> None:
        self._log(LogLevel.ERROR, component, msg, data)

    def critical(self, component: str, msg: str, data: Dict = None) -> None:
        self._log(LogLevel.CRITICAL, component, msg, data)

    def recent(self, n: int = 20, level: Optional[LogLevel] = None) -> List[LogEntry]:
        entries = list(self._entries)[-n * 2:]
        if level:
            entries = [e for e in entries if e.level == level]
        return entries[-n:]

    def errors(self, n: int = 20) -> List[LogEntry]:
        return [e for e in self._entries
                if e.level in (LogLevel.ERROR, LogLevel.CRITICAL)][-n:]

    def export_json(self, n: int = 100) -> str:
        return json.dumps([e.to_dict() for e in list(self._entries)[-n:]], indent=2)


class ObservabilityHub:
    """
    Central observability hub — single entry point for all
    logging, tracing, and metrics across the organism.
    """

    def __init__(self):
        self.hub_id = str(uuid.uuid4())[:8]
        self.logger = OrganismLogger()
        self.metrics = MetricsCollector()
        self.tracer = Tracer()
        self._component_health: Dict[str, Dict] = {}
        self._alert_rules: List[Dict] = []
        self._alerts_fired: List[Dict] = []
        self.start_time = time.time()

    # ─── Component Health ────────────────────────────────────────────────

    def report_health(self, component: str, status: str,
                      details: Optional[Dict] = None) -> None:
        self._component_health[component] = {
            "status": status,
            "details": details or {},
            "last_seen": time.time(),
        }
        if status == "degraded":
            self.logger.warn(component, f"Component degraded", details)
        elif status == "down":
            self.logger.error(component, f"Component down", details)

    # ─── Alerting ────────────────────────────────────────────────────────

    def add_alert_rule(self, name: str, metric: str,
                       threshold: float, comparator: str = ">") -> None:
        self._alert_rules.append({
            "name": name, "metric": metric,
            "threshold": threshold, "comparator": comparator,
        })

    def check_alerts(self) -> List[Dict]:
        fired = []
        for rule in self._alert_rules:
            value = self.metrics.get_gauge(rule["metric"])
            comp = rule["comparator"]
            threshold = rule["threshold"]
            triggered = (
                (comp == ">" and value > threshold) or
                (comp == "<" and value < threshold) or
                (comp == "==" and value == threshold)
            )
            if triggered:
                alert = {
                    "alert": rule["name"],
                    "metric": rule["metric"],
                    "value": value,
                    "threshold": threshold,
                    "fired_at": time.time(),
                }
                fired.append(alert)
                self._alerts_fired.append(alert)
                self.logger.warn("alerting", f"Alert fired: {rule['name']}", alert)
        return fired

    # ─── Dashboard ───────────────────────────────────────────────────────

    def dashboard(self) -> Dict:
        uptime = time.time() - self.start_time
        return {
            "hub_id": self.hub_id,
            "uptime_seconds": round(uptime, 1),
            "component_health": self._component_health,
            "active_alerts": len(self._alerts_fired),
            "recent_errors": [
                {"component": e.component, "message": e.message}
                for e in self.logger.errors(5)
            ],
            "metrics_snapshot": self.metrics.all_metrics(),
            "slow_operations": [
                s.to_dict() for s in self.tracer.slow_spans(threshold_ms=50.0)[:5]
            ],
        }

    def __repr__(self) -> str:
        return f"ObservabilityHub(id={self.hub_id}, uptime={time.time()-self.start_time:.0f}s)"