"""
Decides when to repair vs. when to learn.
"""
from .anomaly_classifier import AnomalyType

class RepairPolicy:
    def should_repair(self, anomaly: AnomalyType) -> bool:
        return True
