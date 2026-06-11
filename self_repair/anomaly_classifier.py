"""
Classifies drifts (e.g., PerformanceDrift, PromptCollapse).
"""
from enum import Enum

class AnomalyType(Enum):
    PERFORMANCE_DRIFT = "PerformanceDrift"
    MEMORY_CORRUPTION = "MemoryCorruption"
    TOOL_FAILURE = "ToolFailure"
    PROMPT_COLLAPSE = "PromptCollapse"
    GOAL_LOOP = "GoalLoop"

class AnomalyClassifier:
    def classify(self, state_data: dict) -> AnomalyType:
        return AnomalyType.PERFORMANCE_DRIFT
