"""
Event Bus for Pub/Sub architecture.
"""
from typing import Callable, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class EventType(Enum):
    GOAL_CREATED = "GoalCreated"
    MEMORY_ADDED = "MemoryAdded"
    TOOL_DISCOVERED = "ToolDiscovered"
    MUTATION_REQUESTED = "MutationRequested"
    REPAIR_TRIGGERED = "RepairTriggered"

@dataclass
class Event:
    event_type: EventType
    payload: Dict[str, Any]

class EventBus:
    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable[[Event], None]]] = {e: [] for e in EventType}

    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        self.subscribers[event_type].append(callback)

    def publish(self, event: Event):
        for callback in self.subscribers[event.event_type]:
            callback(event)
