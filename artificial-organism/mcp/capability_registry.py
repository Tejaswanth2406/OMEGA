"""
Tracks what the organism can currently do.
"""
class CapabilityRegistry:
    def __init__(self):
        self.capabilities = {}

    def register_capability(self, name: str, details: dict):
        self.capabilities[name] = details
