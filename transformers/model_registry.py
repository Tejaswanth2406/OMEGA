"""
Registers specific LLM providers.
"""
from typing import Dict, Type
from .provider import ModelProvider

class ModelRegistry:
    def __init__(self):
        self._providers: Dict[str, Type[ModelProvider]] = {}

    def register(self, name: str, provider: Type[ModelProvider]):
        self._providers[name] = provider
