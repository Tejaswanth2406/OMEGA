"""
Protocol defining the interface for intelligence engines.
"""
from typing import Protocol, List, Dict, Any

class ModelProvider(Protocol):
    async def generate(self, prompt: str, **kwargs) -> str:
        ...

    async def embed(self, text: str) -> List[float]:
        ...
