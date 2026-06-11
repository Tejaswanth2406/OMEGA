"""
Manages token budgets per request.
"""
class ContextWindowManager:
    def __init__(self, max_tokens: int = 8192):
        self.max_tokens = max_tokens
        self.current_usage = 0
