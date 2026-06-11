"""
Selects the next best action based on policy and state.
"""
class ActionSelector:
    def select_best_action(self, candidate_actions: list) -> dict:
        return candidate_actions[0] if candidate_actions else {}
