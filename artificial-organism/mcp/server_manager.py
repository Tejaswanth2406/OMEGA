"""
Dynamic attachment/detachment of external context servers.
"""
class ServerManager:
    def connect(self, server_url: str):
        pass
    def disconnect(self, server_url: str):
        pass
    def health_check(self) -> bool:
        return True
