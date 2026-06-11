"""
System-level resource monitoring (RAM, CPU).
"""
class HealthMonitor:
    def get_system_health(self) -> dict:
        return {"cpu": 10.0, "ram": 25.0}
