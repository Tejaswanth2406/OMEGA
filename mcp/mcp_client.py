"""
Interface to external MCP protocol servers.
"""
class MCPClient:
    def call_tool(self, server_name: str, tool_name: str, args: dict):
        pass
