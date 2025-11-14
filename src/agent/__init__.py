"""Agent module for MCP orchestration with LlamaIndex.

This module provides a modular, async-first orchestrator for managing
MCP server tools with LlamaIndex agents. It supports both STDIO and HTTP
transports and provides concurrent request handling.

Key Components:
    - MCPAgentOrchestrator: Main orchestrator class
    - MCPClientBase: Abstract base for MCP clients
    - MCPClientStdio: STDIO transport implementation
    - MCPClientHttp: HTTP transport implementation
    - MCPClientFactory: Factory for creating clients
    - AgentConfig: Configuration management
    - SystemPrompts: Prompt templates

Usage:
    >>> from src.agent import get_orchestrator
    >>> orchestrator = await get_orchestrator()
    >>> response = await orchestrator.execute_query("Find patients...")
    >>> await orchestrator.shutdown()

Or with context manager:
    >>> from src.agent import MCPAgentOrchestrator
    >>> async with MCPAgentOrchestrator() as orchestrator:
    ...     response = await orchestrator.execute_query("Find patients...")
"""

from src.mcp_client.mcp_client_base import MCPClientBase
from src.mcp_client.mcp_client_factory import MCPClientFactory
from src.mcp_client.mcp_client_http import MCPClientHttp
from src.mcp_client.mcp_client_stdio import MCPClientStdio

from .config import AgentConfig, MCPTransport, agent_config
from .orchestrator import (
    MCPAgentOrchestrator,
    QueryValidationError,
    get_orchestrator,
    shutdown_orchestrator,
)
from .prompts import (
    QueryPromptTemplates,
    SystemPrompts,
    format_query,
    get_query_template,
    get_system_prompt,
)

__all__ = [
    # Orchestrator
    "MCPAgentOrchestrator",
    "QueryValidationError",
    "get_orchestrator",
    "shutdown_orchestrator",
    # Configuration
    "AgentConfig",
    "MCPTransport",
    "agent_config",
    # MCP Clients
    "MCPClientBase",
    "MCPClientStdio",
    "MCPClientHttp",
    "MCPClientFactory",
    # Prompts
    "SystemPrompts",
    "QueryPromptTemplates",
    "get_system_prompt",
    "get_query_template",
    "format_query",
]
