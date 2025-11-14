"""Conversation management for MCP Agent Orchestrator.

This package provides enterprise-grade conversation management with:
- Session-based chat history
- State persistence (in-memory and Redis)
- Multi-turn conversation support
- Automatic session cleanup
"""

from .manager import ConversationManager
from .persistence import InMemorySessionStore, RedisSessionStore, SessionStore
from .session import ConversationMessage, ConversationSession

__all__ = [
    "ConversationManager",
    "ConversationMessage",
    "ConversationSession",
    "SessionStore",
    "InMemorySessionStore",
    "RedisSessionStore",
]
