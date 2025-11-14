"""Session data structures for conversation management.

This module defines the core data structures for managing conversations:
- ConversationMessage: Individual message in a conversation
- ConversationSession: Complete conversation session with history

Enterprise Pattern:
- Immutable message history
- Timestamps for all events
- Metadata for extensibility
- JSON serialization support
"""

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from llama_index.core.base.llms.types import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a single message in a conversation.

    This is the fundamental unit of conversation history. Each message
    captures the role (user/assistant/system), content, and metadata.

    Attributes:
        role: Message role ('user', 'assistant', 'system', 'tool')
        content: Message content/text
        timestamp: When the message was created (Unix timestamp)
        metadata: Additional metadata (tool calls, results, tokens, etc.)

    Example:
        >>> msg = ConversationMessage(
        ...     role="user",
        ...     content="Find patients with diabetes"
        ... )
        >>> msg.to_llama_message()
        ChatMessage(role=<MessageRole.USER: 'user'>, content='Find patients...')
    """

    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llama_message(self) -> ChatMessage:
        """Convert to LlamaIndex ChatMessage format.

        This enables integration with LlamaIndex agents that expect
        ChatMessage objects for conversation history.

        Returns:
            ChatMessage compatible with LlamaIndex agents

        Note:
            The role is mapped to LlamaIndex MessageRole enum:
            - 'user' -> MessageRole.USER
            - 'assistant' -> MessageRole.ASSISTANT
            - 'system' -> MessageRole.SYSTEM
            - 'tool' -> MessageRole.FUNCTION
        """
        role_mapping = {
            "user": MessageRole.USER,
            "assistant": MessageRole.ASSISTANT,
            "system": MessageRole.SYSTEM,
            "tool": MessageRole.FUNCTION,
        }
        return ChatMessage(
            role=role_mapping.get(self.role, MessageRole.USER),
            content=self.content,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the message
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        """Create ConversationMessage from dictionary.

        Args:
            data: Dictionary with message data

        Returns:
            ConversationMessage instance
        """
        return cls(**data)


@dataclass
class ConversationSession:
    """Manages a conversation session with chat history.

    This is the primary data structure for conversation state management.
    Each session represents a unique conversation thread with a user,
    maintaining full history and metadata.

    Enterprise Pattern:
    - Unique session ID for tracking and auditing
    - Full conversation history with timestamps
    - Last activity tracking for TTL-based cleanup
    - Extensible metadata for user context, permissions, etc.
    - JSON serialization for persistence

    Attributes:
        session_id: Unique session identifier (UUID)
        messages: List of conversation messages (chronological order)
        created_at: Session creation timestamp
        last_activity: Last activity timestamp (for TTL)
        metadata: Session metadata (user_id, context, tags, etc.)

    Example:
        >>> session = ConversationSession()
        >>> session.add_message("user", "Hello!")
        >>> session.add_message("assistant", "Hi! How can I help?")
        >>> len(session.messages)
        2
        >>> session.get_conversation_summary()
        {'session_id': '...', 'message_count': 2, ...}
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[ConversationMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a message to the conversation history.

        This is the primary method for building conversation history.
        Each message is timestamped and appended to the message list.
        Last activity is automatically updated.

        Args:
            role: Message role ('user', 'assistant', 'system', 'tool')
            content: Message content
            metadata: Optional metadata (tool calls, results, etc.)

        Example:
            >>> session.add_message("user", "Find patients with diabetes")
            >>> session.add_message("assistant", "I found 5 patients...",
            ...                     metadata={"tool_used": "search_patients"})
        """
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.last_activity = time.time()
        logger.debug(
            f"Added {role} message to session {self.session_id[:8]} "
            f"(total: {len(self.messages)})"
        )

    def get_chat_history(
        self, max_messages: int | None = None, as_llama_messages: bool = True
    ) -> list[ChatMessage] | list[ConversationMessage]:
        """Get chat history for agent context.

        This method retrieves conversation history, optionally limiting
        the number of messages to prevent token bloat.

        Args:
            max_messages: Maximum messages to return (None for all)
            as_llama_messages: Return as LlamaIndex ChatMessage objects

        Returns:
            List of messages (most recent if limited)

        Example:
            >>> # Get last 10 messages for agent context
            >>> history = session.get_chat_history(max_messages=10)
            >>> # Get all messages as ConversationMessage objects
            >>> all_msgs = session.get_chat_history(as_llama_messages=False)
        """
        messages_to_return = self.messages
        if max_messages and len(self.messages) > max_messages:
            # Return most recent messages (maintains conversation context)
            messages_to_return = self.messages[-max_messages:]
            logger.debug(
                f"Returning last {max_messages} messages " f"(total: {len(self.messages)})"
            )

        if as_llama_messages:
            return [msg.to_llama_message() for msg in messages_to_return]
        return messages_to_return

    def get_conversation_summary(self) -> dict[str, Any]:
        """Get summary of conversation for monitoring/debugging.

        This provides key metrics about the conversation session,
        useful for monitoring, debugging, and analytics.

        Returns:
            Dictionary with conversation statistics

        Example:
            >>> summary = session.get_conversation_summary()
            >>> print(f"Messages: {summary['message_count']}")
            >>> print(f"Duration: {summary['duration_seconds']}s")
        """
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "duration_seconds": self.last_activity - self.created_at,
            "user_messages": sum(1 for m in self.messages if m.role == "user"),
            "assistant_messages": sum(1 for m in self.messages if m.role == "assistant"),
            "metadata": self.metadata,
        }

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if session has expired based on TTL.

        Args:
            ttl_seconds: Time-to-live in seconds

        Returns:
            True if session is expired, False otherwise

        Example:
            >>> session.is_expired(ttl_seconds=3600)  # 1 hour TTL
            False
        """
        return time.time() - self.last_activity > ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for serialization.

        This enables JSON serialization for persistence (Redis, database, etc.)

        Returns:
            Dictionary representation of the session

        Example:
            >>> session_dict = session.to_dict()
            >>> json_str = json.dumps(session_dict)
        """
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSession":
        """Create ConversationSession from dictionary.

        This enables deserialization from JSON (Redis, database, etc.)

        Args:
            data: Dictionary with session data

        Returns:
            ConversationSession instance

        Example:
            >>> session_dict = json.loads(json_str)
            >>> session = ConversationSession.from_dict(session_dict)
        """
        messages = [ConversationMessage.from_dict(m) for m in data.get("messages", [])]
        return cls(
            session_id=data["session_id"],
            messages=messages,
            created_at=data["created_at"],
            last_activity=data["last_activity"],
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert session to JSON string.

        Returns:
            JSON string representation

        Example:
            >>> json_str = session.to_json()
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ConversationSession":
        """Create ConversationSession from JSON string.

        Args:
            json_str: JSON string with session data

        Returns:
            ConversationSession instance

        Example:
            >>> session = ConversationSession.from_json(json_str)
        """
        return cls.from_dict(json.loads(json_str))
