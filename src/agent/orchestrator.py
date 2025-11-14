"""MCP Agent Orchestrator - Corrected for FunctionAgent.run().

Key corrections:
1. FunctionAgent.run() is already async - just await it
2. It returns a dict with 'response' key, not a string directly
3. ReActAgent uses .chat() / .achat() (different API)
4. Removed incorrect aquery/achat references
"""

import asyncio
import logging
import random
from typing import Any

from llama_index.core.agent import ReActAgent
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI

from src.config.settings import settings

from .config import agent_config
from .mcp_client_base import MCPClientBase
from .mcp_client_factory import MCPClientFactory

# Import conversation management (optional)
try:
    from .conversation import ConversationManager

    CONVERSATION_SUPPORT = True
except ImportError:
    CONVERSATION_SUPPORT = False
    ConversationManager = None

logger = logging.getLogger(__name__)


class QueryValidationError(ValueError):
    """Raised when query validation fails."""

    pass


class MCPAgentOrchestrator:
    """Orchestrator for MCP agent with LlamaIndex.

    Supports both FunctionAgent (workflow-based) and ReActAgent (traditional).
    """

    # Query validation constants
    MAX_QUERY_LENGTH = 10000
    MAX_CONCURRENT_QUERIES = 50

    # Retry configuration
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0
    MAX_RETRY_DELAY = 10.0
    RETRY_BACKOFF_MULTIPLIER = 2.0

    def __init__(self):
        """Initialize the MCP agent orchestrator."""
        self.mcp_client: MCPClientBase | None = None
        self.agent: FunctionAgent | ReActAgent | None = None
        self.tools: list[Any] = []
        self._initialized = False
        self._semaphore: asyncio.Semaphore | None = None
        self._active_requests = 0
        self._shutdown_event = asyncio.Event()
        self._shutdown_event.set()
        self.conversation_manager: ConversationManager | None = None

        logger.info("Initialized MCPAgentOrchestrator")

    def _validate_query(self, query: str) -> None:
        """Validate a single query."""
        if not isinstance(query, str):
            raise QueryValidationError(f"Query must be a string, got {type(query)}")
        if not query.strip():
            raise QueryValidationError("Query cannot be empty or whitespace only")
        if len(query) > self.MAX_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query too long: {len(query)} characters (maximum: {self.MAX_QUERY_LENGTH})"
            )

    def _validate_query_list(self, queries: list[str]) -> None:
        """Validate a list of queries."""
        if not isinstance(queries, list):
            raise QueryValidationError(f"Queries must be a list, got {type(queries)}")
        if not queries:
            raise QueryValidationError("Query list cannot be empty")
        if len(queries) > self.MAX_CONCURRENT_QUERIES:
            raise QueryValidationError(
                f"Too many queries: {len(queries)} (maximum: {self.MAX_CONCURRENT_QUERIES})"
            )
        for i, query in enumerate(queries):
            try:
                self._validate_query(query)
            except QueryValidationError as e:
                raise QueryValidationError(f"Query {i}: {e}") from e

    async def _retry_with_backoff(
        self,
        func,
        operation_name: str,
        max_retries: int | None = None,
        base_delay: float | None = None,
    ):
        """Execute a function with exponential backoff retry logic."""
        if max_retries is None:
            max_retries = self.MAX_RETRIES
        if base_delay is None:
            base_delay = self.BASE_RETRY_DELAY

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = min(
                        base_delay * (self.RETRY_BACKOFF_MULTIPLIER ** (attempt - 1)),
                        self.MAX_RETRY_DELAY,
                    )
                    jitter = random.uniform(0.1, 1.0) * delay * 0.1
                    total_delay = delay + jitter

                    logger.info(
                        f"Retrying {operation_name} in {total_delay:.2f}s "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    await asyncio.sleep(total_delay)

                return await func()

            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                    )
                else:
                    logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")

        raise last_exception

    async def initialize(self) -> None:
        """Initialize the orchestrator and connect to MCP server."""
        if self._initialized:
            logger.debug("Orchestrator already initialized")
            return

        try:
            logger.info("Initializing MCPAgentOrchestrator...")

            # Create MCP client
            logger.info(f"Creating MCP client with transport: {agent_config.mcp_transport.value}")
            self.mcp_client = MCPClientFactory.create_from_config()

            # Connect to MCP server
            logger.info("Connecting to MCP server...")
            await self.mcp_client.connect()

            # Load tools
            logger.info("Loading tools from MCP server...")
            await self._load_tools()

            # Create agent
            logger.info("Creating LlamaIndex agent...")
            await self._create_agent()

            # Initialize semaphore for concurrency control
            self._semaphore = asyncio.Semaphore(agent_config.agent_max_concurrent_requests)

            # Initialize conversation manager if using conversational agent
            if agent_config.agent_type.lower() == "react" and CONVERSATION_SUPPORT:
                logger.info("Initializing ConversationManager...")
                self.conversation_manager = ConversationManager()
                await self.conversation_manager.initialize()
                logger.info("ConversationManager initialized successfully")

            self._initialized = True
            logger.info(
                f"MCPAgentOrchestrator initialized successfully ({len(self.tools)} tools loaded)"
            )

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            await self.shutdown()
            raise

    async def _load_tools(self) -> None:
        """Load tools from MCP server."""
        if self.mcp_client is None:
            raise RuntimeError("MCP client not initialized")

        try:
            logger.debug("Retrieving tools from MCP server...")
            self.tools = await self.mcp_client.get_tools()
            logger.info(f"Loaded {len(self.tools)} tools from MCP server")

            if not self.tools:
                logger.warning("No tools loaded from MCP server")

        except Exception as e:
            logger.error(f"Failed to load tools: {e}")
            raise RuntimeError(f"Tool loading failed: {e}") from e

    async def _create_agent(self) -> None:
        """Create LlamaIndex agent (FunctionAgent or ReActAgent)."""
        if not self.tools:
            raise RuntimeError("No tools available to create agent")

        try:
            agent_type = agent_config.agent_type.lower()
            logger.info(f"Creating LlamaIndex agent (type: {agent_type})...")

            # Get LLM instance
            llm = OpenAI(
                model=agent_config.agent_llm_model,
                temperature=agent_config.agent_llm_temperature,
                api_key=settings.get_api_key(),
            )

            # Get system prompt
            system_prompt = agent_config.agent_system_prompt

            if agent_type == "react":
                # Create ReActAgent with memory
                logger.debug("Creating ReActAgent with memory buffer...")

                memory = None
                if agent_config.agent_memory_enabled:
                    memory = ChatMemoryBuffer.from_defaults(
                        token_limit=agent_config.agent_memory_token_limit
                    )
                    logger.debug(
                        f"Created memory buffer (token_limit={agent_config.agent_memory_token_limit})"
                    )

                self.agent = ReActAgent.from_tools(
                    tools=self.tools,
                    llm=llm,
                    memory=memory,
                    verbose=agent_config.agent_verbose,
                    max_iterations=agent_config.agent_max_iterations,
                    system_prompt=system_prompt,
                )

                logger.info(
                    f"ReActAgent created (memory: {agent_config.agent_memory_enabled}, "
                    f"max_iterations: {agent_config.agent_max_iterations})"
                )

            else:
                # Create FunctionAgent (workflow-based, stateless by default)
                logger.debug("Creating FunctionAgent (workflow-based)...")

                self.agent = FunctionAgent(
                    tools=self.tools,
                    llm=llm,
                    system_prompt=system_prompt,
                    verbose=agent_config.agent_verbose,
                )

                logger.info("FunctionAgent created successfully")

        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise RuntimeError(f"Agent creation failed: {e}") from e

    async def execute_query(self, query: str) -> str:
        """Execute a query using the agent.

        CORRECTED: Uses proper agent.run() for FunctionAgent or agent.chat() for ReActAgent.

        Args:
            query: The query to execute

        Returns:
            The agent's response
        """
        # Validate query
        self._validate_query(query)

        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        if self.agent is None:
            raise RuntimeError("Agent not created")
        if self._semaphore is None:
            raise RuntimeError("Semaphore not initialized")

        try:
            async with self._semaphore:
                if not self._shutdown_event.is_set():
                    raise RuntimeError("Orchestrator is shutting down")

                self._active_requests += 1
                logger.info(
                    f"Executing query: {query[:100]}... (active requests: {self._active_requests})"
                )

                try:

                    async def execute_query_once() -> str:
                        # THE CORRECT WAY: Different agents have different APIs
                        if isinstance(self.agent, ReActAgent):
                            # ReActAgent uses .chat() for stateless or .achat() if exists
                            # Standard API: agent.chat(message)
                            response = self.agent.chat(query)
                            return str(response)

                        elif isinstance(self.agent, FunctionAgent):
                            # FunctionAgent is workflow-based
                            # Uses: await agent.run(user_msg="query")
                            # Returns: dict with 'response' key
                            result = await asyncio.wait_for(
                                self.agent.run(user_msg=query),
                                timeout=agent_config.mcp_request_timeout,
                            )

                            # Extract response from result
                            # FunctionAgent.run() returns a dict-like object
                            if isinstance(result, dict):
                                return str(result.get("response", result))
                            else:
                                # Fallback: convert to string
                                return str(result)

                        else:
                            # Generic fallback
                            response = await asyncio.wait_for(
                                self.agent.run(user_msg=query),
                                timeout=agent_config.mcp_request_timeout,
                            )
                            return str(response)

                    # Execute with retry logic
                    response = await self._retry_with_backoff(
                        execute_query_once, f"query execution: {query[:50]}..."
                    )

                    logger.info("Query executed successfully")
                    return response

                finally:
                    self._active_requests -= 1
                    logger.debug(f"Query completed (active requests: {self._active_requests})")

        except TimeoutError:
            logger.error(f"Query execution timeout after {agent_config.mcp_request_timeout}s")
            raise
        except QueryValidationError:
            raise
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise RuntimeError(f"Query execution failed: {e}") from e

    async def chat(
        self,
        session_id: str,
        user_message: str,
        create_session_if_missing: bool = True,
    ) -> str:
        """Execute a conversational query with session management.

        CORRECTED: Uses proper agent methods based on agent type.
        """
        # Validate query
        self._validate_query(user_message)

        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")
        if not self.conversation_manager:
            raise RuntimeError(
                "Conversation manager not initialized. "
                "Set agent_type='react' in configuration to use chat()."
            )
        if self.agent is None:
            raise RuntimeError("Agent not created")

        try:
            # Get or create session
            session = await self.conversation_manager.get_session(session_id)
            if session is None:
                if create_session_if_missing:
                    logger.info(f"Creating new session: {session_id[:8]}")
                    session = await self.conversation_manager.create_session(session_id=session_id)
                else:
                    raise ValueError(f"Session {session_id} not found")

            # Add user message to session history
            await self.conversation_manager.add_message(
                session_id=session_id, role="user", content=user_message
            )

            # Get conversation history
            history = session.get_chat_history(
                max_messages=agent_config.session_max_history_messages
            )

            logger.info(
                f"Executing chat for session {session_id[:8]} (history: {len(history)} messages)"
            )

            async with self._semaphore:
                if not self._shutdown_event.is_set():
                    raise RuntimeError("Orchestrator is shutting down")

                self._active_requests += 1

                try:

                    async def execute_chat_once() -> str:
                        # Use appropriate method based on agent type
                        if isinstance(self.agent, ReActAgent):
                            # ReActAgent uses .chat() method
                            response = self.agent.chat(user_message)
                            return str(response)

                        elif isinstance(self.agent, FunctionAgent):
                            # FunctionAgent uses .run() method
                            result = await asyncio.wait_for(
                                self.agent.run(user_msg=user_message),
                                timeout=agent_config.mcp_request_timeout,
                            )

                            if isinstance(result, dict):
                                return str(result.get("response", result))
                            return str(result)

                        else:
                            # Generic fallback
                            result = await asyncio.wait_for(
                                self.agent.run(user_msg=user_message),
                                timeout=agent_config.mcp_request_timeout,
                            )
                            return str(result)

                    # Execute with retry logic
                    response = await self._retry_with_backoff(
                        execute_chat_once, f"chat execution: {user_message[:50]}..."
                    )

                    # Add assistant response to session history
                    await self.conversation_manager.add_message(
                        session_id=session_id, role="assistant", content=response
                    )

                    logger.info(f"Chat completed for session {session_id[:8]}")
                    return response

                finally:
                    self._active_requests -= 1

        except TimeoutError:
            logger.error(f"Chat execution timeout after {agent_config.mcp_request_timeout}s")
            raise
        except Exception as e:
            logger.error(f"Chat execution failed: {e}")
            raise RuntimeError(f"Chat execution failed: {e}") from e

    async def execute_queries(self, queries: list[str]) -> list[str]:
        """Execute multiple queries concurrently."""
        self._validate_query_list(queries)

        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        logger.info(f"Executing {len(queries)} queries concurrently...")

        try:
            tasks = [self.execute_query(query) for query in queries]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.error(f"Query {i} failed: {response}")
                    results.append(f"Error: {str(response)}")
                else:
                    results.append(response)

            logger.info(f"Executed {len(queries)} queries successfully")
            return results

        except Exception as e:
            logger.error(f"Concurrent query execution failed: {e}")
            raise RuntimeError(f"Concurrent execution failed: {e}") from e

    async def get_tools_info(self) -> list[dict[str, Any]]:
        """Get information about available tools."""
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        try:
            logger.debug("Retrieving tool information...")
            tools_info = []

            for tool in self.tools:
                # Extract tool name
                tool_name = "unknown"
                if hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
                    tool_name = tool.metadata.name
                elif hasattr(tool, "name"):
                    tool_name = tool.name

                # Extract description
                tool_description = ""
                if hasattr(tool, "metadata") and hasattr(tool.metadata, "description"):
                    tool_description = tool.metadata.description or ""
                elif hasattr(tool, "description"):
                    tool_description = tool.description or ""

                # Extract parameters
                tool_params = {}
                if hasattr(tool, "metadata") and hasattr(tool.metadata, "fn_schema"):
                    schema = tool.metadata.fn_schema
                    if hasattr(schema, "model_json_schema"):
                        tool_params = schema.model_json_schema()
                    elif hasattr(schema, "__dict__"):
                        tool_params = schema.__dict__

                info = {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": tool_params,
                }
                tools_info.append(info)

            logger.info(f"Retrieved information for {len(tools_info)} tools")
            return tools_info

        except Exception as e:
            logger.error(f"Failed to retrieve tool information: {e}")
            raise RuntimeError(f"Tool information retrieval failed: {e}") from e

    async def health_check(self) -> bool:
        """Perform a health check on the orchestrator."""
        try:
            if not self._initialized:
                logger.warning("Orchestrator not initialized")
                return False

            if self.mcp_client is None:
                logger.warning("MCP client not initialized")
                return False

            is_healthy = await self.mcp_client.health_check()

            if is_healthy:
                logger.info("Orchestrator health check passed")
            else:
                logger.warning("Orchestrator health check failed")

            return is_healthy

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def shutdown(self, timeout: float = 30.0) -> None:
        """Shutdown the orchestrator and cleanup resources."""
        try:
            logger.info("Shutting down MCPAgentOrchestrator...")

            self._shutdown_event.clear()
            logger.info("Shutdown signal sent, waiting for active requests...")

            # Wait for active requests
            start_time = asyncio.get_event_loop().time()
            while self._active_requests > 0:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    logger.warning(
                        f"Shutdown timeout after {timeout}s, "
                        f"{self._active_requests} requests still active"
                    )
                    break

                logger.info(f"Waiting for {self._active_requests} active requests...")
                await asyncio.sleep(0.5)

            # Shutdown conversation manager
            if self.conversation_manager is not None:
                await self.conversation_manager.shutdown()

            # Disconnect MCP client
            if self.mcp_client is not None:
                await self.mcp_client.disconnect()

            self.agent = None
            self.tools = []
            self._initialized = False

            self._shutdown_event.set()

            logger.info("MCPAgentOrchestrator shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            self._shutdown_event.set()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# Global orchestrator instance
_orchestrator: MCPAgentOrchestrator | None = None
_orchestrator_lock = asyncio.Lock()


async def get_orchestrator() -> MCPAgentOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator

    async with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = MCPAgentOrchestrator()
            await _orchestrator.initialize()

    return _orchestrator


async def shutdown_orchestrator(timeout: float = 30.0) -> None:
    """Shutdown the global orchestrator instance."""
    global _orchestrator

    async with _orchestrator_lock:
        if _orchestrator is not None:
            await _orchestrator.shutdown(timeout=timeout)
            _orchestrator = None
