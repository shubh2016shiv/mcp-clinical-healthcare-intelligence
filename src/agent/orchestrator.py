"""MCP Agent Orchestrator.

This module provides the main orchestrator for managing MCP server tools
with LlamaIndex agents. It handles:
- MCP client connection and lifecycle management
- Tool loading and caching
- Agent creation and execution
- Concurrent request handling
- Error handling and logging

The orchestrator supports both STDIO and HTTP transports and provides
a clean async API for agent interactions.
"""

import asyncio
import logging
import random
from collections.abc import Callable
from typing import Any, TypeVar

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI

from src.config.settings import settings

from .config import agent_config
from .mcp_client_base import MCPClientBase
from .mcp_client_factory import MCPClientFactory

logger = logging.getLogger(__name__)

T = TypeVar("T")


class QueryValidationError(ValueError):
    """Raised when query validation fails."""

    pass


class MCPAgentOrchestrator:
    """Orchestrator for MCP agent with LlamaIndex.

    This class manages the lifecycle of an MCP agent, including:
    - Connection to MCP server
    - Tool loading and caching
    - Agent creation and configuration
    - Request execution and concurrency management

    Attributes:
        mcp_client: The MCP client instance
        agent: The LlamaIndex FunctionAgent instance
        tools: Cached list of available tools
    """

    # Query validation constants
    MAX_QUERY_LENGTH = 10000  # Maximum characters per query
    MAX_CONCURRENT_QUERIES = 50  # Maximum queries in batch execution

    # Retry configuration
    MAX_RETRIES = 3  # Maximum number of retries
    BASE_RETRY_DELAY = 1.0  # Base delay in seconds
    MAX_RETRY_DELAY = 10.0  # Maximum delay between retries
    RETRY_BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier

    def __init__(self):
        """Initialize the MCP agent orchestrator."""
        self.mcp_client: MCPClientBase | None = None
        self.agent: FunctionAgent | None = None
        self.tools: list[Any] = []
        self._initialized = False
        self._semaphore: asyncio.Semaphore | None = None
        self._active_requests = 0  # Counter for active requests
        self._shutdown_event = asyncio.Event()  # Event to signal shutdown
        self._shutdown_event.set()  # Initially not shutting down

        logger.info("Initialized MCPAgentOrchestrator")

    def _validate_query(self, query: str) -> None:
        """Validate a single query.

        Args:
            query: The query to validate

        Raises:
            QueryValidationError: If validation fails
        """
        if not isinstance(query, str):
            raise QueryValidationError(f"Query must be a string, got {type(query)}")

        if not query.strip():
            raise QueryValidationError("Query cannot be empty or whitespace only")

        if len(query) > self.MAX_QUERY_LENGTH:
            raise QueryValidationError(
                f"Query too long: {len(query)} characters " f"(maximum: {self.MAX_QUERY_LENGTH})"
            )

    def _validate_query_list(self, queries: list[str]) -> None:
        """Validate a list of queries.

        Args:
            queries: The list of queries to validate

        Raises:
            QueryValidationError: If validation fails
        """
        if not isinstance(queries, list):
            raise QueryValidationError(f"Queries must be a list, got {type(queries)}")

        if not queries:
            raise QueryValidationError("Query list cannot be empty")

        if len(queries) > self.MAX_CONCURRENT_QUERIES:
            raise QueryValidationError(
                f"Too many queries: {len(queries)} " f"(maximum: {self.MAX_CONCURRENT_QUERIES})"
            )

        # Validate each query
        for i, query in enumerate(queries):
            try:
                self._validate_query(query)
            except QueryValidationError as e:
                raise QueryValidationError(f"Query {i}: {e}") from e

    async def _retry_with_backoff(
        self,
        func: Callable[[], T],
        operation_name: str,
        max_retries: int | None = None,
        base_delay: float | None = None,
    ) -> T:
        """Execute a function with exponential backoff retry logic.

        Args:
            func: Async function to execute
            operation_name: Name of the operation for logging
            max_retries: Maximum number of retries (default: self.MAX_RETRIES)
            base_delay: Base delay in seconds (default: self.BASE_RETRY_DELAY)

        Returns:
            Result of the function call

        Raises:
            The last exception encountered if all retries fail
        """
        if max_retries is None:
            max_retries = self.MAX_RETRIES
        if base_delay is None:
            base_delay = self.BASE_RETRY_DELAY

        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        base_delay * (self.RETRY_BACKOFF_MULTIPLIER ** (attempt - 1)),
                        self.MAX_RETRY_DELAY,
                    )
                    # Add jitter to prevent thundering herd
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

        # All retries failed, raise the last exception
        raise last_exception

    async def initialize(self) -> None:
        """Initialize the orchestrator and connect to MCP server.

        This method must be called before using the orchestrator.
        It connects to the MCP server, loads tools, and creates the agent.

        Raises:
            ConnectionError: If MCP server connection fails
            RuntimeError: If agent creation fails
        """
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

            self._initialized = True
            logger.info(
                f"MCPAgentOrchestrator initialized successfully "
                f"({len(self.tools)} tools loaded)"
            )

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            await self.shutdown()
            raise

    async def _load_tools(self) -> None:
        """Load tools from MCP server.

        Raises:
            RuntimeError: If tool loading fails
        """
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
        """Create LlamaIndex FunctionAgent.

        Raises:
            RuntimeError: If agent creation fails
        """
        if not self.tools:
            raise RuntimeError("No tools available to create agent")

        try:
            logger.debug("Creating LlamaIndex FunctionAgent...")

            # Get LLM instance
            llm = OpenAI(
                model=agent_config.agent_llm_model,
                temperature=agent_config.agent_llm_temperature,
                api_key=settings.get_api_key(),
            )

            # Get system prompt
            system_prompt = agent_config.agent_system_prompt

            # Create agent
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

        Args:
            query: The query to execute

        Returns:
            The agent's response

        Raises:
            QueryValidationError: If query validation fails
            RuntimeError: If orchestrator not initialized or execution fails
            asyncio.TimeoutError: If execution exceeds timeout
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
                # Check if shutdown is in progress
                if not self._shutdown_event.is_set():
                    raise RuntimeError("Orchestrator is shutting down")

                # Track active request
                self._active_requests += 1
                logger.info(
                    f"Executing query: {query[:100]}... (active requests: {self._active_requests})"
                )

                try:
                    # Define the query execution function for retry logic
                    async def execute_query_once() -> str:
                        # Run agent with timeout
                        response = await asyncio.wait_for(
                            self.agent.arun(query),
                            timeout=agent_config.mcp_request_timeout,
                        )
                        return str(response)

                    # Execute with retry logic for transient failures
                    response = await self._retry_with_backoff(
                        execute_query_once, f"query execution: {query[:50]}..."
                    )

                    logger.info("Query executed successfully")
                    return response

                finally:
                    # Decrement active request counter
                    self._active_requests -= 1
                    logger.debug(f"Query completed (active requests: {self._active_requests})")

        except TimeoutError:
            logger.error(f"Query execution timeout after {agent_config.mcp_request_timeout}s")
            raise
        except QueryValidationError:
            # Don't retry validation errors
            raise
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise RuntimeError(f"Query execution failed: {e}") from e

    async def execute_queries(self, queries: list[str]) -> list[str]:
        """Execute multiple queries concurrently.

        Args:
            queries: List of queries to execute

        Returns:
            List of responses corresponding to each query

        Raises:
            QueryValidationError: If query list validation fails
            RuntimeError: If orchestrator not initialized
        """
        # Validate query list
        self._validate_query_list(queries)

        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        logger.info(f"Executing {len(queries)} queries concurrently...")

        try:
            # Note: Individual query validation is handled by execute_query
            tasks = [self.execute_query(query) for query in queries]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in responses
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
        """Get information about available tools.

        Returns:
            List of tool information dictionaries

        Raises:
            RuntimeError: If orchestrator not initialized
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized. Call initialize() first.")

        try:
            logger.debug("Retrieving tool information...")
            tools_info = []

            for tool in self.tools:
                # Extract tool name - FunctionTool from LlamaIndex stores name in metadata
                # Try metadata.name first (most common for LlamaIndex FunctionTool)
                tool_name = None
                if hasattr(tool, "metadata") and hasattr(tool.metadata, "name"):
                    tool_name = tool.metadata.name
                else:
                    # Fallback to other possible attribute names
                    tool_name = (
                        getattr(tool, "name", None)
                        or getattr(tool, "fn_name", None)
                        or getattr(tool, "tool_name", None)
                        or getattr(tool, "__name__", None)
                        or (str(tool).split("(")[0] if "(" in str(tool) else None)
                        or "unknown"
                    )

                # Extract description - FunctionTool stores description in metadata
                tool_description = None
                if hasattr(tool, "metadata") and hasattr(tool.metadata, "description"):
                    tool_description = tool.metadata.description
                else:
                    # Fallback to other possible attribute names
                    tool_description = (
                        getattr(tool, "description", None)
                        or getattr(tool, "fn_description", None)
                        or getattr(tool, "tool_description", None)
                        or getattr(tool, "__doc__", None)
                        or ""
                    )

                # Extract parameters/schema - FunctionTool stores schema in metadata
                tool_params = None
                if hasattr(tool, "metadata") and hasattr(tool.metadata, "fn_schema"):
                    # Convert schema to dict if it's a Pydantic model
                    schema = tool.metadata.fn_schema
                    if hasattr(schema, "model_json_schema"):
                        tool_params = schema.model_json_schema()
                    elif hasattr(schema, "__dict__"):
                        tool_params = schema.__dict__
                    else:
                        tool_params = str(schema)
                else:
                    # Fallback to other possible attribute names
                    tool_params = (
                        getattr(tool, "fn_schema", None)
                        or getattr(tool, "parameters", None)
                        or getattr(tool, "schema", None)
                        or {}
                    )

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
        """Perform a health check on the orchestrator.

        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self._initialized:
                logger.warning("Orchestrator not initialized")
                return False

            if self.mcp_client is None:
                logger.warning("MCP client not initialized")
                return False

            # Check MCP server health
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
        """Shutdown the orchestrator and cleanup resources.

        This method waits for active requests to complete before shutting down.

        Args:
            timeout: Maximum time to wait for active requests (seconds)

        This method should be called when the orchestrator is no longer needed.
        """
        try:
            logger.info("Shutting down MCPAgentOrchestrator...")

            # Signal that shutdown is in progress
            self._shutdown_event.clear()
            logger.info("Shutdown signal sent, waiting for active requests to complete...")

            # Wait for active requests to complete with timeout
            start_time = asyncio.get_event_loop().time()
            while self._active_requests > 0:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    logger.warning(
                        f"Shutdown timeout reached after {timeout}s, "
                        f"{self._active_requests} requests still active"
                    )
                    break

                logger.info(f"Waiting for {self._active_requests} active requests to complete...")
                await asyncio.sleep(0.5)

            # Disconnect MCP client
            if self.mcp_client is not None:
                await self.mcp_client.disconnect()

            self.agent = None
            self.tools = []
            self._initialized = False

            # Signal shutdown complete
            self._shutdown_event.set()

            logger.info("MCPAgentOrchestrator shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            # Ensure shutdown event is set even on error
            self._shutdown_event.set()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# Global orchestrator instance and initialization lock
_orchestrator: MCPAgentOrchestrator | None = None
_orchestrator_lock = asyncio.Lock()


async def get_orchestrator() -> MCPAgentOrchestrator:
    """Get or create the global orchestrator instance.

    Returns:
        The global MCPAgentOrchestrator instance

    Example:
        >>> orchestrator = await get_orchestrator()
        >>> response = await orchestrator.execute_query("Find patients...")
    """
    global _orchestrator

    # Use lock to prevent race conditions during initialization
    async with _orchestrator_lock:
        if _orchestrator is None:
            _orchestrator = MCPAgentOrchestrator()
            await _orchestrator.initialize()

    return _orchestrator


async def shutdown_orchestrator(timeout: float = 30.0) -> None:
    """Shutdown the global orchestrator instance.

    Args:
        timeout: Maximum time to wait for active requests (seconds)

    Example:
        >>> await shutdown_orchestrator()
    """
    global _orchestrator

    # Use lock to prevent race conditions during shutdown
    async with _orchestrator_lock:
        if _orchestrator is not None:
            await _orchestrator.shutdown(timeout=timeout)
            _orchestrator = None
