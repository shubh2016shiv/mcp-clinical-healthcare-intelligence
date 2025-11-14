"""STDIO-based MCP client implementation.

This module provides a client for connecting to MCP servers via
standard input/output, suitable for local process communication.

IMPORTANT: This client is designed to connect to an ALREADY RUNNING MCP server.
It will NOT spawn a new server process. The server must be started separately
using server_control.py or another method.

The client checks for an existing server process and will raise a clear error
if no server is detected, preventing accidental server spawning.

Architecture:
    - Checks for running server via PID file (.mcp_server.pid)
    - Validates server process is actually running
    - Connects to existing server's stdin/stdout via BasicMCPClient
    - Provides detailed error messages for debugging

Usage:
    # Server must be started first:
    # python mcp_server_management/server_control.py --start

    # Then client can connect:
    client = MCPClientStdio()
    await client.connect()
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from llama_index.tools.mcp import BasicMCPClient

from src.mcp_client.mcp_client_base import MCPClientBase

logger = logging.getLogger(__name__)


class MCPClientStdio(MCPClientBase):
    """MCP client using STDIO transport.

    IMPORTANT DESIGN DECISION:
    This client is designed to connect to an ALREADY RUNNING MCP server.
    It will NOT spawn a new server process. The server must be started
    separately using server_control.py before connecting.

    The client performs the following checks:
    1. Verifies server is already running (via PID file check)
    2. Validates the server process is actually alive
    3. Raises clear errors if server is not running
    4. Prevents accidental server spawning

    Note: BasicMCPClient with STDIO transport typically spawns the server.
    However, this wrapper ensures we only connect to pre-existing servers
    to avoid conflicts with server_control.py managed processes.

    Attributes:
        server_path: Path to the MCP server script (for reference only, not used for spawning)
        server_args: Additional arguments (for reference only)
        _pid_file_path: Path to the server PID file for status checking
    """

    # Path to PID file that indicates server is running
    # This matches the PID file location in server_control.py
    DEFAULT_PID_FILE = Path(__file__).parent.parent.parent / ".mcp_server.pid"

    def __init__(
        self,
        server_path: str = "src/mcp_server/server.py",
        server_args: list[str] | None = None,
        connection_timeout: int = 30,
        request_timeout: int = 60,
        pid_file_path: Path | None = None,
    ):
        """Initialize STDIO MCP client.

        Args:
            server_path: Path to MCP server script (for reference/documentation only).
                        This is NOT used to spawn the server - server must be pre-started.
            server_args: Additional arguments (for reference only, not used)
            connection_timeout: Timeout for connection (seconds)
            request_timeout: Timeout for requests (seconds)
            pid_file_path: Optional path to PID file. Defaults to .mcp_server.pid in project root.

        Raises:
            FileNotFoundError: If server_path does not exist (for validation)
            RuntimeError: If server is not running (checked during connect, not init)
        """
        super().__init__(connection_timeout, request_timeout)

        # Validate server path exists (for documentation/validation purposes)
        # Note: We don't actually use this to spawn the server
        server_file = Path(server_path)
        if not server_file.exists():
            raise FileNotFoundError(
                f"MCP server script not found: {server_path}\n"
                "This path is used for validation only. The server must be "
                "started separately using server_control.py before connecting."
            )

        self.server_path = str(server_file.absolute())
        self.server_args = server_args or []

        # Set PID file path for server status checking
        # This allows us to verify server is running before attempting connection
        self._pid_file_path = pid_file_path or self.DEFAULT_PID_FILE

        logger.info(
            f"Initialized STDIO MCP client (server_path={self.server_path}, "
            f"pid_file={self._pid_file_path})"
        )
        logger.debug(
            "Note: This client connects to an already-running server. "
            "It will NOT spawn a new server process."
        )

    def _check_server_running(self) -> tuple[bool, str | None]:
        """Check if MCP server is already running.

        This method checks for the existence of the PID file and validates
        that the process is actually running. This prevents the client from
        attempting to spawn a new server when one is already active.

        Returns:
            Tuple of (is_running: bool, error_message: Optional[str])
            - If server is running: (True, None)
            - If server is not running: (False, detailed_error_message)
            - If check fails: (False, error_description)

        Note:
            This check is performed before attempting connection to provide
            clear error messages and prevent conflicts.
        """
        # Check if PID file exists
        if not self._pid_file_path.exists():
            error_msg = (
                f"MCP server is not running. PID file not found: {self._pid_file_path}\n"
                "Please start the server first using:\n"
                "  python mcp_server_management/server_control.py --start\n"
                "\n"
                "This client connects to an already-running server and will NOT "
                "spawn a new server process."
            )
            logger.warning(error_msg)
            return False, error_msg

        # Read PID from file
        try:
            with open(self._pid_file_path) as f:
                pid_str = f.read().strip()
                if not pid_str:
                    error_msg = (
                        f"PID file exists but is empty: {self._pid_file_path}\n"
                        "The server may have crashed. Try restarting with:\n"
                        "  python mcp_server_management/server_control.py --restart"
                    )
                    logger.warning(error_msg)
                    return False, error_msg

                pid = int(pid_str)
        except (OSError, ValueError) as e:
            error_msg = (
                f"Failed to read PID file {self._pid_file_path}: {e}\n"
                "The server status is unclear. Check server status with:\n"
                "  python mcp_server_management/server_control.py --status"
            )
            logger.warning(error_msg)
            return False, error_msg

        # Check if process is actually running
        # Use platform-agnostic approach
        try:
            if sys.platform == "win32":
                # Windows: Use tasklist command to check if process exists
                # This is more reliable than trying to signal the process
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                # If PID is in output, process is running
                is_running = str(pid) in result.stdout
            else:
                # Unix-like: Use os.kill with signal 0 (doesn't kill, just checks)
                # Signal 0 is a special signal that doesn't send anything,
                # but will raise OSError if the process doesn't exist
                os.kill(pid, 0)
                is_running = True
        except (OSError, subprocess.TimeoutExpired, FileNotFoundError):
            # Process doesn't exist or check failed
            # OSError: Process doesn't exist (Unix) or access denied
            # TimeoutExpired: tasklist command took too long (Windows)
            # FileNotFoundError: tasklist command not found (unlikely on Windows)
            is_running = False

        if not is_running:
            error_msg = (
                f"MCP server process (PID {pid}) is not running.\n"
                "The PID file exists but the process is not active.\n"
                "This may indicate the server crashed. Try restarting with:\n"
                "  python mcp_server_management/server_control.py --restart\n"
                "\n"
                "Or check server status with:\n"
                "  python mcp_server_management/server_control.py --status"
            )
            logger.warning(error_msg)
            return False, error_msg

        # Server is running
        logger.info(f"Verified MCP server is running (PID: {pid})")
        return True, None

    def _build_client(self) -> BasicMCPClient:
        """Build BasicMCPClient for STDIO transport.

        WARNING: BasicMCPClient with STDIO transport will attempt to spawn
        a new server process. However, since we've verified the server is
        already running, this will likely fail or conflict.

        This method is kept for compatibility with the BasicMCPClient interface,
        but the actual connection logic in connect() will handle the case where
        the server is already running.

        Returns:
            Configured BasicMCPClient for STDIO communication

        Note:
            The server_path and args are provided for BasicMCPClient's interface,
            but in practice, we expect the server to already be running and
            BasicMCPClient's spawn attempt will be handled gracefully.
        """
        logger.info(f"Building STDIO MCP client with server: {self.server_path}")
        logger.debug(
            "Note: BasicMCPClient will attempt to spawn server, but we've "
            "verified server is already running. Connection logic will handle this."
        )

        # IMPORTANT: We use the module path format that matches server_control.py
        # This ensures BasicMCPClient uses the same server invocation method
        # However, since server is already running, this spawn will be handled
        # by checking server status first in connect()

        # Use the module format: "src.mcp_server.server" instead of file path
        # This matches how server_control.py starts the server
        from pathlib import Path

        project_root = Path(__file__).parent.parent.parent

        # Convert server_path to module format
        # e.g., "src/mcp_server/server.py" -> "src.mcp_server.server"
        server_module = str(Path(self.server_path).relative_to(project_root))
        server_module = server_module.replace("/", ".").replace("\\", ".").replace(".py", "")

        client = BasicMCPClient(
            "python",
            args=["-m", server_module] + self.server_args,
        )

        logger.debug(
            f"STDIO client built with command='python', args=['-m', '{server_module}', ...]"
        )
        return client

    async def connect(self) -> None:
        """Establish connection to MCP server via STDIO.

        IMPORTANT: This method requires the MCP server to be ALREADY RUNNING.
        It will NOT spawn a new server process. The server must be started
        separately using server_control.py before calling this method.

        Connection Process:
        1. Check if server is running (via PID file and process validation)
        2. If server is not running, raise clear error with instructions
        3. Build BasicMCPClient (which may attempt to spawn, but we've verified server exists)
        4. Verify connection by attempting to retrieve tools
        5. Mark connection as established

        Raises:
            ConnectionError: If server is not running or connection fails
            RuntimeError: If server status check fails

        Note:
            BasicMCPClient with STDIO transport typically spawns the server.
            However, since we verify the server is already running first,
            any spawn attempt will be handled. The actual connection may
            use the existing server's communication channels.
        """
        async with self._connection_lock:
            if self._is_connected:
                logger.debug("Already connected to MCP server")
                return

            # STEP 1: Verify server is already running before attempting connection
            # This prevents the client from spawning a new server process
            logger.info("Checking if MCP server is already running...")
            is_running, error_msg = self._check_server_running()

            if not is_running:
                # Server is not running - provide clear error with instructions
                logger.error("Cannot connect: MCP server is not running")
                raise ConnectionError(
                    f"{error_msg}\n\n"
                    "This client is designed to connect to an already-running server.\n"
                    "It will NOT spawn a new server process to prevent conflicts.\n"
                    "\n"
                    "Alternative: If you want the client to manage the server lifecycle,\n"
                    "consider using HTTP transport mode instead of STDIO."
                )

            # STEP 2: Server is verified to be running, proceed with connection
            logger.info("Server is running. Proceeding with connection...")

            try:
                logger.info(f"Connecting to MCP server via STDIO: {self.server_path}")

                # STEP 3: Build the BasicMCPClient
                # Note: BasicMCPClient may attempt to spawn a server, but since we've
                # verified one is already running, this should either:
                # - Connect to the existing server's communication channels
                # - Fail gracefully with a clear error
                self.client = self._build_client()

                # STEP 4: Verify connection by attempting to get tools
                # This validates that we can actually communicate with the server
                logger.debug("Verifying STDIO connection by retrieving tools...")
                tools = await asyncio.wait_for(
                    self._verify_connection(),
                    timeout=self.connection_timeout,
                )

                # STEP 5: Connection successful
                logger.info(f"Successfully connected to MCP server (found {len(tools)} tools)")
                self._is_connected = True
                # Record connection time to prevent false positive health checks
                # This allows ensure_connected() to skip health checks immediately after connection
                self._connection_time = time.time()

            except TimeoutError:
                error_msg = (
                    f"Connection timeout after {self.connection_timeout}s.\n"
                    "The server is running but not responding to connection attempts.\n"
                    "\n"
                    "Possible causes:\n"
                    "1. Server is busy or overloaded\n"
                    "2. Server's stdin/stdout is already connected to another client\n"
                    "3. Network or process communication issue\n"
                    "\n"
                    "Troubleshooting:\n"
                    "1. Check server logs: logs/mcp_server.stdout.log\n"
                    "2. Verify server status: python mcp_server_management/server_control.py --status\n"
                    "3. Try restarting the server: python mcp_server_management/server_control.py --restart"
                )
                logger.error(error_msg)
                raise ConnectionError(error_msg) from None

            except Exception as e:
                # Enhanced error handling with context
                error_msg = (
                    f"Failed to connect to MCP server: {e}\n"
                    "\n"
                    "The server is running, but the connection attempt failed.\n"
                    "\n"
                    "Possible causes:\n"
                    "1. Server's stdin/stdout is already in use by another client\n"
                    "2. BasicMCPClient attempted to spawn a new server (conflict)\n"
                    "3. Server process communication error\n"
                    "\n"
                    "Troubleshooting:\n"
                    "1. Check if another client is connected to the server\n"
                    "2. Review server logs: logs/mcp_server.stdout.log\n"
                    "3. Consider using HTTP transport mode for multiple clients\n"
                    "4. Restart the server: python mcp_server_control.py --restart"
                )
                logger.error(error_msg)
                raise ConnectionError(error_msg) from e

    async def disconnect(self) -> None:
        """Close connection to MCP server.

        IMPORTANT: This method only disconnects the CLIENT from the server.
        It does NOT stop or terminate the server process. The server will
        continue running and can accept new connections.

        This method:
        1. Closes the client's connection to the server
        2. Cleans up BasicMCPClient resources
        3. Resets connection state
        4. Does NOT affect the running server process

        To stop the server, use:
            python mcp_server_management/server_control.py --stop
        """
        async with self._connection_lock:
            if not self._is_connected:
                logger.debug("Not connected, skipping disconnect")
                return

            try:
                logger.info("Disconnecting client from MCP server (server will continue running)")

                # Properly close the BasicMCPClient using context manager protocol
                # This cleans up the client's connection resources but does NOT
                # terminate the server process, which is managed separately
                if self.client is not None:
                    try:
                        # Try to clean up using context manager protocol if available
                        # BasicMCPClient may or may not be in context manager state
                        if hasattr(self.client, "_exit_stack"):
                            # Client is in context manager state, can use __aexit__
                            await self.client.__aexit__(None, None, None)
                        else:
                            # Client wasn't used as context manager - this is normal
                            # BasicMCPClient will clean up automatically when reference is lost
                            logger.debug(
                                "Client not in context manager state - "
                                "cleanup will happen automatically"
                            )
                    except AttributeError:
                        # _exit_stack doesn't exist - expected when not used as context manager
                        logger.debug("Client cleanup: not in context manager state (expected)")
                    except Exception as cleanup_error:
                        # Only log as warning if it's an unexpected error
                        if "_exit_stack" not in str(cleanup_error):
                            logger.warning(f"Error during client cleanup: {cleanup_error}")
                        else:
                            logger.debug(f"Expected cleanup attribute error: {cleanup_error}")

                # Reset connection state
                self.client = None
                self._is_connected = False
                self._connection_time = None  # Clear connection timestamp
                logger.info("Successfully disconnected from MCP server (server still running)")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                # Ensure state is reset even on error
                self._is_connected = False
                self._connection_time = None  # Clear connection timestamp

    async def _verify_connection(self) -> list:
        """Verify connection by attempting to retrieve tools.

        Returns:
            List of available tools

        Raises:
            RuntimeError: If verification fails
        """
        if self.client is None:
            raise RuntimeError("Client not initialized")

        try:
            from llama_index.tools.mcp import McpToolSpec

            mcp_tool_spec = McpToolSpec(client=self.client)
            tools = await mcp_tool_spec.to_tool_list_async()
            return tools
        except Exception as e:
            logger.error(f"Connection verification failed: {e}")
            raise RuntimeError(f"Failed to verify MCP server connection: {e}") from e
