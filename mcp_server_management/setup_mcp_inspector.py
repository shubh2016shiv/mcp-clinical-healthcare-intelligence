"""Setup script for MCP Inspector debugging tool.

This script helps set up and run MCP Inspector for debugging the MCP server.
MCP Inspector is the official tool from the Model Context Protocol team.

Features:
- Cross-platform support (Windows, macOS, Linux)
- Robust process handling with proper shell resolution
- Comprehensive error handling and logging
- Interactive and CLI modes
"""

import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project configuration
PROJECT_ROOT = Path(__file__).parent.parent
PLATFORM = platform.system()
IS_WINDOWS = PLATFORM == "Windows"

# Platform-specific command configurations
NPX_COMMANDS = ["npx.cmd", "npx"] if IS_WINDOWS else ["npx"]
NODE_COMMANDS = ["node.exe", "node"] if IS_WINDOWS else ["node"]


class CommandExecutor:
    """Handles cross-platform command execution."""

    @staticmethod
    def run_command(
        command: list[str],
        capture_output: bool = True,
        check: bool = True,
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> subprocess.CompletedProcess:
        """
        Execute a command with cross-platform compatibility.

        Args:
            command: Command and arguments to execute
            capture_output: Whether to capture stdout/stderr
            check: Whether to raise exception on non-zero exit
            cwd: Working directory for command execution
            timeout: Command timeout in seconds

        Returns:
            CompletedProcess instance

        Raises:
            subprocess.CalledProcessError: If command fails and check=True
            subprocess.TimeoutExpired: If command exceeds timeout
        """
        try:
            logger.debug(f"Executing command: {' '.join(command)}")
            logger.debug(f"Working directory: {cwd or os.getcwd()}")

            result = subprocess.run(
                command,
                capture_output=capture_output,
                text=True,
                check=check,
                cwd=str(cwd) if cwd else None,
                shell=IS_WINDOWS,  # Use shell on Windows for PATH resolution
                timeout=timeout,
            )

            logger.debug(f"Command succeeded with exit code: {result.returncode}")
            return result

        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with exit code {e.returncode}: {' '.join(command)}")
            if e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
            raise
        except FileNotFoundError:
            logger.error(f"Command not found: {command[0]}")
            raise


class PrerequisiteChecker:
    """Checks system prerequisites for running MCP Inspector."""

    @staticmethod
    def check_node() -> tuple[bool, str | None]:
        """
        Check if Node.js is installed.

        Returns:
            Tuple of (is_installed, version_string)
        """
        for cmd in NODE_COMMANDS:
            try:
                result = CommandExecutor.run_command([cmd, "--version"], timeout=5)
                version = result.stdout.strip()
                logger.info(f"✓ Node.js installed: {version}")
                return True, version
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue

        logger.error("✗ Node.js not found")
        logger.error("  Please install Node.js from: https://nodejs.org/")
        return False, None

    @staticmethod
    def check_npx() -> tuple[bool, str | None]:
        """
        Check if npx is installed (comes with Node.js).

        Returns:
            Tuple of (is_installed, version_string)
        """
        for cmd in NPX_COMMANDS:
            try:
                result = CommandExecutor.run_command([cmd, "--version"], timeout=5)
                version = result.stdout.strip()
                logger.info(f"✓ npx installed: {version}")
                return True, version
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue

        # NPX comes with Node.js, so if Node works, npx should work
        # Issue a warning but don't fail completely
        logger.warning("⚠ Could not verify npx, but will attempt to use it (Node.js is installed)")
        logger.warning("  If npx fails, try reinstalling Node.js from: https://nodejs.org/")
        return True, None  # Return True to continue

    @staticmethod
    def check_python_version() -> bool:
        """Check if Python version is compatible."""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            logger.error(
                f"✗ Python 3.8+ required, found {version.major}.{version.minor}.{version.micro}"
            )
            return False
        logger.info(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True

    @staticmethod
    def check_project_structure() -> bool:
        """Verify project structure exists."""
        server_path = PROJECT_ROOT / "src" / "mcp_server" / "server.py"
        if not server_path.exists():
            logger.error(f"✗ MCP server not found at: {server_path}")
            logger.error("  Please ensure you're running this script from the correct directory")
            return False
        logger.info(f"✓ MCP server found at: {server_path}")
        return True

    @classmethod
    def check_all(cls) -> bool:
        """
        Run all prerequisite checks.

        Returns:
            True if all checks pass, False otherwise
        """
        logger.info("=" * 80)
        logger.info("Checking prerequisites...")
        logger.info("=" * 80)

        checks = [
            cls.check_python_version(),
            cls.check_project_structure(),
            cls.check_node()[0],
            cls.check_npx()[0],
        ]

        if all(checks):
            logger.info("\n✓ All prerequisites satisfied\n")
            return True
        else:
            logger.error("\n✗ Some prerequisites failed. Please fix the issues above.\n")
            return False


class MCPInspector:
    """Manages MCP Inspector execution in different modes."""

    def __init__(self):
        self.project_root = PROJECT_ROOT

    def get_npx_command(self) -> str:
        """Get the appropriate npx command for the platform."""
        return NPX_COMMANDS[0]

    def run_stdio_mode(self) -> int:
        """
        Run MCP Inspector in STDIO mode.

        Returns:
            Exit code (0 for success)
        """
        logger.info("=" * 80)
        logger.info("Starting MCP Inspector in STDIO mode")
        logger.info("=" * 80)
        logger.info("\nThis will start the inspector and connect to your MCP server.")
        logger.info("The inspector will open in your browser automatically.\n")

        server_command = ["python", "-m", "src.mcp_server.server"]

        inspector_command = [
            self.get_npx_command(),
            "--yes",
            "@modelcontextprotocol/inspector",
        ] + server_command

        logger.info(f"Running: {' '.join(inspector_command)}")
        logger.info(f"Working directory: {self.project_root}")
        logger.info("\nPress Ctrl+C to stop the inspector\n")

        try:
            result = CommandExecutor.run_command(
                inspector_command,
                capture_output=False,  # Show output in real-time
                cwd=self.project_root,
                timeout=None,  # No timeout for interactive session
            )
            return result.returncode
        except KeyboardInterrupt:
            logger.info("\n\nInspector stopped by user")
            return 0
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ Inspector failed with exit code: {e.returncode}")
            return e.returncode
        except Exception as e:
            logger.error(f"\n✗ Unexpected error: {e}")
            return 1

    def run_http_mode(self, host: str = "127.0.0.1", port: int = 8000) -> int:
        """
        Run MCP Inspector in HTTP mode.

        Args:
            host: Server host address
            port: Server port number

        Returns:
            Exit code (0 for success)
        """
        logger.info("=" * 80)
        logger.info("Starting MCP Inspector in HTTP mode")
        logger.info("=" * 80)
        logger.info(f"\nMake sure your MCP server is running in HTTP mode on {host}:{port}")
        logger.info("Start the server with:")

        if IS_WINDOWS:
            logger.info("  set MCP_TRANSPORT=http")
            logger.info(f"  set MCP_PORT={port}")
        else:
            logger.info("  export MCP_TRANSPORT=http")
            logger.info(f"  export MCP_PORT={port}")

        logger.info("  python -m src.mcp_server.server\n")

        try:
            input("Press Enter when your server is running...")
        except KeyboardInterrupt:
            logger.info("\n\nOperation cancelled by user")
            return 0

        inspector_command = [
            self.get_npx_command(),
            "--yes",
            "@modelcontextprotocol/inspector",
            f"http://{host}:{port}",
        ]

        logger.info(f"\nRunning: {' '.join(inspector_command)}")
        logger.info("The inspector will open in your browser automatically.")
        logger.info("\nPress Ctrl+C to stop the inspector\n")

        try:
            result = CommandExecutor.run_command(
                inspector_command,
                capture_output=False,  # Show output in real-time
                cwd=self.project_root,
                timeout=None,  # No timeout for interactive session
            )
            return result.returncode
        except KeyboardInterrupt:
            logger.info("\n\nInspector stopped by user")
            return 0
        except subprocess.CalledProcessError as e:
            logger.error(f"\n✗ Inspector failed with exit code: {e.returncode}")
            return e.returncode
        except Exception as e:
            logger.error(f"\n✗ Unexpected error: {e}")
            return 1


def interactive_mode() -> int:
    """
    Run in interactive mode with user prompts.

    Returns:
        Exit code
    """
    print("=" * 80)
    print("MCP Inspector Setup and Launcher")
    print("=" * 80)
    print("\nMCP Inspector is the official tool for debugging MCP servers.")
    print("It provides an interactive GUI to test tools, resources, and prompts.\n")

    # Check prerequisites
    if not PrerequisiteChecker.check_all():
        return 1

    # Mode selection
    print("=" * 80)
    print("Select mode:")
    print("=" * 80)
    print("1. STDIO mode (default, recommended for development)")
    print("   - Inspector manages the server process")
    print("   - Easier to set up")
    print("   - Single command to run")
    print()
    print("2. HTTP mode (for testing running server)")
    print("   - Connect to already-running server")
    print("   - Useful for production debugging")
    print("   - Requires server to be running separately")
    print()

    try:
        choice = input("Enter choice (1 or 2, default: 1): ").strip() or "1"
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        return 0

    inspector = MCPInspector()

    if choice == "1":
        return inspector.run_stdio_mode()
    elif choice == "2":
        try:
            host = input("Enter server host (default: 127.0.0.1): ").strip() or "127.0.0.1"
            port_input = input("Enter server port (default: 8000): ").strip() or "8000"
            try:
                port = int(port_input)
            except ValueError:
                logger.warning(f"Invalid port: {port_input}, using default 8000")
                port = 8000
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            return 0

        return inspector.run_http_mode(host, port)
    else:
        logger.error(f"Invalid choice: {choice}")
        return 1


def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="MCP Inspector Setup and Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python setup_mcp_inspector.py

  # STDIO mode (non-interactive)
  python setup_mcp_inspector.py --mode stdio

  # HTTP mode with custom host and port
  python setup_mcp_inspector.py --mode http --host 0.0.0.0 --port 8080

  # Verbose logging
  python setup_mcp_inspector.py --mode stdio --verbose
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["stdio", "http"],
        help="Inspector mode (stdio or http). If not specified, runs in interactive mode.",
    )

    parser.add_argument(
        "--host", default="127.0.0.1", help="Server host for HTTP mode (default: 127.0.0.1)"
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Server port for HTTP mode (default: 8000)"
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--check-only", action="store_true", help="Only check prerequisites, don't run inspector"
    )

    return parser


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug(f"Platform: {PLATFORM}")
        logger.debug(f"Python version: {sys.version}")
        logger.debug(f"Project root: {PROJECT_ROOT}")

    # Check prerequisites
    if not PrerequisiteChecker.check_all():
        return 1

    # If check-only mode, exit after checks
    if args.check_only:
        logger.info("✓ All checks passed. Ready to run inspector.")
        return 0

    inspector = MCPInspector()

    # Run in specified mode or interactive mode
    if args.mode == "stdio":
        return inspector.run_stdio_mode()
    elif args.mode == "http":
        return inspector.run_http_mode(args.host, args.port)
    else:
        # Interactive mode
        return interactive_mode()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
