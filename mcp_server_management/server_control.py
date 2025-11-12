"""Utility script to start and stop the MCP server.

This script manages the healthcare MCP server process using the local
virtual environment Python interpreter.

Supports two transport modes:
- STDIO (default): Runs with a persistent wrapper process that maintains stdin
- HTTP: Runs as a standalone daemon on specified host:port

Usage:
    python server_control.py --start                    # Start in STDIO mode
    python server_control.py --start --http             # Start in HTTP mode
    python server_control.py --start --http --port 8080 # Custom port
    python server_control.py --stop                     # Stop the server
    python server_control.py --restart                  # Restart the server
    python server_control.py --status                   # Check server status
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Import validation functions
# Add project root to path for validate_server imports
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from mcp_server_management.validate_server import run_integrity_checks
except ImportError:
    # Fallback if import fails
    run_integrity_checks = None
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
SERVER_MODULE = "src.mcp_server.server"
PID_FILE = PROJECT_ROOT / ".mcp_server.pid"
CONFIG_FILE = PROJECT_ROOT / ".mcp_server.config"
INFRASTRUCTURE_SCRIPT = PROJECT_ROOT / "infrastructure" / "manage_infrastructure.py"


def get_python_executable():
    r"""Get the Python executable path from .venv\Scripts (required)."""
    if not VENV_PYTHON.exists():
        print(f"[ERROR] Virtual environment Python not found at: {VENV_PYTHON}")
        print("   Please ensure .venv is set up and activated")
        sys.exit(1)
    return str(VENV_PYTHON)


def save_server_config(mode, host=None, port=None):
    """Save server configuration to file.

    Args:
        mode: Transport mode ('stdio' or 'http')
        host: HTTP host (only for HTTP mode)
        port: HTTP port (only for HTTP mode)
    """
    config = {"mode": mode, "host": host, "port": port}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)


def load_server_config():
    """Load server configuration from file.

    Returns:
        dict: Configuration dict or None if file doesn't exist
    """
    if not CONFIG_FILE.exists():
        return None
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def is_server_running():
    """Check if the server process is running.

    Uses reliable process checking methods for each platform:
    - Windows: Uses Get-Process PowerShell cmdlet (more reliable than tasklist)
    - Unix: Uses os.kill with signal 0 (standard POSIX method)
    """
    if not PID_FILE.exists():
        return False

    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())

        # Check if process exists (Windows)
        if sys.platform == "win32":
            try:
                # Use PowerShell Get-Process for more reliable checking
                # This is more accurate than tasklist which can have parsing issues
                result = subprocess.run(
                    [
                        "powershell",
                        "-Command",
                        f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5,
                )
                # If process exists, Get-Process returns the PID
                return result.returncode == 0 and str(pid) in result.stdout.strip()
            except (subprocess.TimeoutExpired, Exception):
                # Fallback to tasklist if PowerShell fails
                try:
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV"],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=5,
                    )
                    # Check if PID appears in output (more reliable with CSV format)
                    return result.returncode == 0 and f'"{pid}"' in result.stdout
                except Exception:
                    return False
        else:
            # Unix-like systems: use os.kill with signal 0
            try:
                os.kill(pid, 0)  # Signal 0 doesn't kill, just checks existence
                return True
            except ProcessLookupError:
                # Process doesn't exist
                return False
            except PermissionError:
                # Process exists but we don't have permission (still means it's running)
                return True
            except OSError:
                # Other OS error - assume process doesn't exist
                return False
    except (ValueError, FileNotFoundError):
        return False


def check_mongodb_connectivity(host="localhost", port=27017, timeout=2) -> bool:
    """Check if MongoDB is accessible on the specified host and port.

    Args:
        host: MongoDB host (default: localhost)
        port: MongoDB port (default: 27017)
        timeout: Connection timeout in seconds (default: 2)

    Returns:
        bool: True if MongoDB is accessible, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def start_infrastructure() -> bool:
    """Start the infrastructure (MongoDB, Redis) using the infrastructure management script.

    Returns:
        bool: True if infrastructure started successfully, False otherwise
    """
    if not INFRASTRUCTURE_SCRIPT.exists():
        print(f"[WARNING] Infrastructure script not found at: {INFRASTRUCTURE_SCRIPT}")
        print("[INFO]  Skipping infrastructure auto-start")
        return False

    print("[INFO]  MongoDB is not available")
    print("[INFO]  Starting infrastructure (MongoDB, Redis)...")
    print("-" * 60)

    try:
        # Run the infrastructure management script from the infrastructure directory
        # The script expects to be run from the directory containing docker-compose.yml
        infra_dir = INFRASTRUCTURE_SCRIPT.parent
        result = subprocess.run(
            [sys.executable, str(INFRASTRUCTURE_SCRIPT), "--start"],
            cwd=str(infra_dir),
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode == 0:
            print("-" * 60)
            print("[SUCCESS] Infrastructure started successfully")
            return True
        else:
            print("-" * 60)
            print(f"[ERROR] Infrastructure startup failed (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("-" * 60)
        print("[ERROR] Infrastructure startup timed out")
        return False
    except Exception as e:
        print("-" * 60)
        print(f"[ERROR] Failed to start infrastructure: {e}")
        return False


def wait_for_mongodb(host="localhost", port=27017, max_wait=120, check_interval=2) -> bool:
    """Wait for MongoDB to become available.

    Args:
        host: MongoDB host (default: localhost)
        port: MongoDB port (default: 27017)
        max_wait: Maximum time to wait in seconds (default: 120)
        check_interval: Time between checks in seconds (default: 2)

    Returns:
        bool: True if MongoDB becomes available, False if timeout
    """
    print(f"[INFO]  Waiting for MongoDB to become available on {host}:{port}...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        if check_mongodb_connectivity(host, port):
            elapsed = int(time.time() - start_time)
            print(f"[SUCCESS] MongoDB is now available (waited {elapsed}s)")
            return True

        # Show progress every 10 seconds
        elapsed = int(time.time() - start_time)
        if elapsed % 10 == 0 and elapsed > 0:
            print(f"[INFO]  Still waiting for MongoDB... ({elapsed}s elapsed)")

        time.sleep(check_interval)

    print(f"[ERROR] MongoDB did not become available within {max_wait}s")
    return False


def ensure_mongodb_available() -> bool:
    """Ensure MongoDB is available, starting infrastructure if needed.

    This function:
    1. Checks if MongoDB is already available
    2. If not, starts the infrastructure
    3. Waits for MongoDB to become available

    Returns:
        bool: True if MongoDB is available, False otherwise
    """
    print("[INFO]  Checking MongoDB availability...")

    # First check if MongoDB is already available
    if check_mongodb_connectivity():
        print("[SUCCESS] MongoDB is already available")
        return True

    # MongoDB is not available, try to start infrastructure
    print("[INFO]  MongoDB is not available")

    if not start_infrastructure():
        print("[ERROR] Failed to start infrastructure")
        print("[INFO]  You can manually start infrastructure with:")
        print(f"        python {INFRASTRUCTURE_SCRIPT} --start")
        return False

    # Wait for MongoDB to become available
    if not wait_for_mongodb():
        print("[ERROR] MongoDB did not become available after starting infrastructure")
        print("[INFO]  Please check infrastructure logs and try again")
        return False

    return True


def stdio_keepalive_thread(process):
    """Background thread that keeps the STDIO server alive.

    This thread maintains the stdin connection to prevent EOF, allowing
    the STDIO-based MCP server to run as a daemon. It monitors the process
    and exits when the process terminates.

    Args:
        process: subprocess.Popen object of the server process
    """
    try:
        # Keep stdin open and monitor process health
        while True:
            # Check if process is still running
            poll_result = process.poll()
            if poll_result is not None:
                # Process has exited
                print(f"[WARNING] MCP server process exited with code: {poll_result}")
                break

            # Sleep to avoid busy-waiting
            time.sleep(1)
    except Exception as e:
        print(f"[ERROR] Keepalive thread error: {e}")
    finally:
        # Ensure stdin is closed when thread exits
        try:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        except Exception:
            pass


def start_server_stdio():
    """Start the MCP server in STDIO mode with persistent wrapper.

    STDIO mode requires a persistent parent process to maintain the stdin connection.
    This function starts the server and spawns a background thread to keep it alive.
    The wrapper process (this script) continues running as a daemon.

    Returns:
        bool: True if server started successfully, False otherwise
    """
    if is_server_running():
        print("[ERROR] MCP server is already running")
        return False

    # Ensure MongoDB is available before starting server
    print()
    if not ensure_mongodb_available():
        print("[ERROR] Cannot start MCP server: MongoDB is not available")
        return False
    print()

    # Run integrity validation checks
    print("[INFO]  Running server integrity validation...")
    print("-" * 60)
    if run_integrity_checks is None:
        print("[WARNING] Validation module not available, skipping checks...")
    else:
        validation_passed = run_integrity_checks()
        print("-" * 60)
        if not validation_passed:
            print("[ERROR] Server integrity validation failed")
            print("[ERROR] Cannot start MCP server: Validation checks did not pass")
            print("[INFO]  Please fix the failed checks before starting the server")
            return False
        print("[SUCCESS] All integrity checks passed")
    print()

    python_exe = get_python_executable()
    print(f"[START] Starting MCP server in STDIO mode using: {python_exe}")
    print(f"[DIR]   Working directory: {PROJECT_ROOT}")
    print("[INFO]  Mode: STDIO transport (persistent wrapper)")

    try:
        # Prepare log files for server output
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout_log = log_dir / "mcp_server.stdout.log"
        stderr_log = log_dir / "mcp_server.stderr.log"

        # Open log files for appending
        stdout_file = open(stdout_log, "a")
        stderr_file = open(stderr_log, "a")

        # For STDIO mode, we need to keep the parent process alive
        # Start server with stdin=PIPE so we control the connection
        if sys.platform == "win32":
            # Windows: use CREATE_NEW_PROCESS_GROUP for proper process management
            process = subprocess.Popen(
                [python_exe, "-m", SERVER_MODULE],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                bufsize=0,  # Unbuffered
            )
        else:
            # Unix-like systems: use process group for management
            def preexec_fn():
                os.setpgrp()
                os.umask(0o077)

            process = subprocess.Popen(
                [python_exe, "-m", SERVER_MODULE],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                preexec_fn=preexec_fn,
                bufsize=0,
            )

        # Close our copies of stdout/stderr - child has its own
        stdout_file.close()
        stderr_file.close()

        # Save PID of server process
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

        # Save configuration
        save_server_config("stdio")

        # Give server a moment to initialize
        time.sleep(2)

        # Verify process is still running
        if process.poll() is None:
            print(f"[SUCCESS] MCP server started successfully (PID: {process.pid})")
            print("[INFO]    Wrapper maintaining stdin connection")
            print(f"[INFO]    Log output: {stdout_log}")

            # Start keepalive thread to monitor the process
            keepalive = threading.Thread(
                target=stdio_keepalive_thread,
                args=(process,),
                daemon=True,
                name="MCP-STDIO-Keepalive",
            )
            keepalive.start()

            # Keep this process running as the wrapper daemon
            # It maintains the stdin connection for the STDIO server
            try:
                # Wait for the server process to complete
                process.wait()
            except KeyboardInterrupt:
                print("\n[INFO] Wrapper process interrupted, shutting down server...")
                stop_server()
            finally:
                # Clean up on exit
                if PID_FILE.exists():
                    PID_FILE.unlink()
                if CONFIG_FILE.exists():
                    CONFIG_FILE.unlink()

            return True
        else:
            # Process exited during startup
            print("[ERROR] MCP server exited unexpectedly after startup")
            print(f"[INFO]  Check logs at {stderr_log} for error details")
            if PID_FILE.exists():
                PID_FILE.unlink()
            if CONFIG_FILE.exists():
                CONFIG_FILE.unlink()
            return False

    except Exception as e:
        print(f"[ERROR] Failed to start MCP server: {e}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        return False


def start_server_http(host="127.0.0.1", port=8000):
    """Start the MCP server in HTTP mode as a standalone daemon.

    HTTP mode servers can run independently without a persistent parent process.
    They listen on a TCP socket and handle multiple client connections.

    Args:
        host: Host address to bind to (default: 127.0.0.1)
        port: Port number to listen on (default: 8000)

    Returns:
        bool: True if server started successfully, False otherwise
    """
    if is_server_running():
        print("[ERROR] MCP server is already running")
        return False

    # Ensure MongoDB is available before starting server
    print()
    if not ensure_mongodb_available():
        print("[ERROR] Cannot start MCP server: MongoDB is not available")
        return False
    print()

    # Run integrity validation checks
    print("[INFO]  Running server integrity validation...")
    print("-" * 60)
    if run_integrity_checks is None:
        print("[WARNING] Validation module not available, skipping checks...")
    else:
        validation_passed = run_integrity_checks()
        print("-" * 60)
        if not validation_passed:
            print("[ERROR] Server integrity validation failed")
            print("[ERROR] Cannot start MCP server: Validation checks did not pass")
            print("[INFO]  Please fix the failed checks before starting the server")
            return False
        print("[SUCCESS] All integrity checks passed")
    print()

    python_exe = get_python_executable()
    print(f"[START] Starting MCP server in HTTP mode using: {python_exe}")
    print(f"[DIR]   Working directory: {PROJECT_ROOT}")
    print("[INFO]  Mode: HTTP transport")
    print(f"[INFO]  Endpoint: http://{host}:{port}")

    try:
        # Prepare log files for server output
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        stdout_log = log_dir / "mcp_server.stdout.log"
        stderr_log = log_dir / "mcp_server.stderr.log"

        # Open log files for appending
        stdout_file = open(stdout_log, "a")
        stderr_file = open(stderr_log, "a")

        # For HTTP mode, pass transport parameters via environment variables
        env = os.environ.copy()
        env["MCP_TRANSPORT"] = "http"
        env["MCP_HOST"] = host
        env["MCP_PORT"] = str(port)

        # Start server in true daemon mode (detached from parent)
        if sys.platform == "win32":
            # Windows: CREATE_NO_WINDOW prevents console window
            process = subprocess.Popen(
                [python_exe, "-m", SERVER_MODULE],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,  # No stdin needed for HTTP
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                bufsize=0,
            )
        else:
            # Unix-like systems: fully detach from parent
            def preexec_fn():
                os.setpgrp()
                os.umask(0o077)

            process = subprocess.Popen(
                [python_exe, "-m", SERVER_MODULE],
                cwd=str(PROJECT_ROOT),
                stdin=subprocess.DEVNULL,  # No stdin needed for HTTP
                stdout=stdout_file,
                stderr=stderr_file,
                env=env,
                start_new_session=True,
                preexec_fn=preexec_fn,
                bufsize=0,
            )

        # Close our copies of stdout/stderr
        stdout_file.close()
        stderr_file.close()

        # Save PID immediately
        with open(PID_FILE, "w") as f:
            f.write(str(process.pid))

        # Save configuration
        save_server_config("http", host, port)

        # Give server time to bind to port and start listening
        time.sleep(2)

        # Verify process is still running
        if is_server_running():
            print(f"[SUCCESS] MCP server started successfully (PID: {process.pid})")
            print(f"[INFO]    Listening on http://{host}:{port}")
            print(f"[INFO]    Log output: {stdout_log}")
            return True
        else:
            # Process exited during startup
            print("[ERROR] MCP server exited unexpectedly after startup")
            print(f"[INFO]  Check logs at {stderr_log} for error details")
            if PID_FILE.exists():
                PID_FILE.unlink()
            if CONFIG_FILE.exists():
                CONFIG_FILE.unlink()
            return False

    except Exception as e:
        print(f"[ERROR] Failed to start MCP server: {e}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        return False


def start_server(mode="stdio", host="127.0.0.1", port=8000):
    """Start the MCP server in the specified mode.

    Args:
        mode: Transport mode ('stdio' or 'http')
        host: HTTP host (only used in HTTP mode)
        port: HTTP port (only used in HTTP mode)

    Returns:
        bool: True if server started successfully, False otherwise
    """
    if mode == "http":
        return start_server_http(host, port)
    else:
        return start_server_stdio()


def stop_server():
    """Stop the running MCP server."""
    if not is_server_running():
        print("[INFO]  MCP server is not running")
        if PID_FILE.exists():
            PID_FILE.unlink()  # Clean up stale PID file
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()  # Clean up stale config
        return True

    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())

        # Load config to show mode
        config = load_server_config()
        mode_str = f" ({config['mode'].upper()} mode)" if config else ""

        print(f"[STOP] Stopping MCP server{mode_str} (PID: {pid})...")

        if sys.platform == "win32":
            # Windows: use taskkill with /T to kill process tree
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)], check=True, capture_output=True
                )
                print("[SUCCESS] MCP server stopped successfully")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to stop server: {e}")
                return False
        else:
            # Unix-like systems
            try:
                # Try graceful shutdown first
                os.kill(pid, signal.SIGTERM)
                # Wait for graceful shutdown
                time.sleep(2)
                # Force kill if still running
                if is_server_running():
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(1)
                print("[SUCCESS] MCP server stopped successfully")
            except ProcessLookupError:
                print("[INFO]  Process already terminated")
            except Exception as e:
                print(f"[ERROR] Failed to stop server: {e}")
                return False

        # Clean up files
        if PID_FILE.exists():
            PID_FILE.unlink()
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()

        return True

    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] Error reading PID file: {e}")
        if PID_FILE.exists():
            PID_FILE.unlink()
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()
        return False


def restart_server(mode="stdio", host="127.0.0.1", port=8000):
    """Restart the MCP server.

    Args:
        mode: Transport mode ('stdio' or 'http')
        host: HTTP host (only used in HTTP mode)
        port: HTTP port (only used in HTTP mode)

    Returns:
        bool: True if server restarted successfully, False otherwise
    """
    print("[RESTART] Restarting MCP server...")
    stop_server()
    time.sleep(1)
    return start_server(mode, host, port)


def show_status():
    """Show the current status of the MCP server."""
    if is_server_running():
        try:
            with open(PID_FILE) as f:
                pid = int(f.read().strip())

            # Load and display configuration
            config = load_server_config()
            if config:
                mode = config["mode"].upper()
                if config["mode"] == "http":
                    endpoint = f"http://{config['host']}:{config['port']}"
                    print(f"[SUCCESS] MCP server is running (PID: {pid})")
                    print(f"[INFO]    Mode: {mode}")
                    print(f"[INFO]    Endpoint: {endpoint}")
                else:
                    print(f"[SUCCESS] MCP server is running (PID: {pid})")
                    print(f"[INFO]    Mode: {mode} (wrapper maintaining connection)")
            else:
                print(f"[SUCCESS] MCP server is running (PID: {pid})")
                print("[INFO]    Mode: Unknown (config file missing)")

            return True
        except Exception:
            print("[SUCCESS] MCP server is running (PID unknown)")
            return True
    else:
        print("[ERROR] MCP server is not running")
        return False


def run_validation():
    """Run integrity checks on the MCP server.

    Returns:
        True if validation passed, False otherwise
    """
    python_exe = get_python_executable()
    validation_script = PROJECT_ROOT / "mcp_server_management" / "validate_server.py"

    if not validation_script.exists():
        print(f"[WARNING]  Validation script not found at: {validation_script}")
        print("   Skipping validation...")
        return True

    print("[INFO] Running integrity checks...")
    print("-" * 60)

    try:
        result = subprocess.run(
            [python_exe, str(validation_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        # Print validation output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        print("-" * 60)

        if result.returncode == 0:
            print("[PASS] Validation passed")
            return True
        else:
            print("[FAIL] Validation failed")
            return False

    except Exception as e:
        print(f"[ERROR] Error running validation: {e}")
        return False


def run_health_check():
    """Run quick health check on MCP server components.

    Returns:
        True if health check passed, False otherwise
    """
    python_exe = get_python_executable()

    print("[INFO] Running health check...")
    print("-" * 60)

    try:
        # Run integrity test for quick check
        test_script = PROJECT_ROOT / "mcp_server_management" / "validate_server.py"

        if not test_script.exists():
            print("[WARNING]  Test script not found. Running basic checks...")
            # Just check if we can import the server module
            result = subprocess.run(
                [
                    python_exe,
                    "-c",
                    "from src.mcp_server import server; print('[PASS] Server module OK')",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print("[PASS] Basic health check passed")
                return True
            else:
                print("[FAIL] Basic health check failed")
                return False

        result = subprocess.run(
            [python_exe, str(test_script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

        print("-" * 60)

        if result.returncode == 0:
            print("[PASS] Health check passed")
            return True
        else:
            print("[FAIL] Health check failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"[ERROR] Error running health check: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage the Healthcare MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # STDIO mode (default)
  python server_control.py --start           Start in STDIO mode
  python server_control.py --restart         Restart in STDIO mode

  # HTTP mode
  python server_control.py --start --http                      Start in HTTP mode (default: 127.0.0.1:8000)
  python server_control.py --start --http --port 8080          Start on custom port
  python server_control.py --start --http --host 0.0.0.0       Start on all interfaces
  python server_control.py --restart --http --port 8080        Restart in HTTP mode

  # Management
  python server_control.py --stop            Stop the server
  python server_control.py --status          Check server status
  python server_control.py --check           Run health check
        """,
    )

    parser.add_argument("--start", action="store_true", help="Start the MCP server")
    parser.add_argument("--stop", action="store_true", help="Stop the MCP server")
    parser.add_argument("--restart", action="store_true", help="Restart the MCP server")
    parser.add_argument("--status", action="store_true", help="Show server status")
    parser.add_argument(
        "--validate", action="store_true", help="Run integrity checks before starting server"
    )
    parser.add_argument(
        "--check", action="store_true", help="Run quick health check without starting server"
    )
    parser.add_argument(
        "--http", action="store_true", help="Use HTTP transport mode (default: STDIO)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="HTTP host address (default: 127.0.0.1, only for --http mode)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port number (default: 8000, only for --http mode)",
    )

    args = parser.parse_args()

    # Check if at least one action is specified
    if not any([args.start, args.stop, args.restart, args.status, args.check]):
        parser.print_help()
        print(
            "\n[ERROR] Error: Please specify an action (--start, --stop, --restart, --status, or --check)"
        )
        sys.exit(1)

    # Check if multiple actions are specified (validate can be combined with start/restart)
    actions = [args.stop, args.status, args.check]
    if args.start or args.restart:
        # start/restart can be combined with validate
        if sum(actions) > 0:
            print(
                "[ERROR] Error: --start/--restart can only be combined with --validate, --http, --host, and --port"
            )
            sys.exit(1)
    else:
        if sum(actions) > 1:
            print("[ERROR] Error: Please specify only one action at a time")
            sys.exit(1)

    # Validate HTTP-specific arguments
    if not args.http and (args.host != "127.0.0.1" or args.port != 8000):
        print("[WARNING] --host and --port are only used with --http mode")

    # Determine transport mode
    mode = "http" if args.http else "stdio"

    # Execute the requested action
    try:
        if args.start:
            # Run validation if requested
            if args.validate:
                if not run_validation():
                    print("\n[WARNING]  Validation failed. Start server anyway? (y/N): ", end="")
                    try:
                        response = input().strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n[ERROR] Server start cancelled")
                        sys.exit(1)
                    if response != "y":
                        print("[ERROR] Server start cancelled")
                        sys.exit(1)
                print()  # Add blank line

            success = start_server(mode, args.host, args.port)

        elif args.stop:
            success = stop_server()

        elif args.restart:
            # Run validation before restart if requested
            if args.validate:
                if not run_validation():
                    print("\n[WARNING]  Validation failed. Restart server anyway? (y/N): ", end="")
                    try:
                        response = input().strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n[ERROR] Server restart cancelled")
                        sys.exit(1)
                    if response != "y":
                        print("[ERROR] Server restart cancelled")
                        sys.exit(1)
                print()  # Add blank line

            success = restart_server(mode, args.host, args.port)

        elif args.status:
            success = show_status()

        elif args.check:
            success = run_health_check()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[WARNING]  Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
