#!/usr/bin/env python3
"""
Infrastructure Management Script for text_to_mongo_query_language
Manages Docker Compose infrastructure with comprehensive error handling and validation.
"""

import subprocess
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import shutil


# ANSI color codes for better terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message: str):
    """Print formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")


def check_docker_installed() -> bool:
    """Check if Docker is installed and accessible."""
    if not shutil.which("docker"):
        print_error("Docker is not installed or not in PATH")
        print_info("Please install Docker Desktop from: https://www.docker.com/products/docker-desktop")
        return False
    return True


def check_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode != 0:
            print_error("Docker daemon is not running")
            print_info("Please start Docker Desktop and try again")
            return False
        return True
    except subprocess.TimeoutExpired:
        print_error("Docker daemon check timed out")
        print_info("Docker might be starting up. Please wait and try again")
        return False
    except Exception as e:
        print_error(f"Error checking Docker daemon: {e}")
        return False


def check_docker_compose_file() -> bool:
    """Check if docker-compose.yml exists."""
    compose_file = Path("docker-compose.yml")
    if not compose_file.exists():
        print_error("docker-compose.yml not found in current directory")
        print_info(f"Current directory: {Path.cwd()}")
        print_info("Please ensure you're in the project root directory")
        return False
    return True


def check_required_images() -> Tuple[bool, List[str]]:
    """Check if required Docker images are available."""
    required_images = {
        "mongo:7.0": "MongoDB 7.0",
        "redis:7.2-alpine": "Redis 7.2 Alpine"
    }

    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        available_images = set(result.stdout.strip().split('\n'))
        missing_images = []

        for image, name in required_images.items():
            if image not in available_images:
                missing_images.append(f"{name} ({image})")

        if missing_images:
            print_warning("Some required images are missing:")
            for img in missing_images:
                print(f"  • {img}")
            print_info("Docker Compose will pull these images automatically on first start")

        return True, missing_images
    except Exception as e:
        print_warning(f"Could not verify images: {e}")
        return True, []


def get_container_status() -> Dict[str, str]:
    """Get status of project containers."""
    containers = {
        "text_to_mongo_db": "MongoDB",
        "text_to_mongo_redis": "Redis"
    }

    status = {}
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}:{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        for line in result.stdout.strip().split('\n'):
            if ':' in line:
                name, state = line.split(':', 1)
                if name in containers:
                    status[containers[name]] = state

        return status
    except Exception as e:
        print_warning(f"Could not get container status: {e}")
        return {}


def check_port_availability(port: int) -> bool:
    """Check if a port is available."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0


def check_ports() -> Tuple[bool, List[int]]:
    """Check if required ports are available."""
    required_ports = {27017: "MongoDB", 6379: "Redis"}
    occupied_ports = []

    for port, service in required_ports.items():
        if not check_port_availability(port):
            print_warning(f"Port {port} ({service}) is already in use")
            occupied_ports.append(port)

    if occupied_ports:
        print_info("Occupied ports might be from existing containers or other services")
        print_info("Use --stop first to stop existing containers")
        return False, occupied_ports

    return True, []


def run_docker_compose_command(command: List[str], operation: str) -> bool:
    """Run docker compose command with error handling."""
    try:
        print_info(f"Executing: docker compose {' '.join(command)}")
        result = subprocess.run(
            ["docker", "compose"] + command,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print_error(f"Failed to {operation}")
            if result.stderr:
                print(f"\n{Colors.FAIL}Error output:{Colors.ENDC}")
                print(result.stderr)
            return False

        if result.stdout:
            print(result.stdout)

        return True
    except subprocess.TimeoutExpired:
        print_error(f"Operation timed out while trying to {operation}")
        return False
    except Exception as e:
        print_error(f"Error during {operation}: {e}")
        return False


def wait_for_healthy_containers(timeout: int = 60) -> bool:
    """Wait for containers to become healthy."""
    print_info(f"Waiting for containers to become healthy (timeout: {timeout}s)...")

    start_time = time.time()
    containers = ["text_to_mongo_db", "text_to_mongo_redis"]

    while time.time() - start_time < timeout:
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=text_to_mongo",
                 "--format", "{{.Names}}:{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )

            status_map = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line:
                    name, status = line.split(':', 1)
                    status_map[name] = status

            all_healthy = True
            for container in containers:
                if container not in status_map:
                    all_healthy = False
                    break
                status = status_map[container]
                if "starting" in status.lower():
                    all_healthy = False
                    break
                if "unhealthy" in status.lower():
                    print_error(f"Container {container} is unhealthy")
                    return False

            if all_healthy and len(status_map) == len(containers):
                print_success("All containers are healthy")
                return True

            time.sleep(2)
        except Exception as e:
            print_warning(f"Error checking health: {e}")
            time.sleep(2)

    print_warning("Containers did not become healthy within timeout")
    print_info("Containers might still be starting. Check logs with: docker compose logs")
    return True  # Don't fail, just warn


def print_connection_details():
    """Print connection details for all services."""
    print_header("SERVICE CONNECTION DETAILS")

    services = {
        "MongoDB": {
            "Host": "localhost",
            "Port": "27017",
            "Username": "admin",
            "Password": "mongopass123",
            "Database": "text_to_mongo_db",
            "Connection String": "mongodb://admin:mongopass123@localhost:27017/text_to_mongo_db?authSource=admin",
            "Container": "text_to_mongo_db"
        },
        "Redis": {
            "Host": "localhost",
            "Port": "6379",
            "Password": "redispass123",
            "Connection String": "redis://:redispass123@localhost:6379/0",
            "Container": "text_to_mongo_redis"
        }
    }

    for service_name, details in services.items():
        print(f"{Colors.OKCYAN}{Colors.BOLD}{service_name}:{Colors.ENDC}")
        for key, value in details.items():
            if key == "Connection String":
                print(f"  {Colors.BOLD}{key}:{Colors.ENDC}")
                print(f"    {value}")
            else:
                print(f"  {Colors.BOLD}{key}:{Colors.ENDC} {value}")
        print()

    print(f"{Colors.OKBLUE}Quick Test Commands:{Colors.ENDC}")
    print(
        f"  MongoDB: docker exec -it text_to_mongo_db mongosh -u admin -p mongopass123 --authenticationDatabase admin")
    print(f"  Redis:   docker exec -it text_to_mongo_redis redis-cli -a redispass123")
    print()


def start_infrastructure(skip_checks: bool = False) -> bool:
    """Start the infrastructure."""
    print_header("STARTING TEXT_TO_MONGO_QUERY_LANGUAGE INFRASTRUCTURE")

    # Pre-flight checks
    if not skip_checks:
        print_info("Running pre-flight checks...")

        if not check_docker_installed():
            return False

        if not check_docker_running():
            return False

        if not check_docker_compose_file():
            return False

        check_required_images()

        # Check if containers are already running
        status = get_container_status()
        if status:
            running = [name for name, state in status.items() if "Up" in state]
            if running:
                print_warning(f"Some containers are already running: {', '.join(running)}")
                print_info("Use --restart to restart them or --stop to stop first")
                return False

        print_success("Pre-flight checks passed")

    # Start containers
    print_info("Starting containers...")
    if not run_docker_compose_command(["up", "-d"], "start containers"):
        return False

    print_success("Containers started successfully")

    # Wait for health
    wait_for_healthy_containers()

    # Print connection details
    print_connection_details()

    print_success("Infrastructure is ready!")
    return True


def stop_infrastructure() -> bool:
    """Stop the infrastructure."""
    print_header("STOPPING TEXT_TO_MONGO_QUERY_LANGUAGE INFRASTRUCTURE")

    if not check_docker_installed() or not check_docker_running():
        return False

    if not check_docker_compose_file():
        return False

    # Check if containers are running
    status = get_container_status()
    if not status:
        print_warning("No containers found for this project")
        print_info("Infrastructure might already be stopped")
        return True

    running = [name for name, state in status.items() if "Up" in state]
    if not running:
        print_info("All containers are already stopped")
        return True

    print_info(f"Stopping containers: {', '.join(running)}")

    if not run_docker_compose_command(["down"], "stop containers"):
        return False

    print_success("Infrastructure stopped successfully")
    return True


def restart_infrastructure() -> bool:
    """Restart the infrastructure."""
    print_header("RESTARTING TEXT_TO_MONGO_QUERY_LANGUAGE INFRASTRUCTURE")

    if not check_docker_installed() or not check_docker_running():
        return False

    if not check_docker_compose_file():
        return False

    print_info("Stopping containers...")
    run_docker_compose_command(["down"], "stop containers")

    time.sleep(2)

    print_info("Starting containers...")
    return start_infrastructure(skip_checks=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Infrastructure Management for text_to_mongo_query_language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_infrastructure.py --start     Start the infrastructure
  python manage_infrastructure.py --stop      Stop the infrastructure
  python manage_infrastructure.py --restart   Restart the infrastructure
        """
    )

    parser.add_argument("--start", action="store_true", help="Start the infrastructure")
    parser.add_argument("--stop", action="store_true", help="Stop the infrastructure")
    parser.add_argument("--restart", action="store_true", help="Restart the infrastructure")

    args = parser.parse_args()

    # Check if at least one action is specified
    if not any([args.start, args.stop, args.restart]):
        parser.print_help()
        print(f"\n{Colors.FAIL}Error: Please specify an action (--start, --stop, or --restart){Colors.ENDC}")
        sys.exit(1)

    # Check if multiple actions are specified
    if sum([args.start, args.stop, args.restart]) > 1:
        print_error("Please specify only one action at a time")
        sys.exit(1)

    # Execute the requested action
    try:
        if args.start:
            success = start_infrastructure()
        elif args.stop:
            success = stop_infrastructure()
        elif args.restart:
            success = restart_infrastructure()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Operation cancelled by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()