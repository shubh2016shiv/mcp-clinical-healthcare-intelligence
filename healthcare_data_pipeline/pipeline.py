#!/usr/bin/env python3
"""
Healthcare Data Generation and Ingestion Orchestrator

Fixed version that properly handles Docker Compose command arguments.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for formatted terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(message: str) -> None:
    """Print formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(message: str) -> None:
    """Print success message with green checkmark."""
    print(f"{Colors.OKGREEN}[✓]{Colors.ENDC} {message}")


def print_error(message: str) -> None:
    """Print error message with red X."""
    print(f"{Colors.FAIL}[✗]{Colors.ENDC} {message}")


def print_warning(message: str) -> None:
    """Print warning message with yellow exclamation."""
    print(f"{Colors.WARNING}[!]{Colors.ENDC} {message}")


def print_info(message: str) -> None:
    """Print info message with blue i."""
    print(f"{Colors.OKBLUE}[ℹ]{Colors.ENDC} {message}")


def check_docker_installed() -> bool:
    """Check if Docker is installed and accessible."""
    if not shutil.which("docker"):
        print_error("Docker is not installed or not in PATH")
        print_info(
            "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
        )
        return False
    return True


def check_docker_running() -> bool:
    """Check if Docker daemon is running."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
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


def get_container_status() -> dict[str, str]:
    """Get status of project containers."""
    containers = {"text_to_mongo_db": "MongoDB", "text_to_mongo_redis": "Redis"}

    status = {}
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}:{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        for line in result.stdout.strip().split("\n"):
            if ":" in line:
                name, state = line.split(":", 1)
                if name in containers:
                    status[containers[name]] = state

        return status
    except Exception as e:
        print_warning(f"Could not get container status: {e}")
        return {}


def start_infrastructure(infrastructure_dir: Path) -> bool:
    """Start MongoDB and Redis infrastructure using docker-compose."""
    print_info("Starting MongoDB and Redis...")

    # Check if containers are already running
    status = get_container_status()
    running = [name for name, state in status.items() if "Up" in state]

    if running:
        print_success(f"Infrastructure already running: {', '.join(running)}")
        return True

    # Start infrastructure
    try:
        compose_file = infrastructure_dir / "docker-compose.yml"
        result = subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=infrastructure_dir,
        )

        if result.returncode != 0:
            print_error("Failed to start infrastructure")
            if result.stderr:
                print(result.stderr[:500])
            return False

        print_success("Infrastructure containers started")
        print_info("Waiting for MongoDB and Redis to be healthy...")
        time.sleep(5)
        return True

    except Exception as e:
        print_error(f"Error starting infrastructure: {e}")
        return False


def check_synthea_image() -> bool:
    """Check if Java Docker image is available for running Synthea."""
    image_name = "eclipse-temurin:17-jdk"
    print_info("Checking for Java Docker image (for Synthea)...")

    try:
        # Check if image exists
        result = subprocess.run(
            ["docker", "images", image_name, "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.stdout.strip():
            print_success("Java image found")
            return True

        # Image not found, pull it
        print_info("Java image not found. Pulling from Docker Hub...")
        print_warning("This may take a few minutes on first run\n")

        pull_result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=False,  # Show progress
            text=True,
            timeout=600,
        )

        if pull_result.returncode != 0:
            print_error("Failed to pull Java image")
            return False

        print_success("Java image pulled successfully")
        return True

    except subprocess.TimeoutExpired:
        print_error("Image pull timed out")
        return False
    except Exception as e:
        print_error(f"Error checking/pulling Java image: {e}")
        return False


def generate_synthea_data(num_patients: int, state: str, output_dir: Path) -> bool:
    """
    Generate synthetic healthcare data using Synthea Docker directly.

    This bypasses docker-compose and runs Synthea directly with docker run.
    """
    print_header("GENERATING SYNTHETIC HEALTHCARE DATA")
    print_info(f"Patients: {num_patients}")
    print_info(f"State: {state}\n")

    # Check and pull Synthea image if needed
    if not check_synthea_image():
        print_error("Cannot proceed without Synthea image")
        return False
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Synthea using docker run (more reliable than docker-compose for this)
    print_info("Running Synthea Docker container...")
    print_warning("This may take several minutes depending on the number of patients")
    print_info(f"Output directory: {output_dir.absolute()}\n")

    try:
        # Use docker run with Java to download and run Synthea Java JAR
        # Synthea JAR will be downloaded from GitHub releases
        output_path = str(output_dir.absolute())
        # On Windows, Docker Desktop needs forward slashes or proper escaping
        if sys.platform == "win32":
            # Convert Windows path to Docker-friendly format
            # Docker Desktop on Windows can handle both formats, but forward slashes are safer
            output_path = output_path.replace("\\", "/")
            # If path has spaces, ensure it's properly quoted in the volume mount
            if " " in output_path:
                # Docker handles spaces in paths, but we'll use the path as-is
                pass

        # Download Synthea JAR and run it
        # Using the latest release from GitHub
        synthea_jar_url = "https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"

        # Synthea outputs to output/fhir/ by default
        # We run from /output directory, and Synthea will create fhir/ subdirectory
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{output_path}:/output",
            "-w",
            "/app",
            "eclipse-temurin:17-jdk",
            "sh",
            "-c",
            f"(curl -sL {synthea_jar_url} -o /app/synthea.jar 2>&1 || "
            f"(apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq wget > /dev/null 2>&1 && wget -q {synthea_jar_url} -O /app/synthea.jar 2>&1)) && "
            f"cd /output && java -jar /app/synthea.jar -p {num_patients} {state} && "
            f"ls -la /output/fhir/ 2>&1 | head -5 || echo 'Checking output...' && "
            f"find /output -name '*.json' -type f | head -5",
        ]

        print_info("Executing: docker run with Java to run Synthea\n")
        print_info("Downloading Synthea JAR from GitHub and generating data...\n")

        result = subprocess.run(
            cmd,
            capture_output=False,  # Show Synthea output in real-time
            text=True,
            timeout=1800,  # 30 minute timeout for large datasets
        )

        if result.returncode != 0:
            print_error("Synthea data generation failed")
            return False

        print()
        print_success("Synthea data generation complete")

        # Wait a moment for file system to sync (especially on Windows)
        time.sleep(2)

        # Verify data was generated - check multiple possible locations
        # Synthea creates output/fhir/ inside the mounted directory
        possible_dirs = [
            output_dir / "fhir",  # Direct fhir/ subdirectory
            output_dir / "output" / "fhir",  # Synthea's output/fhir/ structure
            output_dir / "fhir_r4",  # Alternative naming
            output_dir / "output" / "fhir_r4",  # Alternative with output/
            output_dir,  # Root directory
        ]

        fhir_files = []
        fhir_dir = None

        for possible_dir in possible_dirs:
            if possible_dir.exists():
                files = list(possible_dir.glob("*.json"))
                if files:
                    fhir_files = files
                    fhir_dir = possible_dir
                    break

        if not fhir_files:
            print_error("No FHIR files found after generation")
            print_info(f"Checked directories: {[str(d) for d in possible_dirs]}")
            print_info("This might be a Docker volume mount issue on Windows.")
            print_info("Troubleshooting steps:")
            print_info("  1. Check Docker Desktop File Sharing settings")
            print_info("  2. Ensure the project directory is shared in Docker Desktop")
            print_info(
                '  3. Try running: docker run --rm -v "${PWD}/synthea_output:/output" eclipse-temurin:17-jdk ls -la /output'
            )

            # Try to check if directory exists but is empty
            if output_dir.exists():
                all_files = list(output_dir.rglob("*"))
                if all_files:
                    print_info(f"Found {len(all_files)} files/directories in output folder:")
                    for f in all_files[:10]:
                        print_info(f"  - {f}")

            return False

        print_success(f"Generated {len(fhir_files)} FHIR bundle files in {fhir_dir}")
        return True

    except subprocess.TimeoutExpired:
        print_error("Synthea generation timed out")
        return False
    except Exception as e:
        print_error(f"Error running Synthea: {e}")
        import traceback

        traceback.print_exc()
        return False


def ingest_data(synthea_output_dir: Path) -> bool:
    """Ingest FHIR data into MongoDB using the ingestion script."""
    print_header("INGESTING DATA INTO MONGODB")

    # Check multiple possible locations for FHIR files
    # Synthea may create output/fhir/ structure
    possible_dirs = [
        synthea_output_dir / "fhir",  # Direct fhir/ subdirectory
        synthea_output_dir / "output" / "fhir",  # Synthea's output/fhir/ structure
        synthea_output_dir / "fhir_r4",  # Alternative naming
        synthea_output_dir / "output" / "fhir_r4",  # Alternative with output/
        synthea_output_dir,  # Root directory
    ]

    fhir_dir = None
    for possible_dir in possible_dirs:
        if possible_dir.exists():
            files = list(possible_dir.glob("*.json"))
            if files:
                fhir_dir = possible_dir
                break

    if not fhir_dir:
        print_error(f"FHIR directory not found. Checked: {[str(d) for d in possible_dirs]}")
        return False

    ingestion_script = Path(__file__).parent / "ingest.py"
    if not ingestion_script.exists():
        print_error(f"Ingestion script not found: {ingestion_script}")
        print_info(f"Expected location: {ingestion_script}")
        return False

    try:
        print_info(f"Running ingestion script: {ingestion_script}")
        print_info(f"Data source: {fhir_dir}\n")

        result = subprocess.run(
            [sys.executable, str(ingestion_script), str(fhir_dir)],
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print_error("Data ingestion failed")
            return False

        print_success("Data ingestion complete")
        return True

    except subprocess.TimeoutExpired:
        print_error("Data ingestion timed out")
        return False
    except Exception as e:
        print_error(f"Error running ingestion script: {e}")
        return False


def verify_data(
    host: str = "localhost",
    port: int = 27017,
    user: str = "admin",
    password: str = "mongopass123",
    db_name: str = "text_to_mongo_db",
) -> bool:
    """Verify data was ingested correctly by counting documents in collections."""
    print_header("VERIFYING DATA")

    try:
        from pymongo import MongoClient

        connection_string = f"mongodb://{user}:{password}@{host}:{port}/{db_name}?authSource=admin"
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)

        # Test connection
        client.admin.command("ping")
        db = client[db_name]

        # Collections to verify
        collections = [
            "patients",
            "encounters",
            "conditions",
            "observations",
            "medicationrequests",
            "allergyintolerances",
            "procedures",
            "immunizations",
            "careplans",
            "diagnosticreports",
        ]

        total_docs = 0
        has_data = False

        for collection_name in collections:
            if collection_name in db.list_collection_names():
                count = db[collection_name].count_documents({})
                if count > 0:
                    print_success(f"{collection_name}: {count:,} documents")
                    total_docs += count
                    has_data = True

        client.close()

        if not has_data:
            print_warning("No data found in any collections")
            return False

        print()
        print_success(f"Total: {total_docs:,} documents ingested")
        return True

    except ImportError:
        print_error("pymongo not installed. Run: pip install pymongo")
        return False
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False


def print_connection_details(
    host: str = "localhost",
    port: int = 27017,
    user: str = "admin",
    password: str = "mongopass123",
    db_name: str = "text_to_mongo_db",
) -> None:
    """Print connection details and sample queries."""
    print_header("CONNECTION DETAILS & NEXT STEPS")

    print(f"{Colors.OKCYAN}{Colors.BOLD}MongoDB Connection:{Colors.ENDC}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Username: {user}")
    print(f"  Database: {db_name}")
    print()

    connection_string = f"mongodb://{user}:{password}@{host}:{port}/{db_name}?authSource=admin"
    print(f"{Colors.BOLD}Connection String:{Colors.ENDC}")
    print(f"  {connection_string}")
    print()

    print(f"{Colors.OKCYAN}{Colors.BOLD}Quick Commands:{Colors.ENDC}")
    print("  Connect to MongoDB:")
    print(
        f"    docker exec -it text_to_mongo_db mongosh -u {user} -p {password} --authenticationDatabase admin {db_name}"
    )
    print()

    print(f"{Colors.OKCYAN}{Colors.BOLD}Sample Queries:{Colors.ENDC}")
    print("  # Count patients")
    print("  db.patients.countDocuments()")
    print()
    print("  # Find diabetic patients")
    print("  db.conditions.find({'code.coding.display': /Diabetes/i}).limit(5)")
    print()
    print("  # Most common conditions")
    print("  db.conditions.aggregate([")
    print("    {$unwind: '$code.coding'},")
    print("    {$group: {_id: '$code.coding.display', count: {$sum: 1}}},")
    print("    {$sort: {count: -1}},")
    print("    {$limit: 10}")
    print("  ])")
    print()


def main() -> None:
    """Main entry point for the orchestration script."""
    parser = argparse.ArgumentParser(
        description="Healthcare Data Generation and Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py 100 Massachusetts
  python pipeline.py 500 California
  python pipeline.py 50 Texas --skip-verify
        """,
    )

    parser.add_argument(
        "num_patients",
        type=int,
        nargs="?",
        default=100,
        help="Number of synthetic patients to generate (default: 100)",
    )
    parser.add_argument(
        "state",
        nargs="?",
        default="Massachusetts",
        help="US state for patient data (default: Massachusetts)",
    )
    parser.add_argument(
        "--skip-infra",
        action="store_true",
        help="Skip infrastructure startup (assumes already running)",
    )
    parser.add_argument(
        "--skip-synthea", action="store_true", help="Skip Synthea generation (assume data exists)"
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip data verification")

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    infrastructure_dir = project_root / "infrastructure"
    synthea_output_dir = project_root / "synthea_output"

    print_header("HEALTHCARE DATA GENERATION & INGESTION PIPELINE")

    # Pre-flight checks
    print_info("Running pre-flight checks...\n")

    if not check_docker_installed():
        sys.exit(1)

    if not check_docker_running():
        sys.exit(1)

    print_success("Docker is available and running\n")

    # Start infrastructure
    if not args.skip_infra:
        if not start_infrastructure(infrastructure_dir):
            print_error("Failed to start infrastructure")
            sys.exit(1)
        print()
    else:
        print_info("Skipping infrastructure startup\n")

    # Generate Synthea data
    if not args.skip_synthea:
        if not generate_synthea_data(args.num_patients, args.state, synthea_output_dir):
            print_error("Failed to generate Synthea data")
            sys.exit(1)
        print()
    else:
        print_info("Skipping Synthea generation\n")

    # Ingest data
    if not ingest_data(synthea_output_dir):
        print_error("Failed to ingest data")
        sys.exit(1)
    print()

    # Verify data
    if not args.skip_verify:
        if not verify_data():
            print_warning("Data verification failed or found no data")
        print()
    else:
        print_info("Skipping data verification\n")

    # Print connection details
    print_connection_details()

    print_success("Pipeline completed successfully!")
    print_info("Your healthcare data is ready for analysis\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Pipeline cancelled by user{Colors.ENDC}")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
