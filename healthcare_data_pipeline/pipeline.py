#!/usr/bin/env python3
"""
Healthcare Data Pipeline - Complete Workflow

Enhanced version that includes FHIR data transformation to clean medical records.
Orchestrates: Infrastructure -> Data Generation -> Ingestion -> Transformation
"""

import argparse
import os
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
    """Generate synthetic healthcare data using Synthea Docker."""
    print_header("GENERATING SYNTHETIC HEALTHCARE DATA")
    print_info(f"Patients: {num_patients}")
    print_info(f"State: {state}\n")

    if not check_synthea_image():
        print_error("Cannot proceed without Synthea image")
        return False
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

    print_info("Running Synthea Docker container...")
    print_warning("This may take several minutes depending on the number of patients")
    print_info(f"Output directory: {output_dir.absolute()}\n")

    try:
        output_path = str(output_dir.absolute())
        if sys.platform == "win32":
            output_path = output_path.replace("\\", "/")

        synthea_jar_url = "https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar"

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

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            print_error("Synthea data generation failed")
            return False

        print()
        print_success("Synthea data generation complete")
        time.sleep(2)

        # Verify data
        possible_dirs = [
            output_dir / "fhir",
            output_dir / "output" / "fhir",
            output_dir / "fhir_r4",
            output_dir / "output" / "fhir_r4",
            output_dir,
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
    print_header("INGESTING FHIR DATA INTO MONGODB")

    possible_dirs = [
        synthea_output_dir / "fhir",
        synthea_output_dir / "output" / "fhir",
        synthea_output_dir / "fhir_r4",
        synthea_output_dir / "output" / "fhir_r4",
        synthea_output_dir,
    ]

    fhir_dir = None
    for possible_dir in possible_dirs:
        if possible_dir.exists():
            files = list(possible_dir.glob("*.json"))
            if files:
                fhir_dir = possible_dir
                break

    if not fhir_dir:
        print_error("FHIR directory not found")
        return False

    ingestion_script = Path(__file__).parent / "ingest.py"
    if not ingestion_script.exists():
        print_error(f"Ingestion script not found: {ingestion_script}")
        return False

    try:
        print_info(f"Running ingestion script: {ingestion_script}")
        print_info(f"Data source: {fhir_dir}\n")

        result = subprocess.run(
            [sys.executable, str(ingestion_script), str(fhir_dir)],
            capture_output=False,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print_error("Data ingestion failed")
            return False

        print_success("FHIR data ingestion complete")
        return True

    except subprocess.TimeoutExpired:
        print_error("Data ingestion timed out")
        return False
    except Exception as e:
        print_error(f"Error running ingestion script: {e}")
        return False


def ingest_drug_data() -> bool:
    """Ingest RxNav drug data with ATC classifications."""
    print_header("INGESTING RXNAV DRUG DATA")

    try:
        from healthcare_data_pipeline.ingest import ingest_drug_data as ingest_drugs

        print_info("Starting RxNav drug data ingestion (3-5 minutes with optimizations)...\n")

        stats = ingest_drugs()

        if stats.get("errors", 0) > 0:
            print_warning(f"Drug ingestion completed with {stats['errors']} error(s)")
        else:
            print_success("Drug data ingestion completed successfully")

        return True

    except ImportError as e:
        print_warning(f"Could not import drug ingestion module: {e}")
        print_info("Continuing without drug data")
        return True

    except Exception as e:
        print_warning(f"Drug ingestion failed: {e}")
        print_info("Continuing without drug data")
        return True


def transform_data(gemini_api_key: str | None = None, use_gemini: bool = True) -> bool:
    """Transform FHIR data to clean medical records."""
    print_header("TRANSFORMING TO CLEAN MEDICAL RECORDS")

    transform_script = Path(__file__).parent / "transform.py"

    if not transform_script.exists():
        print_error(f"Transformation script not found: {transform_script}")
        print_info("Please ensure transform.py is in the same directory")
        return False

    try:
        cmd = [sys.executable, str(transform_script)]

        if use_gemini and gemini_api_key:
            cmd.extend(["--gemini-key", gemini_api_key])
            print_info("Gemini enrichment enabled")
        else:
            cmd.append("--no-gemini")
            print_info("Running without Gemini enrichment (faster)")

        print_info(f"Running transformation script: {transform_script}\n")

        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=1800,  # 30 minutes
        )

        if result.returncode != 0:
            print_error("Data transformation failed")
            return False

        print_success("Data transformation complete")
        return True

    except subprocess.TimeoutExpired:
        print_error("Data transformation timed out")
        return False
    except Exception as e:
        print_error(f"Error running transformation script: {e}")
        return False


def verify_data(
    host: str = "localhost",
    port: int = 27017,
    user: str = "admin",
    password: str = "mongopass123",
    db_name: str = "text_to_mongo_db",
    check_clean: bool = True,
) -> bool:
    """Verify data was ingested and transformed correctly."""
    print_header("VERIFYING DATA")

    try:
        from pymongo import MongoClient

        connection_string = f"mongodb://{user}:{password}@{host}:{port}/{db_name}?authSource=admin"
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)

        client.admin.command("ping")
        db = client[db_name]

        # Collections to verify
        raw_collections = [
            "patients",
            "encounters",
            "conditions",
            "observations",
            "medicationrequests",
        ]

        clean_collections = [
            "clean_patients",
            "clean_encounters",
            "clean_conditions",
            "clean_observations",
            "clean_medications",
        ]

        print_info("Raw FHIR Collections:")
        raw_total = 0
        for collection_name in raw_collections:
            if collection_name in db.list_collection_names():
                count = db[collection_name].count_documents({})
                if count > 0:
                    print_success(f"  {collection_name}: {count:,} documents")
                    raw_total += count

        if check_clean:
            print()
            print_info("Clean Medical Record Collections:")
            clean_total = 0
            for collection_name in clean_collections:
                if collection_name in db.list_collection_names():
                    count = db[collection_name].count_documents({})
                    if count > 0:
                        print_success(f"  {collection_name}: {count:,} documents")
                        clean_total += count

            if clean_total == 0:
                print_warning("No clean collections found - transformation may have failed")

        client.close()

        if raw_total == 0:
            print_warning("No data found in any collections")
            return False

        print()
        print_success("Verification complete - data is accessible")
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

    print(f"{Colors.BOLD}Clean Collections (Transformed Data):{Colors.ENDC}")
    print("  • clean_patients - Patient demographics")
    print("  • clean_conditions - Medical conditions/diagnoses")
    print("  • clean_medications - Medication prescriptions")
    print("  • clean_observations - Lab results & vital signs")
    print("  • clean_allergies - Allergy information")
    print("  • clean_immunizations - Vaccination records")
    print("  • clean_procedures - Medical procedures")
    print("  • clean_encounters - Healthcare visits")
    print("  • clean_care_plans - Treatment plans")
    print()

    print(f"{Colors.OKCYAN}{Colors.BOLD}Sample Queries (Clean Data):{Colors.ENDC}")
    print("  # Find all diabetic patients")
    print("  db.clean_conditions.find({'condition_name': /Diabetes/i})")
    print()
    print("  # Get patient's complete medical record")
    print("  patient_id = 'PATIENT_ID'")
    print("  db.clean_conditions.find({'patient_id': patient_id})")
    print("  db.clean_medications.find({'patient_id': patient_id})")
    print()
    print("  # Count patients by condition")
    print("  db.clean_conditions.aggregate([")
    print("    {$group: {_id: '$condition_name', count: {$sum: 1}}},")
    print("    {$sort: {count: -1}},")
    print("    {$limit: 10}")
    print("  ])")
    print()


def main() -> None:
    """Main entry point for the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete Healthcare Data Pipeline with Transformation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with Gemini enrichment
  python pipeline.py 100 Massachusetts --gemini-key YOUR_KEY

  # Full pipeline without Gemini (faster)
  python pipeline.py 100 Massachusetts --no-gemini

  # Skip transformation step
  python pipeline.py 100 Massachusetts --skip-transform

  # Only run transformation (data already ingested)
  python pipeline.py --only-transform --gemini-key YOUR_KEY
        """,
    )

    parser.add_argument(
        "num_patients",
        type=int,
        nargs="?",
        default=100,
        help="Number of synthetic patients (default: 100)",
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
        help="Skip infrastructure startup",
    )
    parser.add_argument(
        "--skip-synthea",
        action="store_true",
        help="Skip Synthea generation",
    )
    parser.add_argument(
        "--skip-transform",
        action="store_true",
        help="Skip data transformation",
    )
    parser.add_argument(
        "--only-transform",
        action="store_true",
        help="Only run transformation (skip generation/ingestion)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip data verification",
    )
    parser.add_argument(
        "--gemini-key",
        help="Google AI API key for Gemini enrichment",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Gemini enrichment",
    )

    args = parser.parse_args()

    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    infrastructure_dir = project_root / "infrastructure"
    synthea_output_dir = project_root / "synthea_output"

    print_header("COMPLETE HEALTHCARE DATA PIPELINE")

    # Check for Gemini API key from environment if not provided
    if not args.no_gemini and not args.gemini_key:
        args.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if args.gemini_key:
            print_info("Using Gemini API key from environment variable")

    # If only transforming, skip to transformation
    if args.only_transform:
        print_info("Running transformation only (skipping generation/ingestion)\n")

        if not transform_data(args.gemini_key, not args.no_gemini):
            print_error("Transformation failed")
            sys.exit(1)

        if not args.skip_verify:
            verify_data(check_clean=True)

        print_connection_details()
        print_success("Pipeline completed successfully!")
        return

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

    # Ingest raw FHIR data
    if not ingest_data(synthea_output_dir):
        print_error("Failed to ingest data")
        sys.exit(1)
    print()

    # Ingest drug data (optional)
    ingest_drug_data()
    print()

    # Transform to clean medical records
    if not args.skip_transform:
        if not transform_data(args.gemini_key, not args.no_gemini):
            print_warning("Transformation failed or incomplete")
            print_info("Raw FHIR data is still available in original collections")
        print()
    else:
        print_info("Skipping data transformation\n")

    # Verify data
    if not args.skip_verify:
        verify_data(check_clean=not args.skip_transform)
        print()

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
