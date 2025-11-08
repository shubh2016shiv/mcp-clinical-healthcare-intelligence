"""RxNav ATC drug data extraction module.

This module extracts drug ingredient and ATC classification data from the RxNav API
(National Library of Medicine). It handles API communication with retry logic and
processes the ATC hierarchy to extract ingredient-to-classification mappings.

The module is designed to be imported and called programmatically as part of the
healthcare pipeline, not just run as a standalone CLI script.
"""

import logging
import re
import time
from typing import Any

import httpx
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# RXNAV API Constants
RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# ATC code format patterns
ATC_LEVEL_2_PATTERN = re.compile(r"^[A-Z]\d{2}$")  # Example: A01
ATC_LEVEL_3_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z]$")  # Example: A01A
ATC_LEVEL_4_PATTERN = re.compile(r"^[A-Z]\d{2}[A-Z]{2}$")  # Example: A01AA

# All valid ATC Level 1 root categories
DEFAULT_ATC_ROOTS = [
    "A",
    "B",
    "C",
    "D",
    "G",
    "H",
    "J",
    "L",
    "M",
    "N",
    "P",
    "R",
    "S",
    "V",
]


class RxNavHttpClient:
    """HTTP client for RxNav API with retry logic and rate limiting.

    This client handles communication with the RxNav REST API, implementing
    exponential backoff for retries and rate limiting to respect API quotas.

    Attributes:
        request_delay: Seconds to wait between requests (rate limiting)
        request_timeout: Maximum seconds to wait for response
        max_retries: Maximum number of retry attempts
        backoff_multiplier: Exponential backoff multiplier for retries
    """

    def __init__(
        self,
        request_delay: float = 0.3,
        request_timeout: float = 40.0,
        max_retries: int = 3,
        backoff_multiplier: float = 1.8,
    ):
        """Initialize HTTP client with configurable parameters.

        Args:
            request_delay: Delay between requests in seconds (reduced from 0.9 for efficiency)
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            backoff_multiplier: Exponential backoff factor
        """
        self.request_delay = request_delay
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.backoff_multiplier = backoff_multiplier

        self.client = httpx.Client(timeout=request_timeout, headers=HTTP_HEADERS)

    def fetch_json(
        self, endpoint_path: str, query_parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Fetch JSON data from RxNav API with retry logic.

        Implements exponential backoff for retries and rate limiting delays.

        Args:
            endpoint_path: API endpoint path (e.g., "/rxclass/classTree.json")
            query_parameters: Optional query parameters dictionary

        Returns:
            JSON response as dictionary

        Raises:
            RuntimeError: If all retry attempts fail
        """
        url = f"{RXNAV_BASE_URL}{endpoint_path}"
        last_error = None

        for retry_attempt in range(self.max_retries):
            try:
                response = self.client.get(url, params=query_parameters or {})
                response.raise_for_status()
                data = response.json()

                if self.request_delay:
                    time.sleep(self.request_delay)

                return data

            except httpx.HTTPError as error:
                last_error = error
                backoff_delay = (self.backoff_multiplier**retry_attempt) * 0.4
                time.sleep(backoff_delay)

        # All retries failed
        status_code = getattr(getattr(last_error, "response", None), "status_code", "unknown")
        error_body = ""

        try:
            error_body = last_error.response.text[:500]
        except Exception:
            pass

        error_message = (
            f"HTTP request failed after {self.max_retries} retries:\n"
            f"URL: {url}\n"
            f"Parameters: {query_parameters}\n"
            f"Status Code: {status_code}\n"
            f"Response: {error_body}"
        )

        raise RuntimeError(error_message) from last_error

    def try_fetch_json(
        self, endpoint_path: str, query_parameters: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Attempt to fetch JSON data without raising exceptions.

        Args:
            endpoint_path: API endpoint path
            query_parameters: Query parameters dictionary

        Returns:
            JSON response dictionary or None if request fails
        """
        url = f"{RXNAV_BASE_URL}{endpoint_path}"

        try:
            response = self.client.get(url, params=query_parameters)
            response.raise_for_status()
            data = response.json()

            if self.request_delay:
                time.sleep(self.request_delay)

            return data

        except httpx.HTTPError:
            return None

    def close(self) -> None:
        """Close the HTTP client connection."""
        try:
            self.client.close()
        except Exception:
            pass


def fetch_atc_classification_tree(
    http_client: RxNavHttpClient, root_category: str
) -> dict[str, Any]:
    """Fetch the complete ATC classification tree for a root category.

    Args:
        http_client: Configured HTTP client instance
        root_category: ATC Level 1 root category (A-V)

    Returns:
        Complete classification tree as nested dictionary
    """
    return http_client.fetch_json(
        "/rxclass/classTree.json", {"classId": root_category, "classType": "ATC1-4"}
    )


def extract_child_nodes(parent_node: Any) -> list[Any]:
    """Extract child nodes from a parent node in the classification tree.

    Args:
        parent_node: Parent node dictionary

    Returns:
        List of child node dictionaries
    """
    if not isinstance(parent_node, dict):
        return []

    children = parent_node.get("rxclassTree")

    if not children:
        return []

    return children if isinstance(children, list) else [children]


def extract_node_classification(node: Any) -> dict[str, str] | None:
    """Extract classification information from a tree node.

    Args:
        node: Tree node dictionary

    Returns:
        Dictionary with 'classId' and 'className' or None if invalid
    """
    if not isinstance(node, dict):
        return None

    item = node.get("rxclassMinConceptItem")

    if isinstance(item, dict) and item.get("classId"):
        return {"classId": item["classId"], "className": item.get("className", "")}

    return None


def update_atc_hierarchy(
    atc_code: str,
    atc_name: str,
    current_level_2: str | None,
    current_level_3: str | None,
) -> tuple[str | None, str | None]:
    """Update ATC hierarchy levels based on current code.

    Maintains context as the tree is traversed, tracking which level of the
    ATC hierarchy we are currently processing.

    Args:
        atc_code: Current ATC code being processed
        atc_name: Name of the current ATC classification
        current_level_2: Current Level 2 classification
        current_level_3: Current Level 3 classification

    Returns:
        Tuple of (updated_level_2, updated_level_3)
    """
    if ATC_LEVEL_2_PATTERN.match(atc_code):
        return (atc_name or atc_code), None

    if ATC_LEVEL_3_PATTERN.match(atc_code):
        return current_level_2, (atc_name or atc_code)

    return current_level_2, current_level_3


def parse_class_members_response(api_response: dict[str, Any]) -> list[dict[str, str]]:
    """Parse class members from API response handling multiple response formats.

    The RxNav API has multiple response formats across versions. This function
    attempts to parse all known formats to ensure robustness across API changes.

    Args:
        api_response: Raw API response dictionary

    Returns:
        List of member dictionaries with 'rxcui', 'name', and 'tty' keys
    """
    members: list[dict[str, str]] = []

    if not isinstance(api_response, dict):
        return members

    # Format 1: drugMemberGroup.drugMember (current API format)
    drug_member_group = api_response.get("drugMemberGroup") or {}
    drug_members = drug_member_group.get("drugMember", []) or []

    for drug_member in drug_members:
        if not isinstance(drug_member, dict):
            continue

        # Check for nested minConcept structure
        min_concept = drug_member.get("minConcept")
        if isinstance(min_concept, dict) and min_concept.get("rxcui") and min_concept.get("name"):
            members.append(
                {
                    "rxcui": str(min_concept["rxcui"]),
                    "name": min_concept["name"],
                    "tty": min_concept.get("tty", ""),
                }
            )
            continue

        # Check for direct structure
        if drug_member.get("rxcui") and drug_member.get("name"):
            members.append(
                {
                    "rxcui": str(drug_member["rxcui"]),
                    "name": drug_member["name"],
                    "tty": drug_member.get("tty", ""),
                }
            )

    if members:
        return members

    # Format 2: rxclassDrugInfoList.rxclassDrugInfo (legacy format)
    drug_info_list = api_response.get("rxclassDrugInfoList") or {}
    drug_infos = drug_info_list.get("rxclassDrugInfo", []) or []

    for drug_info in drug_infos:
        if not isinstance(drug_info, dict):
            continue

        # Check drugMember
        drug_member = drug_info.get("drugMember")
        if isinstance(drug_member, dict) and drug_member.get("rxcui") and drug_member.get("name"):
            members.append(
                {
                    "rxcui": str(drug_member["rxcui"]),
                    "name": drug_member["name"],
                    "tty": drug_member.get("tty", ""),
                }
            )
            continue

        # Check various minConcept fields
        min_concept = (
            drug_info.get("minConcept")
            or drug_info.get("rxclassMinConcept")
            or drug_info.get("minConceptItem")
        )

        if isinstance(min_concept, dict) and min_concept.get("rxcui") and min_concept.get("name"):
            members.append(
                {
                    "rxcui": str(min_concept["rxcui"]),
                    "name": min_concept["name"],
                    "tty": min_concept.get("tty", ""),
                }
            )

    if members:
        return members

    # Format 3: rxclassMinConceptList.rxclassMinConcept
    min_concept_list = api_response.get("rxclassMinConceptList") or {}
    min_concepts = min_concept_list.get("rxclassMinConcept", []) or []

    for min_concept in min_concepts:
        if not isinstance(min_concept, dict):
            continue

        if min_concept.get("rxcui") and min_concept.get("name"):
            members.append(
                {
                    "rxcui": str(min_concept["rxcui"]),
                    "name": min_concept["name"],
                    "tty": min_concept.get("tty", ""),
                }
            )

    return members


def fetch_ingredient_members(
    http_client: RxNavHttpClient, atc_level_4_code: str
) -> list[dict[str, str]]:
    """Fetch ingredient members for an ATC Level 4 classification.

    Attempts multiple query strategies to retrieve ingredients from the API.

    Args:
        http_client: Configured HTTP client
        atc_level_4_code: ATC Level 4 code (e.g., "A01AA")

    Returns:
        List of ingredient member dictionaries
    """
    # Strategy 1: Request only ingredients (TTY=IN)
    query_params_ingredient = {
        "classId": atc_level_4_code,
        "relaSource": "ATC",
        "ttys": "IN",
    }

    # Strategy 2: Request all members, filter ingredients later
    query_params_all = {"classId": atc_level_4_code, "relaSource": "ATC"}

    for query_parameters in [query_params_ingredient, query_params_all]:
        api_response = http_client.try_fetch_json("/rxclass/classMembers.json", query_parameters)

        members = parse_class_members_response(api_response) if api_response else []

        if members:
            # If we didn't specifically request ingredients, filter them now
            if "ttys" not in query_parameters:
                ingredient_members = [m for m in members if m.get("tty") == "IN"]
                if ingredient_members:
                    return ingredient_members

            return members

    return []


def extract_drug_data(
    atc_root_categories: list[str] | None = None,
    request_delay: float = 0.3,
    max_concurrent: int = 4,
) -> list[dict[str, Any]]:
    """Extract drug ingredient and ATC classification data from RxNav API.

    This is the main programmatic entry point for the extraction module. It
    retrieves all drug ingredients and their ATC classifications, returning
    them as a list of dictionaries ready for ingestion into MongoDB.

    Optimized with concurrent requests and reduced delays for faster extraction
    while remaining respectful of API rate limits.

    Args:
        atc_root_categories: List of ATC Level 1 root categories to process.
            Defaults to all available categories (A-V).
        request_delay: Seconds to wait between API requests for rate limiting.
            Reduced default from 0.9 to 0.3 seconds for efficiency.
        max_concurrent: Maximum concurrent requests for ingredient fetching.
            Defaults to 4 to balance speed and API respect.

    Returns:
        List of dictionaries with keys: ingredient_rxcui, primary_drug_name,
        therapeutic_class_l2, drug_class_l3, drug_subclass_l4

    Raises:
        RuntimeError: If critical API errors prevent data extraction
    """
    if atc_root_categories is None:
        atc_root_categories = DEFAULT_ATC_ROOTS

    http_client = RxNavHttpClient(request_delay=request_delay)
    drug_data = []

    try:
        processed_entries: set = set()

        logger.info(
            f"Starting RxNav extraction for {len(atc_root_categories)} root categories "
            f"(delay={request_delay}s, concurrent={max_concurrent})"
        )

        # First pass: collect all Level 4 ATC codes
        atc_level_4_tasks = []

        for root_category in tqdm(atc_root_categories, desc="Collecting ATC codes"):
            try:
                classification_tree = fetch_atc_classification_tree(http_client, root_category)
                level_4_codes = collect_atc_level_4_codes(
                    classification_tree,
                    processed_entries,
                )
                atc_level_4_tasks.extend(level_4_codes)
            except Exception as error:
                logger.error(f"Failed to process root {root_category}: {error}")
                continue

        logger.info(f"Found {len(atc_level_4_tasks)} ATC Level 4 codes to process")

        # Second pass: fetch ingredients concurrently
        if atc_level_4_tasks:
            drug_data = fetch_ingredients_concurrently(
                http_client,
                atc_level_4_tasks,
                max_concurrent,
            )

        logger.info(f"Extraction complete: {len(drug_data)} drug records extracted")
        return drug_data

    finally:
        http_client.close()


def collect_atc_level_4_codes(
    classification_tree: dict[str, Any],
    processed_entries: set,
) -> list[dict[str, Any]]:
    """Collect all ATC Level 4 codes with their hierarchy context.

    Performs depth-first traversal to collect Level 4 codes without making
    API calls, storing hierarchy context for later ingredient fetching.

    Args:
        classification_tree: Root of classification tree
        processed_entries: Set to track processed codes (for deduplication)

    Returns:
        List of dictionaries with atc_code, atc_name, level_2, level_3
    """
    level_4_codes = []

    def depth_first_search(
        current_node: dict[str, Any],
        level_2_classification: str | None,
        level_3_classification: str | None,
    ) -> None:
        """Recursively traverse classification tree nodes."""
        classification = extract_node_classification(current_node)

        if classification:
            atc_code = classification["classId"]
            atc_name = classification["className"]

            # Update hierarchy context
            updated_level_2, updated_level_3 = update_atc_hierarchy(
                atc_code, atc_name, level_2_classification, level_3_classification
            )

            # Collect Level 4 classifications
            if ATC_LEVEL_4_PATTERN.match(atc_code):
                # Use code as key to avoid duplicates across roots
                code_key = atc_code
                if code_key not in processed_entries:
                    processed_entries.add(code_key)
                    level_4_codes.append(
                        {
                            "atc_code": atc_code,
                            "atc_name": atc_name,
                            "level_2": updated_level_2 or "",
                            "level_3": updated_level_3 or "",
                        }
                    )

            # Continue traversing children
            for child_node in extract_child_nodes(current_node):
                depth_first_search(child_node, updated_level_2, updated_level_3)
        else:
            # Node has no classification data, traverse children with current context
            for child_node in extract_child_nodes(current_node):
                depth_first_search(child_node, level_2_classification, level_3_classification)

    # Start traversal from root(s)
    root_nodes = classification_tree.get("rxclassTree")

    if isinstance(root_nodes, list):
        for root_node in root_nodes:
            depth_first_search(root_node, None, None)
    elif isinstance(root_nodes, dict):
        depth_first_search(root_nodes, None, None)

    return level_4_codes


def fetch_ingredients_concurrently(
    http_client: RxNavHttpClient,
    atc_tasks: list[dict[str, Any]],
    max_concurrent: int = 4,
) -> list[dict[str, Any]]:
    """Fetch ingredients for ATC Level 4 codes with controlled concurrency.

    Uses threading to fetch multiple ingredient lists concurrently while
    respecting API rate limits through a semaphore.

    Args:
        http_client: Configured HTTP client
        atc_tasks: List of ATC Level 4 code dictionaries with hierarchy
        max_concurrent: Maximum concurrent requests

    Returns:
        List of drug data dictionaries
    """
    import threading
    from queue import Empty, Queue

    drug_data = []
    drug_data_lock = threading.Lock()
    processed_entries: set = set()
    entries_lock = threading.Lock()
    semaphore = threading.Semaphore(max_concurrent)
    task_queue = Queue()

    # Add all tasks to queue
    for task in atc_tasks:
        task_queue.put(task)

    def worker():
        """Worker thread to process ATC codes."""
        while True:
            try:
                task = task_queue.get_nowait()
            except Empty:
                break

            atc_code = task["atc_code"]
            atc_name = task["atc_name"]
            level_2 = task["level_2"]
            level_3 = task["level_3"]

            with semaphore:
                try:
                    ingredient_members = fetch_ingredient_members(http_client, atc_code)

                    with drug_data_lock, entries_lock:
                        for member in ingredient_members:
                            entry_key = (member["rxcui"], atc_code)

                            # Skip duplicates
                            if entry_key in processed_entries:
                                continue

                            processed_entries.add(entry_key)

                            drug_data.append(
                                {
                                    "ingredient_rxcui": member["rxcui"],
                                    "primary_drug_name": member["name"],
                                    "therapeutic_class_l2": level_2,
                                    "drug_class_l3": level_3,
                                    "drug_subclass_l4": atc_name or atc_code,
                                }
                            )

                except Exception as error:
                    logger.debug(f"Error processing ATC code {atc_code}: {error}")

            task_queue.task_done()

    # Start worker threads
    threads = []
    num_workers = min(max_concurrent, len(atc_tasks))

    for _ in range(num_workers):
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        threads.append(thread)

    # Wait for all tasks with progress bar
    with tqdm(total=len(atc_tasks), desc="Fetching ingredients") as pbar:
        initial_size = len(atc_tasks)
        while not task_queue.empty():
            time.sleep(0.2)
            remaining = task_queue.qsize()
            completed = initial_size - remaining
            pbar.n = completed
            pbar.refresh()

        task_queue.join()
        pbar.n = initial_size
        pbar.refresh()

    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=5)

    return drug_data
