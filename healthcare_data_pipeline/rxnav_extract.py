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
        request_delay: float = 0.9,
        request_timeout: float = 40.0,
        max_retries: int = 3,
        backoff_multiplier: float = 1.8,
    ):
        """Initialize HTTP client with configurable parameters.

        Args:
            request_delay: Delay between requests in seconds
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
    request_delay: float = 0.9,
) -> list[dict[str, Any]]:
    """Extract drug ingredient and ATC classification data from RxNav API.

    This is the main programmatic entry point for the extraction module. It
    retrieves all drug ingredients and their ATC classifications, returning
    them as a list of dictionaries ready for ingestion into MongoDB.

    Args:
        atc_root_categories: List of ATC Level 1 root categories to process.
            Defaults to all available categories (A-V).
        request_delay: Seconds to wait between API requests for rate limiting.

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

        logger.info(f"Starting RxNav extraction for {len(atc_root_categories)} root categories")

        for root_category in tqdm(atc_root_categories, desc="Processing ATC roots"):
            try:
                classification_tree = fetch_atc_classification_tree(http_client, root_category)
                traverse_and_extract_tree(
                    http_client,
                    classification_tree,
                    drug_data,
                    processed_entries,
                )

            except Exception as error:
                logger.error(f"Failed to process root {root_category}: {error}")
                continue

        logger.info(f"Extraction complete: {len(drug_data)} drug records extracted")
        return drug_data

    finally:
        http_client.close()


def traverse_and_extract_tree(
    http_client: RxNavHttpClient,
    classification_tree: dict[str, Any],
    drug_data: list[dict[str, Any]],
    processed_entries: set,
) -> None:
    """Traverse ATC classification tree and extract ingredient mappings.

    Performs depth-first traversal of the classification tree, extracting
    ingredient-to-classification mappings at ATC Level 4.

    Args:
        http_client: Configured HTTP client
        classification_tree: Root of classification tree
        drug_data: List to accumulate extracted drug records
        processed_entries: Set of processed (rxcui, atc_code) tuples to avoid duplicates
    """

    def depth_first_search(
        current_node: dict[str, Any],
        level_2_classification: str | None,
        level_3_classification: str | None,
    ) -> None:
        """Recursively traverse classification tree nodes.

        Args:
            current_node: Current tree node
            level_2_classification: Current ATC Level 2 name
            level_3_classification: Current ATC Level 3 name
        """
        classification = extract_node_classification(current_node)

        if classification:
            atc_code = classification["classId"]
            atc_name = classification["className"]

            # Update hierarchy context
            updated_level_2, updated_level_3 = update_atc_hierarchy(
                atc_code, atc_name, level_2_classification, level_3_classification
            )

            # Process Level 4 classifications (leaf nodes with ingredients)
            if ATC_LEVEL_4_PATTERN.match(atc_code):
                try:
                    ingredient_members = fetch_ingredient_members(http_client, atc_code)

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
                                "therapeutic_class_l2": updated_level_2 or "",
                                "drug_class_l3": updated_level_3 or "",
                                "drug_subclass_l4": atc_name or atc_code,
                            }
                        )

                except Exception as error:
                    logger.error(f"Error processing ATC code {atc_code}: {error}")

            # Continue traversing children with updated context
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
