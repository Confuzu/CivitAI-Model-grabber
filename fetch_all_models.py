import re
import requests
import logging
import argparse
import getpass
import os
import sys
import tempfile
import threading
from urllib.parse import urlparse

# Constants
ALLOWED_API_HOSTS = {'civitai.com', 'www.civitai.com'}
MAX_PAGES = 1000  # Safety limit to prevent infinite pagination
REQUEST_TIMEOUT = 30  # seconds
MAX_NAME_LENGTH = 500
MAX_ITEMS_PER_CATEGORY = 30_000

# Windows reserved device names (full set)
WINDOWS_RESERVED_NAMES = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Thread-safe logger initialization
_logger_lock = threading.Lock()
_logger_initialized = False


def _get_logger():
    """Get or create logger for this module (thread-safe singleton)."""
    global _logger_initialized

    _logger = logging.getLogger(__name__)

    # Fast path: already initialized
    if _logger_initialized:
        return _logger

    # Slow path: acquire lock and double-check
    with _logger_lock:
        if not _logger_initialized:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            log_file_path = os.path.join(script_dir, "fetch_all_models_ERROR_LOG.txt")
            handler = logging.FileHandler(log_file_path, encoding='utf-8')
            handler.setLevel(logging.ERROR)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            _logger.addHandler(handler)
            _logger.setLevel(logging.ERROR)
            _logger_initialized = True

    return _logger


logger = _get_logger()

# Module-level session for connection pooling
_api_session = None
_session_lock = threading.Lock()


def _get_api_session():
    """Get or create a reusable requests Session with connection pooling."""
    global _api_session
    if _api_session is None:
        with _session_lock:
            if _api_session is None:
                _api_session = requests.Session()
    return _api_session


def sanitize_url_for_logging(url):
    """Remove query parameters from URL before logging to prevent token leakage."""
    try:
        parsed = urlparse(url)
        safe_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return safe_url
    except Exception:
        return "[invalid URL]"


def sanitize_username(username):
    """Validate username to prevent path traversal and injection attacks."""
    if not username or not isinstance(username, str):
        raise ValueError("Username must be a non-empty string")

    # Remove all non-alphanumeric except underscore/hyphen
    safe = re.sub(r'[^a-zA-Z0-9_\-]', '_', username)

    # Strip leading/trailing underscores and dots
    safe = safe.strip('_.')

    # Reject if empty after sanitization
    if not safe:
        raise ValueError(f"Username '{username}' cannot be sanitized to a valid filename")

    # Also reject if only underscores/hyphens remain (no alphanumeric content)
    if not safe.strip('_-'):
        raise ValueError(f"Username '{username}' contains no alphanumeric characters")

    # Prevent names that are just numbers (could conflict with IDs)
    if safe.isdigit():
        safe = f"user_{safe}"

    # Prevent reserved names (full Windows device name set)
    if safe.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Username '{username}' is a reserved system name")

    # Enforce length limit
    if len(safe) > 64:
        safe = safe[:64]

    return safe


def validate_next_page_url(url):
    """Ensure pagination URL belongs to CivitAI and uses HTTPS."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        safe_url = sanitize_url_for_logging(url)

        if parsed.scheme != 'https':
            logger.error(f"Rejected non-HTTPS pagination URL: {safe_url}")
            return None
        if parsed.netloc not in ALLOWED_API_HOSTS:
            logger.error(f"Rejected pagination URL with unexpected host: {parsed.netloc}")
            return None
        if not parsed.path.startswith('/api/'):
            logger.error(f"Rejected pagination URL with unexpected path: {parsed.path}")
            return None
        return url
    except Exception as e:
        logger.error(f"Error validating URL structure: {e}")
        return None


def categorize_item(item):
    """Categorize the item based on JSON type."""
    item_type = item.get("type", "").upper()

    type_to_category = {
        'CHECKPOINT': 'Checkpoints',
        'TEXTUALINVERSION': 'Embeddings',
        'LORA': 'Lora',
        'TRAINING_DATA': 'Training_Data',
    }
    return type_to_category.get(item_type, 'Other')


def search_for_training_data_files(item):
    """Search for files with type 'Training Data' in the item's model versions.

    Returns validated, sanitized file names.
    """
    training_files = []
    for version in item.get("modelVersions", []):
        for file in version.get("files", []):
            if file.get("type") == "Training Data":
                name = file.get("name", "")
                if not isinstance(name, str):
                    logger.warning(f"Invalid training data filename type: {type(name)}")
                    continue
                if len(name) > MAX_NAME_LENGTH:
                    name = name[:MAX_NAME_LENGTH]
                # Sanitize path components from filenames
                if '/' in name or '\\' in name or name.startswith('.'):
                    logger.warning(f"Suspicious training data filename: {name[:50]}")
                    name = name.replace('/', '_').replace('\\', '_').lstrip('.')
                if name:
                    training_files.append(name)
    return training_files


def fetch_page(url, headers, safe_username, page_count):
    """Fetch a single page from the API using connection pooling.

    Returns:
        tuple: (data dict, error message or None)
    """
    session = _get_api_session()
    try:
        response = session.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, 'status_code', 'unknown') if e.response else 'unknown'
        logger.error(f"HTTP {status} error fetching page {page_count} for {safe_username}: {e}")
        # Generic user-facing messages
        if e.response is not None:
            if e.response.status_code == 401:
                return None, "Authentication failed. Please check your API token."
            elif e.response.status_code == 403:
                return None, "Access forbidden. Verify account permissions."
            elif e.response.status_code == 429:
                retry_after = e.response.headers.get('Retry-After', '')
                if retry_after.isdigit():
                    wait_time = int(retry_after)
                    logger.error(f"Rate limited for {safe_username}, retry after {wait_time}s")
                    return None, f"Rate limit exceeded. Wait {wait_time} seconds before retrying."
                return None, "Rate limit exceeded. Try again later."
            elif e.response.status_code >= 500:
                return None, "CivitAI service error. Try again later."
        return None, f"HTTP error {status} occurred."
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error on page {page_count} for {safe_username}: {type(e).__name__}")
        return None, "Network error occurred. Check your connection."

    try:
        return response.json(), None
    except requests.exceptions.JSONDecodeError:
        logger.error(f"Invalid JSON on page {page_count} for {safe_username}")
        return None, "Received invalid data from API."


def process_items(items, categorized_items, other_item_types):
    """Process a list of items and categorize them."""
    for item in items:
        try:
            # Validate item structure
            if not isinstance(item, dict):
                logger.error(f"Invalid item type: {type(item)}")
                continue

            name = item.get("name", "")
            if not isinstance(name, str):
                logger.error(f"Invalid name type for item: {type(name)}")
                continue

            # Truncate excessively long names
            if len(name) > MAX_NAME_LENGTH:
                logger.warning(f"Truncating long name: {name[:50]}...")
                name = name[:MAX_NAME_LENGTH]

            category = categorize_item(item)

            # Prevent memory exhaustion
            if len(categorized_items[category]) >= MAX_ITEMS_PER_CATEGORY:
                logger.error(f"Category {category} exceeded max items, skipping")
                continue

            categorized_items[category].append(name)

            # Check for deep nested "Training Data" files (with limit enforcement)
            training_data_files = search_for_training_data_files(item)
            if training_data_files:
                current_count = len(categorized_items['Training_Data'])
                remaining_capacity = MAX_ITEMS_PER_CATEGORY - current_count

                if remaining_capacity <= 0:
                    logger.error(f"Training_Data category exceeded max items, skipping files from '{name}'")
                else:
                    files_to_add = training_data_files[:remaining_capacity]
                    categorized_items['Training_Data'].extend(files_to_add)
                    if len(training_data_files) > remaining_capacity:
                        logger.warning(
                            f"Truncated {len(training_data_files) - remaining_capacity} "
                            f"training data files from '{name}' due to limit"
                        )

            if category == 'Other':
                # Cap other_item_types to match MAX_ITEMS_PER_CATEGORY
                if len(other_item_types) < MAX_ITEMS_PER_CATEGORY:
                    other_item_types.append((name, item.get("type", None)))

        except ValueError as e:
            # Expected: malformed data from API
            logger.error(f"Validation error for item '{item.get('name', 'unknown')}': {e}")
        except (KeyError, TypeError) as e:
            # Unexpected: programming error or severe API format change
            logger.error(
                f"UNEXPECTED ERROR processing item '{item.get('name', 'unknown')}': "
                f"{type(e).__name__}: {e}"
            )


def paginate_api(base_url, username, headers, safe_username):
    """Paginate through API results.

    Yields:
        dict: Page data containing items and metadata
    """
    next_page = f"{base_url}?username={username}&nsfw=true"
    seen_pages = set()
    page_count = 0

    while next_page and page_count < MAX_PAGES:
        if next_page in seen_pages:
            logger.error(f"Circular pagination detected for {safe_username}: {sanitize_url_for_logging(next_page)}")
            print("Circular pagination detected, stopping.")
            break

        seen_pages.add(next_page)
        page_count += 1

        data, error = fetch_page(next_page, headers, safe_username, page_count)
        if error:
            print(f"Error: {error}")
            break

        if not data:
            break

        yield data

        metadata = data.get('metadata', {})
        if not metadata:
            break

        raw_next_page = metadata.get('nextPage')
        next_page = validate_next_page_url(raw_next_page)

    if page_count >= MAX_PAGES:
        logger.error(f"Maximum page limit ({MAX_PAGES}) reached for {safe_username}")
        print(f"Warning: Maximum page limit ({MAX_PAGES}) reached. Some data may not have been fetched.")


def format_summary(categorized_items, other_item_types):
    """Format categorized items into a summary report.

    Returns:
        str: Formatted summary text
    """
    lines = ["Summary:\n"]
    total_count = sum(len(items) for items in categorized_items.values())
    lines.append(f"Total - Count: {total_count}\n")

    for category, items in categorized_items.items():
        lines.append(f"{category} - Count: {len(items)}\n")

    lines.append("\nDetailed Listing:\n")
    for category, items in categorized_items.items():
        lines.append(f"\n{category}:\n")
        if category == 'Other':
            for item_name, item_type in other_item_types:
                lines.append(f"  {item_name} - Type: {item_type}\n")
        else:
            for item_name in items:
                lines.append(f"  {item_name}\n")

    return ''.join(lines)


def write_summary(safe_username, categorized_items, other_item_types):
    """Write categorized items summary to a text file using atomic write."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(script_dir, f"{safe_username}.txt")

    # Verify the resolved path is still in script_dir (defense in depth)
    real_path = os.path.realpath(summary_path)
    real_script_dir = os.path.realpath(script_dir)
    if not (real_path == real_script_dir or real_path.startswith(real_script_dir + os.sep)):
        logger.error(f"Path traversal attempt detected: {summary_path}")
        raise ValueError("Invalid filename generated")

    content = format_summary(categorized_items, other_item_types)

    temp_path = None
    try:
        # Write to temporary file first, then atomic rename
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                          dir=script_dir,
                                          delete=False,
                                          suffix='.tmp') as tmpfile:
            tmpfile.write(content)
            tmpfile.flush()
            os.fsync(tmpfile.fileno())
            temp_path = tmpfile.name

        # Atomic rename — os.replace() is explicitly atomic on POSIX
        # (tempfile dir=script_dir ensures same-filesystem operation)
        os.replace(temp_path, summary_path)
        temp_path = None  # Mark as successfully moved
    except OSError as e:
        logger.error(f"Failed to write summary file for {safe_username}: {e}")
        print(f"Error: Could not write summary file: {e}")
    finally:
        # Guaranteed cleanup of temp file if move didn't succeed
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass  # Best effort cleanup


def fetch_all_models(token, username):
    """Fetch and categorize all models for a given username from CivitAI.

    Args:
        token: CivitAI API token (sent via Authorization header, never in URL)
        username: CivitAI username to fetch models for

    Returns:
        Dictionary of categorized item names, or empty dict on fatal error.
    """
    base_url = "https://civitai.com/api/v1/models"
    try:
        safe_username = sanitize_username(username)
    except ValueError as e:
        logger.error(f"Invalid username provided: {e}")
        print(f"Error: {e}")
        return {}

    categorized_items = {
        'Checkpoints': [],
        'Embeddings': [],
        'Lora': [],
        'Training_Data': [],
        'Other': []
    }
    other_item_types = []

    # Token goes in headers, NOT in the URL
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Paginate and process all items
    for page_data in paginate_api(base_url, username, headers, safe_username):
        items = page_data.get("items", [])
        if items:
            process_items(items, categorized_items, other_item_types)

    # Write results to file
    write_summary(safe_username, categorized_items, other_item_types)

    return categorized_items


def get_token_securely():
    """Retrieve API token from environment variable or secure prompt."""
    # First try environment variable
    token = os.environ.get('CIVITAI_API_TOKEN')
    if token:
        return token

    # Fall back to secure prompt (doesn't echo)
    try:
        token = getpass.getpass("Enter your CivitAI API token: ")
        if not token:
            raise ValueError("Token cannot be empty")
        return token
    except (KeyboardInterrupt, EOFError):
        print("\nToken input cancelled.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fetch and categorize models.")
    parser.add_argument("--username", type=str, required=True, help="Username to fetch models for.")
    args = parser.parse_args()

    token = get_token_securely()
    fetch_all_models(token, args.username)


if __name__ == "__main__":
    main()
