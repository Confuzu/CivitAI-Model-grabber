import re
import requests
import logging
import os
import getpass
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import argparse
from fetch_all_models import fetch_all_models, paginate_api, sanitize_url_for_logging, WINDOWS_RESERVED_NAMES
import sys

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "civitAI_Model_downloader.txt")
OUTPUT_DIR = "model_downloads"
MAX_PATH_LENGTH = 200
MIN_SAFETENSORS_SIZE = 4 * 1024 * 1024  # 4 MB — typical minimum for valid safetensors
VALID_DOWNLOAD_TYPES = ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', 'All']
BASE_URL = "https://civitai.com/api/v1/models"
ALLOWED_API_HOSTS = {'civitai.com', 'www.civitai.com'}


logger_md = logging.getLogger('md')
logger_md.setLevel(logging.DEBUG)
file_handler_md = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler_md.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_md.setFormatter(formatter)
if not logger_md.handlers:
    logger_md.addHandler(file_handler_md)

# Thread-local storage for sessions
_thread_local = threading.local()

# Per-file lock mechanism for thread-safe file writes
_file_locks = {}
_file_locks_lock = threading.Lock()


def _get_file_lock(filepath):
    """Get or create a lock for a specific file path (thread-safe)."""
    with _file_locks_lock:
        if filepath not in _file_locks:
            _file_locks[filepath] = threading.Lock()
        return _file_locks[filepath]


def _append_to_file_locked(filepath, content):
    """Append content to a file with per-file locking (thread-safe)."""
    lock = _get_file_lock(filepath)
    with lock:
        with open(filepath, "a", encoding='utf-8') as f:
            f.write(content)


def get_session():
    """Get or create a thread-local requests session (thread-safe EAFP pattern)."""
    try:
        return _thread_local.session
    except AttributeError:
        _thread_local.session = requests.Session()
        return _thread_local.session


def sanitize_username_for_path(username):
    """Validate username to prevent path traversal in directory creation."""
    if not username or not isinstance(username, str):
        raise ValueError("Username must be a non-empty string")

    # Remove all non-alphanumeric except underscore/hyphen/dot
    safe = re.sub(r'[^a-zA-Z0-9_\-.]', '_', username)

    # Check for path traversal sequences
    if '..' in safe or '/' in safe or '\\' in safe:
        raise ValueError(f"Invalid username: path traversal detected in '{username}'")

    # Strip leading/trailing underscores and dots
    safe = safe.strip('_.')

    if not safe:
        raise ValueError(f"Invalid username: '{username}' is empty after sanitization")

    # Prevent reserved names (full Windows device name set)
    if safe.upper() in WINDOWS_RESERVED_NAMES:
        raise ValueError(f"Invalid username: '{username}' is a reserved system name")

    if len(safe) > 50:
        safe = safe[:50]

    return safe


def safe_path_join(base_dir, *parts):
    """Join paths and verify result stays within base_dir (prevents path traversal).

    Uses realpath() to resolve symlinks and commonpath() for robust comparison.
    """
    full_path = os.path.realpath(os.path.join(base_dir, *parts))
    base_dir_real = os.path.realpath(base_dir)

    # Verify full_path is base_dir or a subdirectory
    try:
        common = os.path.commonpath([base_dir_real, full_path])
        if common != base_dir_real:
            raise ValueError(f"Path traversal blocked: {full_path}")
    except ValueError:
        # commonpath raises ValueError if paths are on different drives (Windows)
        raise ValueError(f"Path traversal blocked: {full_path}")

    return full_path


def sanitize_filename_strict(filename):
    """Strict filename validation to prevent path traversal from API responses."""
    if not filename:
        return filename

    # Extract just the basename (removes any directory components)
    filename = os.path.basename(filename)

    # Check for path traversal attempts that survived basename()
    if '..' in filename:
        raise ValueError(f"Path traversal detected in filename: {filename}")

    # Remove dangerous characters
    filename = re.sub(r'[<>:"|?*\x00-\x1f\x7f-\x9f]', '_', filename)

    # Prevent empty after sanitization
    if not filename.strip('_. '):
        raise ValueError("Filename invalid after sanitization")

    return filename.strip()



def sanitize_name(name, max_length=MAX_PATH_LENGTH, subfolder=None, output_dir=None, username=None):
    """Sanitize a name for use as a file or folder name."""
    base_name, extension = os.path.splitext(name)

    # Normalize and check for path traversal
    base_name = os.path.basename(base_name)  # Strip any directory components
    if base_name.startswith('..') or os.path.isabs(base_name):
        base_name = 'invalid_name'

    # Remove problematic characters and control characters
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)

    # Handle reserved names (full Windows set)
    if base_name.upper() in WINDOWS_RESERVED_NAMES:
        base_name = '_'

    # Reduce multiple underscores to single and trim leading/trailing underscores and dots
    base_name = re.sub(r'__+', '_', base_name).strip('_.')

    # Calculate max length of base name considering the path length
    if subfolder and output_dir and username:
        # Include path separator before filename in calculation
        path_length = len(os.path.join(output_dir, username, subfolder)) + len(os.sep)
        max_base_length = max_length - len(extension) - path_length
        if max_base_length > 0:
            base_name = base_name[:max_base_length].rsplit('_', 1)[0]
        else:
            logger_md.error(f"Path too long for {username}/{subfolder}, cannot fit filename")
            base_name = base_name[:10]  # Fallback to very short name

    sanitized_name = base_name + extension
    return sanitized_name.strip()


def determine_subfolder(file_name, item_type):
    """Determine the download subfolder based on file extension and item type.

    Args:
        file_name: Name of the file being downloaded
        item_type: The 'type' field from the CivitAI item

    Returns:
        str: Subfolder name ('Lora', 'Checkpoints', 'Embeddings', 'Training_Data', or 'Other')
    """
    extension = os.path.splitext(file_name)[1].lower()

    SUBFOLDER_MAP = {
        '.zip': {
            'LORA': 'Lora',
            'Training_Data': 'Training_Data',
        },
        '.safetensors': {
            'Checkpoint': 'Checkpoints',
            'TextualInversion': 'Embeddings',
            'VAE': 'Other',
            'LoCon': 'Other',
        },
        '.pt': {
            'TextualInversion': 'Embeddings',
        }
    }

    # Default for .safetensors without a type is 'Lora'
    if extension == '.safetensors' and not item_type:
        return 'Lora'

    # Look up by extension then type
    if extension in SUBFOLDER_MAP:
        type_map = SUBFOLDER_MAP[extension]
        if item_type and item_type in type_map:
            return type_map[item_type]
        # Default for .safetensors with unknown type
        if extension == '.safetensors':
            return 'Lora'
        return 'Other'

    return 'Other'


def log_download_failure(url, username, max_retries, error=None):
    """Log download failures without exposing tokens (thread-safe)."""
    download_errors_log = os.path.join(SCRIPT_DIR, f'{username}.download_errors.log')
    content = f"Failed to download {sanitize_url_for_logging(url)} after {max_retries} attempts.\n"
    if error:
        content += f"Error: {error}\n"
    try:
        _append_to_file_locked(download_errors_log, content)
    except OSError as e:
        logger_md.error(f"Could not write to download error log: {e}")


def download_file_or_image(url, output_path, token, username, retry_count=0, max_retries=3, retry_delay=10):
    """Download a file or image using Authorization header for authentication.

    Args:
        url: Download URL (token is NOT appended to this)
        output_path: Local file path to save to
        token: API token for Authorization header
        username: Username (for error log filenames)
        retry_count: Current retry attempt
        max_retries: Maximum number of retries
        retry_delay: Seconds to wait between retries

    Returns:
        str: "downloaded" on success, "skipped" if file exists, "failed" on error
    """
    # Check if the file already exists (skip leftover .tmp files)
    if os.path.exists(output_path):
        return "skipped"

    temp_path = output_path + '.tmp'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    progress_bar = None

    # Build headers with Authorization instead of appending token to URL
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'

    # Add nsfw parameter safely to URL (no token)
    separator = '&' if '?' in url else '?'
    url_with_params = f"{url}{separator}nsfw=true"

    # Pre-compute safe URL for logging (prevents token leakage even if sanitize_url_for_logging fails)
    try:
        safe_url = sanitize_url_for_logging(url)
    except Exception:
        safe_url = "[URL sanitization failed]"

    try:
        session = get_session()
        response = session.get(url_with_params, headers=headers, stream=True, timeout=(20, 40))

        if response.status_code == 404:
            logger_md.warning(f"File not found (404): {safe_url}")
            print(f"File not found: {safe_url}")
            return "failed"

        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, leave=False)

        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)

        progress_bar.close()
        progress_bar = None

        # Validate .safetensors files
        if output_path.endswith('.safetensors') and os.path.getsize(temp_path) < MIN_SAFETENSORS_SIZE:
            # Remove the undersized temp file before retrying
            try:
                os.remove(temp_path)
            except OSError:
                pass
            if retry_count < max_retries:
                logger_md.warning(f"Undersized .safetensors file, retrying: {output_path} (attempt {retry_count + 1})")
                print(f"File {output_path} is smaller than expected. Retrying (attempt {retry_count + 1}).")
                time.sleep(retry_delay)
                return download_file_or_image(url, output_path, token, username, retry_count + 1, max_retries, retry_delay)
            else:
                log_download_failure(url, username, max_retries)
                return "failed"

        # Atomic rename on success
        os.replace(temp_path, output_path)
        return "downloaded"

    except requests.HTTPError as e:
        # HTTP errors (4xx, 5xx) — don't retry auth failures
        if progress_bar:
            progress_bar.close()
        logger_md.error(f"HTTP error for {safe_url}: {e}")
        return "failed"

    except (requests.Timeout, TimeoutError) as e:
        # Network timeout — retry
        if progress_bar:
            progress_bar.close()
        if retry_count < max_retries:
            logger_md.warning(f"Timeout, retrying: {safe_url} (attempt {retry_count + 1})")
            print(f"Timeout downloading {safe_url}. Retrying (attempt {retry_count + 1}).")
            time.sleep(retry_delay)
            return download_file_or_image(url, output_path, token, username, retry_count + 1, max_retries, retry_delay)
        else:
            log_download_failure(url, username, max_retries, error=e)
            return "failed"

    except (requests.ConnectionError, ConnectionResetError) as e:
        # Connection failures — retry
        if progress_bar:
            progress_bar.close()
        if retry_count < max_retries:
            logger_md.warning(f"Connection error, retrying: {safe_url} (attempt {retry_count + 1})")
            print(f"Connection error. Retrying (attempt {retry_count + 1}).")
            time.sleep(retry_delay)
            return download_file_or_image(url, output_path, token, username, retry_count + 1, max_retries, retry_delay)
        else:
            log_download_failure(url, username, max_retries, error=e)
            return "failed"

    except OSError as e:
        # File system errors (disk full, permission denied)
        if progress_bar:
            progress_bar.close()
        logger_md.error(f"File system error for {output_path}: {e}")
        return "failed"

    except Exception as e:
        # Unexpected errors — log with full traceback
        if progress_bar:
            progress_bar.close()
        logger_md.exception(f"Unexpected error downloading {safe_url}: {e}")
        # Re-raise programming errors so they fail loudly in dev/test
        if isinstance(e, (AttributeError, NameError, TypeError)):
            raise
        return "failed"

    finally:
        # Clean up temp file if it still exists (download failed or was interrupted)
        if os.path.exists(temp_path) and not os.path.exists(output_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def download_model_files(item_name, model_version, item, download_type, failed_downloads_file,
                         username, token, output_dir, max_retries, retry_delay, base_model=None):
    """Download related image and model files for each model version.

    Returns:
        tuple: (item_name, counts) where counts is a dict with
               'downloaded', 'skipped', and 'failed' integer values.
    """
    files = model_version.get('files', [])
    images = model_version.get('images', [])
    counts = {'downloaded': 0, 'skipped': 0, 'failed': 0}
    model_id = item.get('id', 'unknown')
    model_url = f"https://civitai.com/models/{model_id}"
    item_name_sanitized = sanitize_name(item_name, max_length=MAX_PATH_LENGTH)
    version_name = sanitize_name(model_version.get('name', ''), max_length=MAX_PATH_LENGTH)
    if not version_name:
        version_name = str(model_version.get('id', 'unknown'))
    item_dir = None
    subfolder = None

    # Extract the description and trigger words
    description = item.get('description') or ''
    trigger_words = model_version.get('trainedWords', [])

    for file in files:
        file_name = file.get('name', '')
        file_url = file.get('downloadUrl', '')

        # Validate file_name from API to prevent path traversal
        try:
            file_name = sanitize_filename_strict(file_name)
        except ValueError as e:
            logger_md.error(f"Rejected unsafe filename from API: {e}")
            continue

        # Determine subfolder using extracted function
        item_type = item.get('type')
        subfolder = determine_subfolder(file_name, item_type)

        if download_type != 'All' and download_type != subfolder:
            continue

        # Create folder structure (version subdirectory prevents filename collisions across versions)
        try:
            if base_model:
                item_dir = safe_path_join(output_dir, username, subfolder, base_model, item_name_sanitized, version_name)
                logger_md.info(f"Using baseModel folder structure for {item_name}: {base_model}/{version_name}")
            else:
                item_dir = safe_path_join(output_dir, username, subfolder, item_name_sanitized, version_name)
                logger_md.info(f"No baseModel found for {item_name}, using standard folder structure/{version_name}")
        except ValueError as e:
            logger_md.error(f"Path traversal blocked for {item_name}: {e}")
            continue

        try:
            os.makedirs(item_dir, exist_ok=True)
        except OSError as e:
            logger_md.error(f"Error creating directory for {item_name}: {str(e)}")
            _append_to_file_locked(
                failed_downloads_file,
                f"Item Name: {item_name}\nModel URL: {model_url}\n---\n"
            )
            return item_name, counts

        # Create and write to the description file (using safe_path_join)
        try:
            description_file = safe_path_join(item_dir, "description.html")
            with open(description_file, "w", encoding='utf-8') as f:
                f.write(description)
        except ValueError as e:
            logger_md.error(f"Path traversal blocked for description.html: {e}")
        except OSError as e:
            logger_md.error(f"Error writing description for {item_name}: {e}")

        try:
            trigger_words_file = safe_path_join(item_dir, "triggerWords.txt")
            with open(trigger_words_file, "w", encoding='utf-8') as f:
                f.write('\n'.join(trigger_words) + '\n' if trigger_words else '')
        except ValueError as e:
            logger_md.error(f"Path traversal blocked for triggerWords.txt: {e}")
        except OSError as e:
            logger_md.error(f"Error writing trigger words for {item_name}: {e}")

        # NOTE: Token is NOT appended to URL — it's sent via Authorization header
        file_name_sanitized = sanitize_name(file_name, max_length=MAX_PATH_LENGTH, subfolder=subfolder)

        try:
            file_path = safe_path_join(item_dir, file_name_sanitized)
        except ValueError as e:
            logger_md.error(f"Path traversal blocked for file {file_name}: {e}")
            continue

        if not file_name or not file_url:
            print(f"Invalid file entry: {file}")
            continue

        result = download_file_or_image(file_url, file_path, token, username,
                                        max_retries=max_retries, retry_delay=retry_delay)
        counts[result] += 1
        if result == "failed":
            _append_to_file_locked(
                failed_downloads_file,
                f"Item Name: {item_name}\nFile URL: {sanitize_url_for_logging(file_url)}\n---\n"
            )

        # Write details file — URLs are logged WITHOUT token (thread-safe)
        details_file = os.path.join(item_dir, "details.txt")
        try:
            _append_to_file_locked(
                details_file,
                f"Model URL: {model_url}\nFile Name: {file_name}\nFile URL: {sanitize_url_for_logging(file_url)}\n"
            )
        except OSError as e:
            logger_md.error(f"Error writing details for {item_name}: {e}")

    if item_dir is not None:
        for image in images:
            image_id = image.get('id', '')
            image_url = image.get('url', '')

            image_filename_raw = f"{item_name_sanitized}_{image_id}_for_{file_name}.jpeg"
            image_filename_sanitized = sanitize_name(image_filename_raw, max_length=MAX_PATH_LENGTH, subfolder=subfolder)

            try:
                image_path = safe_path_join(item_dir, image_filename_sanitized)
            except ValueError as e:
                logger_md.error(f"Path traversal blocked for image {image_filename_raw}: {e}")
                continue

            if not image_id or not image_url:
                print(f"Invalid image entry: {image}")
                continue

            result = download_file_or_image(image_url, image_path, token, username,
                                            max_retries=max_retries, retry_delay=retry_delay)
            counts[result] += 1
            if result == "failed":
                _append_to_file_locked(
                    failed_downloads_file,
                    f"Item Name: {item_name}\nImage URL: {sanitize_url_for_logging(image_url)}\n---\n"
                )

            # Write image details (thread-safe)
            details_file = os.path.join(item_dir, "details.txt")
            try:
                _append_to_file_locked(
                    details_file,
                    f"Image ID: {image_id}\nImage URL: {sanitize_url_for_logging(image_url)}\n"
                )
            except OSError as e:
                logger_md.error(f"Error writing image details for {item_name}: {e}")

    return item_name, counts


def process_username(username, download_type, token, max_tries, retry_delay_val, max_threads, output_dir):
    """Process a username and download the specified type of content."""
    # Validate username for path safety
    try:
        safe_username = sanitize_username_for_path(username)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Processing username: {username}, Download type: {download_type}")

    # Fetch and categorize all models (returns categorized dict directly)
    categorized_items = fetch_all_models(token, username)
    total_items = sum(len(items) for items in categorized_items.values())

    if download_type == 'All':
        selected_type_count = total_items
        intentionally_skipped = 0
    else:
        selected_type_count = len(categorized_items.get(download_type, []))
        intentionally_skipped = total_items - selected_type_count

    # Token in Authorization header, NOT in URL query params
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    failed_downloads_file = os.path.join(SCRIPT_DIR, f"failed_downloads_{safe_username}.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write(f"Failed Downloads for Username: {username}\n\n")

    # Track downloads across all pages (not reset per page)
    downloaded_item_names = set()
    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    # Use paginate_api for robust pagination (circular detection, page limit, URL validation)
    for page_data in paginate_api(BASE_URL, username, headers, safe_username):
        items = page_data.get('items', [])

        # Use context manager to guarantee executor shutdown
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            download_futures = []

            for item in items:
                # Validate item structure
                if not isinstance(item, dict):
                    logger_md.warning(f"Skipping non-dict item: {type(item)}")
                    continue

                item_name = item.get('name')
                if not item_name or not isinstance(item_name, str):
                    logger_md.warning(f"Skipping item with invalid name: {item.get('id', 'unknown')}")
                    continue

                model_versions = item.get('modelVersions', [])
                if item_name in downloaded_item_names:
                    continue
                downloaded_item_names.add(item_name)

                for version in model_versions:
                    future = executor.submit(
                        download_model_files, item_name, version, item,
                        download_type, failed_downloads_file, username, token, output_dir,
                        max_tries, retry_delay_val, base_model=version.get('baseModel')
                    )
                    download_futures.append(future)

            # Collect results with exception handling
            for future in tqdm(download_futures, desc="Downloading Files", unit="file", leave=False):
                try:
                    _, counts = future.result()
                    total_downloaded += counts['downloaded']
                    total_skipped += counts['skipped']
                    total_failed += counts['failed']
                except Exception as e:
                    logger_md.exception(f"Unhandled error in download worker: {e}")
                    # Continue processing other downloads

    print(f"\nResults for username {username}:")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Skipped (already existed): {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Type filter skipped: {intentionally_skipped}")


def get_token_securely(args_token):
    """Retrieve API token from args, environment variable, or secure prompt.

    Priority: CLI arg > environment variable > interactive prompt (hidden input)
    """
    if args_token:
        return args_token

    # Try environment variable
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
    """Main entry point — all argument parsing and user interaction happens here."""
    parser = argparse.ArgumentParser(description="Download models from CivitAI.")
    parser.add_argument('--token', type=str, help='CivitAI API token (prefer CIVITAI_API_TOKEN env var instead)')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries (default: 3)')
    parser.add_argument('--retry-delay', type=int, default=10, help='Delay between retries in seconds (default: 10)')
    parser.add_argument('--max-threads', type=int, default=3, help='Maximum number of concurrent downloads (default: 3)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help=f'Output directory (default: {OUTPUT_DIR})')

    args = parser.parse_args()

    token = get_token_securely(args.token)

    print("Enter a username (or multiple usernames separated by commas):")
    usernames_input = input("Username(s): ")
    usernames = [u.strip() for u in usernames_input.split(',') if u.strip()]

    if not usernames:
        print("No usernames provided. Exiting.")
        sys.exit(1)

    print(f"Select a download type from: {VALID_DOWNLOAD_TYPES}")
    download_type = input("Download type: ").strip()

    if download_type not in VALID_DOWNLOAD_TYPES:
        print(f"Invalid download type. Must be one of: {VALID_DOWNLOAD_TYPES}")
        sys.exit(1)

    for username in usernames:
        process_username(username, download_type, token, args.max_retries, args.retry_delay, args.max_threads, args.output_dir)


if __name__ == "__main__":
    main()
