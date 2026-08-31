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
import errno
import tempfile
from fetch_all_models import (fetch_all_models, paginate_api, sanitize_url_for_logging,
                              WINDOWS_RESERVED_NAMES, load_env_file)
import sys

# Constants
VERSION = "0.9.1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "civitAI_Model_downloader.txt")
OUTPUT_DIR = "model_downloads"
MAX_PATH_LENGTH = 200
TEMP_SUFFIX = '.tmp'
# Most filesystems allow 255 bytes per name, but not all: an encrypted home
# (ecryptfs) caps them at 143. The real limit is probed at startup and can be
# overridden with --max-filename-length.
DEFAULT_FILENAME_LENGTH_LIMIT = 255
MIN_FILENAME_LENGTH_LIMIT = 20
FILENAME_LENGTH_LIMIT = DEFAULT_FILENAME_LENGTH_LIMIT
MIN_SAFETENSORS_SIZE = 4 * 1024 * 1024  # 4 MB — typical minimum for valid safetensors
VALID_DOWNLOAD_TYPES = ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', 'All', 'All_except_Checkpoints']
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



def detect_max_filename_length(directory, upper_bound=DEFAULT_FILENAME_LENGTH_LIMIT):
    """Probe how long a single filename may be inside `directory`.

    Filesystems disagree: ext4 allows 255 bytes, an ecryptfs-encrypted home
    only 143. Probing avoids guessing wrong on the user's machine.

    Returns:
        int: the largest accepted name length, or `upper_bound` if the
             directory cannot be probed.
    """
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as e:
        logger_md.warning(f"Cannot probe filename length in {directory}: {e}")
        return upper_bound

    def accepts(length):
        path = os.path.join(directory, 'p' * length)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except OSError as e:
            if e.errno == errno.ENAMETOOLONG:
                return False
            raise
        os.close(fd)
        os.remove(path)
        return True

    try:
        if accepts(upper_bound):
            return upper_bound
        low, high = MIN_FILENAME_LENGTH_LIMIT, upper_bound
        while low < high:
            mid = (low + high + 1) // 2
            if accepts(mid):
                low = mid
            else:
                high = mid - 1
        return low
    except OSError as e:
        logger_md.warning(f"Cannot probe filename length in {directory}: {e}")
        return upper_bound


def configure_filename_length_limit(output_dir, override=None):
    """Set the global filename budget, leaving room for the .tmp suffix."""
    global FILENAME_LENGTH_LIMIT

    if override:
        detected = override
        source = "set via --max-filename-length"
    else:
        detected = detect_max_filename_length(output_dir)
        source = "detected"

    # Downloads are written as "<name>.tmp" first, so that suffix has to fit too.
    FILENAME_LENGTH_LIMIT = max(MIN_FILENAME_LENGTH_LIMIT, detected - len(TEMP_SUFFIX))

    message = (f"Filename length limit for {output_dir}: {detected} characters "
               f"({source}); names are shortened to {FILENAME_LENGTH_LIMIT} "
               f"to leave room for the {TEMP_SUFFIX} suffix.")
    print(message)
    logger_md.info(message)
    return FILENAME_LENGTH_LIMIT


_truncation_lock = threading.Lock()
_truncated_names = set()


def _report_truncation(original, shortened):
    """Tell the user once per name that it had to be shortened."""
    with _truncation_lock:
        if original in _truncated_names:
            return
        _truncated_names.add(original)

    message = (f"Name too long for this filesystem (limit {FILENAME_LENGTH_LIMIT} "
               f"characters), shortened: {original!r} -> {shortened!r}")
    print(f"Note: {message}")
    logger_md.warning(message)


def _truncate_to_bytes(text, max_bytes):
    """Cut `text` so its UTF-8 encoding fits into `max_bytes`."""
    encoded = text.encode('utf-8')
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode('utf-8', 'ignore')


def sanitize_name(name, max_length=MAX_PATH_LENGTH, subfolder=None, output_dir=None, username=None):
    """Sanitize a name for use as a file or folder name."""
    base_name, extension = os.path.splitext(name)

    # Normalize and check for path traversal.
    base_name = base_name.replace('/', '_').replace(os.sep, '_')
    if os.altsep:
        base_name = base_name.replace(os.altsep, '_')
    base_name = os.path.basename(base_name)  # Strip any directory components
    if base_name and (base_name.strip('.') == '' or os.path.isabs(base_name)):
        base_name = 'invalid_name'

    # Remove problematic characters and control characters
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)

    # Reduce multiple underscores to single and trim leading/trailing underscores and dots
    base_name = re.sub(r'__+', '_', base_name).strip('_.')

    # Handle reserved names (full Windows set). This runs after the strip
    # above: a bare '_' replacement would be stripped again and leave an
    # empty path component, silently merging two models into one folder.
    if base_name.upper() in WINDOWS_RESERVED_NAMES:
        base_name = f"{base_name}_file"

    # A non-empty input must never sanitize down to nothing, otherwise
    # os.path.join() drops the component and unrelated models collide.
    if name.strip() and not base_name:
        base_name = 'unnamed'

    # Enforce the length budget. This used to run only when the caller passed
    # subfolder, output_dir and username together, which no call site did, so
    # no name was ever shortened and long names failed with ENAMETOOLONG.
    # The budget is the filesystem limit, not MAX_PATH_LENGTH: names between
    # MAX_PATH_LENGTH and the filesystem limit are written fine today, and
    # shortening them would change their path. Downloads are skipped by path
    # comparison, so a renamed file counts as missing and is fetched again.
    budget = FILENAME_LENGTH_LIMIT - len(extension.encode('utf-8'))
    if subfolder and output_dir and username:
        # Include path separator before filename in calculation
        path_length = len(os.path.join(output_dir, username, subfolder)) + len(os.sep)
        budget = min(budget, max_length - len(extension) - path_length)

    if budget < MIN_FILENAME_LENGTH_LIMIT:
        logger_md.error(f"Path too long for {username}/{subfolder}, cannot fit filename")
        budget = MIN_FILENAME_LENGTH_LIMIT

    if len(base_name.encode('utf-8')) > budget:
        shortened = _truncate_to_bytes(base_name, budget).rstrip('_. ')
        _report_truncation(name, shortened + extension)
        base_name = shortened

    sanitized_name = (base_name + extension).strip()

    # Windows silently drops trailing dots and spaces from file and folder
    # names, so "Cool Lora." on disk becomes "Cool Lora" and no longer matches
    # the path the script just built. Trim them here instead.
    sanitized_name = sanitized_name.rstrip('. ')

    # The trim must not empty out a name that had content.
    if name.strip() and not sanitized_name:
        sanitized_name = 'unnamed'

    return sanitized_name


_migration_lock = threading.Lock()
_migrated_dirs = set()


def legacy_sanitize_name(name):
    """Reproduce the pre-fix sanitize_name() so its folders can be found.

    The old version called os.path.basename() before replacing separators,
    so a model name containing a slash was reduced to its last component
    (Issue #39). Knowing that name lets the script rename the old folder
    instead of downloading everything again.
    """
    base_name, extension = os.path.splitext(name)
    base_name = os.path.basename(base_name)
    if base_name.startswith('..') or os.path.isabs(base_name):
        base_name = 'invalid_name'
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)
    if base_name.upper() in WINDOWS_RESERVED_NAMES:
        base_name = '_'
    base_name = re.sub(r'__+', '_', base_name).strip('_.')
    return (base_name + extension).strip()


def _rename_legacy_prefixed_files(directory, legacy_name, new_name, subfolder):
    """Rename files whose name starts with the old truncated model name.

    Preview images are stored as "<model name>_<image id>_for_<file>.jpeg",
    so they carry the truncated name too and would otherwise be downloaded
    again after the folder itself was migrated.
    """
    renamed = 0
    prefix = legacy_name + '_'
    for root, _dirs, files in os.walk(directory):
        for file_name in files:
            if not file_name.startswith(prefix):
                continue
            remainder = file_name[len(legacy_name):]
            target = sanitize_name(new_name + remainder, max_length=MAX_PATH_LENGTH, subfolder=subfolder)
            if target == file_name:
                continue
            source_path = os.path.join(root, file_name)
            target_path = os.path.join(root, target)
            if os.path.exists(target_path):
                continue
            try:
                os.rename(source_path, target_path)
                renamed += 1
            except OSError as e:
                logger_md.error(f"Could not rename {source_path}: {e}")
    return renamed


_legacy_name_owners = {}


def register_models_for_migration(item_names):
    """Record which model names collapse to the same pre-fix folder name.

    Several names could produce the same truncated name, e.g. both
    "Hyouuma Style (Anima/Illustrious)" and "Anus Outline / ... (Anima/
    Illustrious)" became "Illustrious)". Those models shared one folder, so
    that folder must not be handed to whichever model is processed first.
    """
    with _migration_lock:
        for item_name in item_names:
            if not item_name or not isinstance(item_name, str):
                continue
            legacy_name = legacy_sanitize_name(item_name)
            if legacy_name:
                _legacy_name_owners.setdefault(legacy_name, set()).add(item_name)


def expected_version_dir_names(item):
    """Version folder names this model would create, used to spot foreign data."""
    names = set()
    for version in item.get('modelVersions', []):
        version_name = sanitize_name(version.get('name', ''), max_length=MAX_PATH_LENGTH)
        if not version_name:
            version_name = str(version.get('id', 'unknown'))
        names.add(version_name)
    return names


def migrate_legacy_item_folder(parent_dir, item_name, new_name, subfolder=None, item=None):
    """Rename a folder left behind by the truncating sanitize_name (Issue #39).

    Downloads are skipped by path comparison, so without this every file of an
    affected model would be fetched again under its corrected name.

    The folder is only renamed when it can be attributed to this model alone:
    no other known model name maps to it, and it holds no version folders this
    model does not have. Otherwise it is left untouched and reported, because
    a shared folder already contains mixed data that cannot be separated
    reliably.

    Returns:
        bool: True when a folder was migrated.
    """
    legacy_name = legacy_sanitize_name(item_name)
    if not legacy_name or legacy_name == new_name:
        return False

    old_path = os.path.join(parent_dir, legacy_name)
    new_path = os.path.join(parent_dir, new_name)

    with _migration_lock:
        if old_path in _migrated_dirs:
            return False
        _migrated_dirs.add(old_path)

        if not os.path.isdir(old_path):
            return False

        if os.path.exists(new_path):
            message = (f"Not migrating {old_path}: {new_path} already exists. "
                       f"Merge or remove one of them manually.")
            print(f"Note: {message}")
            logger_md.warning(message)
            return False

        owners = _legacy_name_owners.get(legacy_name, set())
        if len(owners) > 1:
            message = (f"Not migrating {old_path}: {len(owners)} models share this "
                       f"folder name from the older version, so its contents are "
                       f"already mixed. Sort it out manually or delete it.")
            print(f"Note: {message}")
            logger_md.warning(message)
            return False

        if item is not None:
            expected = expected_version_dir_names(item)
            foreign = {entry for entry in os.listdir(old_path)
                       if os.path.isdir(os.path.join(old_path, entry)) and entry not in expected}
            if foreign:
                message = (f"Not migrating {old_path}: it holds version folders this "
                           f"model does not have ({', '.join(sorted(foreign))}), so it "
                           f"belongs to more than one model.")
                print(f"Note: {message}")
                logger_md.warning(message)
                return False

        try:
            os.rename(old_path, new_path)
        except OSError as e:
            logger_md.error(f"Could not migrate {old_path}: {e}")
            return False

    renamed = _rename_legacy_prefixed_files(new_path, legacy_name, new_name, subfolder)
    message = (f"Migrated folder left by an older version: {legacy_name!r} -> "
               f"{new_name!r} ({renamed} files renamed, nothing re-downloaded)")
    print(f"Note: {message}")
    logger_md.info(message)
    return True


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

    temp_path = output_path + TEMP_SUFFIX
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


def extract_image_meta(item):
    """Extract the actual metadata dict from an API image item, handling nested structure.

    CivitAI API changed structure from:
        item["meta"] = {"prompt": "...", "Model": "..."}
    To:
        item["meta"] = {"id": 123, "meta": {"prompt": "...", "Model": "..."}}

    This function handles both old and new structures.
    Matches the logic in CivitAI_Image_grabber.
    """
    meta_field = item.get("meta")
    if not meta_field or not isinstance(meta_field, dict):
        return {}

    # Check for new nested structure: meta.meta exists and contains generation params
    nested_meta = meta_field.get("meta")
    if nested_meta and isinstance(nested_meta, dict):
        return nested_meta

    # Old structure: check if prompt/Model exists at top level
    if "prompt" in meta_field or "Model" in meta_field or "seed" in meta_field:
        return meta_field

    return {}


def fetch_image_metadata(version_id, headers):
    """Fetch image metadata (prompts, generation params) from the images API.

    Args:
        version_id: Model version ID to fetch images for
        headers: Authorization headers dict

    Returns:
        dict: Mapping of image_id -> extracted meta dict. Empty dict on error.
    """
    if not version_id:
        return {}

    url = f"https://civitai.com/api/v1/images?modelVersionId={version_id}&nsfw=true"
    session = get_session()
    meta_by_id = {}

    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logger_md.warning(f"Could not fetch image metadata for version {version_id}: {e}")
        return {}

    for img in data.get('items', []):
        img_id = img.get('id')
        if img_id:
            meta = extract_image_meta(img)
            base_model = img.get('baseModel')
            # Add Model from baseModel if missing (Bug #38 pattern from Image_grabber)
            if meta and base_model and 'Model' not in meta:
                meta = {'Model': base_model, **meta}
            meta_by_id[img_id] = meta

    return meta_by_id


def write_image_meta_file(meta, image_id, item_dir, username):
    """Write per-image metadata to a separate {image_id}_meta.txt file.

    Matches the file format used by CivitAI_Image_grabber:
    - {image_id}_meta.txt when metadata is available
    - {image_id}_no_meta.txt with fallback URL when not

    Args:
        meta: Extracted metadata dict (from extract_image_meta)
        image_id: Image ID (numeric or string)
        item_dir: Directory to write the file in
        username: Username for fallback URL
    """
    if meta and not all(str(v).strip() == '' for v in meta.values()):
        filename = f"{image_id}_meta.txt"
        content_lines = [f"{k}: {str(v) if v is not None else ''}" for k, v in meta.items()]
    else:
        filename = f"{image_id}_no_meta.txt"
        content_lines = [
            "No metadata available.",
            f"URL: https://civitai.com/images/{image_id}?username={username}"
        ]

    try:
        meta_path = safe_path_join(item_dir, filename)
    except ValueError as e:
        logger_md.error(f"Path traversal blocked for meta file {filename}: {e}")
        return

    try:
        with open(meta_path, "w", encoding='utf-8') as f:
            f.write("\n".join(content_lines))
    except OSError as e:
        logger_md.error(f"Error writing metadata file {meta_path}: {e}")


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
    # baseModel comes straight from the API and is used as a folder name,
    # so it needs the same sanitizing as the model and version names.
    base_model_dir = sanitize_name(base_model, max_length=MAX_PATH_LENGTH) if base_model else None
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

        if download_type == 'All_except_Checkpoints':
            if subfolder == 'Checkpoints':
                continue
        elif download_type != 'All' and download_type != subfolder:
            continue

        # Rename folders created by the pre-fix version before building the
        # path, so existing downloads are kept instead of fetched again.
        try:
            if base_model_dir:
                parent_dir = safe_path_join(output_dir, username, subfolder, base_model_dir)
            else:
                parent_dir = safe_path_join(output_dir, username, subfolder)
            migrate_legacy_item_folder(parent_dir, item_name, item_name_sanitized, subfolder, item)
        except ValueError as e:
            logger_md.error(f"Path traversal blocked while migrating {item_name}: {e}")

        # Create folder structure (version subdirectory prevents filename collisions across versions)
        try:
            if base_model_dir:
                item_dir = safe_path_join(output_dir, username, subfolder, base_model_dir, item_name_sanitized, version_name)
                logger_md.info(f"Using baseModel folder structure for {item_name}: {base_model_dir}/{version_name}")
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
        # Fetch image generation metadata (prompts, sampler, etc.) from images API
        version_id = model_version.get('id')
        image_headers = {}
        if token:
            image_headers["Authorization"] = f"Bearer {token}"
        image_meta_by_id = fetch_image_metadata(version_id, image_headers)

        for idx, image in enumerate(images):
            image_url = image.get('url', '')
            if not image_url:
                print(f"Invalid image entry (no URL): {image}")
                continue

            # Use image ID if available, otherwise derive from URL filename
            image_id = image.get('id', '')
            if not image_id:
                # Extract filename from URL (e.g., "38543822.jpeg" from the URL path)
                url_basename = os.path.basename(image_url.split('?')[0]).split('.')[0]
                image_id = url_basename if url_basename else f"img_{idx}"

            image_filename_raw = f"{item_name_sanitized}_{image_id}_for_{file_name}.jpeg"
            image_filename_sanitized = sanitize_name(image_filename_raw, max_length=MAX_PATH_LENGTH, subfolder=subfolder)

            try:
                image_path = safe_path_join(item_dir, image_filename_sanitized)
            except ValueError as e:
                logger_md.error(f"Path traversal blocked for image {image_filename_raw}: {e}")
                continue

            result = download_file_or_image(image_url, image_path, token, username,
                                            max_retries=max_retries, retry_delay=retry_delay)
            counts[result] += 1
            if result == "failed":
                _append_to_file_locked(
                    failed_downloads_file,
                    f"Item Name: {item_name}\nImage URL: {sanitize_url_for_logging(image_url)}\n---\n"
                )

            # Write image details to details.txt (thread-safe)
            details_file = os.path.join(item_dir, "details.txt")
            try:
                _append_to_file_locked(
                    details_file,
                    f"Image ID: {image_id}\nImage URL: {sanitize_url_for_logging(image_url)}\n"
                )
            except OSError as e:
                logger_md.error(f"Error writing image details for {item_name}: {e}")

            # Write separate {image_id}_meta.txt file (matches Image_grabber format)
            meta_key = int(image_id) if str(image_id).isdigit() else None
            meta = image_meta_by_id.get(meta_key) if meta_key else None
            write_image_meta_file(meta, image_id, item_dir, username)

    return item_name, counts


def process_username(username, download_type, token, max_tries, retry_delay_val, max_threads, output_dir, base_model_filters=None):
    """Process a username and download the specified type of content."""
    base_model_filters = base_model_filters or []
    # Validate username for path safety
    try:
        safe_username = sanitize_username_for_path(username)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Processing username: {username}, Download type: {download_type}")
    if base_model_filters:
        print(f"Base model filter: {', '.join(base_model_filters)}")

    # Fetch and categorize all models (returns categorized dict directly)
    categorized_items = fetch_all_models(token, username)
    total_items = sum(len(items) for items in categorized_items.values())

    # Know all model names before migrating folders, so a folder shared by
    # several models under the pre-fix naming is recognised as ambiguous.
    register_models_for_migration(
        entry.get('name') if isinstance(entry, dict) else entry
        for entries in categorized_items.values() for entry in entries
    )

    if download_type == 'All':
        selected_type_count = total_items
        intentionally_skipped = 0
    elif download_type == 'All_except_Checkpoints':
        checkpoint_count = len(categorized_items.get('Checkpoints', []))
        selected_type_count = total_items - checkpoint_count
        intentionally_skipped = checkpoint_count
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
    base_model_skipped = 0

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
                    version_base_model = version.get('baseModel')
                    if not base_model_matches(version_base_model, base_model_filters):
                        base_model_skipped += 1
                        continue

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
    if base_model_filters:
        print(f"  Base model filter skipped versions: {base_model_skipped}")


def fetch_model_by_id(model_id, headers):
    """Fetch a single model by ID from the CivitAI API.

    Returns:
        tuple: (model data dict, error message or None)
    """
    url = f"{BASE_URL}/{model_id}"
    session = get_session()
    try:
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        status = getattr(e.response, 'status_code', 'unknown') if e.response else 'unknown'
        if e.response is not None and e.response.status_code == 404:
            return None, f"Model {model_id} not found."
        return None, f"HTTP error {status} fetching model {model_id}."
    except requests.exceptions.RequestException as e:
        logger_md.error(f"Network error fetching model {model_id}: {type(e).__name__}")
        return None, f"Network error fetching model {model_id}."
    except requests.exceptions.JSONDecodeError:
        return None, f"Invalid JSON response for model {model_id}."


def process_model_ids(model_ids, download_type, token, max_tries, retry_delay_val, max_threads, output_dir, base_model_filters=None):
    """Fetch and download specific models by their IDs."""
    base_model_filters = base_model_filters or []
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    failed_downloads_file = os.path.join(SCRIPT_DIR, "failed_downloads_by_id.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write("Failed Downloads for Model IDs\n\n")

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0
    base_model_skipped = 0

    if base_model_filters:
        print(f"Base model filter: {', '.join(base_model_filters)}")

    for model_id in model_ids:
        print(f"\nFetching model {model_id}...")
        item, error = fetch_model_by_id(model_id, headers)
        if error:
            print(f"  Error: {error}")
            continue

        item_name = item.get('name')
        if not item_name or not isinstance(item_name, str):
            print(f"  Skipping model {model_id}: invalid name")
            continue

        register_models_for_migration([item_name])

        creator = item.get('creator', {})
        username = creator.get('username', 'unknown_user')
        print(f"  Model: {item_name} (by {username})")

        model_versions = item.get('modelVersions', [])
        if not model_versions:
            print(f"  No versions found for model {model_id}")
            continue

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            download_futures = []
            for version in model_versions:
                version_base_model = version.get('baseModel')
                if not base_model_matches(version_base_model, base_model_filters):
                    base_model_skipped += 1
                    continue

                future = executor.submit(
                    download_model_files, item_name, version, item,
                    download_type, failed_downloads_file, username, token, output_dir,
                    max_tries, retry_delay_val, base_model=version.get('baseModel')
                )
                download_futures.append(future)

            for future in tqdm(download_futures, desc=f"  Downloading {item_name}", unit="file", leave=False):
                try:
                    _, counts = future.result()
                    total_downloaded += counts['downloaded']
                    total_skipped += counts['skipped']
                    total_failed += counts['failed']
                except Exception as e:
                    logger_md.exception(f"Unhandled error in download worker: {e}")

    print(f"\nResults for model IDs:")
    print(f"  Downloaded: {total_downloaded}")
    print(f"  Skipped (already existed): {total_skipped}")
    print(f"  Failed: {total_failed}")
    if base_model_filters:
        print(f"  Base model filter skipped versions: {base_model_skipped}")


def get_token_securely(args_token):
    """Retrieve API token from args, environment, .env file, or secure prompt.

    Priority: CLI arg > environment variable > .env file > interactive prompt
    """
    if args_token:
        return args_token

    # Try environment variable, then the .env file next to the script
    token = os.environ.get('CIVITAI_API_TOKEN')
    if not token:
        load_env_file(SCRIPT_DIR)
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


def split_cli_values(values):
    """Split positional or comma-separated CLI values into a clean list."""
    if not values:
        return []
    if isinstance(values, str):
        values = [values]

    result = []
    for value in values:
        result.extend(part.strip() for part in value.split(',') if part.strip())
    return result


def normalize_filter_value(value):
    """Normalize user-provided filter text for case-insensitive matching."""
    return " ".join(value.strip().lower().split())


def parse_base_model_filters(raw_values):
    """Parse base model filters from comma-separated CLI values."""
    filters = []
    for value in split_cli_values(raw_values):
        normalized = normalize_filter_value(value)
        if normalized:
            filters.append(normalized)
    return filters


def base_model_matches(base_model, base_model_filters):
    """Return True when a model version baseModel matches any requested filter."""
    if not base_model_filters:
        return True
    if not base_model or not isinstance(base_model, str):
        return False

    normalized_base_model = normalize_filter_value(base_model)
    for filter_value in base_model_filters:
        if filter_value in normalized_base_model:
            return True
        relaxed_filter = filter_value.rstrip("aeiou")
        if len(relaxed_filter) >= 5 and relaxed_filter in normalized_base_model:
            return True
    return False


def parse_model_ids(raw_values):
    """Parse comma-separated model IDs from CLI or interactive input."""
    model_ids = []
    for raw_id in split_cli_values(raw_values):
        if not raw_id.isdigit():
            print(f"Invalid model ID: {raw_id} (must be a number)")
            sys.exit(1)
        model_ids.append(int(raw_id))
    return model_ids


def resolve_download_type(cli_download_type=None):
    """Use CLI download type when present, otherwise ask interactively."""
    if cli_download_type:
        return cli_download_type

    print(f"Select a download type from: {VALID_DOWNLOAD_TYPES}")
    download_type = input("Download type: ").strip()

    if download_type not in VALID_DOWNLOAD_TYPES:
        print(f"Invalid download type. Must be one of: {VALID_DOWNLOAD_TYPES}")
        sys.exit(1)

    return download_type


def main():
    """Main entry point — all argument parsing and user interaction happens here."""
    parser = argparse.ArgumentParser(description="Download models from CivitAI.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    parser.add_argument('usernames', nargs='*', help='One or more CivitAI usernames. Restores the pre-0.8 positional username CLI.')
    parser.add_argument('--username', '--usernames', dest='usernames_option', help='Username or comma-separated usernames to download.')
    parser.add_argument('--model-id', '--model-ids', '--model_id', '--model_ids', dest='model_ids_option', help='Model ID or comma-separated model IDs to download.')
    parser.add_argument('--download-type', '--download_type', choices=VALID_DOWNLOAD_TYPES, help='Content type to download.')
    parser.add_argument('--base-model', '--base-models', '--base_model', '--base_models', dest='base_models', help='Base model name or comma-separated names to include, matched case-insensitively against modelVersions[].baseModel.')
    parser.add_argument('--token', type=str, help='CivitAI API token (prefer CIVITAI_API_TOKEN env var instead)')
    parser.add_argument('--max-retries', '--max_tries', dest='max_retries', type=int, default=3, help='Maximum number of retries (default: 3)')
    parser.add_argument('--retry-delay', '--retry_delay', dest='retry_delay', type=int, default=10, help='Delay between retries in seconds (default: 10)')
    parser.add_argument('--max-threads', '--max_threads', dest='max_threads', type=int, default=3, help='Maximum number of concurrent downloads (default: 3)')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--max-filename-length', '--max_filename_length', dest='max_filename_length', type=int,
                        help='Maximum filename length the target filesystem accepts. Probed automatically when omitted '
                             '(255 on most filesystems, 143 on an ecryptfs-encrypted home).')

    args = parser.parse_args()

    usernames = split_cli_values(args.usernames) + split_cli_values(args.usernames_option)
    model_ids = parse_model_ids(args.model_ids_option)
    base_model_filters = parse_base_model_filters(args.base_models)

    if usernames and model_ids:
        print("Error: choose either usernames or model IDs, not both.")
        sys.exit(1)

    if args.max_filename_length is not None and args.max_filename_length < MIN_FILENAME_LENGTH_LIMIT:
        print(f"Error: --max-filename-length must be at least {MIN_FILENAME_LENGTH_LIMIT}.")
        sys.exit(1)

    configure_filename_length_limit(args.output_dir, args.max_filename_length)

    token = get_token_securely(args.token)

    if model_ids:
        download_type = resolve_download_type(args.download_type)
        process_model_ids(model_ids, download_type, token, args.max_retries, args.retry_delay, args.max_threads, args.output_dir, base_model_filters)
        return

    if usernames:
        download_type = resolve_download_type(args.download_type)
        for username in usernames:
            process_username(username, download_type, token, args.max_retries, args.retry_delay, args.max_threads, args.output_dir, base_model_filters)
        return

    print("Download mode: (1) By username  (2) By model ID")
    mode = input("Select mode [1]: ").strip() or "1"

    if mode == "2":
        print("Enter model IDs separated by commas (e.g., 12345, 67890):")
        ids_input = input("Model ID(s): ")
        model_ids = parse_model_ids(ids_input)

        if not model_ids:
            print("No model IDs provided. Exiting.")
            sys.exit(1)

        download_type = resolve_download_type(args.download_type)
        process_model_ids(model_ids, download_type, token, args.max_retries, args.retry_delay, args.max_threads, args.output_dir, base_model_filters)
    else:
        print("Enter a username (or multiple usernames separated by commas):")
        usernames_input = input("Username(s): ")
        usernames = split_cli_values(usernames_input)

        if not usernames:
            print("No usernames provided. Exiting.")
            sys.exit(1)

        download_type = resolve_download_type(args.download_type)
        for username in usernames:
            process_username(username, download_type, token, args.max_retries, args.retry_delay, args.max_threads, args.output_dir, base_model_filters)


if __name__ == "__main__":
    main()
