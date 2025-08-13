import re
import json
import requests
import logging
import urllib.parse
import os
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import argparse
from fetch_all_models import fetch_all_models
import sys

# Constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE_PATH = os.path.join(SCRIPT_DIR, "civitAI_Model_downloader.txt")
OUTPUT_DIR = "model_downloads"
MAX_PATH_LENGTH = 200
VALID_DOWNLOAD_TYPES = ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', 'All']
BASE_URL = "https://civitai.com/api/v1/models"

logger_md = logging.getLogger('md')
logger_md.setLevel(logging.DEBUG)
file_handler_md = logging.FileHandler(LOG_FILE_PATH, encoding='utf-8')
file_handler_md.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler_md.setFormatter(formatter)
logger_md.addHandler(file_handler_md)

# Argument parsing
parser = argparse.ArgumentParser(description="Download model files and images from Civitai API.")
parser.add_argument("usernames", nargs='+', type=str, help="Enter one or more usernames you want to download from.")
parser.add_argument("--retry_delay", type=int, default=10, help="Retry delay in seconds.")
parser.add_argument("--max_tries", type=int, default=3, help="Maximum number of retries.")
parser.add_argument("--max_threads", type=int, default=5, help="Maximum number of concurrent threads. Too many produces API Failure.")
parser.add_argument("--token", type=str, default=None, help="API Token for Civitai.")
parser.add_argument("--download_type", type=str, default=None, help="Specify the type of content to download: 'Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', or 'All' (default).")
parser.add_argument("--verify_hashes", action="store_true", help="Verify downloaded files using available hashes (BLAKE3 > CRC32 > SHA256 priority).")
parser.add_argument("--force_redownload", action="store_true", help="Force redownload all files, ignoring existing ones.")
args = parser.parse_args()

# Prompt the user for the token if it's not provided via command line
if args.token is None:
    args.token = input("Please enter your Civitai API token: ")

# Initialize variables
usernames = args.usernames
retry_delay = args.retry_delay
max_tries = args.max_tries
max_threads = args.max_threads
token = args.token

def calculate_file_hash(file_path, hash_type="BLAKE3"):
    """
    Calculate hash of a file using specified algorithm.
    
    Args:
        file_path: Path to file
        hash_type: "BLAKE3", "CRC32", or "SHA256"
    
    Returns:
        str: Uppercase hex hash or None on error
    """
    try:
        with open(file_path, "rb") as f:
            if hash_type == "CRC32":
                import zlib
                crc = 0
                for byte_block in iter(lambda: f.read(65536), b""):  # Larger chunks for CRC32
                    crc = zlib.crc32(byte_block, crc)
                return f"{crc & 0xffffffff:08X}"
            
            elif hash_type == "BLAKE3":
                try:
                    import blake3
                    hasher = blake3.blake3()
                    for byte_block in iter(lambda: f.read(65536), b""):
                        hasher.update(byte_block)
                    return hasher.hexdigest().upper()
                except ImportError:
                    logger_md.warning("BLAKE3 not available, falling back to CRC32")
                    return calculate_file_hash(file_path, "CRC32")
            
            elif hash_type == "SHA256":
                sha256_hash = hashlib.sha256()
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
                return sha256_hash.hexdigest().upper()
            
            else:
                logger_md.error(f"Unsupported hash type: {hash_type}")
                return None
                
    except (IOError, OSError) as e:
        logger_md.error(f"Error calculating {hash_type} hash for {file_path}: {e}")
        return None

def get_best_hash_for_verification(hashes_dict):
    """
    Get the best available hash for verification from API response.
    Priority: BLAKE3 > CRC32 > SHA256 (fastest to slowest)
    
    Args:
        hashes_dict: Dictionary of hashes from API response
    
    Returns:
        tuple: (hash_type, hash_value) or (None, None) if no suitable hash found
    """
    if not hashes_dict:
        return None, None
    
    # Priority order: BLAKE3 (fastest), CRC32 (very fast), SHA256 (slow)
    hash_priority = ["BLAKE3", "CRC32", "SHA256"]
    
    for hash_type in hash_priority:
        if hash_type in hashes_dict:
            return hash_type, hashes_dict[hash_type]
    
    # If none of the preferred hashes are available, log available hashes
    logger_md.warning(f"No preferred hash types found. Available hashes: {list(hashes_dict.keys())}")
    return None, None
def verify_file_integrity(file_path, expected_size_kb=None, expected_hashes=None):
    """
    Verify if a downloaded file is complete and valid.
    
    Args:
        file_path: Path to the file to verify
        expected_size_kb: Expected file size in KB from API
        expected_hashes: Dictionary of hashes from API response
    
    Returns:
        tuple: (is_valid, reason)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    try:
        # Check file size
        actual_size = os.path.getsize(file_path)
        if expected_size_kb is not None:
            expected_size_bytes = expected_size_kb * 1024
            # Allow 1% tolerance for size differences
            size_tolerance = max(1024, expected_size_bytes * 0.01)  # At least 1KB tolerance
            
            if abs(actual_size - expected_size_bytes) > size_tolerance:
                return False, f"Size mismatch: expected {expected_size_bytes} bytes, got {actual_size} bytes"
        
        # Check for minimum file size for certain types
        if file_path.endswith('.safetensors') and actual_size < 4 * 1024 * 1024:  # 4MB minimum
            return False, f"File too small: {actual_size} bytes (minimum 4MB expected for .safetensors)"
        
        # Verify hash if provided and hash verification is enabled
        if args.verify_hashes and expected_hashes:
            hash_type, expected_hash = get_best_hash_for_verification(expected_hashes)
            
            if hash_type and expected_hash:
                logger_md.info(f"Verifying {file_path} using {hash_type} hash...")
                actual_hash = calculate_file_hash(file_path, hash_type)
                
                if actual_hash and actual_hash != expected_hash.upper():
                    return False, f"{hash_type} hash mismatch: expected {expected_hash}, got {actual_hash}"
                elif actual_hash:
                    logger_md.info(f"{hash_type} hash verification passed for {file_path}")
            else:
                logger_md.warning(f"No suitable hash available for verification of {file_path}")
        
        return True, "File is valid"
        
    except (OSError, IOError) as e:
        return False, f"Error accessing file: {e}"

def should_skip_download(file_path, file_info):
    """
    Determine if a file should be skipped based on existing file verification.
    
    Args:
        file_path: Path where the file would be downloaded
        file_info: Dictionary containing file information from API
    
    Returns:
        bool: True if file should be skipped, False if it should be downloaded
    """
    if args.force_redownload:
        return False
    
    if not os.path.exists(file_path):
        return False
    
    # Extract file information
    expected_size_kb = file_info.get('sizeKB')
    expected_hashes = file_info.get('hashes', {})
    
    # Verify file integrity
    is_valid, reason = verify_file_integrity(file_path, expected_size_kb, expected_hashes)
    
    if is_valid:
        logger_md.info(f"Skipping {file_path}: {reason}")
        return True
    else:
        logger_md.warning(f"Re-downloading {file_path}: {reason}")
        # Remove the invalid file
        try:
            os.remove(file_path)
        except OSError as e:
            logger_md.error(f"Could not remove invalid file {file_path}: {e}")
        return False

# Function to sanitize directory names
def sanitize_directory_name(name):
    return name.rstrip()  # Remove trailing whitespace characters

# Create output directory
OUTPUT_DIR = sanitize_directory_name(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create session
session = requests.Session()

# Validate download type
if args.download_type:
    download_type = args.download_type
    if download_type not in VALID_DOWNLOAD_TYPES:
        print("Error: Invalid download type specified.")
        print("Valid download types are: 'Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other', or 'All'.")
        sys.exit(1)
else:
    while True:
        download_type = input("Please enter the type of content to download (Lora, Checkpoints, Embeddings, 'Training_Data', Other, or All): ")
        if download_type in VALID_DOWNLOAD_TYPES:
            break
        else:
            print("Invalid download type. Please try again.")

def read_summary_data(username):
    """Read summary data from a file."""
    summary_path = os.path.join(SCRIPT_DIR, f"{username}.txt")
    data = {}
    try:
        with open(summary_path, 'r', encoding='utf-8') as file:
            for line in file:
                if 'Total - Count:' in line:
                    total_count = int(line.strip().split(':')[1].strip())
                    data['Total'] = total_count
                elif ' - Count:' in line:
                    category, count = line.strip().split(' - Count:')
                    data[category.strip()] = int(count.strip())
    except FileNotFoundError:
        print(f"File {summary_path} not found.")
    return data

def sanitize_name(name, folder_name=None, max_length=MAX_PATH_LENGTH, subfolder=None, output_dir=None, username=None):
    """Sanitize a name for use as a file or folder name."""
    base_name, extension = os.path.splitext(name)

    if folder_name and base_name == folder_name:
        return name

    if folder_name:
        base_name = base_name.replace(folder_name, "").strip("_")

    # Remove problematic characters and control characters
    base_name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', base_name)

    # Handle reserved names (Windows specific)
    reserved_names = {"CON", "PRN", "AUX", "NUL"} | {f"COM{i}" for i in range(1, 10)} | {f"LPT{i}" for i in range(1, 10)}
    if base_name.upper() in reserved_names:
        base_name = '_'

    # Reduce multiple underscores to single and trim leading/trailing underscores and dots
    base_name = re.sub(r'__+', '_', base_name).strip('_.')
    
    # Calculate max length of base name considering the path length
    if subfolder and output_dir and username:
        path_length = len(os.path.join(output_dir, username, subfolder))
        max_base_length = max_length - len(extension) - path_length
        base_name = base_name[:max_base_length].rsplit('_', 1)[0]

    sanitized_name = base_name + extension
    return sanitized_name.strip()

def download_file_or_image(url, output_path, file_info=None, retry_count=0, max_retries=max_tries):
    """Download a file or image from the provided URL with integrity verification."""
    
    # Check if we should skip this download
    if file_info and should_skip_download(output_path, file_info):
        return True  # File exists and is valid, skip download
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    progress_bar = None
    temp_path = output_path + '.tmp'  # Use temporary file during download
    
    try:
        response = session.get(url, stream=True, timeout=(20, 40))
        if response.status_code == 404:
            print(f"File not found: {url}")
            return False
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, leave=False, 
                           desc=f"Downloading {os.path.basename(output_path)}")
        
        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    progress_bar.update(len(chunk))
                    file.write(chunk)
        progress_bar.close()
        
        # Verify the downloaded file
        if file_info:
            is_valid, reason = verify_file_integrity(temp_path, 
                                                   file_info.get('sizeKB'), 
                                                   file_info.get('hashes', {}))
            if not is_valid:
                os.remove(temp_path)
                if retry_count < max_retries:
                    print(f"Download verification failed for {output_path}: {reason}. Retrying in {retry_delay} seconds (attempt {retry_count + 1}).")
                    time.sleep(retry_delay)
                    return download_file_or_image(url, output_path, file_info, retry_count + 1, max_retries)
                else:
                    print(f"Download verification failed for {output_path} after {max_retries} attempts: {reason}")
                    return False
        
        # Move temp file to final location
        os.rename(temp_path, output_path)
        logger_md.info(f"Successfully downloaded and verified: {output_path}")
        return True
        
    except (requests.RequestException, Exception) as e:
        if progress_bar:
            progress_bar.close()
        
        # Clean up temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        
        if retry_count < max_retries:
            print(f"Error downloading {url}: {e}. Retrying in {retry_delay} seconds (attempt {retry_count + 1}).")
            time.sleep(retry_delay)
            return download_file_or_image(url, output_path, file_info, retry_count + 1, max_retries)
        else:
            download_errors_log = os.path.join(SCRIPT_DIR, f'{username}.download_errors.log')
            with open(download_errors_log, 'a', encoding='utf-8') as log_file:
                log_file.write(f"Failed to download {url} after {max_retries} attempts. Error: {e}\n")
            return False

def download_model_files(item_name, model_version, item, download_type, failed_downloads_file):
    """Download related image and model files for each model version."""
    files = model_version.get('files', [])
    images = model_version.get('images', [])
    downloaded = False
    model_id = item['id']
    version_id = model_version.get('id', '')
    version_name = model_version.get('name', 'Unknown Version')
    model_url = f"https://civitai.com/models/{model_id}"
    
    # Create unique folder name combining item name and version name
    combined_name = f"{item_name} - {version_name}"
    item_name_sanitized = sanitize_name(combined_name, max_length=MAX_PATH_LENGTH)
    model_images = {}
    item_dir = None

    # Extract the description and baseModel from the version
    description = item.get('description') or ''
    version_description = model_version.get('description') or ''
    # Combine both descriptions
    full_description = f"<h1>Model Description</h1>\n{description}\n\n<h1>Version: {version_name}</h1>\n{version_description}"
    
    base_model = model_version.get('baseModel')  # Get baseModel from version, not item
    trigger_words = model_version.get('trainedWords', [])

    for file in files:
        file_name = file.get('name', '')
        file_url = file.get('downloadUrl', '')

        # Determine subfolder (existing logic)
        if file_name.endswith('.zip'):
            if 'type' in item and item['type'] == 'LORA':
                subfolder = 'Lora'
            elif 'type' in item and item['type'] == 'Training_Data':
                subfolder = 'Training_Data'
            else:
                subfolder = 'Other'
        elif file_name.endswith('.safetensors'):
            if 'type' in item:
                if item['type'] == 'Checkpoint':
                    subfolder = 'Checkpoints'
                elif item['type'] == 'TextualInversion':
                    subfolder = 'Embeddings'
                elif item['type'] in ['VAE', 'LoCon']:
                    subfolder = 'Other'
                else:
                    subfolder = 'Lora'
            else:
                subfolder = 'Lora'
        elif file_name.endswith('.pt'):
            if 'type' in item and item['type'] == 'TextualInversion':
                subfolder = 'Embeddings'
            else:
                subfolder = 'Other'
        else:
            subfolder = 'Other'

        if download_type != 'All' and download_type != subfolder:
            continue

        # Create folder structure with version-specific folders
        if base_model:
            item_dir = os.path.join(OUTPUT_DIR, username, subfolder, base_model, item_name_sanitized)
            logging.info(f"Using baseModel folder structure for {item_name} - {version_name}: {base_model}")
        else:
            item_dir = os.path.join(OUTPUT_DIR, username, subfolder, item_name_sanitized)
            logging.info(f"No baseModel found for {item_name} - {version_name}, using standard folder structure")

        try:
            os.makedirs(item_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"Error creating directory for {item_name} - {version_name}: {str(e)}")
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name} - {version_name}\n")
                f.write(f"Model URL: {model_url}\n")
                f.write("---\n")
            return f"{item_name} - {version_name}", False, model_images

        # Create and write to the description file (version-specific)
        description_file = os.path.join(item_dir, "description.html")
        if not os.path.exists(description_file) or args.force_redownload:
            with open(description_file, "w", encoding='utf-8') as f:
                f.write(full_description)

        # Write version-specific trigger words
        trigger_words_file = os.path.join(item_dir, "triggerWords.txt")
        if not os.path.exists(trigger_words_file) or args.force_redownload:
            with open(trigger_words_file, "w", encoding='utf-8') as f:
                for word in trigger_words:
                    f.write(f"{word}\n")

        # Write version info file
        version_info_file = os.path.join(item_dir, "version_info.txt")
        if not os.path.exists(version_info_file) or args.force_redownload:
            with open(version_info_file, "w", encoding='utf-8') as f:
                f.write(f"Model ID: {model_id}\n")
                f.write(f"Version ID: {version_id}\n")
                f.write(f"Version Name: {version_name}\n")
                f.write(f"Base Model: {base_model}\n")
                f.write(f"Published At: {model_version.get('publishedAt', 'Unknown')}\n")
                f.write(f"Download Count: {model_version.get('stats', {}).get('downloadCount', 'Unknown')}\n")
                f.write(f"Rating: {model_version.get('stats', {}).get('rating', 'Unknown')}\n")

        if '?' in file_url:
            file_url += f"&token={token}&nsfw=true"
        else:
            file_url += f"?token={token}&nsfw=true"

        file_name_sanitized = sanitize_name(file_name, item_name, max_length=MAX_PATH_LENGTH, subfolder=subfolder)
        file_path = os.path.join(item_dir, file_name_sanitized)

        if not file_name or not file_url:
            print(f"Invalid file entry: {file}")
            continue

        # Pass file info for verification
        success = download_file_or_image(file_url, file_path, file)
        if success:
            downloaded = True
        else:
            with open(failed_downloads_file, "a", encoding='utf-8') as f:
                f.write(f"Item Name: {item_name} - {version_name}\n")
                f.write(f"File URL: {file_url}\n")
                f.write("---\n")

        details_file = sanitize_directory_name(os.path.join(item_dir, "details.txt"))
        with open(details_file, "a", encoding='utf-8') as f:
            f.write(f"Model URL: {model_url}\n")
            f.write(f"Version ID: {version_id}\n")
            f.write(f"Version Name: {version_name}\n")
            f.write(f"File Name: {file_name}\n")
            f.write(f"File URL: {file_url}\n")

    if item_dir is not None:
        for image in images:
            image_id = image.get('id', '')
            image_url = image.get('url', '')

            image_filename_raw = f"{item_name}_{version_name}_{image_id}.jpeg"
            image_filename_sanitized = sanitize_name(image_filename_raw, item_name, max_length=MAX_PATH_LENGTH, subfolder=subfolder)
            image_path = os.path.join(item_dir, image_filename_sanitized)
            if not image_id or not image_url:
                print(f"Invalid image entry: {image}")
                continue

            # Images don't have size/hash info from API, so pass None
            success = download_file_or_image(image_url, image_path, None)
            if success:
                downloaded = True
            else:
                with open(failed_downloads_file, "a", encoding='utf-8') as f:
                    f.write(f"Item Name: {item_name} - {version_name}\n")
                    f.write(f"Image URL: {image_url}\n")
                    f.write("---\n")

            details_file = sanitize_directory_name(os.path.join(item_dir, "details.txt"))
            with open(details_file, "a", encoding='utf-8') as f:
                f.write(f"Image ID: {image_id}\n")
                f.write(f"Image URL: {image_url}\n")

    return f"{item_name} - {version_name}", downloaded, model_images

def process_username(username, download_type):
    """Process a username and download the specified type of content."""
    print(f"Processing username: {username}, Download type: {download_type}")
    if args.verify_hashes:
        print("Hash verification enabled - using BLAKE3/CRC32 for fast verification")
    
    fetch_user_data = fetch_all_models(token, username)
    summary_data = read_summary_data(username)
    total_items = summary_data.get('Total', 0)

    if download_type == 'All':
        selected_type_count = total_items
        intentionally_skipped = 0
    else:
        selected_type_count = summary_data.get(download_type, 0)
        intentionally_skipped = total_items - selected_type_count

    params = {
        "username": username,
        "token": token
    }
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}&nsfw=true"

    headers = {
        "Content-Type": "application/json"
    }

    failed_downloads_file = os.path.join(SCRIPT_DIR, f"failed_downloads_{username}.txt")
    with open(failed_downloads_file, "w", encoding='utf-8') as f:
        f.write(f"Failed Downloads for Username: {username}\n\n")

    initial_url = url
    next_page = url
    first_next_page = None

    while True:
        if next_page is None:
            print("End of pagination reached: 'next_page' is None.")
            break

        retry_count = 0
        max_retries = max_tries
        retry_delay = args.retry_delay

        while retry_count < max_retries:
            try:
                response = session.get(next_page, headers=headers)
                response.raise_for_status()
                data = response.json()
                break  # Exit retry loop on successful response
            except (requests.RequestException, TimeoutError, json.JSONDecodeError) as e:
                print(f"Error making API request or decoding JSON response: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("Maximum retries exceeded. Exiting.")
                    exit()

        items = data['items']
        metadata = data.get('metadata', {})
        next_page = metadata.get('nextPage')

        if not metadata and not items:
            print("Termination condition met: 'metadata' is empty.")
            break

        if first_next_page is None:
            first_next_page = next_page

        executor = ThreadPoolExecutor(max_workers=max_threads)
        download_futures = []
        downloaded_item_names = set()

        for item in items:
            item_name = item['name']
            model_versions = item['modelVersions']
            if item_name in downloaded_item_names:
                continue
            downloaded_item_names.add(item_name)

            for version in model_versions:
                # Include baseModel in the item dictionary
                item_with_base_model = item.copy()
                item_with_base_model['baseModel'] = version.get('baseModel')
                
                future = executor.submit(download_model_files, item_name, version, item_with_base_model, download_type, failed_downloads_file)
                download_futures.append(future)

        for future in tqdm(download_futures, desc="Downloading Files", unit="file", leave=False):
            future.result()

        executor.shutdown()
    
    if download_type == 'All':
        downloaded_count = sum(len(os.listdir(os.path.join(OUTPUT_DIR, username, category))) for category in ['Lora', 'Checkpoints', 'Embeddings', 'Training_Data', 'Other'] if os.path.exists(os.path.join(OUTPUT_DIR, username, category)))
    else:
        downloaded_count = len(os.listdir(os.path.join(OUTPUT_DIR, username, download_type))) if os.path.exists(os.path.join(OUTPUT_DIR, username, download_type)) else 0

    failed_count = selected_type_count - downloaded_count

    print(f"Total items for username {username}: {total_items}")
    print(f"Downloaded items for username {username}: {downloaded_count}")
    print(f"Intentionally skipped items for username {username}: {intentionally_skipped}")
    print(f"Failed items for username {username}: {failed_count}")

if __name__ == "__main__":
    for username in usernames:
        process_username(username, download_type)
