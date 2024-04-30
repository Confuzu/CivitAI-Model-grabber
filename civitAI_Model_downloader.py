import re
import json
import requests
import urllib.parse
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
import time
import argparse

#logging only for debugging not productive 
log_file_path = "civitAI_Model_downloader.txt"
logging.basicConfig(filename=log_file_path, level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Download model files and images from Civitai API.")
parser.add_argument("usernames", nargs='+', type=str, help="Enter one or more usernames you want to download from.")
parser.add_argument("--retry_delay", type=int, default=10, help="Retry delay in seconds.")
parser.add_argument("--max_tries", type=int, default=3, help="Maximum number of retries.")
parser.add_argument("--max_threads", type=int, default=5, help="Maximum number of concurrent threads. Too many produces API Failure.")
parser.add_argument("--token", type=str, default=None, help="API Token for Civitai.")
parser.add_argument("--download_type", type=str, default=None, help="Specify the type of content to download: 'Lora', 'Checkpoints', 'Embeddings', 'Other', or 'All' (default).")
args = parser.parse_args()

# Prompt the user for the token if it's not provided via command line
if args.token is None:
    args.token = input("Please enter your Civitai API token: ")

# Command-line arguments
usernames = args.usernames
retry_delay = args.retry_delay
max_tries = args.max_tries
max_threads = args.max_threads
token = args.token

max_path_length = 200

def process_username(username, download_type):
    print(f"Processing username: {username}, Download type: {download_type}")
    # Format the URL with username, types, and nsfw parameter
    base_url = "https://civitai.com/api/v1/models"
    params = {
        "username": username,
        "token": token
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}&nsfw=true"

    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Create a session object for making multiple requests
    session = requests.Session()

    failed_downloads_file = f"failed_downloads_{username}.txt"
    with open(failed_downloads_file, "w") as f:
        f.write(f"Failed Downloads for Username: {username}\n\n")

    def sanitize_name(name, folder_name=None, max_length=200, subfolder=None):
        # Split the name into base name and extension
        base_name, extension = os.path.splitext(name)

        # Check if the base name matches the folder name
        if folder_name and base_name == folder_name:
            # If the base name matches the folder name, use the original name
            sanitized_name = name
        else:
            # Remove the folder name from the base name if provided
            if folder_name:
                base_name = base_name.replace(folder_name, "").strip("_")

            # Replace problematic characters with an underscore
            base_name = re.sub(r'[<>:"/\\|?*\'\’\‘\“\”]', '_', base_name)
            base_name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '_', base_name)

            # Replace reserved names with an underscore
            reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"]
            base_name = '_' if base_name.upper() in reserved_names else base_name

            # Replace multiple underscores with a single one
            base_name = re.sub(r'__+', '_', base_name)

            # Remove leading and trailing underscores and dots
            base_name = base_name.strip('_.')

            # Calculate the maximum allowed length for the base name
            if subfolder:
                max_base_length = max_length - len(extension) - len(os.path.join(output_dir, username, subfolder))
            else:
                max_base_length = max_length - len(extension)

            # Truncate the base name if it exceeds the maximum allowed length
            if len(base_name) > max_base_length:
                base_name = base_name[:max_base_length]

            # Combine the sanitized base name and extension
            sanitized_name = base_name + extension

        return sanitized_name

    # Function to download a file or image from the provided URL
    def download_file_or_image(url, output_path, retry_count=0, max_retries=3):
        progress_bar = None
        try:
            response = session.get(url, stream=True)
            if response.status_code == 404:
                print(f"File not found: {url}")
                return
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, leave=False)
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        progress_bar.update(len(chunk))
                        file.write(chunk)
            progress_bar.close()
            # Check file size after download, only for safetensor files
            if output_path.endswith('.safetensor') and os.path.getsize(output_path) < 4 * 1024 * 1024:  # 4MB
                if retry_count < max_retries:
                    print(f"File {output_path} is smaller than expected. Try to download again (attempt {retry_count}).")
                    time.sleep(retry_delay)
                    download_file_or_image(url, output_path, retry_count + 1, max_retries)
                else:
                    with open('download_errors.log', 'a') as log_file:
                        log_file.write(f"Error downloading file {output_path} from URL {url}: File too small after {max_retries} attempts\n")
        except (requests.RequestException, TimeoutError, ConnectionResetError) as e:
            if progress_bar:
                progress_bar.close()
            if retry_count < max_retries:
                print(f"Error during download: {url}, attempt {retry_count + 1}")
                time.sleep(retry_delay)
                download_file_or_image(url, output_path, retry_count + 1, max_retries)
            else:
                with open('download_errors.log', 'a') as log_file:
                    log_file.write(f"Error downloading file {output_path} from URL {url}: {e} after {max_retries} attempts\n")

        # New function to download the related image and model files for each model version
    
    def download_model_files(item_name, model_version, item, download_type, failed_downloads_file):
        files = model_version.get('files', [])
        images = model_version.get('images', [])
        downloaded = False
        model_id = item['id']
        model_url = f"https://civitai.com/models/{model_id}"
        item_name_sanitized = sanitize_name(item_name, max_length=260)
        model_images = {}  # Initialize the model_images dictionary
        item_dir = None  # Initialize item_dir to None

        for file in files:
            file_name = file.get('name', '')
            file_url = file.get('downloadUrl', '')

            # Determine the appropriate subfolder based on the file extension and JSON response
            if file_name.endswith('.safetensors'):
                if 'type' in item and item['type'] == 'Checkpoint':
                    subfolder = 'Checkpoints'
                elif 'type' in item and item['type'] == 'TextualInversion':
                    subfolder = 'Embeddings'
                else:
                    subfolder = 'Lora'
            elif file_name.endswith('.pt'):
                if 'type' in item and item['type'] == 'TextualInversion':
                    subfolder = 'Embeddings'
                else:
                    subfolder = 'Other_Model_types'
            else:
                subfolder = 'Other_Model_types'

            # Check if the user-specified download_type matches the determined subfolder
            if download_type != 'All' and download_type != subfolder:
                continue

            item_dir = os.path.join(output_dir, username, subfolder, item_name_sanitized)
            try:
                os.makedirs(item_dir, exist_ok=True)
            except OSError as e:
                with open(failed_downloads_file, "a") as f:
                    f.write(f"Item Name: {item_name}\n")
                    f.write(f"Model URL: {model_url}\n")
                    f.write("---\n")
                return item_name, False, model_images
    
            # Add token to the file URL
            if '?' in file_url:
                file_url += f"&token={token}&nsfw=true"
            else:
                file_url += f"?token={token}&nsfw=true"
    
            # Skip download if the file already exists
            file_name_sanitized = sanitize_name(file_name, item_name, max_length=260, subfolder=subfolder)
            file_path = os.path.join(item_dir, file_name_sanitized)
            if os.path.exists(file_path):
                continue
            
            # Skip if 'name' or 'downloadUrl' keys are missing
            if not file_name or not file_url:
                print(f"Invalid file entry: {file}")
                continue
            
            # Download the file
            try:
                download_file_or_image(file_url, file_path)
                downloaded = True
            except (requests.RequestException, TimeoutError):
                with open(failed_downloads_file, "a") as f:
                    f.write(f"Item Name: {item_name}\n")
                    f.write(f"File URL: {file_url}\n")
                    f.write("---\n")

    
            # Update the details file
            details_file = os.path.join(item_dir, "details.txt")
            with open(details_file, "a") as f:
                f.write(f"Model URL: {model_url}\n")
                f.write(f"File Name: {file_name}\n")
                f.write(f"File URL: {file_url}\n")
    
        if item_dir is not None:
            for image in images:
                image_id = image.get('id', '')
                image_url = image.get('url', '')

                # Generate the image filename
                image_filename_raw = f"{item_name}_{image_id}_for_{file_name}.jpeg"
                image_filename_sanitized = sanitize_name(image_filename_raw, item_name, max_length=260, subfolder=subfolder)
                image_path = os.path.join(item_dir, image_filename_sanitized)

                if os.path.exists(image_path):
                    continue
                
                # Skip if 'id' or 'url' keys are missing
                if not image_id or not image_url:
                    print(f"Invalid image entry: {image}")
                    continue
                
                # Download the image
                try:
                    download_file_or_image(image_url, image_path)
                    downloaded = True
                except (requests.RequestException, TimeoutError):
                    with open(failed_downloads_file, "a") as f:
                        f.write(f"Item Name: {item_name}\n")
                        f.write(f"Image URL: {image_url}\n")
                        f.write("---\n")

                # Update the details file
                details_file = os.path.join(item_dir, "details.txt")
                with open(details_file, "a") as f:
                    f.write(f"Image ID: {image_id}\n")
                    f.write(f"Image URL: {image_url}\n")

                # Store the image filename in the model_images dictionary
                if item_name not in model_images:
                    model_images[item_name] = []
                model_images[item_name].append(image_filename_raw)
    
        return item_name, downloaded, model_images

    # Create a directory for the username
    output_dir = "model_downloads"
    output_dir = os.path.join(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    initial_url = url
    next_page = url
    first_next_page = None

    while True:
        # Check if next_page is None before making the request
        if next_page is None:
            print("End of pagination reached: 'next_page' is None.")
            break

        retry_count = 0
        max_retries = 3
        retry_delay = 10

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
        # Process items here (e.g., downloading content)
        metadata = data.get('metadata', {})
        next_page = metadata.get('nextPage')

        # After processing the items, then check if metadata is empty
        if not metadata and not items:  # Adding `not items` ensures we only break if there's truly nothing left to process
            print("Termination condition met: 'metadata' is empty.")
            break

        if first_next_page is None:
            first_next_page = next_page  # Store the first nextPage URL

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
                future = executor.submit(download_model_files, item_name, version, item, download_type, failed_downloads_file)
                download_futures.append(future)
        
        download_results = []
        for future in tqdm(download_futures, desc="Downloading Files", unit="file", leave=False):
            result = future.result()
            download_results.append(result)

        executor.shutdown()
    
        all_item_names_sanitized = {sanitize_name(item['name']) for item in items}
        downloaded_item_names_sanitized = {sanitize_name(item_name) for item_name, downloaded, _ in download_results if downloaded}
        missing_items = all_item_names_sanitized - downloaded_item_names_sanitized
    
        print(f"Download completed for username: {username}")
        print(f"Total items for username: {username}: {len(items)}")
        print(f"Downloaded items for username: {username}: {len(downloaded_item_names)}")
        print(f"Missing items for username: {username}: {len(missing_items)}")
    
        if missing_items:
            print("Missing item names:")
            for item_name in missing_items:
                print(item_name)

if args.download_type:
    download_type = args.download_type
else:
    # Prompt the user for the download type if not provided via command line
    download_type = input("Please enter the type of content to download (Lora, Checkpoints, Embeddings, Other, or All): ")

for username in usernames:
    process_username(username, download_type)
    print(f"Download completed for username: {username}")
    
