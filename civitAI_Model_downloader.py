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

def process_username(username):
    # Format the URL with username, types, and nsfw parameter
    base_url = "https://civitai.com/api/v1/models"
    params = {
        "username": username,
        "token": token
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    # Set the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Create a session object for making multiple requests
    session = requests.Session()

    def sanitize_name(name):
        # Replace problematic characters with an underscore
        name = re.sub(r'[\\/*?:"<>|]', '_', name)
        # Replace multiple underscores with a single one
        name = re.sub(r'__+', '_', name)
        # Remove leading and trailing underscores
        name = name.strip('_')
        # Optionally truncate the name if it's too long
        max_length = 255
        if len(name) > max_length:
            name = name[:max_length]
        return name

    # Function to download a file or image from the provided URL
    def download_file_or_image(url, output_path, retry_count=0, max_retries=3):
        progress_bar = None
        try:
            response = session.get(url, stream=True)
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
    def download_model_files(item_name, model_version, item):
        files = model_version.get('files', [])
        images = model_version.get('images', [])
        downloaded = False
        model_id = item['id']
        model_url = f"https://civitai.com/models/{model_id}"
        item_name_sanitized = sanitize_name(item_name)
        item_dir = os.path.join(output_dir, username, item_name_sanitized)  # Use the username here
        os.makedirs(item_dir, exist_ok=True)
        existing_files_count = sum(os.path.exists(os.path.join(item_dir, sanitize_name(file.get('name', '')))) for file in files)
        if existing_files_count == len(files):
            downloaded = True
        model_images = {}  # Dictionary to store image filenames associated with the model
        for file in files:
            file_name = file.get('name', '')  # Use empty string as default if 'name' key is missing
            file_url = file.get('downloadUrl', '')  # Use empty string as default if 'downloadUrl' key is missing
            # Add token to the file URL
            if '?' in file_url:
                file_url += f"&token={token}"
            else:
                file_url += f"?token={token}"
            # Skip download if the file already exists
            file_name_sanitized = sanitize_name(file_name)
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
                print(f"Error downloading file: {file_url}")
            # Update the details file
            details_file = os.path.join(item_dir, "details.txt")
            with open(details_file, "a") as f:
                f.write(f"Model URL: {model_url}\n")
                f.write(f"File Name: {file_name}\n")
                f.write(f"File URL: {file_url}\n")
        for image in images:
            image_id = image.get('id', '')  # Use empty string as default if 'id' key is missing
            image_url = image.get('url', '')  # Use empty string as default if 'url' key is missing
            # Skip download if the image already exists
            image_filename_raw = f"{item_name}_{image_id}_for_{file_name}.jpeg"
            image_filename_sanitized = sanitize_name(image_filename_raw)
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
                print(f"Error downloading image: {image_url}")
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
                future = executor.submit(download_model_files, item_name, version, item)
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



# Process each username one by one
for username in usernames:
    process_username(username)

    print(f"Download completed for username: {username}")
