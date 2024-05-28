import requests
import logging
import argparse
import os

# Setting up logger for errors only
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "fetch_all_models_ERROR_LOG.txt")
logging.basicConfig(filename=file_path, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def categorize_item(item):
    """Categorize the item based on JSON type."""
    item_type = item.get("type", "").upper()
    file_name = item.get("name", "")

    if item_type == 'CHECKPOINT':
        return 'Checkpoints'
    elif item_type == 'TEXTUALINVERSION':
        return 'Embeddings'
    elif item_type == 'LORA':
        return 'Lora'
    elif item_type == 'TRAINING_DATA':
        return 'Training_Data'
    else:
        return 'Other'

def search_for_training_data_files(item):
    """Search for files with type 'Training Data' in the item's model versions."""
    training_data_files = []
    model_versions = item.get("modelVersions", [])
    for version in model_versions:
        for file in version.get("files", []):
            if file.get("type") == "Training Data":
                training_data_files.append(file.get("name", ""))
    return training_data_files

def fetch_all_models(token, username):
    base_url = "https://civitai.com/api/v1/models"
    categorized_items = {
        'Checkpoints': [],
        'Embeddings': [],
        'Lora': [],
        'Training_Data': [],
        'Other': []
    }
    other_item_types = []

    next_page = f"{base_url}?username={username}&token={token}&nsfw=true"
    first_next_page = None

    while next_page:
        response = requests.get(next_page)
        data = response.json()
        for item in data.get("items", []):
            try:
                # Check for top-level categorization
                category = categorize_item(item)
                categorized_items[category].append(item.get("name", ""))
                
                # Check for deep nested "Training Data" files
                training_data_files = search_for_training_data_files(item)
                if training_data_files:
                    categorized_items['Training_Data'].extend(training_data_files)

                if category == 'Other':
                    other_item_types.append((item.get("name", ""), item.get("type", None)))
            except Exception as e:
                logger.error(f"Error categorizing item: {item} - {e}")

        metadata = data.get('metadata', {})
        next_page = metadata.get('nextPage')
        if first_next_page is None:
            first_next_page = next_page
        if next_page and next_page == first_next_page and next_page != next_page or not metadata:
            logger.error("Termination condition met: first nextPage URL repeated.")
            break    

    total_count = sum(len(items) for items in categorized_items.values())

    # Write the summary
    file_path = os.path.join(script_dir, f"{username}.txt")
    with open(file_path, "w", encoding='utf-8') as file:
        file.write("Summary:\n")
        file.write(f"Total - Count: {total_count}\n")
        for category, items in categorized_items.items():
            file.write(f"{category} - Count: {len(items)}\n")
        file.write("\nDetailed Listing:\n")

        # Write the detailed listing
        for category, items in categorized_items.items():
            file.write(f"{category} - Count: {len(items)}\n")
            if category == 'Other':
                for item_name, item_type in other_item_types:
                    file.write(f"{category} - Item: {item_name} - Type: {item_type}\n")
            else:
                for item_name in items:
                    file.write(f"{category} - Item: {item_name}\n")
            file.write("\n")

    return categorized_items

def main():
    parser = argparse.ArgumentParser(description="Fetch and categorize models.")
    parser.add_argument("--token", type=str, help="API token.")
    parser.add_argument("--username", type=str, help="Username to fetch models for.")
    args = parser.parse_args()

    fetch_all_models(args.token, args.username)

if __name__ == "__main__":
    main()
