# Civit-Model-grabber
The script Supports different download types: Lora, Checkpoints, Embeddings, Training Data, Other, or All and related images from a given CivitAI username, organizing them into appropriate directories and maintaining details in a text file. 

It's designed to download only the files that are not already present in the specified username's folder.
If the user uploads new models, running the script again will download only the newly uploaded files.

**Example of Details.txt** 
```
Model URL: https://civitai.com/models/ID
File Name: Name of the Model.ending
File URL: https://civitai.com/api/download/models/ID
Image ID: ID
Image URL: https://image.civitai.com/Random_characters/width=450/ID.jpeg
```

**File Structure**  <br /> 
The downloaded files will be organized in the following structure:
```
model_downloads/
├── username1/
│   ├── Lora/
│   │   ├── SDXL 1.0/
│   │   │   └── model1/
│   │   │       ├── v1.0/
│   │   │       │   ├── file1.safetensors
│   │   │       │   ├── image1.jpeg
│   │   │       │   ├── 12345_meta.txt    
│   │   │       │   ├── details.txt
│   │   │       │   ├── triggerWords.txt
│   │   │       │   └── description.html
│   │   │       └── v2.0/
│   │   │           ├── file1.safetensors
│   │   │           ├── image2.jpeg
│   │   │           ├── 67890_meta.txt 
│   │   │           ├── details.txt
│   │   │           ├── triggerWords.txt
│   │   │           └── description.html
│   │   └── SD 1.5/
│   │       └── model2/
│   │           └── v1.0/
│   │               ├── file3.safetensors
│   │               ├── image2.jpeg
│   │               ├── details.txt
│   │               ├── triggerWords.txt
│   │               └── description.html
│   ├── Checkpoints/
│   │   └── FLUX/
│   │       └── model1/
│   │           └── v1.0/
│   │               ├── file.safetensors
│   │               ├── image.jpeg
│   │               ├── details.txt
│   │               ├── triggerWords.txt
│   │               └── description.html
│   ├── Embeddings/
│   ├── Training_Data/
│   └── Other/
└── username2/
    ├── Lora/
    ├── Checkpoints/
    ├── Embeddings/
    ├── Training_Data/
    └── Other/
```

# How to use
```
install Python3
```
```
pip install -r requirements.txt
```

**API Token Setup**

The script needs a CivitAI API token. You can provide it in three ways (checked in this order):
1. CLI argument: `--token YOUR_TOKEN`
2. Environment variable: `CIVITAI_API_TOKEN=YOUR_TOKEN`
3. `.env` file in the script directory containing `CIVITAI_API_TOKEN=YOUR_TOKEN`
4. If none of the above are set, the script will prompt you securely (hidden input)

Without a token, only public models can be downloaded. With a token, models behind the CivitAI login are also accessible.

**Running the script**
```
python civitAI_Model_downloader.py
python civitAI_Model_downloader.py username1 username2 --download-type Lora
python civitAI_Model_downloader.py --username username1,username2 --download-type All
python civitAI_Model_downloader.py --model-ids 12345,67890 --download-type All_except_Checkpoints
python civitAI_Model_downloader.py username1 --download-type Lora --base-model "SDXL 1.0"
python civitAI_Model_downloader.py --version
```
The script will interactively prompt for:
- **Download mode:** Choose (1) By username or (2) By model ID
- **Username(s):** (mode 1) Enter one or multiple usernames separated by commas
- **Model ID(s):** (mode 2) Enter one or multiple model IDs separated by commas (e.g., `12345, 67890`)
- **Download type:** Choose from Lora, Checkpoints, Embeddings, Training_Data, Other, All, or All_except_Checkpoints

**Download by Model ID**

You can download specific models directly by their CivitAI model ID instead of fetching all models from a username. The model ID is the number in the URL, e.g., `https://civitai.com/models/12345` has ID `12345`. Files are organized under the model creator's username automatically.

**Optional CLI Arguments**
```
username1 username2   Positional usernames, compatible with pre-0.8 CLI usage
--username NAMES      Username or comma-separated usernames
--model-ids IDS       Model ID or comma-separated model IDs
--download-type TYPE  Download type: Lora, Checkpoints, Embeddings, Training_Data, Other, All, or All_except_Checkpoints
--base-model NAMES    Base model name or comma-separated names to include, e.g. "SDXL 1.0", "Flux", or "Illustrious"
--version             Print the script version and exit
--token TOKEN         CivitAI API token (prefer env var or .env file instead)
--retry-delay N       Delay between retries in seconds (default: 10)
--max-retries N       Maximum number of retries per file (default: 3)
--max-threads N       Maximum concurrent downloads (default: 3, too many causes API failures)
--output-dir DIR      Output directory (default: model_downloads)
--max-filename-length Overrides the probe for setups where it cannot be measured.
```

**Download Reporting**

After each username, the script reports:
```
Results for username <name>:
  Downloaded: 42
  Skipped (already existed): 100
  Failed: 2
  Type filter skipped: 15
```
Re-running the script will skip files that already exist and only download new or previously failed files.

**Base Model Filter**

Use `--base-model` or `--base-models` to download only model versions whose CivitAI `baseModel` matches the requested text. Matching is case-insensitive and allows partial names, so `--base-model sdxl` will match `SDXL 1.0`. Multiple values can be comma-separated, for example `--base-models "SDXL 1.0,Flux,Illustrious"`.

**Atomic Downloads**

Files are downloaded to a temporary `.tmp` file first, then atomically renamed on completion. This prevents corrupt partial files if the download is interrupted.

**Helper script** `fetch_all_models.py`
```
python fetch_all_models.py --username <USERNAME> --token <API_TOKEN>
```
**Example of username.txt created with helper script fetch_all_models.py**
```
Summary:
Total - Count: 61
Checkpoints - Count: 12
Embeddings - Count: 33
Lora - Count: 11
Training_Data - Count: 2
Other - Count: 3

Detailed Listing:
Checkpoints - Count: 12
Checkpoints - Item: NAME
...

Embeddings - Count: 33
Embeddings - Item: NAME - Embeddings
...

Lora - Count: 11
Lora - Item: NAME
...

Training_Data - Count: 2
Training_Data - Item: NAME_training_data.zip
...

Other - Count: 3
Other - Item: NAME - Type: Other
...
```


You can create your API Key here
 [Account Settings](https://civitai.com/user/account).
 Scoll down until  the end and you  find this Box

![API](https://github.com/Confuzu/CivitAI-Model-grabber/assets/133601702/bc126680-62bd-41db-8211-a47b55d5fd36)


 # Updates & Bugfixes

# 0.9.1  Bugfixes & Improvements

**Folder Names Truncated by a Slash in the Model Name (Issue #39)**
- Model names containing a `/`, e.g. `Hyouuma Style (Anima/Illustrious)`, were treated as filesystem paths, so only the part after the last slash survived and the folder ended up named `Illustrious)`. Image filenames built from the model name were truncated the same way.
- Path separators are now replaced before the name is normalized, so the folder becomes `Hyouuma Style (Anima_Illustrious)`.

**Existing Downloads Are Migrated, Not Fetched Again**
- Downloads are skipped by comparing paths, so the corrected folder name would have made every file of an affected model count as missing.
- Folders left by the previous version are renamed to their corrected name, including the preview images that carry the model name in their own filename.
- A folder is only renamed when it clearly belongs to one model. Under the old naming several models could share one folder, e.g. `Hyouuma Style (Anima/Illustrious)` and `Anus Outline / Visible Through Clothes / Covered Anus (Anima/Illustrious)` both became `Illustrious)`.
- Such folders, folders holding versions the model does not have, and existing target folders are reported and left untouched for manual sorting.

**Filename Length Limit Is now Detected and Enforced**
- The limit is no longer assumed: it is probed once per run against the output directory, because it differs between filesystems. The `.tmp` suffix used for atomic downloads is reserved.
- Only names the filesystem genuinely cannot store are shortened, so names that work today keep their path and stay recognized as already downloaded. Shortening happens on UTF-8 character boundaries and is reported once per name, with the original and the shortened form.
- `--max-filename-length` overrides the probe for setups where it cannot be measured.

# 0.9 Improvements

**Restored CLI Username Support (Issue #36)**
- Usernames can again be passed directly as positional command-line arguments, e.g. `python civitAI_Model_downloader.py username1 username2 --download-type Lora`.

**Base Model Filter (Issue #37)**
- Added `--base-model` to download only model versions to reducing bandwidth for users who only want bases such as `SDXL 1.0`, `Flux`, or `Illustrious`.
- Matching is case-insensitive and supports partial names, so `--base-model sdxl` matches `SDXL 1.0`.
  
**Version Flag**
- Added `--version` so users can confirm the script version without starting the interactive prompt.

# 0.8 Improvements

- Downloads now write to a temporary `.tmp` file and  renamed on completion, prevents partial files seen as finished when the download is interrupted.
- Replaced pagination with `paginate_api()` from `fetch_all_models.py`, gaining circular pagination detection, page limits, and URL validation.
- Download results now distinguish between newly downloaded, skipped (already existed), and failed files instead of lumping skipped files into the failure count.
- Token lookup now supports `CIVITAI_API_TOKEN` environment variable and `.env` file in addition to CLI arg and interactive prompt.
- Usernames and download type are now entered interactively (comma-separated usernames)
- Each version now gets its own directory with its own files, images, trigger words, and description. Falling back to the version ID if the name is empty.
- New Download Mode: All_except_Checkpoints
- Image generation metadata is now saved as separate `{image_id}_meta.txt` files per image, matching the format used by CivitAI_Image_grabber                                                          
    Includes prompt, negative prompt, model, sampler, steps, CFG scale, seed, size, resources, and all other generation parameters                                                                       
    Images without metadata get a `{image_id}_no_meta.txt` with a link to the image on CivitAI    


# 0.7 New Feature
**Triggerwords text File**
- Added functionality to create a "triggerWords.txt" file for each model.
- This file contains the trigger words associated with the model.
- The "triggerWords.txt" file is saved in the same directory as the model files.


# 0.6 New Feature
**Base Model Folder Organization**
- Implemented a new folder structure that organizes downloads based on their base model.
- Downloads are now sorted into subfolders named after their respective base models within each category (Lora, Checkpoints, etc.).
- This organization applies to all categories when base model information is available.
- Folders for categories without base model information remain unchanged
- Improved logging to track base model usage and any related issues.

# 0.5 New Feature 
**Model Description Files**
- These files contain the original description of the model as provided by the creator.
- Description files which are HTML files that can be opened directly in a browser, saving the original descriptions provided by creators in the same directory as the corresponding model files.

# 0.4 New Features & Updates & Bugfixes 

### New features:
- **Download option for Training_Data added:**
  - Automatically creates its own download folder.
  - Saves downloaded ZIP packages, associated images and a `detail.txt` file.
  
- **Introduction of a helper script `fetch_all_models.py`:**
  - Retrieves model information from the CivitAI API based on username and API token.
  - Categorizes the results and summarizes them in a text file `{username}.txt`.
  - Improves the overview of the user content and enables the statistics function.
  - Can also be used standalone with the following command:  
    `python fetch_all_models.py --username <USERNAME> --token <API_TOKEN>`

### Updates:
- **Detection and categorization of new types:**
  - Script now recognizes the types VAE and Locon and assigns them to the category "Other".
  
- **Improvement of the filter for problematic characters:**
  - Optimization of filter functions to better handle problematic characters.

- **Code optimizations:**
  - All global variables are now at the beginning of the script.
  - No more functions inside other functions.
  - Merge lines of code where appropriate for better readability and maintainability.
  
- **Correct allocation of ZIP packages:**
  - ZIP packages are now downloaded to the appropriate folders according to API specifications, e.g. Training_Data, Lora, Other.
  - ZIP packages without a specific category are still downloaded under "Other".

### Bugfixes:
- **Statistics fixed:**
  - The statistics function is now based on the new helper script `fetch_all_models.py`, which improves accuracy and functionality.

# 0.3 Bugfix & Changes

Enhanced Character Filtering: <br /> 
The script has been modified to extensively filter out forbidden and problematic characters to prevent issues during the folder creation process. <br />

Error Handling Improvements: <br />
In cases where the script encounters characters that prevent folder creation, it now logs the name and URL of the affected download. <br /> 
This information is recorded in a pre-existing text file, which is automatically named after the user whose content is being downloaded. This update allows users to manually complete downloads if issues arise.<br />
```
failed_downloads_username.txt
```
Changed from Skipping image to Truncate when path length exceeding the limit. <br /> 


 # 0.2 New Features & Update & Bugfix 
New long awaited Feature <br /> 

Selective Download Options <br />
Users can now choose to download specific content types: <br />
Lora <br />
Checkpoints <br />
Embeddings <br />
Other <br />
Everything but Lora, Checkpoints, Embeddings (grouped under Other_Model_types for less frequently downloaded items) <br /> 
All <br />
is the Default Download Behavior: The default option to download all available content remains if no specific download parameters are set. <br /> 

Command Line and Interactive Enhancements: <br /> 

Command Line Arguments: Users can directly specify their download preference (Lora, Checkpoints, Embedding, Other or All) via command line alongside other startup parameters. <br /> 
Interactive Mode: If no command line arguments are specified, the program will prompt users interactively to select the content they wish to download. Pressing the Enter key activates the default settings to download all content. <br /> 

Folder Structure Update: <br /> 

Organized Storage: The program’s folder structure has been reorganized to support new download options efficiently: <br />
Main directory: model_downloads/ <br />
User-specific subdirectory: Username/ <br />
Content-specific subfolders for Lora, Checkpoints, Embeddings, and Other_Model_types each containing item-specific subfolders. <br />

Bugfix <br /> 
The script will no longer remove the file name if it is written in the same way as the folder name. This could happen from time to time due to the sanitization function of the script. 

# 0.1 Better Errorhandling <br /> 
New function to avoid OSError: [Errno 36] File name too long: <br /> 

Pagination is fixed <br /> 
New Function Multiple Usernames <br /> 




