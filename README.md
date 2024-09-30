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
│   │   │       ├── file1.safetensors
│   │   │       ├── image1.jpeg
│   │   │       ├── details.txt
│   │   │       └── description.html
│   │   └── SD 1.5/
│   │       └── model2/
│   │           ├── file3.safetensors
│   │           ├── image2.jpeg
│   │           ├── details.txt
│   │           └── description.html
│   ├── Checkpoints/
│   │   ├── FLUX/
│   │   │   └── model1/
│   │   │       ├── file.safetensors
│   │   │       ├── image.jpeg
│   │   │       ├── details.txt
│   │   │       └── description.html       
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

# How to  use
```
install Python3
```
```
pip install -r requirements.txt
```
```
python civitAI_Model_downloader.py one or multiple usernames space separated
```
You  can also  give the script this 5 extra Arguments
```
--retry_delay 
```
+ default=10,
+ "Retry delay in seconds."
```
--max_tries
```
+ default=3,
+ "Maximum number of retries."
```
--max_threads
```
 + default=5, 
 + "Maximum number of concurrent threads.Too many produces API Failure."
```
--download_type
```
+ Lora
+ Checkpoints
+ Embeddings
+ Training_Data
+ Other
+ Default = All
```
--token 
```
default=None
+ "It will only Download the Public availabe Models"
+ "Provide a Token and it can also Download those Models behind the CivitAI Login."
+ If you forgot to Provide a Token the Script asks for your token.

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

# 0.7 New Feature
**Triggerwords text File**
- Added functionality to create a "triggerWords.txt" file for each model.
- This file contains the trigger words associated with the model.
- The "trainedWords.txt" file is saved in the same directory as the model files.


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




