# Civit-Model-grabber
The script downloads models (such as Lora, Lycoris, Embedding, etc.) and related images from a given CivitAI username, organizing them into appropriate directories and maintaining details in a text file. 

It's designed to download only the files that are not already present in the specified username's folder.
If the user uploads new models, running the script again will download only the newly uploaded files.

Example of Details.txt 
```
Model URL: https://civitai.com/models/ID
File Name: Name of the Model.ending
File URL: https://civitai.com/api/download/models/ID
Image ID: ID
Image URL: https://image.civitai.com/Random_characters/width=450/ID.jpeg
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
+ Other
+ Default = All
```
--token 
```
default=None
+ "It will only Download the Public availabe Models"
+ "Provide a Token and it can also Download those Models behind the CivitAI Login."
+ If you forgot to Provide a Token the Script asks for your token.

You can create your API Key here
 [Account Settings](https://civitai.com/user/account).
 Scoll down until  the end and you  find this Box

![API](https://github.com/Confuzu/CivitAI-Model-grabber/assets/133601702/bc126680-62bd-41db-8211-a47b55d5fd36)

 # Updates & Bugfixes

# Bugfix

Enhanced Character Filtering: <br /> 
The script has been modified to extensively filter out forbidden and problematic characters to prevent issues during the folder creation process. <br />

Error Handling Improvements: <br />
In cases where the script encounters characters that prevent folder creation, it now logs the name and URL of the affected download. <br /> 
This information is recorded in a pre-existing text file, which is automatically named after the user whose content is being downloaded. This update allows users to manually complete downloads if issues arise.<br />

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




