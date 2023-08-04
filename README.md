# Civit-Model-grabber
The script downloads both models (such as Lora, Lycoris, Embedding, etc.) and related images from a given CivitAI username, organizing them into appropriate directories and maintaining details in a text file. 

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
python civitAI_Model_downloader.py username 
```
You  can also  give the script this 3 extra Arguments
```
--retry_delay= 
```
default=10,"Retry delay in seconds."
```
--max_tries=
```
default=3, "Maximum number of retries."
```
--max_threads=
```
 default=5, "Maximum number of concurrent threads.Too many produces API Failure."

