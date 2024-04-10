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

# Updates & Bugfixes

Better Errorhandling <br /> 
New function to avoid OSError: [Errno 36] File name too long: <br /> 

Pagination is fixed <br /> 
New Function Multiple Usernames <br /> 


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
You  can also  give the script this 3 extra Arguments
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
