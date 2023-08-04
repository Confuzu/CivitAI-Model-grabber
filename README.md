# Civit-Model-grabber
The Script Downloads in bulk both model(Lora,Lycoris,Embeding etc..) and related images,organizing them into appropriate directories in one go from a given CivitAI Username 
 and maintaining details in a text file.

# How to  use
```
python civitai_Model_downloader.py username 
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

