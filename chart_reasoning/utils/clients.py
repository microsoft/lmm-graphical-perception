# import openai
# import os
# from dotenv import load_dotenv
# import pathlib
# from azure.identity import DefaultAzureCredential, get_bearer_token_provider
# from openai import AzureOpenAI


# class OpenAIClients(object):
    
# 	def __init__(self):

# 		FILE_PATH = pathlib.Path(__file__).parent.resolve()
# 		load_dotenv(os.path.join(FILE_PATH, '..', 'openai-keys.env'))
# 		token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")


# 		# setup for gpt3.5
# 		self.gpt35_client = openai.AzureOpenAI(
# 			azure_endpoint = os.getenv("GPT35_ENDPOINT"),  
# 			api_key=os.getenv("GPT35_KEY"),  
# 			api_version="2023-10-01-preview"
# 		)
# 		self.gpt35_model = os.getenv("GPT35_MODEL")

# 		# setup gpt4
# 		# self.gpt4_client = openai.AzureOpenAI(
# 		# 	azure_endpoint = os.getenv("GPT4_ENDPOINT"), 
# 		# 	api_key=os.getenv("GPT4_KEY"),  
# 		# 	api_version="2023-10-01-preview"
# 		# )
# 		self.gpt4_client = openai.AzureOpenAI(
# 			azure_endpoint = os.getenv("GPT4_ENDPOINT"), 
# 			api_key=os.getenv("GPT4_KEY"),  
# 			api_version="2024-02-15-preview")
# 		self.gpt4_model=os.getenv("GPT4_MODEL")
# 		self.gpt4_vision_model=os.getenv("GPT4_VISION_MODEL")


import os
from dotenv import load_dotenv
import pathlib
from openai import OpenAI

class OpenAIClients(object):
    
	def __init__(self):

		FILE_PATH = pathlib.Path(__file__).parent.resolve()
		load_dotenv(os.path.join(FILE_PATH, '..', '..', 'openai-keys.env'))

		self.gpt4_client = OpenAI(
			api_key=os.getenv("OPENAI_API_KEY"))
		self.gpt4_model="gpt-4o-2024-08-06"
		self.gpt4_vision_model="gpt-4o-2024-08-06"
