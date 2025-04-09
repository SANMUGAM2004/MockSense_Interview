import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the GenAI client
genai.configure(api_key=api_key)

# List models and print info
models = genai.list_models()
for model in models:
    print(model.name, "— supports generateContent:", "generateContent" in model.supported_generation_methods)