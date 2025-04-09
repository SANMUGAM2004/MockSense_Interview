import google.generativeai as genai

genai.configure(api_key="AIzaSyCUUmAWPdNRByj4hAtHntFZbZizGXuojVc")
models = genai.list_models()

for model in models:
    print(model.name, "— supports generateContent:", "generateContent" in model.supported_generation_methods)
