from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

for m in genai.list_models():
    if "embedContent" in m.supported_generation_methods:
        print("EMBED:", m.name)
    if "generateContent" in m.supported_generation_methods:
        print("CHAT: ", m.name)