from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vec = emb.embed_query("hello world")
print(len(vec))