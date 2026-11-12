from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np

load_dotenv()


raw_documents = PyPDFDirectoryLoader("docs/").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(raw_documents)
print(f"Total chunks: {len(chunks)}")



emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
test_texts = [c.page_content for c in chunks[:5]]
vectors = emb.embed_documents(test_texts)

print(f"Embedded {len(vectors)} chunks, each with {len(vectors[0])} numbers")


q = emb.embed_query("What is the history of artificial intelligence?")

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

for i, v in enumerate(vectors):
    print(f"Question vs chunk {i}: similarity = {cosine(q, v):.3f}")

