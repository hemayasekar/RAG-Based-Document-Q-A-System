import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()  

BATCH_SIZE = 50       
PAUSE_SECONDS = 40    
EMBED_MODEL = "models/gemini-embedding-001"  


def build_index():
    
    docs = PyPDFDirectoryLoader("docs/").load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    print(f"Loaded {len(docs)} pages -> {len(chunks)} chunks")

    emb = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

    
    store = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i: i + BATCH_SIZE]

        for attempt in range(3):  
            try:
                if store is None:
                    store = FAISS.from_documents(batch, emb)  
                else:
                    store.add_documents(batch)               
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    print("Rate limited, waiting 60s...")
                    time.sleep(60)
                else:
                    raise

        print(f"Embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
        time.sleep(PAUSE_SECONDS)

    
    store.save_local("faiss_index")
    print(f"Indexed {len(chunks)} chunks from {len(docs)} pages.")


if __name__ == "__main__":
    build_index()