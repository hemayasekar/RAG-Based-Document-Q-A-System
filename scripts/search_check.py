from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.load_local(
    "faiss_index", emb, allow_dangerous_deserialization=True
)

questions = [
    "What was the Dartmouth workshop?",
    "What is the Turing Test?",
]

for q in questions:
    print("=" * 60)
    print("QUESTION:", q)
    results = vector_store.similarity_search_with_score(q, k=4)
    for doc, score in results:
        print(f"\n[score={score:.3f}] page {doc.metadata['page']}")
        print(doc.page_content[:300], "...")