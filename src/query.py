from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = FAISS.load_local(
    "faiss_index", emb, allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, max_retries=1)

prompt_template = """Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
Answer:"""

QA_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 8}),
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True,
)

if __name__ == "__main__":
    questions = [
        "What was the Dartmouth workshop?",
        "What is the Turing Test?",
    ]
    for q in questions:
        response = qa_chain.invoke({"query": q})
        print("=" * 60)
        print("Q:", q)
        print("A:", response["result"])
        print("Sources:", [d.metadata["page"] for d in response["source_documents"]])