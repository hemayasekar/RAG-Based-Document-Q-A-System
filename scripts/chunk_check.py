from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFDirectoryLoader("docs/")
raw_documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(raw_documents)

print(f"Split {len(raw_documents)} pages into {len(chunks)} chunks")
print("=" * 40)
for i in [0, 1, 50]:
    print(f"CHUNK {i} | {chunks[i].metadata}")
    print(chunks[i].page_content)
    print("=" * 40)

print(f"Split {len(raw_documents)} pages into {len(chunks)} chunks")