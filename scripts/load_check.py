from langchain_community.document_loaders import PyPDFDirectoryLoader

loader = PyPDFDirectoryLoader("docs/")
raw_documents = loader.load()

print(f"Loaded {len(raw_documents)} pages")
print("-" * 40)
print("Metadata:", raw_documents[0].metadata)
print("-" * 40)
print(raw_documents[0].page_content[:500])