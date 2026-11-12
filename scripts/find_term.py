from langchain_community.document_loaders import PyPDFDirectoryLoader

docs = PyPDFDirectoryLoader("docs/").load()
for d in docs:
    text = d.page_content
    i = text.lower().find("turing")
    if i != -1:
        print(d.metadata)
        print(text[max(0, i - 150): i + 250].replace("\n", " "))
        print("-" * 40)