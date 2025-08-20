from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from pathlib import Path

emb = OpenAIEmbeddings(openai_api_key="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")
faiss_store = FAISS.load_local("faiss_store", emb, allow_dangerous_deserialization=True)

query = "Detailed Material List"
results = faiss_store.similarity_search(query, k=10)

for doc in results:
    print(doc.metadata["source"], doc.metadata["page"])
    print(doc.page_content[:200], "\n")
