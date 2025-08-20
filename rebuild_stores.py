import os
import json
from pathlib import Path
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings  # Bruger den nye anbefalede import
from langchain_community.vectorstores import Chroma, FAISS
from langchain.schema import Document
import shutil

# Stier
JSONL_FOLDER = Path("C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JSONL_data")
CHROMA_FOLDER = Path("chroma_store")
FAISS_FOLDER = Path("faiss_store")

# Slet gamle stores
for folder in [CHROMA_FOLDER, FAISS_FOLDER]:
    if folder.exists():
        print(f"Sletter gammel store: {folder}")
        shutil.rmtree(folder)
    folder.mkdir(exist_ok=True)

# Embeddings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "DIN-OPENAI-API-NØGLE-HER"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Saml alle dokumenter fra JSONL
all_documents = []
jsonl_files = list(JSONL_FOLDER.glob("*.jsonl"))
print(f"Læser {len(jsonl_files)} JSONL-filer fra: {JSONL_FOLDER}")

for file in jsonl_files:
    with open(file, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    docs = [
        Document(
            page_content=chunk["content"],
            metadata={
                "source": chunk.get("source", "?"),
                "page": chunk.get("page", "?"),
                "type": chunk.get("type", "unknown")
            }
        ) for chunk in chunks
    ]
    all_documents.extend(docs)

print(f"Genererer embeddings for {len(all_documents)} chunks...")

# Genopbyg Chroma
vectorstore = Chroma.from_documents(
    documents=tqdm(all_documents, desc="Chroma-embeddings"),
    embedding=embedding_model,
    persist_directory=str(CHROMA_FOLDER)
)
print(f"✅ Ny Chroma-store gemt til: {CHROMA_FOLDER}")

# Genopbyg FAISS
faiss_store = FAISS.from_documents(all_documents, embedding_model)
faiss_store.save_local(str(FAISS_FOLDER))
print(f"✅ Ny FAISS-store gemt til: {FAISS_FOLDER}")

print("\n=== Færdig! ===")
