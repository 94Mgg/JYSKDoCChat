import os
import json
from pathlib import Path
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Stier
JSONL_FOLDER = Path("C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JSONL_data")
CHROMA_FOLDER = Path("chroma_store")
CHROMA_FOLDER.mkdir(exist_ok=True)

# OpenAI API nøgle
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "DIN-OPENAI-API-NØGLE-HER"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def load_chunks_from_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def convert_to_documents(chunks):
    return [
        Document(
            page_content=chunk["content"],
            metadata={
                "source": chunk.get("source", "?"),
                "page": chunk.get("page", "?"),
                "type": chunk.get("type", "unknown")
            }
        )
        for chunk in chunks
    ]

def main():
    # 1. Slet eksisterende Chroma-store først
    if CHROMA_FOLDER.exists():
        print("  Sletter gammel Chroma-store...")
        for item in CHROMA_FOLDER.glob("*"):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                for subitem in item.glob("*"):
                    subitem.unlink()
                item.rmdir()

    # 2. Indlæs alle JSONL-filer
    all_documents = []
    jsonl_files = list(JSONL_FOLDER.glob("*.jsonl"))
    print(f"Fundet {len(jsonl_files)} JSONL-filer i: {JSONL_FOLDER}\n")

    for file in jsonl_files:
        print(f"  Indlæser: {file.name}")
        chunks = load_chunks_from_jsonl_file(file)
        docs = convert_to_documents(chunks)
        all_documents.extend(docs)

    print(f"\nGenererer embeddings for {len(all_documents)} chunks...")

    # 3. Byg og gem ny Chroma-store
    vectorstore = Chroma.from_documents(
        documents=tqdm(all_documents),
        embedding=embedding_model,
        persist_directory=str(CHROMA_FOLDER)
    )

    print(f"\n✅ Ny Chroma-store gemt til: {CHROMA_FOLDER}")

if __name__ == "__main__":
    main()
