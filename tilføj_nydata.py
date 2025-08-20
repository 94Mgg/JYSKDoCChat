import os
import json
import re
from pathlib import Path
import tiktoken
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# --- STIER ---
PDF_FOLDER = Path(r"C:\Users\MickiGrunzig\OneDrive - Zolo International Trading\Dokumenter\RAG model - JYSK V2_24.04\JYSK dokumenter")
OPDATERINGS_FOLDER = PDF_FOLDER / "Opdateringer"
OUTPUT_FOLDER = Path(r"C:\Users\MickiGrunzig\OneDrive - Zolo International Trading\Dokumenter\RAG model - JYSK V2_24.04\JSONL_data")
CHROMA_FOLDER = Path("chroma_store")
CHROMA_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# --- TOKENIZER ---
encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 300
MIN_WORDS = 4

MERGEABLE_TYPES = {"NarrativeText", "ListItem", "Table"}

def simplify_filename(filename: str) -> str:
    name = Path(filename).stem.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name + ".jsonl"

def extract_markdown_from_table(el):
    try:
        html = el.metadata.text_as_html
        if not html:
            return el.text or ""
        soup = BeautifulSoup(html, "html.parser")
        table = pd.read_html(StringIO(str(soup)))[0]
        return table.to_markdown(index=False)
    except Exception:
        return el.text or ""

def split_into_token_chunks(text, max_tokens):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        if len(chunk_text.split()) >= MIN_WORDS:
            chunks.append(chunk_text)
    return chunks

def parse_pdf_to_chunks(pdf_path, use_ocr=False):
    # Robust PDF parser, med/uden OCR
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            infer_table_structure=True,
            strategy="hi_res",
            extract_images_in_pdf=True if use_ocr else False,
            ocr_languages="eng" if use_ocr else None
        )
    except Exception as e:
        print(f"  ❌ PDF parsing fejlede for {pdf_path.name}: {e}")
        return []

    grouped_by_page = {}
    for el in elements:
        # Kan udvides til flere kategorier hvis nødvendigt
        if el.category not in MERGEABLE_TYPES:
            continue
        page = el.metadata.page_number
        if page not in grouped_by_page:
            grouped_by_page[page] = []
        if el.category == "Table":
            content = extract_markdown_from_table(el)
        else:
            content = (el.text or "").strip()
        if content:
            grouped_by_page[page].append(content)

    all_chunks = []
    for page, blocks in grouped_by_page.items():
        full_text = "\n\n".join(blocks)
        token_chunks = split_into_token_chunks(full_text, MAX_TOKENS)
        for chunk in token_chunks:
            all_chunks.append({
                "content": chunk,
                "page": page,
                "source": pdf_path.name,
                "type": "MergedText" if not use_ocr else "OCRText"
            })
    return all_chunks

def save_chunks_to_jsonl(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")

def main():
    print("=== Starter PDF → JSONL process ===")
    # Step 1: Scan alle dokumenter
    all_pdfs = list(PDF_FOLDER.glob("*.pdf"))
    opdaterings_pdfs = list(OPDATERINGS_FOLDER.glob("*.pdf")) if OPDATERINGS_FOLDER.exists() else []
    opdaterings_map = {f.stem.lower(): f for f in opdaterings_pdfs}

    for pdf_path in all_pdfs:
        stem = pdf_path.stem.lower()
        output_filename = simplify_filename(pdf_path.name)
        output_path = OUTPUT_FOLDER / output_filename

        # Hvis denne fil er opdateret
        if stem in opdaterings_map:
            print(f"\n--- {pdf_path.name} ER OPDATERET. Erstatter med ny version. ---")
            new_pdf = opdaterings_map[stem]
            chunks = parse_pdf_to_chunks(new_pdf, use_ocr=True)
            print(f"  {len(chunks)} nye chunks genereret fra opdatering.")
            save_chunks_to_jsonl(chunks, output_path)
            print(f"  Eksisterende data i {output_path.name} erstattet med opdateret info.")
        else:
            print(f"\n--- {pdf_path.name} ikke opdateret. Eksisterende JSONL bibeholdes. ---")
            if not output_path.exists():
                chunks = parse_pdf_to_chunks(pdf_path, use_ocr=True)
                save_chunks_to_jsonl(chunks, output_path)
                print(f"  Skrev {len(chunks)} chunks (første gang for denne fil).")
            else:
                print(f"  Skipper - allerede eksisterende JSONL.")

    print("\n=== Opdatering færdig. Nu genopbygges Chroma-vectorstore. ===")

    # Step 2: Genopbyg Chroma-store
    # (Kan tilpasses til også at opdatere FAISS hvis du ønsker begge)
    try:
        embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY") or "DIN-OPENAI-API-NØGLE-HER")

        # Slet gammel Chroma
        if CHROMA_FOLDER.exists():
            print("  Sletter gammel Chroma-store...")
            for item in CHROMA_FOLDER.glob("*"):
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    for subitem in item.glob("*"):
                        subitem.unlink()
                    item.rmdir()

        all_documents = []
        jsonl_files = list(OUTPUT_FOLDER.glob("*.jsonl"))
        print(f"  Læser {len(jsonl_files)} JSONL-filer fra: {OUTPUT_FOLDER}")

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

        print(f"\n  Genererer embeddings for {len(all_documents)} chunks...")
        vectorstore = Chroma.from_documents(
            documents=tqdm(all_documents),
            embedding=embedding_model,
            persist_directory=str(CHROMA_FOLDER)
        )
        print(f"\n✅ Ny Chroma-store gemt til: {CHROMA_FOLDER}")

    except Exception as e:
        print(f"Fejl ved Chroma-genopbygning: {e}")

    print("\n=== Færdig! ===")

if __name__ == "__main__":
    main()
