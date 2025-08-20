# -*- coding: utf-8 -*-
"""
Formål:
- Læs .env for OPENAI_API_KEY
- Parse PDF'er (normal + OCR) til JSONL
- Genopbyg FAISS + Chroma vectorstores
- Brug ny langchain_openai.OpenAIEmbeddings
- Robust fejlhåndtering og tydelige prints (uden emojis)
"""

import os
import sys
import json
import re
import shutil
from pathlib import Path
from io import StringIO

# Konsol-encoding (Windows): undgå ASCII-fejl i prints
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ------------- Eksterne pakker -------------
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup
import tiktoken
from unstructured.partition.pdf import partition_pdf

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
# -------------------------------------------

# ============ KONFIGURATION ============
PDF_FOLDER = Path(
    "C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JYSK dokumenter/Opdateringer"
)
JSONL_FOLDER = Path(
    "C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JSONL_data"
)
VECTORSTORE_DIR = Path("vectorstore")
CHROMA_FOLDER = Path("chroma_store")

VECTORSTORE_DIR.mkdir(exist_ok=True)
CHROMA_FOLDER.mkdir(exist_ok=True)
JSONL_FOLDER.mkdir(exist_ok=True)

encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 300
MIN_WORDS = 4
MERGEABLE_TYPES = {"NarrativeText", "ListItem", "Table"}

# ============ HJÆLPEFUNKTIONER ============
def ensure_api_key() -> str:
    """Indlæs .env og returnér OPENAI_API_KEY eller stop med tydelig fejl."""
    load_dotenv(find_dotenv())
    key = os.getenv("OPENAI_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "OPENAI_API_KEY mangler. Tilføj den i en .env-fil i projektmappen som:\n"
            "OPENAI_API_KEY=sk-...din_nøgle...\n"
            "Alternativt: sæt miljøvariablen i din terminal før kørsel."
        )
    return key.strip()

def simplify_filename(filename: str) -> str:
    name = Path(filename).stem.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name + ".jsonl"

def extract_markdown_from_table(el) -> str:
    """Konverter Unstructured Table-element til Markdown. Fald tilbage til ren tekst."""
    try:
        html = el.metadata.text_as_html
        if not html:
            return el.text or ""
        soup = BeautifulSoup(html, "html.parser")
        table = pd.read_html(StringIO(str(soup)))[0]
        return table.to_markdown(index=False)
    except Exception:
        return el.text or ""

def split_into_token_chunks(text: str, max_tokens: int):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_text = encoding.decode(tokens[i : i + max_tokens])
        if len(chunk_text.split()) >= MIN_WORDS:
            chunks.append(chunk_text)
    return chunks

def parse_pdf_to_chunks(pdf_path: Path):
    """Hi-res parsing uden OCR. Inkluderer kun relevante kategorier. Samler pr. side og chunker i tokens."""
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            infer_table_structure=True,
            strategy="hi_res",
        )
    except Exception as e:
        print(f"[FEJL] Normal parsing fejlede for {pdf_path.name}: {e}")
        return []

    grouped_by_page = {}
    for el in elements:
        if el.category in MERGEABLE_TYPES:
            page = getattr(el.metadata, "page_number", None)
            if el.category == "Table":
                content = extract_markdown_from_table(el)
            else:
                content = (el.text or "").strip()
            if content:
                grouped_by_page.setdefault(page, []).append(content)

    chunks = []
    for page, blocks in grouped_by_page.items():
        full_text = "\n\n".join(blocks)
        for chunk in split_into_token_chunks(full_text, MAX_TOKENS):
            chunks.append(
                {"content": chunk, "page": page, "source": pdf_path.name, "type": "MergedText"}
            )
    return chunks

def reparse_with_ocr(pdf_path: Path):
    """OCR-parsing som supplement. Sæt sprog efter behov."""
    try:
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            extract_images_in_pdf=True,
            infer_table_structure=True,
            ocr_languages="eng+dan",
        )
    except Exception as e:
        print(f"[FEJL] OCR parsing fejlede for {pdf_path.name}: {e}")
        return []

    chunks = []
    for el in elements:
        txt = (el.text or "").strip()
        if txt and len(txt.split()) >= MIN_WORDS:
            page = getattr(el.metadata, "page_number", None)
            chunks.append(
                {"content": txt, "page": page, "source": pdf_path.name, "type": "OCRText"}
            )
    return chunks

def write_jsonl(chunks, output_path: Path):
    """Overskriv JSONL med UTF-8 og uden ASCII-beskæring."""
    with open(output_path, "w", encoding="utf-8") as f:
        for c in chunks:
            json.dump(c, f, ensure_ascii=False)
            f.write("\n")

def rebuild_vectorstores(all_documents, embedding_model, faiss_dir: Path, chroma_dir: Path):
    """Slet og genbyg FAISS og Chroma med simpel progress og robusthed."""
    print("Sletter gamle FAISS og Chroma stores...")
    shutil.rmtree(faiss_dir, ignore_errors=True)
    shutil.rmtree(chroma_dir, ignore_errors=True)
    faiss_dir.mkdir(exist_ok=True)
    chroma_dir.mkdir(exist_ok=True)

    print("Genererer embeddings og bygger FAISS...")
    try:
        faiss_store = FAISS.from_documents(all_documents, embedding_model)
        faiss_store.save_local(str(faiss_dir))
        print("FAISS gemt.")
    except Exception as e:
        print(f"[FEJL] Kunne ikke bygge/gemme FAISS: {e}")

    print("Genererer embeddings og bygger Chroma...")
    try:
        chroma_store = Chroma.from_documents(
            documents=all_documents,
            embedding=embedding_model,
            persist_directory=str(chroma_dir),
        )
        print("Chroma gemt.")
    except Exception as e:
        print(f"[FEJL] Kunne ikke bygge/gemme Chroma: {e}")

# ============ MAIN ============
def main():
    print("=== Start: PDF -> JSONL -> Vectorstores ===")

    try:
        api_key = ensure_api_key()
        print("OPENAI_API_KEY fundet.")
    except Exception as e:
        print(f"[STOP] {e}")
        return

    # Initialiser embeddings-klient (nyt bibliotek)
    try:
        embedding_model = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-3-small"  # skift til -large hvis du har brug for højere kvalitet
        )
        print("Embeddings-klient initialiseret.")
    except Exception as e:
        print(f"[STOP] Kunne ikke initialisere OpenAIEmbeddings: {e}")
        return

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        print("Ingen PDF'er fundet i:", PDF_FOLDER)
    else:
        print(f"Finder {len(pdf_files)} PDF'er i: {PDF_FOLDER}")

    # Parse alle PDF'er (normal + OCR) og skriv JSONL
    for pdf_path in pdf_files:
        print(f"Parser: {pdf_path.name}")
        chunks_normal = parse_pdf_to_chunks(pdf_path)
        chunks_ocr = reparse_with_ocr(pdf_path)
        all_chunks = chunks_normal + chunks_ocr

        if not all_chunks:
            print(f"[ADVARSEL] Ingen tekstudtræk for {pdf_path.name}")
            continue

        out_path = JSONL_FOLDER / simplify_filename(pdf_path.name)
        try:
            write_jsonl(all_chunks, out_path)
            print(f"Skrev {len(all_chunks)} chunks til {out_path.name}")
        except Exception as e:
            print(f"[FEJL] Kunne ikke skrive JSONL for {pdf_path.name}: {e}")

    # Læs alle JSONL som Documents
    all_documents = []
    jsonl_files = list(JSONL_FOLDER.glob("*.jsonl"))
    print(f"Læser {len(jsonl_files)} JSONL-filer fra: {JSONL_FOLDER}")
    for file in jsonl_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                chunks = [json.loads(line) for line in f]
            docs = [
                Document(page_content=c["content"], metadata=c)
                for c in chunks
                if c.get("content")
            ]
            all_documents.extend(docs)
        except Exception as e:
            print(f"[FEJL] Kunne ikke læse {file.name}: {e}")

    print(f"I alt {len(all_documents)} dokument-chunks klar til embeddings.")
    if not all_documents:
        print("[STOP] Ingen dokumenter at indeksere.")
        return

    rebuild_vectorstores(all_documents, embedding_model, VECTORSTORE_DIR, CHROMA_FOLDER)
    print("=== Færdig ===")

if __name__ == "__main__":
    main()
