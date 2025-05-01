import os
import json
import re
from pathlib import Path
import tiktoken
from unstructured.partition.pdf import partition_pdf

from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

# Stier
PDF_FOLDER = Path("C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JYSK dokumenter/")
OUTPUT_FOLDER = Path("C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JSONL_data")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Tokenizer
encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 300
MIN_WORDS = 4

# Vi vil kun samle disse kategorier
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

def parse_pdf_to_chunks(pdf_path):
    elements = partition_pdf(
        filename=str(pdf_path),
        infer_table_structure=True,
        strategy="hi_res"
    )

    grouped_by_page = {}
    for el in elements:
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
                "type": "MergedText"
            })
    return all_chunks

def main():
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    print(f"Fundet {len(pdf_files)} PDF'er i {PDF_FOLDER}\n")

    for pdf_path in pdf_files:
        print(f"Parser: {pdf_path.name}")
        try:
            chunks = parse_pdf_to_chunks(pdf_path)
            print(f"  {len(chunks)} kontekstuelle chunks oprettet")
            if chunks:
                output_filename = simplify_filename(pdf_path.name)
                output_path = OUTPUT_FOLDER / output_filename
                with open(output_path, "w", encoding="utf-8") as f:
                    for chunk in chunks:
                        json.dump(chunk, f, ensure_ascii=False)
                        f.write("\n")
                print(f"  Gemte til: {output_path}")
        except Exception as e:
            print(f"  ❌ Fejl ved {pdf_path.name}: {e}")

    print("\n✅ Færdig med alle PDF'er.")

if __name__ == "__main__":
    main()
