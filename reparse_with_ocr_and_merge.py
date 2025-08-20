import os
import json
from pathlib import Path
from unstructured.partition.pdf import partition_pdf

# Mapper
PDF_FOLDER = Path("C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JYSK dokumenter/")
JSONL_FOLDER = Path("C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JSONL_data/")

def reparse_with_ocr(pdf_path):
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        ocr_languages="eng"
    )
    chunks = []
    for el in elements:
        if el.text:
            text = el.text.strip()
            if len(text.split()) >= 4:  # Mindst 4 ord
                chunks.append({
                    "content": text,
                    "page": el.metadata.page_number,
                    "source": pdf_path.name,
                    "type": "OCRText"
                })
    return chunks

def merge_into_existing(jsonl_file, new_chunks):
    # Indlæs eksisterende data
    existing_chunks = []
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            existing_chunks = [json.loads(line) for line in f]
    
    # Tilføj nye OCR-chunks
    all_chunks = existing_chunks + new_chunks

    # Gem samlet
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            json.dump(chunk, f, ensure_ascii=False)
            f.write("\n")

def main():
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    print(f"Fundet {len(pdf_files)} PDF'er til OCR-reparse\n")

    for pdf_path in pdf_files:
        print(f"OCR-parsing af: {pdf_path.name}")
        new_chunks = reparse_with_ocr(pdf_path)
        print(f"  Fundet {len(new_chunks)} nye OCR-chunks")
        if new_chunks:
            output_filename = Path(pdf_path.name).stem.lower().replace(" ", "_") + ".jsonl"
            jsonl_path = JSONL_FOLDER / output_filename
            print(f"  Fletter ind i {jsonl_path.name}")
            merge_into_existing(jsonl_path, new_chunks)

    print("\n✅ OCR reparse og fletning færdig.")

if __name__ == "__main__":
    main()
