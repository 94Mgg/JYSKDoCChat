import json
from collections import Counter, defaultdict
from pathlib import Path

INPUT_FILE = Path("rag_input.jsonl")

def main():
    if not INPUT_FILE.exists():
        print("Filen 'rag_input.jsonl' blev ikke fundet.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]

    total = len(lines)
    types = Counter(line["type"] for line in lines)
    sources = Counter(line["source"] for line in lines)

    print(f"\n📊 Analyse af {INPUT_FILE.name}")
    print(f"Total chunks: {total}\n")

    print("Chunks pr. type:")
    for t, count in types.items():
        print(f"  {t:<20}: {count}")

    print("\nChunks pr. dokument:")
    for src, count in sources.items():
        print(f"  {src:<50}: {count}")

    # Valgfrit: Eksempel på første tabel
    table_examples = [line for line in lines if line["type"] == "Table"]
    if table_examples:
        print("\nEksempel på 'Table'-chunk:\n")
        print(table_examples[0]["content"])
    else:
        print("\nIngen 'Table'-chunks fundet.")

if __name__ == "__main__":
    main()
