import os
from pathlib import Path
import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from rapidfuzz import fuzz

# === CONFIGURATION ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY-HERE"
CHROMA_FOLDER = Path("chroma_store")
JSONL_FOLDER = Path("C:/Users/MickiGrunzig/OneDrive - Zolo International Trading/Dokumenter/RAG model - JYSK V2_24.04/JSONL_data")

# === INITIALIZE MODELS ===
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma(persist_directory=str(CHROMA_FOLDER), embedding_function=embedding_model)
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === SMART SYSTEM PROMPT ===
system_prompt = """
You are a compliance specialist for JYSK.

When answering questions:
You are a compliance specialist for JYSK. You must help users understand the requirements stated in JYSK's internal compliance documentation.

When answering a question:

1. Search the provided context for a clear, factual answer.
2. Understand that terms such as "upload", "submission", and "registration" may not all appear in the documents, but they can represent the same process. Treat them as synonyms unless the context clearly distinguishes between them.
3. If a related process is described, reason logically based on the context and explain your conclusion.
4. If no direct or indirect answer is available:
   - List what you looked for.
   - Point out what was missing.
5. If helpful, suggest a clearer or more effective way to phrase the question — based on how the JYSK documents are written and structured.
6. If the answer cannot be found in the context, you may search the raw data using fuzzy keyword matching.
7. Always respond in a short, factual, and structured manner.


Always answer factually, logically, clearly, and concisely.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        system_prompt +
        "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
)

# === BUILD CONVERSATIONAL RAG CHAIN (with correct memory output) ===
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 50}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True,
    output_key="answer"  # Important: Memory stores only 'answer'
)

# === RAW DATA SEARCH FUNCTION WITH FUZZY MATCH ===
def search_raw_data(keyword, jsonl_folder, fuzz_threshold=80):
    keyword = keyword.lower()
    print(f"\n🔎 Searching raw data for keyword: '{keyword}' (fuzzy matching enabled)\n")
    found = False
    for file in jsonl_folder.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                text = data.get("content", "").lower()
                if fuzz.partial_ratio(keyword, text) >= fuzz_threshold:
                    print(f"Found in {file.name} - page {data.get('page', '?')}")
                    print(data.get("content"))
                    print("-" * 80)
                    found = True
    if not found:
        print(f"No raw data found for '{keyword}'.\n")

# === CHAT FUNCTION ===
def chat():
    print("Conversational Guided RAG (with Synonyms, Raw Data, Fuzzy Matching, and Source Listing) is ready!")
    print("ℹ️  Tip: You can type 'rawdata on (search word)' to search raw JSONL files manually.\n")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit"]:
            break

        if query.lower().startswith("rawdata on "):
            keyword = query[10:].strip()
            search_raw_data(keyword, JSONL_FOLDER)
            continue

        # === GET RESULT ===
        raw_result = qa_chain.invoke({"question": query})
        answer = raw_result["answer"]
        sources = raw_result.get("source_documents", [])

        # === PRINT ANSWER ===
        print("\nAnswer:\n")
        print(answer)

        # === PRINT SOURCES ===
        print("\nSources used:")
        if sources:
            for doc in sources:
                metadata = doc.metadata
                source_file = metadata.get("source", "?")
                page = metadata.get("page", "?")
                chunk_type = metadata.get("type", "?")
                print(f"- {source_file} (page {page}, type: {chunk_type})")
        else:
            print("No sources available.")

if __name__ == "__main__":
    chat()
