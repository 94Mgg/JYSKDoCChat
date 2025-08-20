import os
import json
from pathlib import Path
import streamlit as st
from rapidfuzz import fuzz
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR_REAL_KEY_HERE"
JSONL_FOLDER = Path("JSONL_data")
FAISS_STORE = Path("faiss_store")
client = OpenAI(api_key=OPENAI_API_KEY)

# === LOAD DATA & BUILD/LOAD VECTORSTORE ===
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
docs = []
for f in JSONL_FOLDER.glob("*.jsonl"):
    for line in f.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        docs.append(Document(page_content=data["content"], metadata=data))

if FAISS_STORE.exists() and any(FAISS_STORE.iterdir()):
    vectorstore = FAISS.load_local(
        folder_path=str(FAISS_STORE),
        embeddings=emb,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(docs, emb)
    vectorstore.save_local(str(FAISS_STORE))

# === SEARCH FUNCTIONS ===
def semantic_search(query: str, k: int = 5):
    results = vectorstore.similarity_search(query, k=k)
    return [
        {"source": d.metadata["source"], "page": d.metadata["page"], "text": d.page_content}
        for d in results
    ]

def fuzzy_search(keyword: str, threshold: int = 80):
    out = []
    for f in JSONL_FOLDER.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if fuzz.partial_ratio(keyword.lower(), d.get("content", "").lower()) >= threshold:
                out.append({"source": d["source"], "page": d["page"], "text": d["content"]})
    return out

# === STREAMLIT UI & SESSION STATE ===
st.set_page_config(page_title="JYSK Compliance Chat", layout="wide")
st.title("🤖 JYSK Compliance Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === OPENAI FUNCTION CALLING DEFINITIONS ===
functions = [
    {
        "name": "semantic_search",
        "description": "Fetch the most relevant policy snippets from the JYSK compliance docs.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
    },
    {
        "name": "fuzzy_search",
        "description": "Fetch raw fuzzy matches from the original JSONL text.",
        "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}}
    }
]

def chat():
    user_input = st.session_state.input_text.strip()
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    force_rag = False
    if user_input.startswith("[FORCE-RAG]"):
        force_rag = True
        user_input = user_input[len("[FORCE-RAG]"):].strip()

    messages = [
        {"role": "system", "content": "You are a compliance specialist for JYSK. Provide detailed, context-aware responses based on previous interactions."},
    ] + st.session_state.chat_history

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call={"name": "semantic_search", "arguments": json.dumps({"query": user_input})} if force_rag else "auto",
    )

    if response.choices[0].message.function_call:
        func_name = response.choices[0].message.function_call.name
        args = json.loads(response.choices[0].message.function_call.arguments)
        if func_name == "semantic_search":
            search_results = semantic_search(args["query"])
        elif func_name == "fuzzy_search":
            search_results = fuzzy_search(args["keyword"])
        context_text = "\n\n".join([f"{res['text']} (source: {res['source']} p.{res['page']})" for res in search_results])
        messages.append({"role": "function", "name": func_name, "content": context_text})

        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        reply = final_response.choices[0].message.content
    else:
        reply = response.choices[0].message.content

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.input_text = ""

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.chat_history:
    prefix = "You:" if msg["role"] == "user" else "Bot:"
    st.markdown(f"**{prefix}** {msg['content']}")

st.text_input("Your question (use `[FORCE-RAG]` to force semantic search):", key="input_text", on_change=chat)
