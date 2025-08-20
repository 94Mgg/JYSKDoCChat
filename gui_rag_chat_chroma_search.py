import os
import json
from pathlib import Path
import streamlit as st
from rapidfuzz import fuzz
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from openai import OpenAI

# === CONFIG ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "DIN-OPENAI-API-NØGLE-HER"
JSONL_FOLDER = Path("JSONL_data")
CHROMA_FOLDER = Path("chroma_store")
client = OpenAI(api_key=OPENAI_API_KEY)

# === LOAD DATA & BUILD/LOAD VECTORSTORE ===
emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
if CHROMA_FOLDER.exists():
    vectorstore = Chroma(persist_directory=str(CHROMA_FOLDER), embedding_function=emb)
else:
    st.error("Chroma vectorstore ikke fundet. Sørg for, at den findes.")
    st.stop()

# === SEARCH FUNCTIONS ===
def chroma_search(query: str, k: int = 5):
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
st.title("🤖 JYSK Compliance Chat (Chroma med Fuzzy fallback)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat():
    user_input = st.session_state.input_text.strip()
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Først Chroma søgning
    search_results = chroma_search(user_input)
    
    if not search_results:
        st.info("Ingen Chroma-resultater fundet, udfører fuzzy søgning...")
        search_results = fuzzy_search(user_input)

    if search_results:
        context_text = "\n\n".join([f"{res['text']} (kilde: {res['source']} s.{res['page']})" for res in search_results])
    else:
        context_text = "Ingen relevante resultater fundet."

    messages = [
        {"role": "system", "content": "Du er en compliance-specialist for JYSK. Giv detaljerede og kontekstbevidste svar baseret på resultaterne."},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": context_text}
    ]

    response = client.chat.completions.create(model="gpt-4", messages=messages)
    reply = response.choices[0].message.content

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.session_state.input_text = ""

# === DISPLAY CHAT HISTORY ===
for msg in st.session_state.chat_history:
    prefix = "You:" if msg["role"] == "user" else "Bot:"
    st.markdown(f"**{prefix}** {msg['content']}")

st.text_input("Indtast dit spørgsmål:", key="input_text", on_change=chat)
