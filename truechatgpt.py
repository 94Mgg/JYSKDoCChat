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

# === BUILD OR LOAD VECTORSTORE ===
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

# === FUNCTIONS DEFINITION FOR OPENAI ===
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

    # Check if force semantic search
    force_rag = False
    if user_input.startswith("[FORCE-RAG]"):
        force_rag = True
        user_input = user_input[len("[FORCE-RAG]"):].strip()

    # Prepare messages for ChatGPT
    messages = [
        {"role": "system", "content": """
        You are a compliance specialist for JYSK. Provide detailed, context-aware responses based on previous interactions.
         When answering a question:
        1. Maintain full conversational context. If the user asks a follow‑up or refers back to something you said earlier, recall that information and elaborate on it clearly.
        2. Search the provided context for a clear, factual answer. If more than one result is found, always provide all of the results.
        2.a. If the requirement varies by a user attribute (e.g. age, region, category, sex, weight, size), **enumerate all applicable cases**, clearly labeling each (for example “Under 3 years…”, “3–14 years…”, “14+ years…”).
        3. Understand that terms such as "upload", "submission", and "registration" may not all appear in the documents, but they can represent the same process. Treat them as synonyms unless the context clearly distinguishes between them.
        4. If a related process is described, reason logically based on the context and explain your conclusion.
        4a. If you have the JYSK standard; end the answer with the sources of the information you used to answer the question. E.g. "JYSK 1009 - CATEGORY STANDARD - Garden, Page 6, Table 1"
        5. If no direct or indirect answer is available:
      - List what you looked for.
     - Point out what was missing.
        6. If helpful, suggest a clearer or more effective way to phrase the question — based on how the JYSK documents are written and structured.
        7. If the answer still cannot be found in the context, perform a raw‑data fuzzy keyword search across the JSONL.
        8. After your direct answer, provide 5 additional relevant facts you find in other retrieved chunks about the same topic.
        9. Finally, ask the user what aspect they’d like to explore in more detail (e.g. “Which age group would you like to focus on next?”)."""},
    ] + st.session_state.chat_history

    # Request completion from OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        functions=functions,
        function_call={"name": "semantic_search", "arguments": json.dumps({"query": user_input})} if force_rag else "auto",
    )

    # Handle function calls
    if response.choices[0].message.function_call:
        func_name = response.choices[0].message.function_call.name
        args = json.loads(response.choices[0].message.function_call.arguments)
        
        if func_name == "semantic_search":
            search_results = semantic_search(args["query"])
        elif func_name == "fuzzy_search":
            search_results = fuzzy_search(args["keyword"])

        context_text = "\n\n".join([f"{res['text']} (source: {res['source']} p.{res['page']})" for res in search_results])
        messages.append({"role": "function", "name": func_name, "content": context_text})

        # second call to include context in response
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
