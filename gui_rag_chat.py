import os
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
from rapidfuzz import fuzz

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# === CONFIGURATION ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY-HERE"
BASE_DIR = Path(__file__).parent
JSONL_FOLDER = BASE_DIR / "JSONL_data"
FAISS_STORE = BASE_DIR / "faiss_store"

# === STREAMLIT PAGE SETUP ===
st.set_page_config(page_title="JYSK Compliance Chat", layout="wide")
st.title("🤖 JYSK Compliance Chat")

# === SESSION STATE INITIALIZATION ===
if "log_path" not in st.session_state:
    LOG_DIR = BASE_DIR / "Chat log"
    LOG_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%d%m%y")
    existing = sorted(LOG_DIR.glob(f"Chat_log_{date_str}_*.txt"))
    seq = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 1
    st.session_state.log_path = LOG_DIR / f"Chat_log_{date_str}_{seq:02d}.txt"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === VECTORSTORE & LLM INITIALIZATION ===

# 1) Embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 2) Load JSONL chunks into Document objects
docs = []
for jsonl_file in JSONL_FOLDER.glob("*.jsonl"):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            docs.append(Document(
                page_content=data.get("content", ""),
                metadata={
                    "source": data.get("source", "?"),
                    "page": data.get("page", "?"),
                    "type": data.get("type", "unknown"),
                },
            ))
if not docs:
    st.error(f"No JSONL files found in {JSONL_FOLDER}. Please commit your JSONL_data folder.")
    st.stop()

# 3) Build or Load FAISS index
try:
    if FAISS_STORE.exists():
        # NOTE the extra flag here
        vectorstore = FAISS.load_local(
            directory=str(FAISS_STORE),
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(docs, embedding_model)
        vectorstore.save_local(str(FAISS_STORE))
except Exception as e:
    st.error(f"Error building/loading vectorstore: {e}")
    st.stop()


# 4) Initialize the LLM
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# === SYSTEM PROMPT WITH AGE-CASE ENUMERATION & FOLLOW-UP ===
system_prompt = """
You are a compliance specialist for JYSK. You must help users understand the requirements stated in JYSK's internal compliance documentation.

When answering:
1. Search the provided context for a clear, factual answer.
1.a. If the requirement varies by a user attribute (e.g. age, region, category, sex, weight, size), enumerate all applicable cases (“Under 3 years…”, “3–14 years…”, “14+ years…”).
2. Treat synonyms (“upload”, “submission”, “registration”) as equal unless context distinguishes.
3. If related processes exist, explain by reasoning.
4. If no answer:
   - List what you looked for.
   - Note what's missing.
5. Suggest clearer rephrasing if helpful.
6. You may raw-search with fuzzy matching.
7. After your answer, give 5 extra relevant facts from other chunks.
8. Ask the user what aspect to explore next.

Always be factual and structured.
"""
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

# === PERSISTED MEMORY & QA CHAIN ===
if "qa_chain" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 80}),
        memory=st.session_state.memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        output_key="answer"
    )

# === RAW DATA SEARCH FUNCTION ===
def search_raw_data(keyword, folder, fuzz_threshold=80):
    matches = []
    for file in folder.glob("*.jsonl"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if fuzz.partial_ratio(keyword.lower(), data.get("content", "").lower()) >= fuzz_threshold:
                    matches.append({"file": file.name, "page": data.get("page", "?"), "content": data.get("content")})
    return matches

# === USER INPUT PROCESSOR ===
def process_input():
    user_input = st.session_state.query_input.strip()
    if not user_input:
        return

    st.session_state.chat_history.append(("You", user_input))

    if user_input.lower().startswith("rawdata on "):
        kw = user_input[10:].strip()
        results = search_raw_data(kw, JSONL_FOLDER)
        bot_reply = (
            f"Rawdata matches for '{kw}':\n" +
            "\n".join(f"- **{r['file']}** – Page {r['page']}: {r['content']}" for r in results)
            if results else f"No matches found for rawdata on '{kw}'."
        )
    else:
        raw = st.session_state.qa_chain.invoke({"question": user_input})
        bot_reply = raw["answer"]

    st.session_state.chat_history.append(("Bot", bot_reply))

    with open(st.session_state.log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"QUESTION:\n{user_input}\n\nANSWER:\n{bot_reply}\n\n")
        if not user_input.lower().startswith("rawdata on "):
            used = set()
            for doc in raw.get("source_documents", []):
                txt = doc.page_content.replace("\n", " ").lower()
                for sent in txt.split("."):
                    if sent and fuzz.partial_ratio(sent.strip(), bot_reply.lower()) >= 80:
                        m = doc.metadata
                        used.add((m.get("source","?"), m.get("page","?")))
                        break
            for src, pg in used:
                log_f.write(f"SOURCE: {src} (page {pg})\n")
        log_f.write("\n" + "="*80 + "\n")

    st.session_state.query_input = ""

# === DISPLAY CHAT HISTORY ===
for sender, msg in st.session_state.chat_history:
    label = "You:" if sender == "You" else "Bot:"
    st.markdown(f"**{label}** {msg}")

# === INPUT FIELD AT BOTTOM ===
st.text_input(
    "Asking field:",
    key="query_input",
    placeholder="Type your question here...",
    on_change=process_input
)

st.markdown("---")
st.caption("Type 'rawdata on <keyword>' to search the original JSONL files directly.")
