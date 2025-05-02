import os
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
from rapidfuzz import fuzz
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# === CONFIGURATION ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "YOUR-OPENAI-API-KEY-HERE"
BASE_DIR = Path(__file__).parent
CHROMA_FOLDER = BASE_DIR / "chroma_store"
JSONL_FOLDER = BASE_DIR / "JSONL_data"

# Ensure folders exist
CHROMA_FOLDER.mkdir(exist_ok=True)


# === STREAMLIT PAGE SETUP ===
st.set_page_config(page_title="JYSK Compliance Chat", layout="wide")
st.title("🤖 JYSK Compliance Chat")

# === SESSION-STATE INITIALIZATION ===
if "log_path" not in st.session_state:
    LOG_DIR = Path(__file__).parent / "Chat log"
    LOG_DIR.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%d%m%y")
    existing = sorted(LOG_DIR.glob(f"Chat_log_{date_str}_*.txt"))
    seq = int(existing[-1].stem.split("_")[-1]) + 1 if existing else 1
    st.session_state.log_path = LOG_DIR / f"Chat_log_{date_str}_{seq:02d}.txt"

if "chat_history" not in st.session_state:
    # UI history for display
    st.session_state.chat_history = []  # list of tuples (sender, text)

# === VECTORSTORE & LLM INITIALIZATION ===

# 1) Embedding model (same as before)
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# 2) Load JSONL chunks into Document objects
docs = []
for jsonl_file in JSONL_FOLDER.glob("*.jsonl"):
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            docs.append(
                Document(
                    page_content=data["content"],
                    metadata={
                        "source": data.get("source", "?"),
                        "page": data.get("page", "?"),
                        "type": data.get("type", "unknown"),
                    },
                )
            )

# 3) Build FAISS vectorstore (in‑memory)
vectorstore = FAISS.from_documents(docs, embedding_model)

# 4) Initialize the LLM (same as before)
llm = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key=OPENAI_API_KEY)

# === SYSTEM PROMPT WITH AGE-CASE ENUMERATION & FOLLOW-UP ===
system_prompt = """
You are a compliance specialist for JYSK. You must help users understand the requirements stated in JYSK's internal compliance documentation.

When answering a question:
1. Search the provided context for a clear, factual answer.
1.a. If the requirement varies by a user attribute (e.g. age, region, category, sex, weight, size), **enumerate all applicable cases**, clearly labeling each (for example “Under 3 years…”, “3–14 years…”, “14+ years…”).
2. Understand that terms such as "upload", "submission", and "registration" may not all appear in the documents, but they can represent the same process. Treat them as synonyms unless the context clearly distinguishes between them.
3. If a related process is described, reason logically based on the context and explain your conclusion.
4. If no direct or indirect answer is available:
   - List what you looked for.
   - Point out what was missing.
5. If helpful, suggest a clearer or more effective way to phrase the question — based on how the JYSK documents are written and structured.
6. If the answer cannot be found in the context, you may search the raw data using fuzzy keyword matching.
7. After your answer, provide 5 additional relevant facts you find in other retrieved chunks about the same topic.
8. Finally, ask the user what aspect they’d like to explore in more detail (e.g. “Which age group would you like to focus on next?”).

Always respond in a factual, structured manner with as many details necessary to provide a full overview.
"""
qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=system_prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
)

# === PERSISTED MEMORY & QA CHAIN ===
if "qa_chain" not in st.session_state:
    # Create a persistent memory buffer
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    # Build the chain once, reusing memory each run
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
                text = data.get("content", "").lower()
                if fuzz.partial_ratio(keyword.lower(), text) >= fuzz_threshold:
                    matches.append({
                        "file": file.name,
                        "page": data.get("page", "?"),
                        "content": data.get("content")
                    })
    return matches

# === USER INPUT PROCESSOR ===
def process_input():
    user_input = st.session_state.get("query_input", "").strip()
    if not user_input:
        return

    # 1) Save the UI history
    st.session_state.chat_history.append(("You", user_input))

    # 2) Decide branch: raw-data search vs QA chain
    if user_input.lower().startswith("rawdata on "):
        kw = user_input[10:].strip()
        results = search_raw_data(kw, JSONL_FOLDER)
        if results:
            bot_reply = f"Rawdata matches for '{kw}':\n" + "\n".join(
                f"- **{r['file']}** – Page {r['page']}: {r['content']}"
                for r in results
            )
        else:
            bot_reply = f"No matches found for rawdata on '{kw}'."
    else:
        # 3) Invoke the persisted chain (which has memory)
        raw = st.session_state.qa_chain.invoke({"question": user_input})
        bot_reply = raw["answer"]

    # 4) Save the UI history
    st.session_state.chat_history.append(("Bot", bot_reply))

    # 5) Append to your session log
    with open(st.session_state.log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"QUESTION:\n{user_input}\n\nANSWER:\n{bot_reply}\n\n")
        if not user_input.lower().startswith("rawdata on "):
            used = set()
            for doc in raw.get("source_documents", []):
                text = doc.page_content.replace("\n", " ").lower()
                for sent in text.split("."):
                    if sent and fuzz.partial_ratio(sent.strip(), bot_reply.lower()) >= 80:
                        m = doc.metadata
                        used.add((m.get("source","?"), m.get("page","?")))
                        break
            for src, pg in used:
                log_f.write(f"SOURCE: {src} (page {pg})\n")
        log_f.write("\n" + "="*80 + "\n")

    # 6) Clear the input box
    st.session_state.query_input = ""

# === DISPLAY CHAT HISTORY ===
# Chronological: oldest at top, newest right above input
for sender, msg in st.session_state.chat_history:
    prefix = "You:" if sender == "You" else "Bot:"
    st.markdown(f"**{prefix}** {msg}")

# === INPUT FIELD AT BOTTOM ===
st.text_input(
    "Asking field:",
    key="query_input",
    placeholder="Type your question here...",
    on_change=process_input
)

st.markdown("---")
st.caption("Type 'rawdata on <keyword>' to search the original JSONL files directly.")
