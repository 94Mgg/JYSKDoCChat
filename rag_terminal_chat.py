from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# 🔐 Din OpenAI API-nøgle
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "DIN-OPENAI-API-NØGLE-HER"

# 🧠 Indlæs embeddings og vectorstore
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
#retriever = vectorstore.as_retriever(search_kwargs={"k": 16})
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 16, "lambda_mult": 0.25})


# 📋 Prompt template til kontrol
template = """
Du er en faktuel, regelbaseret assistent for JYSK. Brug udelukkende konteksten nedenfor til at besvare spørgsmålet.
Svar kort og præcist. Hvis svaret ikke findes i konteksten, skal du svare: "Det fremgår ikke af dokumenterne."

Kontekst:
{context}

Spørgsmål:
{question}
"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 💬 Sprogmodel
llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=OPENAI_API_KEY)

# 🔄 Retrieval + promptstyret QA-kæde
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

# 🖥️ Terminalchat
def chat():
    print("RAG-chat (med promptkontrol) klar. Skriv 'exit' for at afslutte.\n")
    while True:
        query = input("> ")
        if query.lower() in ["exit", "quit"]:
            break

        result = qa_chain(query)
        print("\nSvar:\n")
        print(result["result"])

        print("\nKilder:")
        for doc in result["source_documents"]:
            meta = doc.metadata
            print(f"- {meta.get('source', '?')} (side {meta.get('page', '?')}, type: {meta.get('type', '')})")
        print()

if __name__ == "__main__":
    chat()
