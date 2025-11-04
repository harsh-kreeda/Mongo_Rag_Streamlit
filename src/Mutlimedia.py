import os
import traceback
from dotenv import load_dotenv

# Optional Streamlit import (for secrets)
try:
    import streamlit as st
except ImportError:
    st = None

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ===========================================
# 🔧 CONFIGURATION
# ===========================================
load_dotenv()

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4
MAX_TOKENS = 4096
IN_MEMORY_CHROMA_LIMIT = 200_000  # Max token count in memory

# ------------------ API Key Handling ------------------
try:
    OPENAI_API_KEY = None
    if st and hasattr(st, "secrets"):
        OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in Streamlit secrets or .env")
except Exception as e:
    print(f"[ERROR] Unable to load API key: {e}")
    OPENAI_API_KEY = None


# ===========================================
# 🧠 LLM + PROMPT SETUP
# ===========================================
PROMPT = PromptTemplate(
    template="""Context:
{context}

Question:
{question}

Answer concisely and only based on the given context.
If the context does not provide sufficient information, say:
"I don't have enough information to answer that question."
""",
    input_variables=["context", "question"]
)


def init_llm():
    """Initialize ChatOpenAI model safely."""
    try:
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
    except Exception as e:
        raise RuntimeError(f"LLM initialization failed: {e}")


# ===========================================
# ⚙️ GLOBAL IN-MEMORY DB CACHE
# ===========================================
CHROMA_CACHE = {}


def load_in_memory_vectorstore(embeddings, docs, collection_name="ram_store"):
    """Create an in-memory Chroma DB and cache it globally."""
    try:
        vectorstore = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
            collection_name=collection_name
        )
        CHROMA_CACHE[collection_name] = vectorstore
        return vectorstore
    except Exception as e:
        raise RuntimeError(f"In-memory vectorstore creation failed: {e}")


def get_cached_vectorstore(collection_name="ram_store"):
    """Retrieve cached Chroma vectorstore if available."""
    return CHROMA_CACHE.get(collection_name)


# ===========================================
# 🧩 RAG RETRIEVAL + ANSWERING
# ===========================================
def policy_handler(query: str, collection_name="ram_store"):
    """
    Perform retrieval + LLM answering using cached in-memory Chroma DB.
    Returns: (answer: str, retrieved_docs: list[str])
    """
    try:
        # 1️⃣ Retrieve cached DB
        vectorstore = get_cached_vectorstore(collection_name)
        if not vectorstore:
            return (
                "ERROR: No in-memory Chroma DB found. "
                "Upload and embed policy documents first before querying.",
                []
            )

        # 2️⃣ Retrieve context manually before QA
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_chunks = [doc.page_content for doc in retrieved_docs]

        if not retrieved_chunks:
            return ("ERROR: No relevant chunks retrieved from vectorstore.", [])

        # 3️⃣ Initialize LLM + chain
        llm = init_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # 4️⃣ Generate response (using invoke instead of deprecated run)
        response = qa.invoke({"query": query})
        if not response or not response.get("result"):
            return ("ERROR: Empty response returned from QA model.", retrieved_chunks)

        final_answer = response["result"].strip()
        return (final_answer, retrieved_chunks)

    except Exception as e:
        err_msg = f"ERROR: Policy handler failed: {e}\n{traceback.format_exc()}"
        return (err_msg, [])


# ===========================================
# 🧹 MEMORY MANAGEMENT HELPERS
# ===========================================
def clear_cache():
    """Safely clear the in-memory Chroma cache."""
    try:
        CHROMA_CACHE.clear()
        return "✅ In-memory Chroma cache cleared."
    except Exception as e:
        return f"ERROR: Failed to clear cache: {e}"


def cache_status():
    """Return info about current cache usage."""
    try:
        collections = list(CHROMA_CACHE.keys())
        return {
            "collections_loaded": collections,
            "total_collections": len(collections)
        }
    except Exception as e:
        return {"error": f"Failed to get cache status: {e}"}
