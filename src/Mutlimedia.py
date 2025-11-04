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
IN_MEMORY_CHROMA_LIMIT = 200_000  # Max total token count to keep in memory

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
    template="""Context: {context}

Question: {question}

Answer concisely and only based on the given context. 
If the context does not provide sufficient information, say:
"I don't have enough information to answer that question." """,
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
# A global cache to hold Chroma vectorstore in memory (RAM only)
CHROMA_CACHE = {}


def load_in_memory_vectorstore(embeddings, docs, collection_name="ram_store"):
    """
    Create an in-memory (non-persistent) Chroma DB from existing embeddings & texts.
    main.py will call this once after embedding step and pass docs to it.
    """
    try:
        vectorstore = Chroma.from_texts(
            texts=docs,
            embedding=embeddings,
            collection_name=collection_name
        )
        # Cache in RAM to reuse for future policy queries
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

def policy_handler(query: str, collection_name="ram_store") -> str:
    """
    Perform retrieval + LLM answering using cached in-memory Chroma DB.
    Returns a plain string result (or 'ERROR: ...' if failure).
    """
    try:
        # 1️⃣ Check for existing vectorstore
        vectorstore = get_cached_vectorstore(collection_name)
        if not vectorstore:
            return (
                "ERROR: No in-memory Chroma DB found.\n"
                "Upload and embed policy documents first before querying."
            )

        # 2️⃣ Initialize LLM
        llm = init_llm()

        # 3️⃣ Build retrieval-based QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT}
        )

        # 4️⃣ Run retrieval QA
        response = qa.run(query)
        if not response:
            return "ERROR: Empty response returned from QA model."

        return response.strip()

    except Exception as e:
        return f"ERROR: Policy handler failed at retrieval stage: {e}\n{traceback.format_exc()}"


# ===========================================
# 🧹 MEMORY MANAGEMENT HELPERS
# ===========================================

def clear_cache():
    """Safely clear the in-memory Chroma cache (for memory limits)."""
    try:
        CHROMA_CACHE.clear()
        return "✅ In-memory Chroma cache cleared."
    except Exception as e:
        return f"ERROR: Failed to clear cache: {e}"


def cache_status():
    """Return info about current cache usage."""
    try:
        collections = list(CHROMA_CACHE.keys())
        status = {
            "collections_loaded": collections,
            "total_collections": len(collections)
        }
        return status
    except Exception as e:
        return {"error": f"Failed to get cache status: {e}"}
