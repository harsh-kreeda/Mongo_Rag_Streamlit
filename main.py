
# # main.py — Streamlit front-end for Router -> (Mongo | Policy) handlers
# import streamlit as st
# from datetime import datetime, timezone
# import traceback
# import os
# import re

# from src.Router_gpt import classify_query, RouteType

# # Lazy import Mutlimedia to prevent heavy init on Streamlit load
# import importlib
# Mutlimedia = importlib.import_module("src.Mutlimedia")
# load_in_memory_vectorstore = Mutlimedia.load_in_memory_vectorstore
# get_cached_vectorstore = Mutlimedia.get_cached_vectorstore
# clear_cache = Mutlimedia.clear_cache
# cache_status = Mutlimedia.cache_status
# policy_handler = Mutlimedia.policy_handler

# from langchain_openai import OpenAIEmbeddings
# from PyPDF2 import PdfReader
# from pptx import Presentation
# from docx import Document
# from langchain.text_splitter import CharacterTextSplitter

# # ===========================================
# # PAGE CONFIG
# # ===========================================
# st.set_page_config(page_title="Mongo_RAG Streamlit", page_icon="📚", layout="wide")
# st.title("📚 Mongo_RAG - Streamlit Frontend")

# # Tabs for Upload and Query
# tab1, tab2 = st.tabs(["📂 Upload & Embed", "💬 Query & Route"])

# # ===========================================
# # HELPER — Process and Embed Files Safely
# # ===========================================
# def process_and_embed_files(uploaded_files):
#     """Extract text, split safely, and embed once into in-memory vectorstore."""
#     try:
#         if get_cached_vectorstore():
#             st.info("Existing in-memory Chroma DB found — clearing old cache.")
#             clear_cache()

#         all_texts = []
#         splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

#         for uploaded_file in uploaded_files:
#             file_ext = os.path.splitext(uploaded_file.name)[1].lower()
#             text = ""

#             if file_ext == ".pdf":
#                 reader = PdfReader(uploaded_file)
#                 for page in reader.pages:
#                     t = page.extract_text()
#                     if t:
#                         text += t + "\n"

#             elif file_ext == ".pptx":
#                 prs = Presentation(uploaded_file)
#                 for slide in prs.slides:
#                     for shape in slide.shapes:
#                         if hasattr(shape, "text"):
#                             text += shape.text + "\n"

#             elif file_ext == ".docx":
#                 doc = Document(uploaded_file)
#                 for para in doc.paragraphs:
#                     text += para.text + "\n"

#             # Clean text and avoid large single-blocks
#             text = re.sub(r"\s+", " ", text).strip()
#             if not text:
#                 continue

#             chunks = splitter.split_text(text)

#             # Enforce max chunk length for token safety
#             safe_chunks = [c[:1000] if len(c) > 1000 else c for c in chunks]
#             all_texts.extend(safe_chunks)

#         if not all_texts:
#             st.error("❌ No valid text found in uploaded files.")
#             return False

#         embeddings = OpenAIEmbeddings()
#         load_in_memory_vectorstore(embeddings, all_texts, collection_name="ram_store")
#         st.success(f"✅ Successfully embedded {len(all_texts)} text chunks in memory!")
#         return True

#     except Exception as e:
#         st.error(f"❌ Embedding process failed: {e}")
#         st.code(traceback.format_exc())
#         return False


# # ===========================================
# # TAB 1 — UPLOAD & EMBED
# # ===========================================
# with tab1:
#     st.markdown("""
#     ### 📁 Upload Policy Documents
#     Upload **PDF**, **DOCX**, or **PPTX** files here to embed them in-memory.
#     These embeddings will later be used for answering Policy-related queries.
#     """)

#     uploaded_files = st.file_uploader(
#         "Upload one or more files",
#         type=["pdf", "docx", "pptx"],
#         accept_multiple_files=True
#     )

#     if uploaded_files:
#         with st.spinner("🔄 Processing and embedding uploaded files..."):
#             success = process_and_embed_files(uploaded_files)
#         if success:
#             st.caption(f"🧮 Cache status: {cache_status()}")
#     if st.button("🧹 Clear Cache"):
#         msg = clear_cache()
#         st.info(msg)

# # ===========================================
# # TAB 2 — QUERY + ROUTING
# # ===========================================
# with tab2:
#     st.markdown("""
#     ### 🧠 Intelligent Query Router
#     Enter a natural-language query.  
#     The router will:
#     1. Classify it as **Policy**, **Document**, or **Both**
#     2. Route to appropriate handler(s)
#     3. Return detailed results + full trace logs
#     """)

#     query = st.text_input("Enter your question", placeholder="e.g. What is the leave encashment policy?")

#     col1, col2 = st.columns([3, 1])
#     with col2:
#         if st.button("Run Query"):
#             if not query or not query.strip():
#                 st.warning("⚠️ Please enter a question.")
#             else:
#                 st.session_state['run_time'] = datetime.now(timezone.utc).isoformat()
#                 st.session_state['query'] = query.strip()

#     if 'run_time' in st.session_state:
#         st.caption(f"Last run UTC: {st.session_state['run_time']}")

#     # ==============================
#     # EXECUTION PIPELINE
#     # ==============================
#     if 'query' in st.session_state and st.session_state['query']:
#         q = st.session_state['query']

#         st.divider()
#         st.subheader("🔍 Step 1 — Classifying Query")

#         with st.spinner("Classifying query via Router..."):
#             try:
#                 router_response = classify_query(q)
#             except Exception as e:
#                 st.error(f"❌ Router crashed: {e}")
#                 st.code(traceback.format_exc())
#                 st.stop()

#         # Handle router output
#         if isinstance(router_response, dict) and "error" in router_response:
#             st.error(f"❌ Router error at stage **{router_response.get('stage')}**")
#             st.code(router_response["error"])
#             st.stop()
#         else:
#             try:
#                 route, confidence, reason, doc_q, pol_q = router_response
#             except Exception as e:
#                 st.error(f"❌ Unexpected Router output format: {e}")
#                 st.write(router_response)
#                 st.stop()

#         with st.expander("Router Output (Raw)", expanded=False):
#             st.json({
#                 "route": getattr(route, "value", str(route)),
#                 "confidence": confidence,
#                 "reason": reason,
#                 "doc_query": doc_q,
#                 "policy_query": pol_q,
#             })

#         st.success(f"✅ Router classified as **{getattr(route, 'value', str(route)).upper()}** (confidence {confidence})")

#         result_text = ""
#         stage_logs = []

#         # ==============================
#         # DOCUMENT HANDLER
#         # ==============================
#         if getattr(route, "value", str(route)).lower() == "document":
#             st.info("🗂️ Running Mongo (Document) handler...")
#             stage_logs.append("Router → Document Handler (Mongo)")
#             try:
#                 from src.Mongo import query_mongo
#                 with st.spinner("Executing Mongo pipeline..."):
#                     result_text = query_mongo(q)
#                 if isinstance(result_text, str) and result_text.startswith("ERROR"):
#                     st.error(result_text)
#                 else:
#                     st.success("✅ Document handler executed successfully.")
#             except Exception as e:
#                 stage_logs.append(f"Mongo Handler Error: {e}")
#                 st.error(f"❌ Document handler failed: {e}")
#                 st.code(traceback.format_exc())

#         # ==============================
#         # POLICY HANDLER
#         # ==============================
#         elif getattr(route, "value", str(route)).lower() == "policy":
#             st.info("📜 Running Policy handler...")
#             stage_logs.append("Router → Policy Handler")
#             try:
#                 if not get_cached_vectorstore():
#                     st.warning("⚠️ No in-memory Chroma DB found — please upload and embed documents first.")
#                     st.stop()

#                 with st.spinner("Executing Policy RAG pipeline..."):
#                     result_text = policy_handler(q)

#                 if not result_text:
#                     st.warning("⚠️ No response generated by Policy handler.")
#                 elif isinstance(result_text, str) and result_text.startswith("ERROR"):
#                     st.error(result_text)
#                 else:
#                     st.success("✅ Policy handler executed successfully.")

#             except Exception as e:
#                 stage_logs.append(f"Policy Handler Error: {e}")
#                 st.error("❌ Policy handler failed:")
#                 st.code(traceback.format_exc())

#         # ==============================
#         # BOTH HANDLERS
#         # ==============================
#         elif getattr(route, "value", str(route)).lower() == "both":
#             st.info("🔄 Running both Mongo & Policy handlers...")
#             stage_logs.append("Router → Both (Document + Policy)")
#             try:
#                 from src.Mongo import query_mongo
#                 if not get_cached_vectorstore():
#                     st.warning("⚠️ No in-memory Chroma DB found — please upload and embed documents first.")
#                     st.stop()

#                 with st.spinner("Executing Document handler..."):
#                     res_doc = query_mongo(doc_q or q)
#                 with st.spinner("Executing Policy handler..."):
#                     res_pol = policy_handler(pol_q or q)

#                 if any(isinstance(x, str) and x.startswith("ERROR") for x in [res_doc, res_pol]):
#                     st.error("⚠️ One or both handlers reported an error.")
#                 else:
#                     st.success("✅ Both handlers executed successfully.")

#                 result_text = f"--- DOCUMENT RESULT ---\n{res_doc}\n\n--- POLICY RESULT ---\n{res_pol}"

#             except Exception as e:
#                 stage_logs.append(f"Both Handler Error: {e}")
#                 st.error("❌ Combined handler failure:")
#                 st.code(traceback.format_exc())

#         # ==============================
#         # UNKNOWN ROUTE
#         # ==============================
#         else:
#             st.warning(f"⚠️ Unknown route type: {route}")
#             stage_logs.append("Unknown Route")

#         # ==============================
#         # OUTPUT DISPLAY
#         # ==============================
#         st.divider()
#         st.subheader("🧾 Final Result")
#         st.code(result_text or "No output generated.", language="text")

#         st.subheader("📋 Execution Log")
#         for log_entry in stage_logs:
#             st.write("-", log_entry)

#         st.caption("✅ Process completed successfully — all stages ran in same thread.")

# # main.py — Streamlit front-end (single-tab query + email)
# import streamlit as st
# from datetime import datetime, timezone
# import traceback
# import os

# # Router (existing)
# from src.Router_gpt import classify_query, RouteType

# # Import local modules (embedding + retriever + multimedia)
# # They live under src/
# try:
#     from src.embedding_Class import RAGIndexer
# except Exception as e:
#     st.error(f"Failed to import embedding_Class: {e}")
#     st.stop()

# try:
#     from src.retrival_class import Retriever, policy_handler_from_retriever
# except Exception as e:
#     st.error(f"Failed to import retrival_class: {e}")
#     st.stop()

# # multimedia.py is optional — policy_handler_from_retriever already produces answers,
# # but we'll import multimedia_response if you want to pass chunks to it in future.
# try:
#     from src.multimedia import multimedia_response
# except Exception:
#     multimedia_response = None  # optional

# # --------------------------
# # Page config
# # --------------------------
# st.set_page_config(page_title="Mongo_RAG - Query", page_icon="📚", layout="wide")
# st.title("📚 Mongo_RAG — Query (Single Tab)")

# # --------------------------
# # Session-state helpers
# # --------------------------
# if "rag_cache" not in st.session_state:
#     st.session_state.rag_cache = None  # will hold {'texts','vectors','metadatas','embed_model'}

# if "last_embedded" not in st.session_state:
#     st.session_state.last_embedded = None

# if "last_run" not in st.session_state:
#     st.session_state.last_run = None

# # --------------------------
# # Controls: Email + Query + Actions
# # --------------------------
# with st.container():
#     col1, col2 = st.columns([3, 1])
#     with col1:
#         email = st.text_input("User Email (for audit / routing)", value=st.session_state.get("user_email", ""))
#         query = st.text_area("Enter your question", height=120, placeholder="e.g. How do I find my leave balance?")
#     with col2:
#         st.write("")  # spacer
#         if st.button("Run Query"):
#             st.session_state.user_email = email.strip()
#             st.session_state.query_to_run = query.strip()
#             st.session_state.last_run = datetime.now(timezone.utc).isoformat()

#         if st.button("Rebuild embeddings (force)"):
#             # Clear stored cache and prompt rebuild on next request
#             st.session_state.rag_cache = None
#             st.info("Embeddings cache cleared — the next Run Query will rebuild embeddings.")

# # --------------------------
# # Ensure we have a (single) embedding index loaded in session
# # - If not present, build it synchronously now (using Dataset/Policies)
# # --------------------------
# def build_index_from_policies():
#     """Build RAG index from local Dataset/Policies and store in session state."""
#     try:
#         with st.spinner("Building embeddings from Dataset/Policies (this runs once per session)..."):
#             idx = RAGIndexer(
#                 local_paths=["Dataset/Policies"],
#                 s3_urls=None,
#                 workdir="rag_work",
#                 embed_model="text-embedding-3-large",
#                 max_tokens=900,
#                 overlap=150,
#                 min_chunk_chars=280,
#             )
#             idx.build()

#             if not idx.texts or idx.vectors is None:
#                 st.error("Embedding pipeline completed but returned no data. Check files in Dataset/Policies.")
#                 return False

#             # Save to session_state
#             st.session_state.rag_cache = {
#                 "texts": idx.texts,
#                 "vectors": idx.vectors,
#                 "metadatas": idx.metadatas,
#                 "embed_model": idx.cfg.embed_model if hasattr(idx, "cfg") else "text-embedding-3-large",
#             }
#             st.session_state.last_embedded = datetime.now(timezone.utc).isoformat()
#             st.success(f"Embeddings ready — {len(idx.texts)} chunks loaded.")
#             return True
#     except Exception as e:
#         st.error("Failed to build embeddings:")
#         st.code(traceback.format_exc())
#         return False

# # If no cache, build now (synchronous)
# if st.session_state.rag_cache is None:
#     build_index_from_policies()

# # --------------------------
# # Run the pipeline when user pressed Run Query
# # --------------------------
# if st.session_state.get("query_to_run"):
#     q = st.session_state["query_to_run"]
#     st.markdown("---")
#     st.subheader("🔎 Execution Trace")

#     st.write("User:", st.session_state.get("user_email", "N/A"))
#     st.write("Query submitted at (UTC):", st.session_state.get("last_run"))

#     # 1) Classify
#     with st.spinner("Classifying query..."):
#         try:
#             router_response = classify_query(q)
#         except Exception as e:
#             st.error(f"Router crashed: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#     # Validate router response format
#     if isinstance(router_response, dict) and "error" in router_response:
#         st.error(f"Router error: {router_response}")
#         st.stop()

#     try:
#         route, confidence, reason, doc_q, pol_q = router_response
#     except Exception:
#         st.error("Router returned unexpected format.")
#         st.write(router_response)
#         st.stop()

#     st.write("Router decision:", getattr(route, "value", str(route)))
#     st.write("Confidence:", confidence)
#     st.write("Reason:", reason)

#     result_text = ""
#     stage_logs = []

#     # Handle routes
#     route_name = getattr(route, "value", str(route)).lower()

#     if route_name == "document":
#         st.info("🗂 Document route selected — currently unavailable.")
#         stage_logs.append("Router → Document Handler (NOT IMPLEMENTED)")
#         result_text = "Document handler is currently unavailable."

#     elif route_name == "policy":
#         stage_logs.append("Router → Policy Handler")
#         st.info("📜 Policy handler — retrieving from in-memory index.")

#         # Ensure embeddings present
#         cache = st.session_state.rag_cache
#         if not cache:
#             st.error("No embeddings present. Please upload/persist or rebuild embeddings first.")
#             st.stop()

#         # Instantiate Retriever once per run (lightweight)
#         try:
#             retr = Retriever(
#                 texts=cache["texts"],
#                 metadatas=cache.get("metadatas", [{}] * len(cache["texts"])),
#                 vectors=cache["vectors"],
#                 embed_model=cache.get("embed_model"),
#             )
#         except Exception as e:
#             st.error(f"Failed to create Retriever: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#         # Use policy_handler_from_retriever for a safe pipeline (returns answer + chunks)
#         try:
#             with st.spinner("Running retrieval + (optional) rerank + answer generation..."):
#                 answer, context_chunks = policy_handler_from_retriever(retr, q, top_k=5, rerank=True)
#         except Exception as e:
#             st.error(f"Policy pipeline failed: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#         # Present the outputs
#         if isinstance(answer, str) and answer.startswith("ERROR"):
#             st.error(answer)
#         else:
#             st.success("✅ Policy handler executed.")
#             st.subheader("📄 Final Answer")
#             st.write(answer)

#             with st.expander("🔎 Retrieved Chunks (for debugging)", expanded=False):
#                 for i, c in enumerate(context_chunks):
#                     st.markdown(f"**Chunk #{i+1}** (first 600 chars):")
#                     st.code(c[:600], language="text")

#             result_text = answer

#     elif route_name == "both":
#         # Run policy; document is unavailable
#         stage_logs.append("Router → Both (Document + Policy)")
#         st.info("Running policy part (document handler is unavailable).")

#         cache = st.session_state.rag_cache
#         if not cache:
#             st.error("No embeddings present. Please rebuild.")
#             st.stop()

#         retr = Retriever(
#             texts=cache["texts"],
#             metadatas=cache.get("metadatas", [{}] * len(cache["texts"])),
#             vectors=cache["vectors"],
#             embed_model=cache.get("embed_model"),
#         )

#         try:
#             with st.spinner("Running policy retrieval..."):
#                 answer, context_chunks = policy_handler_from_retriever(retr, pol_q or q, top_k=5, rerank=True)
#         except Exception as e:
#             st.error(f"Policy part failed: {e}")
#             st.code(traceback.format_exc())
#             st.stop()

#         result_text = f"--- POLICY RESULT ---\n{answer}\n\n--- DOCUMENT RESULT ---\nCurrently unavailable."

#     else:
#         st.warning(f"Unknown route type: {route_name}")
#         stage_logs.append("Unknown Route")
#         result_text = "No handler for this route."

#     # Display final result and logs
#     st.markdown("---")
#     st.subheader("🧾 Final Result")
#     st.code(result_text or "No result produced.", language="text")

#     st.subheader("📋 Execution Log")
#     for entry in stage_logs:
#         st.write("-", entry)

#     # clear the saved query so the user can type a new one
#     del st.session_state["query_to_run"]

# ===========================================
# ✅ main.py — Streamlit Frontend for RAG System
# ===========================================
# main.py — Streamlit front-end (robust import for src modules)
import streamlit as st
from datetime import datetime, timezone
import traceback
import os
import sys
import importlib
import importlib.util

# -----------------------------
# Helper: resilient loader for src.<module>
# -----------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

def load_src_module(module_name: str):
    """
    Try `import src.<module_name>` normally; if that fails, load from file
    src/<module_name>.py and insert into sys.modules as 'src.<module_name>'.
    Returns the module object.
    """
    full_name = f"src.{module_name}"
    # 1) try normal import (works when src is a package)
    try:
        return importlib.import_module(full_name)
    except Exception:
        pass

    # 2) fallback: import from file path
    module_path = os.path.join(SRC_DIR, f"{module_name}.py")
    if not os.path.isfile(module_path):
        raise ImportError(f"Module file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(full_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")

    mod = importlib.util.module_from_spec(spec)
    # Insert into sys.modules under both file-name key and 'src.<module>' to mimic package import
    sys.modules[full_name] = mod
    sys.modules[f"{module_name}"] = mod  # optional convenience
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        # Remove partially-initialized module if exec fails to avoid stale state
        sys.modules.pop(full_name, None)
        sys.modules.pop(module_name, None)
        raise
    return mod

# -----------------------------
# Import router + core modules via resilient loader
# -----------------------------
try:
    Router_mod = load_src_module("Router_gpt")
    classify_query = getattr(Router_mod, "classify_query")
    RouteType = getattr(Router_mod, "RouteType", None)
except Exception as e:
    st.error(f"Failed to load router module (`src/Router_gpt.py`): {e}")
    st.code(traceback.format_exc())
    st.stop()

# Load embedding, retriever, multimedia similarly
try:
    Emb_mod = load_src_module("embedding_Class")
    RAGIndexer = getattr(Emb_mod, "RAGIndexer")
except Exception as e:
    st.error(f"Failed to load embedding_Class (`src/embedding_Class.py`): {e}")
    st.code(traceback.format_exc())
    st.stop()

try:
    Ret_mod = load_src_module("retrival_class")
    # retrival_class may expose Retriever and/or policy_handler_from_retriever
    Retriever = getattr(Ret_mod, "Retriever")
    policy_handler_from_retriever = getattr(Ret_mod, "policy_handler_from_retriever", None)
except Exception as e:
    st.error(f"Failed to load retrival_class (`src/retrival_class.py`): {e}")
    st.code(traceback.format_exc())
    st.stop()

# multimedia may be optional — but we try to load it the same way
try:
    Multi_mod = load_src_module("Multimedia")
    multimedia_response = getattr(Multi_mod, "multimedia_response", None)
except Exception as e:
    # don't hard-stop here; we can still use policy_handler_from_retriever if present
    multimedia_response = None
    st.warning(f"Warning: multimedia module not loaded: {e}")

# -----------------------------
# Streamlit session-state init
# -----------------------------
if "rag_cache" not in st.session_state:
    # rag_cache will be dict: {texts, vectors, metadatas, embed_model (opt)}
    st.session_state.rag_cache = None

if "last_embedded" not in st.session_state:
    st.session_state.last_embedded = None

if "last_run" not in st.session_state:
    st.session_state.last_run = None

# -----------------------------
# UI: Page config
# -----------------------------
st.set_page_config(page_title="Policy RAG — Streamlit", page_icon="📚", layout="wide")
st.title("📚 Policy RAG — Streamlit Frontend")

# -----------------------------
# Controls: email + query + actions
# -----------------------------
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        user_email = st.text_input("User Email (for audit)", value=st.session_state.get("user_email", ""))
        user_query = st.text_area("Enter your question", height=140, placeholder="e.g. How do I find my leave balance?")
    with col2:
        st.write("")  # spacer
        run = st.button("Run Query")
        rebuild = st.button("Rebuild embeddings (force)")

    if run:
        st.session_state.user_email = user_email.strip()
        st.session_state.query_to_run = user_query.strip()
        st.session_state.last_run = datetime.now(timezone.utc).isoformat()

    if rebuild:
        # clear cache so next run rebuilds embeddings
        st.session_state.rag_cache = None
        st.session_state.last_embedded = None
        st.info("Embeddings cache cleared. Next query will rebuild the index.")

# -----------------------------
# Helper: build embeddings (single-run)
# -----------------------------
POLICIES_PATH = os.path.join(ROOT_DIR, "Dataset", "Policies")

def build_index_from_policies():
    """
    Builds the RAG index from Dataset/Policies using RAGIndexer and stores into session_state.
    """
    try:
        with st.spinner("Building embeddings from Dataset/Policies (this runs once per session)..."):
            idx = RAGIndexer(
                local_paths=[POLICIES_PATH],
                s3_urls=None,
                workdir="rag_work",
                embed_model="text-embedding-3-large",
                max_tokens=900,
                overlap=150,
                min_chunk_chars=280,
            )
            idx.build()

            if not idx.texts or idx.vectors is None:
                st.error("Embedding pipeline completed but returned no data. Check files in Dataset/Policies.")
                return False

            # Save to session_state
            st.session_state.rag_cache = {
                "texts": idx.texts,
                "vectors": idx.vectors,
                "metadatas": idx.metadatas,
                "embed_model": getattr(idx.cfg, "embed_model", "text-embedding-3-large"),
            }
            st.session_state.last_embedded = datetime.now(timezone.utc).isoformat()
            st.success(f"Embeddings ready — {len(idx.texts)} chunks loaded into RAM.")
            return True

    except Exception as e:
        st.error("Failed to build embeddings:")
        st.code(traceback.format_exc())
        return False

# If not present, build now (sync)
if st.session_state.rag_cache is None:
    build_index_from_policies()

# -----------------------------
# When user pressed Run Query
# -----------------------------
if st.session_state.get("query_to_run"):
    q = st.session_state["query_to_run"]
    st.markdown("---")
    st.subheader("🔎 Execution Trace")

    st.write("User:", st.session_state.get("user_email", "N/A"))
    st.write("Query submitted at (UTC):", st.session_state.get("last_run"))

    # 1) Classify
    with st.spinner("Classifying query..."):
        try:
            router_response = classify_query(q)
        except Exception as e:
            st.error(f"Router crashed: {e}")
            st.code(traceback.format_exc())
            st.stop()

    # Validate router response
    if isinstance(router_response, dict) and "error" in router_response:
        st.error(f"Router error: {router_response}")
        st.stop()

    try:
        route, confidence, reason, doc_q, pol_q = router_response
    except Exception:
        st.error("Router returned unexpected format.")
        st.write(router_response)
        st.stop()

    st.write("Router decision:", getattr(route, "value", str(route)))
    st.write("Confidence:", confidence)
    st.write("Reason:", reason)

    result_text = ""
    stage_logs = []

    # route_name normalized
    route_name = getattr(route, "value", str(route)).lower()

    # Helper to safely run policy pipeline
    def run_policy_pipeline(use_query: str):
        """
        Returns (answer_str, context_chunks_list) or (error_str, []).
        """
        cache = st.session_state.rag_cache
        if not cache:
            return ("ERROR: No embeddings present. Please upload/persist or rebuild embeddings.", [])

        try:
            retr = Retriever(
                texts=cache["texts"],
                metadatas=cache.get("metadatas", [{}] * len(cache["texts"])),
                vectors=cache["vectors"],
                embed_model=cache.get("embed_model"),
            )
        except Exception as e:
            return (f"ERROR: Failed to create Retriever: {e}", [])

        # Prefer high-level handler if available
        if policy_handler_from_retriever is not None:
            try:
                answer, context_chunks = policy_handler_from_retriever(retr, use_query, top_k=5, rerank=True)
                return (answer, context_chunks)
            except Exception as e:
                return (f"ERROR: policy_handler_from_retriever crashed: {e}\n{traceback.format_exc()}", [])

        # else fallback to manual retrieval + multimedia (if available)
        try:
            ret = retr.retrieve(use_query, top_k=5, rerank=True)
            if "error" in ret:
                return (f"ERROR: Retriever returned error: {ret['error']}", [])
            candidates = ret.get("candidates", [])
            chunks = [c["text"] for c in candidates]
            # If multimedia_response available, use it
            if multimedia_response:
                try:
                    ans = multimedia_response(use_query, chunks)
                    return (ans, chunks)
                except Exception as e:
                    return (f"ERROR: multimedia_response failed: {e}\n{traceback.format_exc()}", chunks)
            else:
                # return concatenated chunks
                combined = "\n\n---\n\n".join(chunks)
                return (combined, chunks)
        except Exception as e:
            return (f"ERROR: Retrieval pipeline failed: {e}\n{traceback.format_exc()}", [])

    # handle routes
    if route_name == "document":
        st.info("🗂 Document route selected — currently unavailable.")
        stage_logs.append("Router → Document Handler (NOT IMPLEMENTED)")
        result_text = "Document handler is currently unavailable. (Planned)"

    elif route_name == "policy":
        stage_logs.append("Router → Policy Handler")
        st.info("📜 Policy handler — retrieving from in-memory index.")

        answer, context_chunks = run_policy_pipeline(pol_q or q)

        if isinstance(answer, str) and answer.startswith("ERROR"):
            st.error(answer)
            result_text = answer
        else:
            st.success("✅ Policy handler executed.")
            st.subheader("📄 Final Answer")
            st.write(answer)
            with st.expander("🔎 Retrieved Chunks (debug)", expanded=False):
                for i, c in enumerate(context_chunks):
                    st.markdown(f"**Chunk #{i+1}** (first 600 chars):")
                    st.code(c[:600], language="text")
            result_text = answer

    elif route_name == "both":
        stage_logs.append("Router → Both (Document + Policy)")
        st.info("Running policy part (document handler is unavailable).")

        answer, context_chunks = run_policy_pipeline(pol_q or q)
        if isinstance(answer, str) and answer.startswith("ERROR"):
            st.error(answer)
            result_text = answer
        else:
            result_text = f"--- POLICY RESULT ---\n{answer}\n\n--- DOCUMENT RESULT ---\nCurrently unavailable."

            st.subheader("📄 Policy Answer (Documents not available)")
            st.write(answer)
            with st.expander("🔎 Retrieved Chunks (debug)", expanded=False):
                for i, c in enumerate(context_chunks):
                    st.markdown(f"**Chunk #{i+1}** (first 600 chars):")
                    st.code(c[:600], language="text")

    else:
        st.warning(f"Unknown route type: {route_name}")
        stage_logs.append("Unknown Route")
        result_text = "No handler for this route."

    # Display final result and logs
    st.markdown("---")
    st.subheader("🧾 Final Result")
    st.code(result_text or "No result produced.", language="text")

    st.subheader("📋 Execution Log")
    for entry in stage_logs:
        st.write("-", entry)

    # clear the saved query so user can type a new one
    if "query_to_run" in st.session_state:
        del st.session_state["query_to_run"]
