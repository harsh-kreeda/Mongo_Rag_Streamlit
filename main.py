# main.py — Streamlit front-end for Router -> (Mongo | Policy) handlers
import streamlit as st
from datetime import datetime
import os
from importlib import import_module

# import Router_gpt
from src.Router_gpt import classify_query, RouteType

st.set_page_config(page_title="Mongo_RAG Streamlit", page_icon="📚", layout="wide")
st.title("Mongo_RAG - Streamlit Frontend")
st.markdown("Enter a natural-language query and the router will decide whether to call the Mongo handler or the Policy handler.")

# Input
query = st.text_input("Enter your question", placeholder="e.g. What is the leave encashment policy?")

col1, col2 = st.columns([3,1])
with col2:
    if st.button("Run Query"):
        if not query or not query.strip():
            st.warning("Please enter a question.")
        else:
            st.session_state['last_query_time'] = datetime.utcnow().isoformat()

# Show last run and a manual run button
if 'last_query_time' in st.session_state:
    st.caption(f"Last run UTC: {st.session_state['last_query_time']}")

# Run pipeline
if st.button("Execute") or ('last_query_time' in st.session_state and query):
    q = query.strip()
    with st.spinner("Classifying query..."):
        try:
            route, confidence, reason, doc_q, pol_q = classify_query(q)
        except Exception as e:
            st.error(f"Router error: {e}")
            route = None
            confidence = 0.0
            reason = f"Router raised exception: {e}"
            doc_q = pol_q = None

    st.subheader("Router decision")
    st.write({
        "route": getattr(route, "value", str(route)),
        "confidence": confidence,
        "reason": reason,
        "doc_query": doc_q,
        "policy_query": pol_q,
    })

    # Route to handlers
    result_text = ""
    log_lines = []
    if route is None:
        st.error("No route returned.")
    else:
        if getattr(route, "value", str(route)).lower() == "document":
            st.info("→ Running Mongo/document handler")
            log_lines.append("Calling src.Mongo.query_mongo(...)")
            try:
                from src.Mongo import query_mongo
                result_text = query_mongo(q)
            except Exception as e:
                result_text = f"ERROR calling Mongo handler: {e}"
        elif getattr(route, "value", str(route)).lower() == "policy":
            st.info("→ Running Policy handler")
            log_lines.append("Calling src.Mutlimedia.policy_handler(...)")
            try:
                from src.Mutlimedia import policy_handler
                result_text = policy_handler(q)
            except Exception as e:
                result_text = f"ERROR calling Policy handler: {e}"
        elif getattr(route, "value", str(route)).lower() == "both":
            st.info("→ BOTH: running both handlers and merging outputs")
            log_lines.append("Calling both Mongo and Policy handlers")
            try:
                from src.Mongo import query_mongo
                from src.Mutlimedia import policy_handler
                res_doc = query_mongo(doc_q or q)
                res_pol = policy_handler(pol_q or q)
                # Simple merge — you can replace with better merging logic
                result_text = f"--- DOCUMENT RESULT ---\n{res_doc}\n\n--- POLICY RESULT ---\n{res_pol}"
            except Exception as e:
                result_text = f"ERROR calling both handlers: {e}"
        else:
            st.warning(f"Unknown route: {route}")
            result_text = "Unknown route returned by router."

    st.subheader("Result")
    st.code(result_text)

    st.subheader("Processing log")
    for ln in log_lines:
        st.write("-", ln)
