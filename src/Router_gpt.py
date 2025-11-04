##############################################################################################
# V3 - Streamlit Optimized Router
# With 3 Functional Direct Enforcement Routes and Full Error Reporting
##############################################################################################

#!/usr/bin/env python3
"""
Router_gpt.py (Streamlit-ready)

Supports three route outcomes:
 - policy
 - document
 - both   (requires subqueries doc_query + policy_query)

Enhancements:
 - Returns all runtime and LLM errors to caller instead of raising.
 - Uses Streamlit secrets for API key (fallback to .env or environment).
 - Thread-safe; runs synchronously in the same process (no subprocess calls).
 - Logs stage-by-stage status for Streamlit display.
"""

import os
import re
import json
import time
from enum import Enum
from typing import Tuple, Dict, Any, Optional, Union

# Optional Streamlit import — only when running inside Streamlit
try:
    import streamlit as st
except ImportError:
    st = None

from dotenv import load_dotenv
load_dotenv("./.env")

# Use OpenAI client API
from openai import OpenAI

# ----------------------------- CONFIG ---------------------------------
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_RETRIES = 3
BACKOFF_BASE = 1.5
TIMEOUT = 20
# ----------------------------------------------------------------------

class RouteType(str, Enum):
    POLICY = "policy"
    DOCUMENT = "document"
    BOTH = "both"

# ------------------------ FALLBACK KEYWORDS ----------------------------
POLICY_KEYWORDS = [
    "policy", "policies", "compliance", "procedure", "guideline",
    "eligibility", "law", "legal", "regulation", "entitlement",
    "disciplinary", "grievance", "confidentiality", "privacy", "appeal",
    "escalate", "approval", "how should", "can i", "do i have to", "notice period",
    "severance", "probation", "maternity", "paternity", "benefits", "reimbursement",
    "leave policy", "attendance policy", "termination", "resignation", "promotion policy"
]

DOCUMENT_KEYWORDS = [
    "report", "show", "list", "rows", "find", "get", "give me", "count",
    "employee", "person number", "emp code", "classroom", "attendance",
    "pms", "pip", "leave", "transaction", "goal", "status", "requests",
    "xls", "xlsx", "csv", "table", "data", "value", "document", "balance", "payroll",
    "performance", "manager email", "assigned on", "start date", "end date"
]
# ----------------------------------------------------------------------

# ------------------------- PRIMARY PROMPT ------------------------------
SYSTEM_PROMPT = """You are an HR assistant *intent classifier* used in production.
Goal: decide whether a user's natural-language query should be routed to exactly one of:
 - "policy"   : requires HR policy interpretation, rules, eligibility or prescriptive guidance.
 - "document" : requires fetching/returning factual data from internal structured sources (Excel reports/db).
 - "both"     : the query legitimately requires both a document lookup AND policy reasoning; in that case the system needs two separate sub-queries (one for docs, one for policy).

Operational rules (enforced in code):
- Final allowed routes: policy, document, both.
- If the user request includes explicit instructions to fetch records or IDs, prefer document.
- If the request asks for rules/eligibility/what-to-do, prefer policy.
- If the request requires both factual lookup and rule interpretation (e.g., "Show my leave balance and tell me if unused leaves can be encashed"), mark BOTH.
- Always output JSON only:
  {"route":"policy"|"document"|"both", "confidence":<0-1 float>, "reason":"short justification"}
"""
USER_PROMPT_TEMPLATE = """User query:
"{query}"

Classify according to the SYSTEM instructions above. Return only JSON.
"""
# ----------------------------------------------------------------------

# ------------------------ Utilities & Fallbacks -------------------------

def _parse_json_safe(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


def keyword_scores(query: str) -> Tuple[int, int]:
    q = query.lower()
    pol_score = sum(1 for kw in POLICY_KEYWORDS if kw in q)
    doc_score = sum(1 for kw in DOCUMENT_KEYWORDS if kw in q)
    return pol_score, doc_score


def keyword_fallback_decision(query: str) -> Tuple[RouteType, float, str, Optional[str], Optional[str]]:
    """
    Deterministic fallback: (route, confidence, reason, doc_query, policy_query)
    """
    pol_score, doc_score = keyword_scores(query)
    if pol_score == 0 and doc_score == 0:
        return RouteType.POLICY, 0.55, "Fallback: no strong keywords; defaulting to policy.", None, None
    if pol_score > 0 and doc_score > 0:
        doc_q, pol_q = fallback_split_queries_by_keywords(query)
        return RouteType.BOTH, 0.8, "Fallback: both policy & document cues present.", doc_q, pol_q
    if pol_score > doc_score:
        return RouteType.POLICY, 0.75, "Fallback: keyword-based policy detection.", None, None
    else:
        return RouteType.DOCUMENT, 0.75, "Fallback: keyword-based document detection.", None, None


def fallback_split_queries_by_keywords(query: str) -> Tuple[str, str]:
    q = query.strip()
    doc_phrases, policy_phrases = [], []
    parts = re.split(r"\band\b|\bthen\b|\b, and\b|\b;|\bor\b", q, flags=re.IGNORECASE)
    for p in parts:
        pl = p.strip()
        if any(kw in pl.lower() for kw in DOCUMENT_KEYWORDS):
            doc_phrases.append(pl)
        elif any(kw in pl.lower() for kw in POLICY_KEYWORDS):
            policy_phrases.append(pl)
        else:
            doc_phrases.append(pl)
            policy_phrases.append(pl)
    doc_q = " ; ".join(doc_phrases) if doc_phrases else q
    pol_q = " ; ".join(policy_phrases) if policy_phrases else q
    return doc_q, pol_q

# ------------------------ Secondary LLM helpers -------------------------

def enforce_binary_decision_with_model(client: OpenAI, query: str, model: str):
    system = ("RETURN ONLY the single word 'policy' or 'document' (lowercase). "
              "Interpret 'document' as requiring structured-data lookup. "
              "Interpret 'policy' as normative HR interpretation.")
    user = f"Decide for query: \"{query}\". Reply only with policy or document."
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=6,
            timeout=TIMEOUT,
        )
        content = resp.choices[0].message.content.strip().lower()
        if content == "policy":
            return RouteType.POLICY, 0.95, "Enforced binary LLM: policy"
        if content == "document":
            return RouteType.DOCUMENT, 0.95, "Enforced binary LLM: document"
    except Exception as e:
        return f"Error in enforce_binary_decision_with_model: {e}"
    return None


def generate_split_queries_with_model(client: OpenAI, query: str, model: str):
    system = ("Rewrite the user's query into two concise sub-queries as JSON: "
              '{"doc_query":"...","policy_query":"..."} — '
              "doc_query targets document retrieval, policy_query targets policy reasoning.")
    user = f"Original query: \"{query}\". Produce JSON now."
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=200,
            timeout=TIMEOUT,
        )
        parsed = _parse_json_safe(resp.choices[0].message.content.strip())
        if parsed and "doc_query" in parsed and "policy_query" in parsed:
            return parsed["doc_query"].strip(), parsed["policy_query"].strip()
    except Exception as e:
        return f"Error in generate_split_queries_with_model: {e}"
    return None

# ------------------------ MAIN CLASSIFIER -------------------------------

def classify_query(
    query: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> Union[Tuple[RouteType, float, str, Optional[str], Optional[str]], Dict[str, str]]:
    """
    Classify query into one of (policy, document, both).
    Returns:
      (route, confidence, reason, doc_query, policy_query)
    OR on error:
      {"error": "...", "stage": "..."}
    """
    stage = "init"
    try:
        if not query or not query.strip():
            return RouteType.POLICY, 0.0, "Empty query; defaulting to policy.", None, None

        stage = "apikey"
        if not api_key:
            if st and hasattr(st, "secrets"):
                api_key = st.secrets.get("OPENAI_API_KEY")
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "Missing OpenAI API key (check Streamlit secrets or .env)", "stage": stage}

        stage = "client_init"
        client = OpenAI(api_key=api_key)

        stage = "llm_call"
        system_msg = SYSTEM_PROMPT
        user_msg = USER_PROMPT_TEMPLATE.format(query=query)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                    timeout=TIMEOUT,
                )
                content = resp.choices[0].message.content.strip()
                parsed = _parse_json_safe(content)
                if not parsed:
                    raise ValueError("Invalid or non-JSON response from model.")

                route_raw = str(parsed.get("route", "")).lower()
                confidence = float(parsed.get("confidence", 0.6))
                reason = str(parsed.get("reason", "")).strip()

                if route_raw in ("policy", "document", "both"):
                    if route_raw == "both":
                        stage = "split_generation"
                        split = generate_split_queries_with_model(client, query, model)
                        if isinstance(split, tuple):
                            doc_q, pol_q = split
                            return RouteType.BOTH, confidence, reason or "Model indicated both", doc_q, pol_q
                        elif isinstance(split, str) and split.startswith("Error"):
                            return {"error": split, "stage": stage}
                        else:
                            doc_q, pol_q = fallback_split_queries_by_keywords(query)
                            return RouteType.BOTH, confidence, reason + " (fallback split)", doc_q, pol_q
                    elif route_raw == "policy":
                        return RouteType.POLICY, confidence, reason, None, None
                    elif route_raw == "document":
                        return RouteType.DOCUMENT, confidence, reason, None, None
                break
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE * (2 ** (attempt - 1)))
                    continue
                return {"error": f"Primary LLM classification failed: {e}", "stage": stage}

        # Enforcement/fallback chain
        stage = "fallback_enforce"
        pol_score, doc_score = keyword_scores(query)
        if pol_score >= 1 and doc_score >= 1:
            split = generate_split_queries_with_model(client, query, model)
            if isinstance(split, tuple):
                doc_q, pol_q = split
            else:
                doc_q, pol_q = fallback_split_queries_by_keywords(query)
            return RouteType.BOTH, 0.9, "Detected both via keywords.", doc_q, pol_q

        enforced = enforce_binary_decision_with_model(client, query, model)
        if isinstance(enforced, tuple):
            return enforced[0], enforced[1], enforced[2], None, None
        elif isinstance(enforced, str) and enforced.startswith("Error"):
            return {"error": enforced, "stage": stage}

        route, conf, reason, doc_q, pol_q = keyword_fallback_decision(query)
        return route, conf, reason, doc_q, pol_q

    except Exception as e:
        return {"error": f"Unhandled error: {e}", "stage": stage}

# --------------------------- CLI ENTRYPOINT ----------------------------
if __name__ == "__main__":
    print("\nQuery Router — Determine if query is POLICY, DOCUMENT or BOTH.")
    user_q = input("Enter your query: ").strip() or \
              "Show my leave balance and tell me if unused leaves can be encashed."
    result = classify_query(user_q)
    print("\n--- Classification Result ---")
    print(result)
