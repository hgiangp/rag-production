"""Streamlit UI for the Production RAG Chatbot.

Features:
- Login / Register
- Document upload (PDF, DOCX, TXT, MD)
- Chat with source citations
- RAG triad scores per response
- Session history
"""

import os
import time
from typing import Optional

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE}/api/v1"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")


# ─── Session state helpers ────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "token": None,
        "session_id": None,
        "messages": [],
        "eval_scores": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ─── Auth forms ───────────────────────────────────────────────────────────────

def _login_page() -> None:
    st.title("🤖 RAG Chatbot")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            try:
                resp = httpx.post(f"{API_V1}/auth/login", json={"email": email, "password": password})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.session_id = data["session_id"]
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Login failed"))
            except Exception as exc:
                st.error(f"Connection error: {exc}")

    with tab_register:
        with st.form("register"):
            email_r = st.text_input("Email", key="reg_email")
            password_r = st.text_input("Password (min 8 chars)", type="password", key="reg_pw")
            submitted_r = st.form_submit_button("Register")
        if submitted_r:
            try:
                resp = httpx.post(f"{API_V1}/auth/register", json={"email": email_r, "password": password_r})
                if resp.status_code == 201:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.session_id = data["session_id"]
                    st.success("Registered! Logging you in...")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Registration failed"))
            except Exception as exc:
                st.error(f"Connection error: {exc}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📄 Upload Documents")
        uploaded = st.file_uploader(
            "PDF, DOCX, TXT, MD",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )
        if uploaded:
            for f in uploaded:
                if st.button(f"Ingest: {f.name}", key=f"ingest_{f.name}"):
                    with st.spinner(f"Uploading {f.name}..."):
                        try:
                            resp = httpx.post(
                                f"{API_V1}/documents",
                                headers=_headers(),
                                files={"file": (f.name, f.read(), f.type)},
                                timeout=30.0,
                            )
                            if resp.status_code in (200, 202):
                                st.success(f"✓ {f.name} accepted for indexing")
                            else:
                                st.error(f"Failed: {resp.json().get('detail')}")
                        except Exception as exc:
                            st.error(str(exc))

        st.divider()

        # Eval summary
        st.markdown("### 📊 Eval Scores (last 10)")
        if st.session_state.eval_scores:
            recent = st.session_state.eval_scores[-10:]
            avg_cr = sum(s.get("context_relevance", 0) or 0 for s in recent) / len(recent)
            avg_gd = sum(s.get("groundedness", 0) or 0 for s in recent) / len(recent)
            avg_ar = sum(s.get("answer_relevance", 0) or 0 for s in recent) / len(recent)
            st.metric("Context Relevance", f"{avg_cr:.2f}")
            st.metric("Groundedness", f"{avg_gd:.2f}")
            st.metric("Answer Relevance", f"{avg_ar:.2f}")
        else:
            st.caption("No eval scores yet")

        st.divider()
        if st.button("🚪 Logout"):
            for key in ["token", "session_id", "messages", "eval_scores"]:
                st.session_state[key] = None if key != "messages" else []
            st.rerun()


# ─── Chat interface ───────────────────────────────────────────────────────────

def _chat_page() -> None:
    st.title("💬 RAG Chat")
    _sidebar()

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📚 {len(msg['citations'])} source(s)"):
                    for cite in msg["citations"]:
                        st.markdown(
                            f"**{cite['filename']}** (score: {cite['score']:.2f})\n"
                            f"> {cite['excerpt']}"
                        )
            if msg.get("eval"):
                e = msg["eval"]
                cols = st.columns(3)
                cols[0].metric("Context Rel.", f"{e.get('context_relevance', 'N/A')}")
                cols[1].metric("Groundedness", f"{e.get('groundedness', 'N/A')}")
                cols[2].metric("Answer Rel.", f"{e.get('answer_relevance', 'N/A')}")

    # User input
    if prompt := st.chat_input("Ask anything about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    start = time.time()
                    resp = httpx.post(
                        f"{API_V1}/chat/invoke",
                        headers=_headers(),
                        json={
                            "message": prompt,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=600.0,
                    )
                    latency = int((time.time() - start) * 1000)

                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["message"]["content"]
                        citations = data.get("citations", [])
                        eval_scores = data.get("eval_scores") or {}

                        st.markdown(answer)

                        if citations:
                            with st.expander(f"📚 {len(citations)} source(s)"):
                                for cite in citations:
                                    st.markdown(
                                        f"**{cite['filename']}** (score: {cite['score']:.2f})\n"
                                        f"> {cite['excerpt']}"
                                    )

                        if eval_scores:
                            cols = st.columns(4)
                            cols[0].metric("Latency", f"{latency}ms")
                            cols[1].metric("Context Rel.", f"{eval_scores.get('context_relevance', 'N/A'):.2f}" if eval_scores.get('context_relevance') else "N/A")
                            cols[2].metric("Groundedness", f"{eval_scores.get('groundedness', 'N/A'):.2f}" if eval_scores.get('groundedness') else "N/A")
                            cols[3].metric("Answer Rel.", f"{eval_scores.get('answer_relevance', 'N/A'):.2f}" if eval_scores.get('answer_relevance') else "N/A")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "eval": eval_scores,
                        })
                        if eval_scores:
                            st.session_state.eval_scores.append(eval_scores)
                    else:
                        error_msg = resp.json().get("detail", "Request failed")
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})

                except Exception as exc:
                    st.error(f"Connection error: {exc}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    if not st.session_state.token:
        _login_page()
    else:
        _chat_page()


if __name__ == "__main__":
    main()
