"""Streamlit UI for the Production RAG Chatbot.

Features:
- Login / Register
- Document upload (PDF, DOCX, TXT, MD)
- Chat with source citations
- RAG triad scores per response
- Session list with history (persisted in PostgreSQL via API)
- Create new session / switch sessions
"""

import os
import time
from typing import List, Optional

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
API_V1 = f"{API_BASE}/api/v1"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")


# ─── Language options ─────────────────────────────────────────────────────────

_LANGUAGES = {
    "English": "en",
    "Japanese": "ja",
    "Vietnamese": "vi",
    "French": "fr",
    "German": "de",
    "Chinese": "zh",
    "Korean": "ko",
    "Spanish": "es",
    "Portuguese": "pt",
    "Thai": "th",
    "Arabic": "ar",
    "Russian": "ru",
    "Indonesian": "id",
}


# ─── Session state helpers ────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "token": None,
        "active_session_id": None,   # currently displayed session
        "default_session_id": None,  # session embedded in JWT (from login)
        "messages": [],              # displayed message objects
        "eval_scores": [],
        "target_language": "en",
        "sessions": [],              # list of SessionInfo dicts from API
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ─── API helpers ──────────────────────────────────────────────────────────────

def _fetch_sessions() -> List[dict]:
    """Load session list from the API."""
    try:
        resp = httpx.get(f"{API_V1}/chat/sessions", headers=_headers(), timeout=10.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return []


def _fetch_history(session_id: str) -> List[dict]:
    """Fetch message history for a session."""
    try:
        resp = httpx.get(
            f"{API_V1}/chat/sessions/{session_id}/history",
            headers=_headers(),
            timeout=10.0,
        )
        if resp.status_code == 200:
            return resp.json().get("messages", [])
    except Exception:
        pass
    return []


def _create_session(name: str) -> Optional[dict]:
    """Create a new session and return its info dict."""
    try:
        resp = httpx.post(
            f"{API_V1}/chat/sessions",
            headers=_headers(),
            json={"name": name},
            timeout=10.0,
        )
        if resp.status_code == 201:
            return resp.json()
    except Exception:
        pass
    return None


def _delete_session(session_id: str) -> bool:
    """Delete a session and all its messages. Returns True on success."""
    try:
        resp = httpx.delete(
            f"{API_V1}/chat/sessions/{session_id}",
            headers=_headers(),
            timeout=10.0,
        )
        return resp.status_code == 204
    except Exception:
        return False


def _switch_session(session_id: str) -> None:
    """Load history for a session and set it as active."""
    messages = _fetch_history(session_id)
    st.session_state.active_session_id = session_id
    st.session_state.messages = _history_to_display(messages)
    st.session_state.eval_scores = [
        m["eval_scores"] for m in messages
        if m.get("role") == "assistant" and m.get("eval_scores")
    ]


def _history_to_display(api_messages: List[dict]) -> List[dict]:
    """Convert API HistoryMessage list to the display format used by the UI."""
    display = []
    for m in api_messages:
        display.append({
            "role": m["role"],
            "content": m["content"],
            "citations": m.get("citations", []),
            "eval": m.get("eval_scores"),
        })
    return display


# ─── Auth forms ───────────────────────────────────────────────────────────────

def _login_page() -> None:
    st.title("RAG Chatbot")
    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            try:
                resp = httpx.post(
                    f"{API_V1}/auth/login",
                    json={"email": email, "password": password},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.default_session_id = data["session_id"]
                    st.session_state.active_session_id = data["session_id"]
                    # Load sessions list
                    st.session_state.sessions = _fetch_sessions()
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
                resp = httpx.post(
                    f"{API_V1}/auth/register",
                    json={"email": email_r, "password": password_r},
                )
                if resp.status_code == 201:
                    data = resp.json()
                    st.session_state.token = data["access_token"]
                    st.session_state.default_session_id = data["session_id"]
                    st.session_state.active_session_id = data["session_id"]
                    st.session_state.sessions = _fetch_sessions()
                    st.success("Registered!")
                    st.rerun()
                else:
                    st.error(resp.json().get("detail", "Registration failed"))
            except Exception as exc:
                st.error(f"Connection error: {exc}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        # ── Document upload ────────────────────────────────────────────────────
        st.markdown("### Upload Documents")
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
                                st.success(f"{f.name} accepted for indexing")
                            else:
                                st.error(f"Failed: {resp.json().get('detail')}")
                        except Exception as exc:
                            st.error(str(exc))

        st.divider()

        # ── Answer language ────────────────────────────────────────────────────
        st.markdown("### Answer Language")
        lang_label = st.selectbox(
            "Respond in",
            options=list(_LANGUAGES.keys()),
            index=list(_LANGUAGES.values()).index(st.session_state.target_language),
            label_visibility="collapsed",
        )
        st.session_state.target_language = _LANGUAGES[lang_label]

        st.divider()

        # ── Session management ─────────────────────────────────────────────────
        st.markdown("### Chat Sessions")

        col_new, col_refresh = st.columns([1, 1])
        with col_new:
            if st.button("＋ New Chat", use_container_width=True, help="Start a new conversation"):
                info = _create_session("")
                if info:
                    st.session_state.sessions = _fetch_sessions()
                    _switch_session(info["session_id"])
                    st.rerun()
        with col_refresh:
            if st.button("↺ Refresh", use_container_width=True, help="Reload session list"):
                st.session_state.sessions = _fetch_sessions()
                st.rerun()

        sessions = st.session_state.sessions
        if sessions:
            for sess in sessions:
                sid = sess["session_id"]
                name = sess.get("name") or "Chat"
                count = sess.get("message_count", 0)
                preview = sess.get("last_message") or ""
                is_active = sid == st.session_state.active_session_id

                label = f"**{name}**" if is_active else name
                tooltip = f"{count // 2} turn(s)"
                if preview:
                    tooltip += f" · {preview[:50]}{'…' if len(preview) > 50 else ''}"

                col_name, col_del = st.columns([5, 1])
                with col_name:
                    if st.button(label, key=f"sess_{sid}", help=tooltip, use_container_width=True):
                        if sid != st.session_state.active_session_id:
                            _switch_session(sid)
                            st.rerun()
                with col_del:
                    if st.button("🗑", key=f"del_{sid}", help="Delete this chat"):
                        if _delete_session(sid):
                            st.session_state.sessions = _fetch_sessions()
                            # If we deleted the active session, switch to the next one
                            if sid == st.session_state.active_session_id:
                                remaining = st.session_state.sessions
                                if remaining:
                                    _switch_session(remaining[0]["session_id"])
                                else:
                                    st.session_state.active_session_id = None
                                    st.session_state.messages = []
                            st.rerun()
        else:
            st.caption("No sessions yet")

        st.divider()

        # ── Eval summary ───────────────────────────────────────────────────────
        st.markdown("### Eval Scores (last 10)")
        if st.session_state.eval_scores:
            recent = [s for s in st.session_state.eval_scores[-10:] if s]
            if recent:
                avg_cr = sum(s.get("context_relevance") or 0 for s in recent) / len(recent)
                avg_gd = sum(s.get("groundedness") or 0 for s in recent) / len(recent)
                avg_ar = sum(s.get("answer_relevance") or 0 for s in recent) / len(recent)
                st.metric("Context Relevance", f"{avg_cr:.2f}")
                st.metric("Groundedness", f"{avg_gd:.2f}")
                st.metric("Answer Relevance", f"{avg_ar:.2f}")
        else:
            st.caption("No eval scores yet")

        st.divider()
        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ─── Chat interface ───────────────────────────────────────────────────────────

def _chat_page() -> None:
    # Ensure sessions list is loaded
    if not st.session_state.sessions:
        st.session_state.sessions = _fetch_sessions()
        # If the active session is the JWT session and has no messages yet, load it
        if st.session_state.active_session_id and not st.session_state.messages:
            _switch_session(st.session_state.active_session_id)

    _sidebar()

    # Session header
    active_sid = st.session_state.active_session_id
    active_name = next(
        (s.get("name", "Chat") for s in st.session_state.sessions if s["session_id"] == active_sid),
        "Chat",
    )
    st.title(f"💬 {active_name}")

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
                cols[0].metric("Context Rel.", f"{e.get('context_relevance', 'N/A'):.2f}" if e.get("context_relevance") is not None else "N/A")
                cols[1].metric("Groundedness", f"{e.get('groundedness', 'N/A'):.2f}" if e.get("groundedness") is not None else "N/A")
                cols[2].metric("Answer Rel.", f"{e.get('answer_relevance', 'N/A'):.2f}" if e.get("answer_relevance") is not None else "N/A")

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
                            "session_id": st.session_state.active_session_id,
                            "target_language": st.session_state.target_language,
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
                            cols[1].metric("Context Rel.", f"{eval_scores.get('context_relevance', 'N/A'):.2f}" if eval_scores.get("context_relevance") is not None else "N/A")
                            cols[2].metric("Groundedness", f"{eval_scores.get('groundedness', 'N/A'):.2f}" if eval_scores.get("groundedness") is not None else "N/A")
                            cols[3].metric("Answer Rel.", f"{eval_scores.get('answer_relevance', 'N/A'):.2f}" if eval_scores.get("answer_relevance") is not None else "N/A")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "citations": citations,
                            "eval": eval_scores,
                        })
                        if eval_scores:
                            st.session_state.eval_scores.append(eval_scores)

                        # Refresh session list so updated_at / preview update
                        st.session_state.sessions = _fetch_sessions()

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
