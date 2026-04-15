"""Streamlit UI for the Production RAG Chatbot.

Features:
- Login / Register
- Document upload (PDF, DOCX, TXT, MD)
- Chat with SSE streaming and live workflow transparency panel
- Source citations and RAG triad scores per response
- Session list with history (persisted in PostgreSQL via API)
- Create new session / switch sessions
"""

import json
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

# Icon per workflow step type
_STEP_ICONS = {
    "query_rewrite": "🔍",
    "tool_retrieval": "📄",
    "cross_ref_detected": "🔗",
    "cross_ref_fetched": "📋",
    "aggregation": "🧩",
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
            "workflow_steps": m.get("workflow_steps", []),
        })
    return display


# ─── Streaming helper ─────────────────────────────────────────────────────────

def _stream_chat(prompt: str, session_id: str, target_language: str):
    """Generator that parses SSE chunks from /chat/stream.

    Yields parsed chunk dicts: {type, content, citations, workflow_step, ...}
    Uses httpx synchronous streaming — no new dependencies needed.
    """
    url = f"{API_V1}/chat/stream"
    payload = {
        "message": prompt,
        "session_id": session_id,
        "target_language": target_language,
    }
    with httpx.stream(
        "POST",
        url,
        headers=_headers(),
        json=payload,
        timeout=600.0,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            raw = line[len("data: "):]
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue


# ─── Workflow step rendering ──────────────────────────────────────────────────

def _render_workflow_step(step: dict) -> None:
    """Render a single completed WorkflowStep dict inside a nested expander."""
    step_type = step.get("step_type", "")
    title = step.get("title", "Step")
    duration = step.get("duration_ms")
    data = step.get("data") or {}
    icon = _STEP_ICONS.get(step_type, "•")
    dur_str = f" — {duration}ms" if duration is not None else ""

    with st.expander(f"{icon} {title}{dur_str}", expanded=False):
        if step_type == "query_rewrite":
            lang = data.get("detected_language", "")
            questions = data.get("sub_questions", [])
            if lang:
                st.caption(f"Detected language: `{lang}`")
            for i, q in enumerate(questions, 1):
                st.markdown(f"**Q{i}:** {q}")

        elif step_type == "tool_retrieval":
            tool = data.get("tool_name", "")
            chunks = data.get("chunks", [])
            if not chunks:
                st.caption(f"No results from `{tool}`")
            else:
                st.caption(f"Tool: `{tool}`")
                for chunk in chunks:
                    score = chunk.get("score")
                    score_str = f" (score: {score:.2f})" if isinstance(score, (int, float)) else ""
                    src = chunk.get("source", "unknown")
                    with st.expander(f"{src}{score_str}", expanded=False):
                        st.text(chunk.get("content", "")[:400])

        elif step_type == "cross_ref_detected":
            refs = data.get("references", [])
            if refs:
                for r in refs:
                    st.markdown(f"- {r}")
            else:
                st.caption("No cross-references detected.")

        elif step_type == "cross_ref_fetched":
            n = data.get("contexts_injected", 0)
            st.markdown(f"{n} context block(s) injected from cross-references.")

        elif step_type == "aggregation":
            n = data.get("sub_answer_count", 1)
            st.markdown(f"Merged {n} sub-answer(s) into the final response.")


def _render_workflow_steps_panel(steps: list, expanded: bool = False) -> None:
    """Render a collapsed/expandable panel of completed workflow steps."""
    if not steps:
        return
    with st.expander(f"Pipeline steps ({len(steps)} step(s))", expanded=expanded):
        for step in steps:
            _render_workflow_step(step)


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
            # Workflow steps from history (collapsed by default)
            _render_workflow_steps_panel(msg.get("workflow_steps", []), expanded=False)
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
            # Live indicator: shows the current pipeline step while streaming
            step_status = st.empty()
            answer_placeholder = st.empty()

            accumulated_tokens: List[str] = []
            completed_steps: List[dict] = []
            final_citations: List[dict] = []

            try:
                for chunk in _stream_chat(
                    prompt,
                    st.session_state.active_session_id,
                    st.session_state.target_language,
                ):
                    chunk_type = chunk.get("type")

                    if chunk_type == "workflow_step":
                        ws = chunk.get("workflow_step") or {}
                        icon = _STEP_ICONS.get(ws.get("step_type", ""), "•")
                        if ws.get("status") == "running":
                            step_status.markdown(f"_{icon} {ws.get('title', 'Working')}..._")
                        elif ws.get("status") == "done":
                            completed_steps.append(ws)
                            step_status.markdown(f"_{icon} {ws.get('title')}_ ✓")

                    elif chunk_type == "token":
                        accumulated_tokens.append(chunk.get("content", ""))
                        # Typing cursor while tokens arrive
                        answer_placeholder.markdown("".join(accumulated_tokens) + "▌")

                    elif chunk_type == "citation":
                        final_citations = chunk.get("citations") or []

                    elif chunk_type == "done":
                        step_status.empty()
                        break

                    elif chunk_type == "error":
                        step_status.empty()
                        st.error(chunk.get("error", "Unknown streaming error"))
                        break

                # Final render — replace cursor with clean answer
                full_answer = "".join(accumulated_tokens)
                answer_placeholder.markdown(full_answer)

                # Workflow steps panel (collapsed, user can expand)
                _render_workflow_steps_panel(completed_steps, expanded=False)

                if final_citations:
                    with st.expander(f"📚 {len(final_citations)} source(s)", expanded=False):
                        for cite in final_citations:
                            score = cite.get("score")
                            score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
                            st.markdown(
                                f"**{cite.get('filename', 'unknown')}** (score: {score_str})\n"
                                f"> {cite.get('excerpt', '')}"
                            )

                # Persist in session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "citations": final_citations,
                    "eval": None,       # eval scores arrive async post-response
                    "workflow_steps": completed_steps,
                })

                # Refresh session list (updated_at / preview)
                st.session_state.sessions = _fetch_sessions()

            except Exception as exc:
                step_status.empty()
                answer_placeholder.empty()
                st.error(f"Stream error: {exc}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_state()
    if not st.session_state.token:
        _login_page()
    else:
        _chat_page()


if __name__ == "__main__":
    main()
