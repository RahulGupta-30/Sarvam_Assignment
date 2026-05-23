
import asyncio


import streamlit as st
from dotenv import load_dotenv

from research_pipeline import run_research_pipeline
import search_tavily
import fetch_page
import build_context
import citation_validator
import agent
import session_store
import rolling_summary
import query_planner
from concurrent.futures import ThreadPoolExecutor


load_dotenv()
session_store.init_db()


def run_async(async_fn, *args, **kwargs):
    """
    Run an async function from Streamlit sync code.

    Important:
    Pass the async function itself, not an already-created coroutine.
    This avoids reusing an already-awaited coroutine.
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # Normal Streamlit case: no event loop is already running here
        return asyncio.run(async_fn(*args, **kwargs))

    # Fallback case: if an event loop is already running,
    # run the async function in a separate thread with its own loop.
    def runner():
        return asyncio.run(async_fn(*args, **kwargs))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner)
        return future.result()


# -------------------------------
# Streamlit UI
# -------------------------------
import streamlit as st
import session_store


# -------------------------------------------------
# Streamlit page config
# IMPORTANT: This should be the first Streamlit command
# -------------------------------------------------

st.set_page_config(
    page_title="Deep Research Agent",
    page_icon="🔎",
    layout="wide"
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background-color: #0f1117;
    }

    section[data-testid="stSidebar"] .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    section[data-testid="stSidebar"] .stSelectbox label {
        font-weight: 600;
    }

    div[data-baseweb="select"] > div {
        border-radius: 10px !important;
    }

    .block-container {
        padding-top: 1.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# -------------------------------------------------
# Database/session initialization
# -------------------------------------------------

session_store.init_db()

if "session_id" not in st.session_state:
    st.session_state.session_id = session_store.create_session()

if "messages" not in st.session_state:
    st.session_state.messages = session_store.get_messages(
        st.session_state.session_id
    )

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# -------------------------------------------------
# Header
# -------------------------------------------------

st.title("Deep Research Agent")
st.caption("Search → select sources → fetch pages → build context → answer with citations")


# -------------------------------------------------
# Sidebar
# -------------------------------------------------

with st.sidebar:
    st.markdown("## 🔎 Research Sessions")

    st.caption("Manage and reopen your previous research chats.")

    # ---- buttons row ----
    col1, col2 = st.columns(2)

    with col1:
        if st.button("➕ New", use_container_width=True):
            st.session_state.session_id = session_store.create_session()
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()

    with col2:
        if st.button("🗑 Delete", use_container_width=True):
            session_store.delete_session(st.session_state.session_id)
            st.session_state.session_id = session_store.create_session()
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()

    st.divider()

    sessions = session_store.list_sessions(limit=30)

    normal_sessions = []
    eval_sessions = []

    for s in sessions:
        title = (s.get("title") or "").strip()
        if title.startswith("eval::"):
            eval_sessions.append(s)
        else:
            normal_sessions.append(s)

    def format_session_label(session: dict) -> str:
        title = (session.get("title") or "Untitled Session").strip()

        if len(title) > 42:
            title = title[:42].rstrip() + "..."

        updated_at = session.get("updated_at", "")
        updated_display = updated_at.replace("T", " ")[:16] if updated_at else ""

        return f"{title} · {updated_display}"

    if normal_sessions:
        st.markdown("### Recent Sessions")

        session_map = {s["session_id"]: s for s in normal_sessions}
        session_ids = [s["session_id"] for s in normal_sessions]

        current_index = 0
        if st.session_state.session_id in session_ids:
            current_index = session_ids.index(st.session_state.session_id)

        selected_session_id = st.selectbox(
            "Choose a session",
            options=session_ids,
            index=current_index,
            format_func=lambda sid: format_session_label(session_map[sid]),
            label_visibility="collapsed"
        )

        if selected_session_id != st.session_state.session_id:
            st.session_state.session_id = selected_session_id
            st.session_state.messages = session_store.get_messages(selected_session_id)
            st.session_state.last_result = None
            st.rerun()
    else:
        st.info("No normal sessions yet.")

    if eval_sessions:
        with st.expander("Evaluation Sessions", expanded=False):
            eval_map = {s["session_id"]: s for s in eval_sessions}
            eval_ids = [s["session_id"] for s in eval_sessions]

            selected_eval_id = st.selectbox(
                "Choose an eval session",
                options=[""] + eval_ids,
                format_func=lambda sid: "Select..." if sid == "" else format_session_label(eval_map[sid]),
                label_visibility="collapsed",
                key="eval_session_selectbox"
            )

            if selected_eval_id and selected_eval_id != st.session_state.session_id:
                st.session_state.session_id = selected_eval_id
                st.session_state.messages = session_store.get_messages(selected_eval_id)
                st.session_state.last_result = None
                st.rerun()

    st.divider()

    st.markdown("### Current Session")
    current_session = session_store.get_session(st.session_state.session_id)

    if current_session:
        st.success(f"**{current_session.get('title', 'Untitled Session')}**")
        st.caption(f"Updated: {current_session.get('updated_at', '')}")

    st.divider()

    show_debug = st.checkbox("Show debug data", value=False)


# -------------------------------------------------
# Display existing chat messages
# -------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------------------------------
# Chat input
# -------------------------------------------------

query = st.chat_input("Ask a deep research question...")


if query:
    # Save user message to DB
    session_store.add_message(
        session_id=st.session_state.session_id,
        role="user",
        content=query
    )

    session_store.maybe_set_session_title_from_first_query(
    st.session_state.session_id,
    query
    )

    # Save user message to Streamlit state
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # Render user message
    with st.chat_message("user"):
        st.markdown(query)

    # Run assistant/research pipeline
    with st.chat_message("assistant"):
        status_box = st.empty()
        progress_bar = st.progress(0)

        progress_state = {"value": 0}

        def update_progress(message: str):
            message_lower = message.lower()

            if "planning" in message_lower or "query" in message_lower:
                next_progress = 10

            elif "search" in message_lower or "tavily" in message_lower:
                next_progress = 30

            elif "select" in message_lower or "url" in message_lower or "gemini" in message_lower:
                next_progress = 45

            elif "fetch" in message_lower or "page" in message_lower:
                next_progress = 60

            elif "context" in message_lower:
                next_progress = 75

            elif "generating" in message_lower or "answer" in message_lower:
                next_progress = 88

            elif "citation" in message_lower or "repair" in message_lower:
                next_progress = 95

            else:
                # Move forward slowly instead of jumping back to 10
                next_progress = progress_state["value"] + 3

            # Never allow progress to move backward
            next_progress = max(progress_state["value"], next_progress)

            # Keep below 100 until the pipeline is actually done
            next_progress = min(next_progress, 95)

            progress_state["value"] = next_progress

            status_box.info(message)
            progress_bar.progress(next_progress)

            

        try:
            result = run_async(
                run_research_pipeline,
                query=query,
                session_id=st.session_state.session_id,
                progress_callback=update_progress
                
            )

            progress_bar.progress(100)
            status_box.success("Research complete.")

            answer = result.get("final_answer", "")

            st.markdown(answer)

            # Save assistant message to DB
            session_store.add_message(
                session_id=st.session_state.session_id,
                role="assistant",
                content=answer
            )

            # Save assistant message to Streamlit state
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # Save latest research result for trace/debug UI
            st.session_state.last_result = result

        except Exception as e:
            progress_bar.empty()
            status_box.error("Something went wrong.")
            st.exception(e)


# -------------------------------------------------
# Research Trace
# -------------------------------------------------

if st.session_state.last_result:
    result = st.session_state.last_result

    st.divider()

    st.subheader("Research Trace")

    with st.expander("Query Plan", expanded=False):
        st.markdown("**Original query:**")
        st.write(result.get("query", ""))

        st.markdown("**Standalone research question:**")
        st.write(result.get("standalone_question", ""))

        st.markdown("**Search queries issued:**")
        search_queries = result.get("search_queries", [])

        if search_queries:
            for q in search_queries:
                st.markdown(f"- {q}")
        else:
            st.caption("No search queries found in result.")

        st.markdown("**History used:**")
        st.write(result.get("query_plan", {}).get("history_used", ""))

    with st.expander("Selected URLs", expanded=False):
        selected_urls = result.get("selected_urls", [])

        if selected_urls:
            for item in selected_urls:
                st.markdown(f"**{item.get('title', 'Untitled')}**")
                st.markdown(item.get("url", ""))
                st.caption(item.get("reason", ""))
        else:
            st.caption("No selected URLs found.")

    with st.expander("Fetched Pages", expanded=False):
        fetched_pages = result.get("fetched_pages", [])

        if fetched_pages:
            for idx, page in enumerate(fetched_pages):
                st.markdown(f"### {page.get('title', 'Untitled')}")
                st.markdown(page.get("url", ""))

                st.caption(
                    f"Domain: {page.get('domain', '')} | "
                    f"Retrieved: {page.get('retrieved_at', '')}"
                )

                preview = page.get("text", "")[:1000]

                st.text_area(
                    label="Preview",
                    value=preview,
                    height=160,
                    key=f"preview_{idx}_{page.get('url', '')}"
                )
        else:
            st.caption("No fetched pages found.")

    with st.expander("Used Context Sources", expanded=False):
        used_sources = result.get("used_sources", [])

        if used_sources:
            for source in used_sources:
                st.markdown(f"**{source.get('title', 'Untitled')}**")
                st.markdown(source.get("url", ""))

                st.caption(
                    f"Domain: {source.get('domain', '')} | "
                    f"Chunk: {source.get('chunk_index', '')} | "
                    f"Score: {source.get('score', '')}"
                )
        else:
            st.caption("No used context sources found.")

    with st.expander("Citation Validation", expanded=False):
        citation_check = result.get("citation_check", {})

        if citation_check.get("is_valid"):
            st.success("Citation validation passed.")
        else:
            st.warning("Citation validation found issues.")

        st.markdown("**Citation count:**")
        st.write(citation_check.get("citation_count", 0))

        st.markdown("**Valid citation count:**")
        st.write(citation_check.get("valid_citation_count", 0))

        st.markdown("**Citation score:**")
        st.write(citation_check.get("citation_score", 0))

        invalid_urls = citation_check.get("invalid_urls", [])

        if invalid_urls:
            st.markdown("**Invalid cited URLs:**")
            for url in invalid_urls:
                st.markdown(f"- {url}")

        warnings = citation_check.get("warnings", [])

        if warnings:
            st.markdown("**Warnings:**")
            for warning in warnings:
                st.markdown(f"- {warning}")

        if show_debug:
            st.json(citation_check)

    if show_debug:
        with st.expander("Raw Debug Data", expanded=False):
            st.json({
                "query": result.get("query"),
                "standalone_question": result.get("standalone_question"),
                "search_queries": result.get("search_queries"),
                "selected_urls": result.get("selected_urls"),
                "used_sources": result.get("used_sources"),
                "timestamp": result.get("timestamp"),
            })

        with st.expander("Built Context", expanded=False):
            st.text_area(
                label="Context sent to Gemini",
                value=result.get("context", ""),
                height=400
            )


# -------------------------------------------------
# Session Turn History
# -------------------------------------------------

st.divider()

with st.expander("Session Turn History", expanded=False):
    turns = session_store.get_turns(
        st.session_state.session_id,
        limit=10
    )

    if turns:
        for turn in turns:
            st.markdown(f"### {turn['query']}")
            st.caption(turn["created_at"])

            st.markdown("**Final answer preview:**")
            st.write((turn.get("final_answer") or "")[:1000])

            st.markdown("**Selected URLs:**")

            selected_urls = turn.get("selected_urls", [])

            if selected_urls:
                for item in selected_urls:
                    title = item.get("title", "Untitled")
                    url = item.get("url", "")

                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")
            else:
                st.caption("No selected URLs stored for this turn.")

            st.divider()
    else:
        st.caption("No turns stored for this session yet.")