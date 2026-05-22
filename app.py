import os
import json
import asyncio
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from research_pipeline import run_research_pipeline
import search_tavily
import fetch_page
import build_context
import citation_validator
import agent
import session_store
import rolling_summary
import query_planner


load_dotenv()
session_store.init_db()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")


client = genai.Client(api_key=GEMINI_API_KEY)






def run_async(coro):
    """
    Helper to run async code from Streamlit button clicks.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


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
    st.header("Sessions")

    st.caption(f"Current session: `{st.session_state.session_id[:8]}`")

    sessions = session_store.list_sessions(limit=20)

    session_options = {
        f"{s['title']} | {s['updated_at']}": s["session_id"]
        for s in sessions
    }

    if session_options:
        selected_label = st.selectbox(
            "Load session",
            options=list(session_options.keys())
        )

        if st.button("Load selected session"):
            st.session_state.session_id = session_options[selected_label]
            st.session_state.messages = session_store.get_messages(
                st.session_state.session_id
            )
            st.session_state.last_result = None
            st.rerun()

    if st.button("New session"):
        st.session_state.session_id = session_store.create_session()
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    if st.button("Delete current session"):
        session_store.delete_session(st.session_state.session_id)
        st.session_state.session_id = session_store.create_session()
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    st.header("Settings")

    with st.expander("Rolling Summary", expanded=False):
        summary = session_store.get_rolling_summary(st.session_state.session_id)

    if summary:
        st.write(summary)
    else:
        st.caption("No rolling summary yet. It will be created after the conversation gets longer.")

    st.info(
        "Current pipeline uses Tavily for search/fetch and Gemini for "
        "query planning, URL selection, and final answer generation."
    )

    show_debug = st.checkbox("Show debug data", value=False)

    st.divider()

    if st.button("Clear conversation / start fresh"):
        st.session_state.session_id = session_store.create_session()
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


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

        progress_steps = {
                    "Planning research query from session history.": 10,
                    "Searching the web with Tavily.": 25,
                    "Selecting the best URLs with Gemini.": 45,
                    "Fetching selected pages concurrently.": 60,
                    "Building research context.": 75,
                    "Generating final answer with citations.": 85,
                    "Repairing answer citations.": 95,
                }

        def update_progress(message: str):
            status_box.info(message)
            progress_bar.progress(progress_steps.get(message, 10))

        try:
            result = run_async(
                run_research_pipeline(
                    query=query,
                    session_id=st.session_state.session_id,
                    progress_callback=update_progress
                )
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