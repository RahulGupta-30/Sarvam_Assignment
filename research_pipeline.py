import os
import json
import asyncio
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

import search_tavily
import fetch_page
import build_context
import citation_validator
import agent
import session_store
import rolling_summary
import query_planner
import llm_provider

load_dotenv()
session_store.init_db()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")


client = genai.Client(api_key=GEMINI_API_KEY)


async def run_research_pipeline(query: str, session_id: str, progress_callback=None):
    if progress_callback:
        progress_callback("Planning research query from session history...")

    previous_turns = session_store.get_turns(session_id, limit=3)

    local_client = genai.Client(api_key=GEMINI_API_KEY)

    async with local_client.aio as aclient:
        await rolling_summary.maybe_update_rolling_summary(
            aclient=aclient,
            session_id=session_id
        )

        current_rolling_summary = session_store.get_rolling_summary(session_id)

        query_plan = await query_planner.plan_research_query(
            aclient=aclient,
            current_query=query,
            previous_turns=previous_turns,
            rolling_summary=current_rolling_summary
        )

    standalone_question = query_plan["standalone_question"]
    search_query = query_plan["search_query"]

    if progress_callback:
        progress_callback("Searching the web with Tavily...")

    search_results = await search_tavily.search_page(search_query)

    if progress_callback:
        progress_callback("Selecting the best URLs with Gemini...")

    selected_urls = await agent.select_urls_with_gemini(
        query=standalone_question,
        search_results=search_results
    )

    if progress_callback:
        progress_callback("Fetching selected pages concurrently...")

    tasks = [
        fetch_page.extract_page(item["url"], query=standalone_question)
        for item in selected_urls
        if item.get("url")
    ]

    page_batches = await asyncio.gather(*tasks, return_exceptions=True)

    fetched_pages = []

    for batch in page_batches:
        if isinstance(batch, Exception):
            continue

        fetched_pages.extend(batch)

    if progress_callback:
        progress_callback("Building research context...")

    context, used_sources = build_context.build_context(
        query=standalone_question,
        pages=fetched_pages,
        max_chars=22000,
        chunk_size=2000,
        max_chunks_per_domain=2
    )

    history_context = session_store.build_history_context(session_id)

    if progress_callback:
        progress_callback("Generating final answer with citations...")

    final_answer = await agent.generate_final_answer(
    query=query,
    standalone_question=standalone_question,
    web_context=context,
    history_context=history_context
)

    citation_check = citation_validator.validate_citations(
        final_answer=final_answer,
        used_sources=used_sources
    )

    if not citation_check["is_valid"]:
        if progress_callback:
            progress_callback("Repairing answer citations...")

        validation_feedback = citation_validator.format_validation_feedback(citation_check)

        final_answer = await agent.repair_answer_with_citations(
            query=query,
            standalone_question=standalone_question,
            web_context=context,
            history_context=history_context,
            previous_answer=final_answer,
            validation_feedback=validation_feedback,
        )

        citation_check = citation_validator.validate_citations(
            final_answer=final_answer,
            used_sources=used_sources
        )

    session_store.add_turn(
        session_id=session_id,
        query=query,
        search_results=search_results,
        selected_urls=selected_urls,
        fetched_pages=fetched_pages,
        used_sources=used_sources,
        final_answer=final_answer
    )

    return {
        "query": query,
        "query_plan": query_plan,
        "standalone_question": standalone_question,
        "search_query": search_query,
        "search_results": search_results,
        "selected_urls": selected_urls,
        "citation_check": citation_check,
        "fetched_pages": fetched_pages,
        "used_sources": used_sources,
        "context": context,
        "final_answer": final_answer
    }