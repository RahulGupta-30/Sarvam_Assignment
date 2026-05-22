import os
import json
import asyncio
import random
from typing import Iterable

from dotenv import load_dotenv
from google import genai
from google.genai import types

import fetch_page
import search_tavily
import build_context
import query_planner



load_dotenv(override=True)

API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

client = genai.Client(api_key=API_KEY)




def get_model_list(env_name: str, default_models: str) -> list[str]:
    """
    Reads a comma-separated model fallback list from .env.

    Example:
        GEMINI_FINAL_ANSWER_MODELS=gemini-3.5-flash,gemini-2.5-flash
    """
    raw_value = os.getenv(env_name, default_models)

    return [
        model.strip()
        for model in raw_value.split(",")
        if model.strip()
    ]


GEMINI_URL_SELECTOR_MODELS = get_model_list(
    env_name="GEMINI_URL_SELECTOR_MODELS",
    default_models="gemini-3.5-flash,gemini-2.5-flash",
)

GEMINI_FINAL_ANSWER_MODELS = get_model_list(
    env_name="GEMINI_FINAL_ANSWER_MODELS",
    default_models="gemini-3.5-flash,gemini-2.5-flash",
)

GEMINI_CITATION_REPAIR_MODELS = get_model_list(
    env_name="GEMINI_CITATION_REPAIR_MODELS",
    default_models="gemini-2.5-flash",
)

MAX_MODEL_RETRIES = int(os.getenv("MAX_MODEL_RETRIES", "4"))
MODEL_RETRY_BASE_SLEEP_SECONDS = float(os.getenv("MODEL_RETRY_BASE_SLEEP_SECONDS", "3"))
MODEL_RETRY_MAX_SLEEP_SECONDS = float(os.getenv("MODEL_RETRY_MAX_SLEEP_SECONDS", "45"))


def _error_text(error: Exception) -> str:
    """
    Converts an exception into uppercase text so marker checks are simple.
    """
    return str(error).upper()


def is_model_access_error(error: Exception) -> bool:
    """
    These errors usually do NOT get fixed by retrying the same model.

    Example:
    - 403 PERMISSION_DENIED
    - Your project has been denied access
    - model not found / unsupported model

    For these, we immediately try the next fallback model.
    """
    text = _error_text(error)

    access_markers = [
        "403",
        "PERMISSION_DENIED",
        "DENIED ACCESS",
        "HAS BEEN DENIED ACCESS",
        "MODEL_NOT_FOUND",
        "MODEL NOT FOUND",
        "NOT FOUND",
        "404",
    ]

    return any(marker in text for marker in access_markers)


def is_retryable_model_error(error: Exception) -> bool:
    """
    These errors are often temporary, so retrying helps.

    Examples:
    - 429 RESOURCE_EXHAUSTED
    - quota/rate limit pressure
    - 500/502/503/504 server errors
    - timeout/deadline errors
    """
    text = _error_text(error)

    retryable_markers = [
        "429",
        "RESOURCE_EXHAUSTED",
        "RATE_LIMIT",
        "RATE LIMIT",
        "QUOTA",
        "500",
        "INTERNAL",
        "502",
        "503",
        "UNAVAILABLE",
        "504",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "TIMED OUT",
    ]

    return any(marker in text for marker in retryable_markers)


async def generate_content_with_fallback(
    aclient,
    *,
    models: Iterable[str],
    contents: str,
    config: types.GenerateContentConfig,
    operation_name: str,
    max_retries: int = MAX_MODEL_RETRIES,
):
    """
    Calls Gemini with retry + fallback logic.

    Algorithm:
    1. Try each model in the configured fallback list.
    2. For retryable errors such as quota/rate/server failures, retry the same model.
    3. For access/model errors such as 403 or model not found, skip to the next model.
    4. If every model fails, raise the last error.
    """
    model_list = list(models)

    if not model_list:
        raise ValueError(f"{operation_name}: no Gemini models configured.")

    last_error = None

    for model in model_list:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"{operation_name}: model={model}, attempt={attempt}")

                response = await aclient.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                return response

            except Exception as error:
                last_error = error

                if is_model_access_error(error):
                    print(
                        f"{operation_name}: access/model error for {model}. "
                        f"Trying fallback model if available. Error: {error}"
                    )
                    break

                if not is_retryable_model_error(error):
                    print(
                        f"{operation_name}: non-retryable error for {model}. "
                        f"Trying fallback model if available. Error: {error}"
                    )
                    break

                if attempt >= max_retries:
                    print(
                        f"{operation_name}: exhausted retries for {model}. "
                        f"Trying fallback model if available. Error: {error}"
                    )
                    break

                exponential_delay = MODEL_RETRY_BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
                capped_delay = min(MODEL_RETRY_MAX_SLEEP_SECONDS, exponential_delay)
                jitter = random.uniform(0, 1.5)
                wait_time = capped_delay + jitter

                print(
                    f"{operation_name}: retryable error for {model}. "
                    f"Sleeping {wait_time:.1f}s before retry. Error: {error}"
                )

                await asyncio.sleep(wait_time)

    raise last_error


URL_SELECTOR_PROMPT = """
You are a URL selector agent.

You will receive:
1. A user query
2. A list of search results with titles, URLs, snippets, and scores.

Select the best URLs based on:
- relevance to the user query
- source quality
- recency if available
- source diversity
Select 6 to 8 high-quality URLs for deep research.
Return exactly this structure:
{
  "selected_urls": [
    {
      "title": "...",
      "url": "...",
      "reason": "..."
    }
  ]
}
"""


FINAL_ANSWER_PROMPT = """
You are a Deep Research Agent.

You will receive:
1. The user's question
2. Selected web context snippets
3. Source metadata including title, domain, URL, and retrieval time

Conflict and uncertainty rules:
- If sources disagree, explicitly say they disagree and cite both sources.
- If the evidence is weak, incomplete, outdated, or missing, say so clearly.
- If no direct conflict is found in the provided context, include:
  "I did not find a direct conflict among the selected sources."
- Do not overstate confidence.

Rules:
- Answer only using the provided web context.
- Do not invent facts.
- Every important factual claim must include a citation.
- Use this citation format: [Title — domain](URL)
- If the sources disagree, clearly mention the disagreement and cite both sources.
- If the evidence is weak or missing, say so honestly.
- Give a clear, useful, well-structured answer.
"""


async def repair_answer_with_citations(
    query: str,
    standalone_question: str,
    web_context: str,
    history_context: str,
    previous_answer: str,
    validation_feedback: str,
):
    prompt = f"""
                The previous answer failed citation validation.

                Original user question:
                {query}

                Standalone research question:
                {standalone_question}

                Session context:
                {history_context}

                Current web context:
                {web_context}

                Previous answer:
                {previous_answer}

                Citation validation feedback:
                {validation_feedback}

                Rewrite the answer so that:
                - Every claim-heavy paragraph has citations.
                - All citations use only URLs from the current web context.
                - No invented citation URLs are used.
                - Source conflicts are explicitly mentioned.
                - Uncertainty or weak evidence is clearly stated.
                - The answer follows the required structure.
                """

    async with client.aio as aclient:
        response = await generate_content_with_fallback(
            aclient,
            models=GEMINI_CITATION_REPAIR_MODELS,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=FINAL_ANSWER_PROMPT,
            ),
            operation_name="Citation repair",
        )

    return response.text


async def select_urls_with_gemini(query: str, search_results: list):
    """
    Uses Gemini to select the best URLs from Tavily search results.
    """

    prompt = f"""
            User query:
            {query}

            Search results:
            {json.dumps(search_results, indent=2)}

            Select the best URLs to fetch for deep research.
            """

    async with client.aio as aclient:
        response = await generate_content_with_fallback(
            aclient,
            models=GEMINI_URL_SELECTOR_MODELS,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=URL_SELECTOR_PROMPT,
                response_mime_type="application/json",
            ),
            operation_name="URL selection",
        )

    try:
        selected_data = json.loads(response.text)
    except json.JSONDecodeError:
        raise ValueError(f"Gemini did not return valid JSON:\n{response.text}")

    return selected_data.get("selected_urls", [])


async def generate_final_answer(
    query: str,
    standalone_question: str,
    web_context: str,
    history_context: str = "",
):
    prompt = f"""
            Original user question:
            {query}

            Standalone research question:
            {standalone_question}

            Session context:
            {history_context}

            Current web context:
            {web_context}

            Answer the original user question.

            Use session context only to understand references in the user's question.
            Use current web context as the evidence for factual claims.
            """

    async with client.aio as aclient:
        response = await generate_content_with_fallback(
            aclient,
            models=GEMINI_FINAL_ANSWER_MODELS,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=FINAL_ANSWER_PROMPT,
            ),
            operation_name="Final answer generation",
        )

    return response.text
