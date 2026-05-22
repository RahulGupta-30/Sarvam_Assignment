import os
import json
import asyncio
from dotenv import load_dotenv
from google import genai
from google.genai import types
import fetch_page
import search_tavily
import build_context
import query_planner

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

client = genai.Client(api_key=API_KEY)

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
        response = await aclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                system_instruction=FINAL_ANSWER_PROMPT,
            ),
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
        response = await aclient.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=URL_SELECTOR_PROMPT,
                response_mime_type="application/json",
            ),
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
    history_context: str = ""
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
        response = await aclient.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=FINAL_ANSWER_PROMPT,
            ),
        )

    return response.text




# async def main():
#     async with client.aio as aclient:
#         while True:
#             query = input("Enter Your Question: ").strip()

#             if query.lower() == "exit":
#                 print("Exiting the program.")
#                 break

#             print("Searching Tavily...")

#             response = await search_tavily.search_page(query)

#             prompt = f"""
#                         User query: {query}

#                         Search results: {json.dumps(response, indent=2)}

#                         Select the best URLs to fetch for deep research.
#                         """

#             print("Selecting best URLs with Gemini...")

#             response_2 = await aclient.models.generate_content(
#                 model="gemini-3.5-flash",
#                 contents=prompt,
#                 config=types.GenerateContentConfig(
#                     temperature=0.2,
#                     system_instruction=SYSTEM_PROMPT_2,
#                     response_mime_type="application/json",
#                 ),
#             )

            


#             selected_data = json.loads(response_2.text)

#             selected_urls = selected_data.get("selected_urls", [])

#             print("Fetching selected pages concurrently...")

#             tasks = [
#                 fetch_page.extract_page(item["url"], query=query)
#                 for item in selected_urls
#                 if item.get("url")
#             ]

#             page_batches = await asyncio.gather(
#                 *tasks,
#                 return_exceptions=True
#             )

#             fetched_pages = []

#             for batch in page_batches:
#                 if isinstance(batch, Exception):
#                     print(f"Error fetching page: {batch}")
#                     continue

#                 fetched_pages.extend(batch)


#             context, used_sources = build_context.build_context(
#                 query=query,
#                 pages=fetched_pages,
#                 max_chars=12000,
#                 chunk_size=1800,
#                 max_chunks_per_domain=2
#             )


#             print("Generating final answer with citations...")

#             final_prompt = f"""
#             User question:
#             {query}

#             Web context:
#             {context}

#             Now answer the user's question using only the provided context.
#             """

#             final_response = await aclient.models.generate_content(
#                 model="gemini-3.5-flash",
#                 contents=final_prompt,
#                 config=types.GenerateContentConfig(
#                     temperature=0.2,
#                     system_instruction=FINAL_ANSWER_PROMPT,
#                 ),
#             )

#             print("\nFinal Answer:")
#             print(final_response.text)
            


# if __name__ == "__main__":
#     asyncio.run(main())