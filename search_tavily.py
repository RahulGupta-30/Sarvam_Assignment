import os
import asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


async def search_page(query: str):
    response_tavily = await asyncio.to_thread(
        tavily_client.search,
        query=query,
        search_depth="advanced",
        max_results=8
    )

    results = []

    for r in response_tavily.get("results", []):
        url = r.get("url", "")

        results.append({
            "title": r.get("title", ""),
            "url": url,
            "domain": get_domain(url),
            "score": r.get("score", 0),
            "snippet": r.get("content", "")
        })

    return results