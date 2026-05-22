import os
from dotenv import load_dotenv
from urllib.parse import urlparse
from datetime import datetime
from tavily import AsyncTavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)


async def extract_page(url: str, query: str = ""):
    response_tavily = await tavily_client.extract(
        urls=[url],
        extract_depth="basic",
        format="text",
        query=query if query else None,
        chunks_per_source=3 if query else None,
    )

    pages = []

    for r in response_tavily.get("results", []):
        page_url = r.get("url", url)
        raw_text = r.get("raw_content", "")

        if not raw_text or not raw_text.strip():
            continue

        pages.append({
            "url": page_url,
            "title": r.get("title", ""),
            "text": raw_text,
            "retrieved_at": datetime.now().isoformat(),
            "domain": urlparse(page_url).netloc.replace("www.", "")
        })

    return pages