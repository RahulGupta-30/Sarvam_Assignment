import re
from typing import List, Dict, Tuple


def clean_text(text: str) -> str:
    """
    Removes extra whitespace so the context is cleaner for the LLM.
    """
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> set:
    """
    Converts text into lowercase words.
    Used for simple keyword-based relevance scoring.
    """
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> List[str]:
    """
    Splits a long page into smaller overlapping chunks.

    overlap helps because useful information may be split across two chunks.
    """
    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def score_chunk(query: str, chunk: str) -> int:
    """
    Simple relevance score:
    counts how many query terms appear in the chunk.
    """
    query_terms = tokenize(query)
    chunk_terms = tokenize(chunk)

    return len(query_terms.intersection(chunk_terms))


def build_context(
    query: str,
    pages: List[Dict],
    max_chars: int = 12000,
    chunk_size: int = 1800,
    max_chunks_per_domain: int = 2
) -> Tuple[str, List[Dict]]:
    """
    Builds LLM context from fetched pages.

    It:
    1. Splits pages into chunks
    2. Scores chunks against the user query
    3. Keeps source metadata for citations
    4. Limits total context size
    5. Encourages source diversity
    """

    candidates = []

    for page in pages:
        title = page.get("title", "Untitled source")
        url = page.get("url", "")
        domain = page.get("domain", "")
        retrieved_at = page.get("retrieved_at", "")
        text = page.get("text", "")

        chunks = chunk_text(text, chunk_size=chunk_size)

        for index, chunk in enumerate(chunks):
            score = score_chunk(query, chunk)

            candidates.append({
                "title": title,
                "url": url,
                "domain": domain,
                "retrieved_at": retrieved_at,
                "chunk_index": index,
                "text": chunk,
                "score": score
            })

    # Highest relevance first
    candidates.sort(key=lambda x: x["score"], reverse=True)

    context_parts = []
    used_sources = []
    domain_counts = {}
    total_chars = 0
    source_number = 1

    for item in candidates:
        domain = item["domain"]

        if domain_counts.get(domain, 0) >= max_chunks_per_domain:
            continue

        source_block = f"""
                        [SOURCE {source_number}]
                        Title: {item["title"]}
                        Domain: {item["domain"]}
                        URL: {item["url"]}
                        Retrieved At: {item["retrieved_at"]}
                        Chunk Index: {item["chunk_index"]}

                        Content:
                        {item["text"]}
                        """.strip()

        if total_chars + len(source_block) > max_chars:
            break

        context_parts.append(source_block)
        used_sources.append(item)

        total_chars += len(source_block)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        source_number += 1

    context = "\n\n---\n\n".join(context_parts)

    return context, used_sources