import re
from urllib.parse import urlsplit, urlunsplit
from typing import List, Dict, Any


URL_RE = re.compile(r"https?://[^\s\)\]\}\"']+")


def normalize_url(url: str) -> str:
    """
    Normalizes URLs so small differences do not break validation.
    Example:
    https://www.example.com/page#section
    becomes:
    https://example.com/page
    """
    if not url:
        return ""

    url = url.strip().rstrip(".,;:")

    try:
        parsed = urlsplit(url)

        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        if netloc.startswith("www."):
            netloc = netloc[4:]

        path = parsed.path.rstrip("/")
        if not path:
            path = "/"

        return urlunsplit((scheme, netloc, path, parsed.query, ""))
    except Exception:
        return url.strip()


def extract_urls_from_answer(answer: str) -> List[str]:
    """
    Extracts all URLs from the generated answer.
    Works for markdown links like:
    [Title — domain](https://example.com/page)
    """
    if not answer:
        return []

    urls = URL_RE.findall(answer)

    cleaned_urls = []

    for url in urls:
        cleaned = url.strip().rstrip(".,;:")
        cleaned_urls.append(cleaned)

    return list(dict.fromkeys(cleaned_urls))


def validate_citations(
    final_answer: str,
    used_sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Validates citation integrity.

    What this checks:
    - Does the answer contain citations?
    - Are cited URLs from the selected context sources?
    - Did the model invent any citation URLs?

    What this does NOT fully check:
    - Whether every sentence is semantically supported by the cited source.
      That requires a stronger LLM-judge or human evaluation.
    """

    cited_urls = extract_urls_from_answer(final_answer)

    source_urls = []

    for source in used_sources:
        url = source.get("url", "")
        if url:
            source_urls.append(url)

    normalized_source_map = {
        normalize_url(url): url
        for url in source_urls
        if url
    }

    normalized_cited_map = {
        normalize_url(url): url
        for url in cited_urls
        if url
    }

    valid_urls = []
    invalid_urls = []

    for normalized_url, original_url in normalized_cited_map.items():
        if normalized_url in normalized_source_map:
            valid_urls.append(original_url)
        else:
            invalid_urls.append(original_url)

    warnings = []

    if not cited_urls:
        warnings.append("No citations found in final answer.")

    if invalid_urls:
        warnings.append("Some cited URLs were not present in the selected context sources.")

    if used_sources and not valid_urls:
        warnings.append("The answer used sources, but no valid source URL citations were found.")

    citation_count = len(cited_urls)
    valid_citation_count = len(valid_urls)

    citation_score = 0.0

    if citation_count > 0:
        citation_score = valid_citation_count / citation_count

    is_valid = (
        citation_count > 0
        and len(invalid_urls) == 0
        and valid_citation_count > 0
    )

    return {
        "is_valid": is_valid,
        "citation_count": citation_count,
        "valid_citation_count": valid_citation_count,
        "citation_score": citation_score,
        "cited_urls": cited_urls,
        "valid_urls": valid_urls,
        "invalid_urls": invalid_urls,
        "source_urls": source_urls,
        "warnings": warnings,
    }


def format_validation_feedback(check: Dict[str, Any]) -> str:
    """
    Converts validation result into text that can be sent back to Gemini
    if we want it to repair the answer.
    """

    lines = []

    lines.append(f"Citation valid: {check.get('is_valid')}")
    lines.append(f"Citation count: {check.get('citation_count')}")
    lines.append(f"Valid citation count: {check.get('valid_citation_count')}")
    lines.append(f"Citation score: {check.get('citation_score')}")

    invalid_urls = check.get("invalid_urls", [])
    warnings = check.get("warnings", [])

    if invalid_urls:
        lines.append("\nInvalid cited URLs:")
        for url in invalid_urls:
            lines.append(f"- {url}")

    if warnings:
        lines.append("\nWarnings:")
        for warning in warnings:
            lines.append(f"- {warning}")

    return "\n".join(lines)