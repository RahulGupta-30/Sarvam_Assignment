import os
import re
import json
import asyncio
from pathlib import Path
from datetime import datetime
from statistics import mean

from dotenv import load_dotenv

import session_store
import citation_validator
from research_pipeline import run_research_pipeline


load_dotenv()

DATASET_PATH = "eval_dataset.json"
OUTPUT_DIR = Path("evaluation_runs")

UNCERTAINTY_TERMS = [
    "uncertain",
    "uncertainty",
    "could not verify",
    "cannot verify",
    "not enough evidence",
    "limited evidence",
    "weak evidence",
    "incomplete",
    "missing",
    "not publicly available",
    "classified",
    "unverified"
]

CONFLICT_TERMS = [
    "conflict",
    "conflicting",
    "disagree",
    "disagreement",
    "differ",
    "different sources",
    "sources vary",
    "varies",
    "both sources",
    "in contrast"
]


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def contains_any(text: str, terms: list[str]) -> bool:
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def score_required_terms(answer: str, required_terms: list[str]) -> float:
    if not required_terms:
        return 1.0

    answer_lower = answer.lower()

    matched = 0

    for term in required_terms:
        if term.lower() in answer_lower:
            matched += 1

    return matched / len(required_terms)


def get_unique_domains(used_sources: list[dict]) -> set[str]:
    domains = set()

    for source in used_sources:
        domain = source.get("domain", "")
        if domain:
            domains.add(domain)

    return domains


def score_turn(result: dict, turn_spec: dict) -> dict:
    answer = result.get("final_answer", "") or ""
    used_sources = result.get("used_sources", []) or []

    citation_check = result.get("citation_check")

    if not citation_check:
        citation_check = citation_validator.validate_citations(
            final_answer=answer,
            used_sources=used_sources
        )

    citation_count = citation_check.get("citation_count", 0)
    citation_score = citation_check.get("citation_score", 0.0)

    has_citations_score = 1.0 if citation_count > 0 else 0.0
    citation_integrity_score = citation_score

    unique_domains = get_unique_domains(used_sources)
    source_diversity_score = min(len(unique_domains) / 3, 1.0)

    used_context_score = min(len(used_sources) / 3, 1.0)

    answer_length_score = 1.0 if len(answer.strip()) >= 500 else 0.5 if answer.strip() else 0.0

    required_terms_score = score_required_terms(
        answer=answer,
        required_terms=turn_spec.get("required_terms", [])
    )

    if turn_spec.get("expect_uncertainty", False):
        uncertainty_score = 1.0 if contains_any(answer, UNCERTAINTY_TERMS) else 0.0
    else:
        uncertainty_score = 1.0

    if turn_spec.get("expect_conflict", False):
        conflict_score = 1.0 if contains_any(answer, CONFLICT_TERMS) else 0.0
    else:
        conflict_score = 1.0

    query_plan = result.get("query_plan", {}) or {}
    standalone_question = result.get("standalone_question", "") or ""
    original_query = result.get("query", "") or ""

    if turn_spec.get("expect_follow_up", False):
        planner_followup_score = 1.0 if query_plan.get("is_follow_up") else 0.0

        standalone_changed_score = (
            1.0
            if standalone_question.strip().lower() != original_query.strip().lower()
            else 0.0
        )

        followup_score = (planner_followup_score + standalone_changed_score) / 2
    else:
        followup_score = 1.0

    search_success_score = 1.0 if result.get("search_results") else 0.0
    fetch_success_score = 1.0 if result.get("fetched_pages") else 0.0

    overall_score = (
        0.20 * citation_integrity_score
        + 0.10 * has_citations_score
        + 0.15 * source_diversity_score
        + 0.10 * used_context_score
        + 0.10 * answer_length_score
        + 0.10 * required_terms_score
        + 0.10 * uncertainty_score
        + 0.10 * conflict_score
        + 0.10 * followup_score
        + 0.025 * search_success_score
        + 0.025 * fetch_success_score
    )

    return {
        "overall_score": round(overall_score, 3),
        "citation_integrity_score": round(citation_integrity_score, 3),
        "has_citations_score": has_citations_score,
        "source_diversity_score": round(source_diversity_score, 3),
        "used_context_score": round(used_context_score, 3),
        "answer_length_score": answer_length_score,
        "required_terms_score": round(required_terms_score, 3),
        "uncertainty_score": uncertainty_score,
        "conflict_score": conflict_score,
        "followup_score": round(followup_score, 3),
        "search_success_score": search_success_score,
        "fetch_success_score": fetch_success_score,
        "citation_check": citation_check,
        "unique_domains": sorted(list(unique_domains)),
        "answer_length": len(answer),
        "used_sources_count": len(used_sources),
        "search_results_count": len(result.get("search_results", []) or []),
        "fetched_pages_count": len(result.get("fetched_pages", []) or []),
    }


def compact_result_for_json(result: dict) -> dict:
    """
    Keeps evaluation output useful without storing massive full page text.
    """

    fetched_pages_light = []

    for page in result.get("fetched_pages", []):
        fetched_pages_light.append({
            "title": page.get("title", ""),
            "url": page.get("url", ""),
            "domain": page.get("domain", ""),
            "retrieved_at": page.get("retrieved_at", ""),
            "text_preview": (page.get("text", "") or "")[:800]
        })

    return {
        "query": result.get("query"),
        "query_plan": result.get("query_plan"),
        "standalone_question": result.get("standalone_question"),
        "search_queries": result.get("search_queries"),
        "selected_urls": result.get("selected_urls"),
        "used_sources": result.get("used_sources"),
        "fetched_pages": fetched_pages_light,
        "final_answer": result.get("final_answer"),
        "citation_check": result.get("citation_check"),
        "timestamp": result.get("timestamp")
    }


async def evaluate_case(case: dict) -> dict:
    session_id = session_store.create_session(
        title=f"eval::{case['id']}"
    )

    case_output = {
        "case_id": case["id"],
        "type": case.get("type", ""),
        "description": case.get("description", ""),
        "session_id": session_id,
        "turns": [],
    }

    for idx, turn_spec in enumerate(case.get("turns", []), start=1):
        query = turn_spec["query"]

        print(f"\n[{case['id']}] Turn {idx}: {query}")

        session_store.add_message(
            session_id=session_id,
            role="user",
            content=query
        )

        try:
            result = await run_research_pipeline(
                query=query,
                session_id=session_id,
                progress_callback=lambda msg: print(f"  - {msg}")
            )

            final_answer = result.get("final_answer", "")

            session_store.add_message(
                session_id=session_id,
                role="assistant",
                content=final_answer
            )

            scores = score_turn(result, turn_spec)

            turn_output = {
                "turn_index": idx,
                "query": query,
                "expectations": turn_spec,
                "scores": scores,
                "result": compact_result_for_json(result),
                "error": None
            }

            print(f"  Score: {scores['overall_score']}")

        except Exception as e:
            turn_output = {
                "turn_index": idx,
                "query": query,
                "expectations": turn_spec,
                "scores": {
                    "overall_score": 0.0
                },
                "result": None,
                "error": str(e)
            }

            print(f"  ERROR: {e}")

        case_output["turns"].append(turn_output)

    turn_scores = [
        turn["scores"].get("overall_score", 0.0)
        for turn in case_output["turns"]
    ]

    case_output["average_score"] = round(mean(turn_scores), 3) if turn_scores else 0.0

    return case_output


def write_markdown_report(results: dict, path: Path):
    lines = []

    lines.append("# Deep Research Agent Evaluation Report\n")
    lines.append(f"Generated at: `{results['generated_at']}`\n")

    lines.append("## Summary\n")
    lines.append(f"- Cases evaluated: **{results['case_count']}**")
    lines.append(f"- Turns evaluated: **{results['turn_count']}**")
    lines.append(f"- Average score: **{results['average_score']}**\n")

    lines.append("## Metrics Used\n")
    lines.append("- **Citation integrity:** cited URLs must come from selected context sources.")
    lines.append("- **Citation presence:** final answer should include source citations.")
    lines.append("- **Source diversity:** answer should use multiple domains where possible.")
    lines.append("- **Used context count:** selected context snippets should be present.")
    lines.append("- **Required term coverage:** answer should include key expected entities/concepts.")
    lines.append("- **Uncertainty handling:** expected uncertainty cases should clearly state limits.")
    lines.append("- **Conflict handling:** expected conflict cases should mention disagreement or variation.")
    lines.append("- **Follow-up handling:** multi-turn questions should be rewritten using session history.")
    lines.append("- **Search/fetch success:** agent should retrieve and fetch web sources.\n")

    lines.append("## Case Results\n")

    for case in results["cases"]:
        lines.append(f"### {case['case_id']} — {case['type']}")
        lines.append(f"{case.get('description', '')}\n")
        lines.append(f"Average score: **{case['average_score']}**\n")

        for turn in case["turns"]:
            lines.append(f"#### Turn {turn['turn_index']}")
            lines.append(f"Query: `{turn['query']}`")

            if turn.get("error"):
                lines.append(f"Error: `{turn['error']}`\n")
                continue

            scores = turn["scores"]
            result = turn["result"]

            lines.append(f"- Overall score: **{scores['overall_score']}**")
            lines.append(f"- Citation integrity: {scores['citation_integrity_score']}")
            lines.append(f"- Source diversity: {scores['source_diversity_score']}")
            lines.append(f"- Uncertainty score: {scores['uncertainty_score']}")
            lines.append(f"- Conflict score: {scores['conflict_score']}")
            lines.append(f"- Follow-up score: {scores['followup_score']}")
            lines.append(f"- Search results: {scores['search_results_count']}")
            lines.append(f"- Fetched pages: {scores['fetched_pages_count']}")
            lines.append(f"- Used sources: {scores['used_sources_count']}")

            citation_check = scores.get("citation_check", {})
            invalid_urls = citation_check.get("invalid_urls", [])

            if invalid_urls:
                lines.append("- Invalid citation URLs:")
                for url in invalid_urls:
                    lines.append(f"  - {url}")

            lines.append("\n**Standalone question:**")
            lines.append(result.get("standalone_question", "") or "")

            lines.append("\n**Search queries:**")
            for q in result.get("search_queries", []) or []:
                lines.append(f"- {q}")

            lines.append("\n**Answer preview:**")
            answer = result.get("final_answer", "") or ""
            lines.append(answer[:1200].replace("\n", "\n\n"))

            lines.append("\n")

    path.write_text("\n".join(lines), encoding="utf-8")


async def main():
    session_store.init_db()

    dataset = load_dataset(DATASET_PATH)

    OUTPUT_DIR.mkdir(exist_ok=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    safe_timestamp = generated_at.replace(":", "-")

    cases = []

    for case in dataset:
        case_result = await evaluate_case(case)
        cases.append(case_result)

    all_turn_scores = []

    for case in cases:
        for turn in case["turns"]:
            all_turn_scores.append(turn["scores"].get("overall_score", 0.0))

    output = {
        "generated_at": generated_at,
        "case_count": len(cases),
        "turn_count": len(all_turn_scores),
        "average_score": round(mean(all_turn_scores), 3) if all_turn_scores else 0.0,
        "cases": cases
    }

    json_path = OUTPUT_DIR / f"eval_results_{safe_timestamp}.json"
    md_path = OUTPUT_DIR / f"eval_report_{safe_timestamp}.md"

    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    write_markdown_report(output, md_path)

    print("\nEvaluation complete.")
    print(f"JSON results: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Average score: {output['average_score']}")


if __name__ == "__main__":
    asyncio.run(main())