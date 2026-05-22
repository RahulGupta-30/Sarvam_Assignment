import os
import json
import asyncio
import random
from pathlib import Path
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

import session_store
import citation_validator
from research_pipeline import run_research_pipeline


# override=True avoids a stale Windows environment variable silently beating .env
load_dotenv(override=True)

DATASET_PATH = os.getenv("EVAL_DATASET_PATH", "eval_dataset.json")
OUTPUT_DIR = Path(os.getenv("EVAL_OUTPUT_DIR", "evaluation_runs"))


def get_model_list(env_name: str, default_models: str) -> list[str]:
    """
    Reads a comma-separated Groq judge model fallback list from .env.

    Preferred:
        EVAL_JUDGE_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant

    Backward compatibility:
    - If EVAL_JUDGE_MODELS is not set, this checks EVAL_JUDGE_MODEL.
    - If EVAL_JUDGE_MODEL is also not set, this checks GROQ_MODEL.
    """
    raw_value = os.getenv(env_name)

    if not raw_value and env_name == "EVAL_JUDGE_MODELS":
        raw_value = os.getenv("EVAL_JUDGE_MODEL")

    if not raw_value and env_name == "EVAL_JUDGE_MODELS":
        raw_value = os.getenv("GROQ_MODEL")

    if not raw_value:
        raw_value = default_models

    return [
        model.strip()
        for model in raw_value.split(",")
        if model.strip()
    ]


JUDGE_PROVIDER = "groq"

JUDGE_MODELS = get_model_list(
    "EVAL_JUDGE_MODELS",
    "llama-3.3-70b-versatile",
)

JUDGE_MODEL = ",".join(JUDGE_MODELS)

MAX_JUDGE_CONTEXT_CHARS = int(os.getenv("MAX_JUDGE_CONTEXT_CHARS", "30000"))
MAX_ANSWER_CHARS = int(os.getenv("MAX_ANSWER_CHARS", "12000"))
EVAL_JUDGE_MAX_TOKENS = int(os.getenv("EVAL_JUDGE_MAX_TOKENS", "4000"))

MAX_JUDGE_RETRIES = int(os.getenv("MAX_JUDGE_RETRIES", "3"))
EVAL_RETRY_BASE_SLEEP_SECONDS = float(os.getenv("EVAL_RETRY_BASE_SLEEP_SECONDS", "8"))
EVAL_RETRY_MAX_SLEEP_SECONDS = float(os.getenv("EVAL_RETRY_MAX_SLEEP_SECONDS", "90"))

EVAL_SLEEP_AFTER_PIPELINE_SECONDS = float(os.getenv("EVAL_SLEEP_AFTER_PIPELINE_SECONDS", "3"))
EVAL_SLEEP_BEFORE_JUDGE_CALL_SECONDS = float(os.getenv("EVAL_SLEEP_BEFORE_JUDGE_CALL_SECONDS", "4"))
EVAL_SLEEP_BETWEEN_TURNS_SECONDS = float(os.getenv("EVAL_SLEEP_BETWEEN_TURNS_SECONDS", "6"))
EVAL_SLEEP_BETWEEN_CASES_SECONDS = float(os.getenv("EVAL_SLEEP_BETWEEN_CASES_SECONDS", "10"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

judge_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
)


JUDGE_SYSTEM_PROMPT = """
You are a strict evaluation judge for a citation-grounded web research agent.

You will receive:
1. The original user question.
2. The standalone research question produced by the agent.
3. Optional dataset expectations such as reference_answer, key_facts, and expected_behavior.
4. The fetched web context available to the agent.
5. The final answer written by the agent.

Your job is to score reading-comprehension-based quality. Do not reward keyword matching.
Only use the provided web context and optional dataset expectations. Do not rely on outside knowledge.

Score each dimension from 1 to 5:
1 = poor or missing
2 = weak
3 = acceptable but incomplete
4 = good
5 = excellent

Dimensions:
- factual_grounding: Are important factual claims supported by citations and by the cited/context sources?
- conflict_handling: Did the answer identify real source disagreement when present, avoid inventing conflicts when absent, and explain tension accurately?
- uncertainty_calibration: Is the confidence/uncertainty level appropriate given the evidence quality, recency, and completeness?
- answer_completeness: Does the answer address every part of the user question and any listed key_facts/reference_answer expectations?
- follow_up_resolution_quality: If this is a follow-up, did the agent resolve pronouns/references correctly? If it is not a follow-up, score 5 and say not applicable.

Return valid JSON only. Do not wrap it in markdown.
Use exactly this structure:
{
  "scores": {
    "factual_grounding": {"score": 1, "justification": "..."},
    "conflict_handling": {"score": 1, "justification": "..."},
    "uncertainty_calibration": {"score": 1, "justification": "..."},
    "answer_completeness": {"score": 1, "justification": "..."},
    "follow_up_resolution_quality": {"score": 1, "justification": "..."}
  },
  "key_fact_coverage": [
    {"fact": "...", "covered": true, "justification": "..."}
  ],
  "notes": "brief overall evaluation notes"
}
""".strip()


HALLUCINATION_SYSTEM_PROMPT = """
You are a hallucination probe for a citation-grounded web research agent.

You will receive:
1. The user question.
2. The fetched web context available to the agent.
3. The final answer.

Task:
- Extract the important factual claims from the final answer.
- For each claim, decide whether it is traceable to at least one fetched source chunk.
- Ignore purely stylistic text, section headings, advice about uncertainty, and generic transition sentences.
- Be strict: if the context does not clearly support a claim, mark it as unsupported or unclear.
- Do not use outside knowledge.

Return valid JSON only. Do not wrap it in markdown.
Use exactly this structure:
{
  "claims": [
    {
      "claim": "...",
      "traceability": "supported | unsupported | unclear",
      "supporting_sources": ["SOURCE 1"],
      "explanation": "..."
    }
  ],
  "summary": {
    "supported_claim_count": 0,
    "unsupported_claim_count": 0,
    "unclear_claim_count": 0,
    "hallucination_risk": "low | medium | high",
    "justification": "..."
  }
}
""".strip()


JUDGE_DIMENSIONS = [
    "factual_grounding",
    "conflict_handling",
    "uncertainty_calibration",
    "answer_completeness",
    "follow_up_resolution_quality",
]


RADAR_DIMENSIONS = [
    "citation_url_validity",
    "factual_grounding",
    "conflict_handling",
    "uncertainty_calibration",
    "answer_completeness",
    "follow_up_resolution_quality",
]


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def truncate_text(text: str, max_chars: int) -> str:
    if not text:
        return ""

    if len(text) <= max_chars:
        return text

    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars

    return (
        text[:head_chars]
        + "\n\n[TRUNCATED FOR EVALUATION]\n\n"
        + text[-tail_chars:]
    )


def clean_json_response(text: str) -> str:
    text = (text or "").strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = clean_json_response(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])

        raise


def clamp_judge_score(value: Any) -> int | None:
    try:
        score = int(value)
    except Exception:
        return None

    if score < 1:
        return 1

    if score > 5:
        return 5

    return score


def normalize_judge_result(data: dict[str, Any]) -> dict[str, Any]:
    scores = data.get("scores", {}) or {}
    normalized_scores = {}

    for dimension in JUDGE_DIMENSIONS:
        item = scores.get(dimension, {}) or {}
        normalized_scores[dimension] = {
            "score": clamp_judge_score(item.get("score")),
            "justification": item.get("justification", "") or "No justification returned."
        }

    key_fact_coverage = data.get("key_fact_coverage", []) or []
    if not isinstance(key_fact_coverage, list):
        key_fact_coverage = []

    return {
        "scores": normalized_scores,
        "key_fact_coverage": key_fact_coverage,
        "notes": data.get("notes", "") or ""
    }


def normalize_hallucination_result(data: dict[str, Any]) -> dict[str, Any]:
    claims = data.get("claims", []) or []
    if not isinstance(claims, list):
        claims = []

    summary = data.get("summary", {}) or {}
    if not isinstance(summary, dict):
        summary = {}

    return {
        "claims": claims,
        "summary": {
            "supported_claim_count": int(summary.get("supported_claim_count") or 0),
            "unsupported_claim_count": int(summary.get("unsupported_claim_count") or 0),
            "unclear_claim_count": int(summary.get("unclear_claim_count") or 0),
            "hallucination_risk": summary.get("hallucination_risk", "unclear") or "unclear",
            "justification": summary.get("justification", "") or "No justification returned."
        }
    }


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
        "401",
        "403",
        "UNAUTHORIZED",
        "FORBIDDEN",
        "INVALID API KEY",
        "API_KEY_INVALID",
        "PERMISSION_DENIED",
        "DENIED ACCESS",
        "HAS BEEN DENIED ACCESS",
        "MODEL_NOT_FOUND",
        "MODEL NOT FOUND",
        "MODEL DECOMMISSIONED",
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
        "RATE_LIMIT_EXCEEDED",
        "TOO MANY REQUESTS",
        "QUOTA",
        "INSUFFICIENT_QUOTA",
        "500",
        "INTERNAL",
        "502",
        "503",
        "UNAVAILABLE",
        "504",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "TIMED OUT",
        "CONNECTION",
        "NETWORK",
    ]

    return any(marker in text for marker in retryable_markers)


async def sleep_before_judge_call(label: str):
    """
    Small pacing delay before judge/probe calls.

    This helps avoid quota spikes because each evaluation turn makes:
    1. normal research pipeline model calls
    2. LLM judge call
    3. hallucination probe call
    """
    if EVAL_SLEEP_BEFORE_JUDGE_CALL_SECONDS <= 0:
        return

    print(f"  - Waiting {EVAL_SLEEP_BEFORE_JUDGE_CALL_SECONDS:.1f}s before {label}...")
    await asyncio.sleep(EVAL_SLEEP_BEFORE_JUDGE_CALL_SECONDS)


async def groq_json_call(
    system_instruction: str,
    user_prompt: str,
    operation_name: str = "Judge call",
) -> dict[str, Any]:
    """
    Calls Groq for JSON evaluation with retry + fallback logic.

    Behavior:
    - 429/quota/rate/server errors: retry the same model with exponential backoff.
    - 401/403/model access/model not found: skip that model and try the fallback model.
    - malformed JSON: retry, because the next response may be valid JSON.
    - if every model fails: raise the last error.

    This intentionally uses Groq instead of Gemini for evaluation, so the
    research agent and the evaluator do not depend on the same judge model.
    """
    last_error = None

    for model in JUDGE_MODELS:
        for attempt in range(1, MAX_JUDGE_RETRIES + 1):
            try:
                print(f"  - {operation_name}: provider=Groq, model={model}, attempt={attempt}")

                def _call_groq():
                    response = judge_client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": system_instruction,
                            },
                            {
                                "role": "user",
                                "content": user_prompt,
                            },
                        ],
                        temperature=0.0,
                        max_tokens=EVAL_JUDGE_MAX_TOKENS,
                        response_format={"type": "json_object"},
                    )

                    return response.choices[0].message.content or ""

                response_text = await asyncio.to_thread(_call_groq)
                return parse_json_object(response_text)

            except json.JSONDecodeError as e:
                last_error = e

                if attempt >= MAX_JUDGE_RETRIES:
                    print(
                        f"  - {operation_name}: JSON parsing failed after retries "
                        f"for {model}. Trying fallback model if available. Error: {e}"
                    )
                    break

                exponential_delay = EVAL_RETRY_BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
                capped_delay = min(EVAL_RETRY_MAX_SLEEP_SECONDS, exponential_delay)
                jitter = random.uniform(0, 2.0)
                wait_time = capped_delay + jitter

                print(
                    f"  - {operation_name}: invalid JSON from Groq model {model}. "
                    f"Sleeping {wait_time:.1f}s before retry. Error: {e}"
                )
                await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e

                if is_model_access_error(e):
                    print(
                        f"  - {operation_name}: access/model error for Groq model {model}. "
                        f"Trying fallback model if available. Error: {e}"
                    )
                    break

                if not is_retryable_model_error(e):
                    print(
                        f"  - {operation_name}: non-retryable error for Groq model {model}. "
                        f"Trying fallback model if available. Error: {e}"
                    )
                    break

                if attempt >= MAX_JUDGE_RETRIES:
                    print(
                        f"  - {operation_name}: exhausted retries for Groq model {model}. "
                        f"Trying fallback model if available. Error: {e}"
                    )
                    break

                exponential_delay = EVAL_RETRY_BASE_SLEEP_SECONDS * (2 ** (attempt - 1))
                capped_delay = min(EVAL_RETRY_MAX_SLEEP_SECONDS, exponential_delay)
                jitter = random.uniform(0, 2.0)
                wait_time = capped_delay + jitter

                print(
                    f"  - {operation_name}: retryable Groq error for model {model}. "
                    f"Sleeping {wait_time:.1f}s before retry. Error: {e}"
                )
                await asyncio.sleep(wait_time)

    raise last_error


def extract_expected_info(turn_spec: dict[str, Any]) -> dict[str, Any]:
    """
    New dataset fields supported by this evaluator:
    - reference_answer: string summary of the expected answer.
    - key_facts: list of specific facts the answer should cover.
    - expected_behavior: desired behavior, useful for adversarial cases.
    - adversarial_type: false_premise, outdated_sources, broad_scope, unfindable, etc.

    The old required_terms field is preserved in output as legacy metadata but is not scored.
    """

    return {
        "reference_answer": turn_spec.get("reference_answer", ""),
        "key_facts": turn_spec.get("key_facts", []) or [],
        "expected_behavior": turn_spec.get("expected_behavior", ""),
        "adversarial_type": turn_spec.get("adversarial_type", ""),
        "expect_conflict": bool(turn_spec.get("expect_conflict", False)),
        "expect_uncertainty": bool(turn_spec.get("expect_uncertainty", False)),
        "expect_follow_up": bool(turn_spec.get("expect_follow_up", False)),
        "legacy_required_terms_not_scored": turn_spec.get("required_terms", []) or [],
    }


def get_web_context_for_judge(result: dict[str, Any]) -> str:
    context = result.get("context", "") or ""

    if context:
        return truncate_text(context, MAX_JUDGE_CONTEXT_CHARS)

    # Fallback in case the pipeline result did not include the built context.
    parts = []
    for idx, source in enumerate(result.get("used_sources", []) or [], start=1):
        parts.append(
            f"[SOURCE {idx}]\n"
            f"Title: {source.get('title', '')}\n"
            f"Domain: {source.get('domain', '')}\n"
            f"URL: {source.get('url', '')}\n"
            f"Content:\n{source.get('text', '')}"
        )

    return truncate_text("\n\n---\n\n".join(parts), MAX_JUDGE_CONTEXT_CHARS)


def build_judge_prompt(
    *,
    question: str,
    standalone_question: str,
    query_plan: dict[str, Any],
    expected_info: dict[str, Any],
    web_context: str,
    final_answer: str,
) -> str:
    payload = {
        "original_question": question,
        "standalone_question": standalone_question,
        "query_plan": query_plan,
        "dataset_expectations": expected_info,
        "web_context": web_context,
        "final_answer": truncate_text(final_answer, MAX_ANSWER_CHARS),
    }

    return (
        "Evaluate this research-agent turn. Return only the requested JSON.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def build_hallucination_prompt(
    *,
    question: str,
    web_context: str,
    final_answer: str,
) -> str:
    payload = {
        "question": question,
        "web_context": web_context,
        "final_answer": truncate_text(final_answer, MAX_ANSWER_CHARS),
    }

    return (
        "Run the hallucination probe on this answer. Return only the requested JSON.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


async def judge_turn(result: dict[str, Any], turn_spec: dict[str, Any]) -> dict[str, Any]:
    question = result.get("query", "") or turn_spec.get("query", "")
    standalone_question = result.get("standalone_question", "") or question
    query_plan = result.get("query_plan", {}) or {}
    final_answer = result.get("final_answer", "") or ""
    web_context = get_web_context_for_judge(result)
    expected_info = extract_expected_info(turn_spec)

    prompt = build_judge_prompt(
        question=question,
        standalone_question=standalone_question,
        query_plan=query_plan,
        expected_info=expected_info,
        web_context=web_context,
        final_answer=final_answer,
    )

    raw = await groq_json_call(
        JUDGE_SYSTEM_PROMPT,
        prompt,
        operation_name="LLM judge",
    )
    return normalize_judge_result(raw)


async def run_hallucination_probe(result: dict[str, Any], turn_spec: dict[str, Any]) -> dict[str, Any]:
    question = result.get("query", "") or turn_spec.get("query", "")
    final_answer = result.get("final_answer", "") or ""
    web_context = get_web_context_for_judge(result)

    prompt = build_hallucination_prompt(
        question=question,
        web_context=web_context,
        final_answer=final_answer,
    )

    raw = await groq_json_call(
        HALLUCINATION_SYSTEM_PROMPT,
        prompt,
        operation_name="Hallucination probe",
    )
    return normalize_hallucination_result(raw)


def get_unique_domains(used_sources: list[dict]) -> set[str]:
    domains = set()

    for source in used_sources:
        domain = source.get("domain", "")
        if domain:
            domains.add(domain)

    return domains


def build_citation_source_pool(result: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Citation URL validity is deterministic: a cited URL is valid if it appears
    in the fetched source set. We include both used_sources and fetched_pages
    because the report should answer: did this URL come from retrieval at all?
    """

    seen_urls = set()
    source_pool = []

    for source in (result.get("used_sources", []) or []) + (result.get("fetched_pages", []) or []):
        url = source.get("url", "")
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        source_pool.append(source)

    return source_pool


def build_deterministic_metrics(result: dict[str, Any], turn_spec: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("final_answer", "") or ""
    used_sources = result.get("used_sources", []) or []
    citation_source_pool = build_citation_source_pool(result)

    citation_check = citation_validator.validate_citations(
        final_answer=answer,
        used_sources=citation_source_pool,
    )

    query_plan = result.get("query_plan", {}) or {}
    standalone_question = result.get("standalone_question", "") or ""
    original_query = result.get("query", "") or ""
    expect_follow_up = bool(turn_spec.get("expect_follow_up", False))

    if expect_follow_up:
        standalone_changed = (
            standalone_question.strip().lower() != original_query.strip().lower()
        )
    else:
        standalone_changed = None

    unique_domains = get_unique_domains(used_sources)

    return {
        "citation_url_validity": {
            "score": round(float(citation_check.get("citation_score", 0.0)), 3),
            "is_valid": bool(citation_check.get("is_valid", False)),
            "citation_count": int(citation_check.get("citation_count", 0)),
            "valid_citation_count": int(citation_check.get("valid_citation_count", 0)),
            "invalid_urls": citation_check.get("invalid_urls", []) or [],
            "warnings": citation_check.get("warnings", []) or [],
            "source_pool_count": len(citation_source_pool),
            "raw_check": citation_check,
        },
        "health_checks": {
            "search_success": bool(result.get("search_results")),
            "fetch_success": bool(result.get("fetched_pages")),
            "search_results_count": len(result.get("search_results", []) or []),
            "fetched_pages_count": len(result.get("fetched_pages", []) or []),
            "used_sources_count": len(used_sources),
            "unique_domains": sorted(list(unique_domains)),
            "unique_domain_count": len(unique_domains),
        },
        "follow_up_rewrite_check": {
            "expected_follow_up": expect_follow_up,
            "planner_marked_follow_up": bool(query_plan.get("is_follow_up", False)),
            "standalone_question_changed": standalone_changed,
            "original_query": original_query,
            "standalone_question": standalone_question,
        },
    }


def build_radar_scores(
    deterministic: dict[str, Any],
    judge: dict[str, Any] | None,
) -> dict[str, float | None]:
    radar = {
        "citation_url_validity": deterministic.get("citation_url_validity", {}).get("score"),
    }

    judge_scores = (judge or {}).get("scores", {}) or {}

    for dimension in JUDGE_DIMENSIONS:
        raw_score = judge_scores.get(dimension, {}).get("score")
        radar[dimension] = round(raw_score / 5, 3) if raw_score else None

    return radar


def compact_result_for_json(result: dict[str, Any]) -> dict[str, Any]:
    """
    Keeps evaluation output useful without storing massive full page text.
    """

    fetched_pages_light = []

    for page in result.get("fetched_pages", []) or []:
        fetched_pages_light.append({
            "title": page.get("title", ""),
            "url": page.get("url", ""),
            "domain": page.get("domain", ""),
            "retrieved_at": page.get("retrieved_at", ""),
            "text_preview": (page.get("text", "") or "")[:800],
        })

    return {
        "query": result.get("query"),
        "query_plan": result.get("query_plan"),
        "standalone_question": result.get("standalone_question"),
        "search_query": result.get("search_query"),
        "search_queries": result.get("search_queries"),
        "selected_urls": result.get("selected_urls"),
        "used_sources": result.get("used_sources"),
        "fetched_pages": fetched_pages_light,
        "final_answer": result.get("final_answer"),
        "citation_check": result.get("citation_check"),
        "timestamp": result.get("timestamp"),
    }


def make_error_scorecard(error: str) -> dict[str, Any]:
    return {
        "deterministic": None,
        "llm_judge": None,
        "hallucination_probe": None,
        "radar_scores": None,
        "error": error,
    }


async def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
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
            content=query,
        )

        try:
            result = await run_research_pipeline(
                query=query,
                session_id=session_id,
                progress_callback=lambda msg: print(f"  - {msg}"),
            )

            if EVAL_SLEEP_AFTER_PIPELINE_SECONDS > 0:
                print(
                    f"  - Waiting {EVAL_SLEEP_AFTER_PIPELINE_SECONDS:.1f}s after pipeline "
                    "to avoid quota spikes..."
                )
                await asyncio.sleep(EVAL_SLEEP_AFTER_PIPELINE_SECONDS)

            final_answer = result.get("final_answer", "") or ""

            session_store.add_message(
                session_id=session_id,
                role="assistant",
                content=final_answer,
            )

            deterministic = build_deterministic_metrics(result, turn_spec)

            print("  - Judging answer quality with Groq...")
            await sleep_before_judge_call("LLM judge")
            llm_judge = await judge_turn(result, turn_spec)

            print("  - Running hallucination probe...")
            await sleep_before_judge_call("hallucination probe")
            hallucination_probe = await run_hallucination_probe(result, turn_spec)

            radar_scores = build_radar_scores(deterministic, llm_judge)

            scorecard = {
                "deterministic": deterministic,
                "llm_judge": llm_judge,
                "hallucination_probe": hallucination_probe,
                "radar_scores": radar_scores,
            }

            turn_output = {
                "turn_index": idx,
                "query": query,
                "expectations": turn_spec,
                "scorecard": scorecard,
                "result": compact_result_for_json(result),
                "error": None,
            }

            citation_score = deterministic["citation_url_validity"]["score"]
            grounding = llm_judge["scores"]["factual_grounding"]["score"]
            completeness = llm_judge["scores"]["answer_completeness"]["score"]
            risk = hallucination_probe["summary"]["hallucination_risk"]

            print(
                "  Scorecard: "
                f"citation_urls={citation_score} | "
                f"grounding={grounding}/5 | "
                f"completeness={completeness}/5 | "
                f"hallucination_risk={risk}"
            )

        except Exception as e:
            error_message = str(e)

            turn_output = {
                "turn_index": idx,
                "query": query,
                "expectations": turn_spec,
                "scorecard": make_error_scorecard(error_message),
                "result": None,
                "error": error_message,
            }

            print(f"  ERROR: {error_message}")

        case_output["turns"].append(turn_output)

        if EVAL_SLEEP_BETWEEN_TURNS_SECONDS > 0:
            print(
                f"  - Waiting {EVAL_SLEEP_BETWEEN_TURNS_SECONDS:.1f}s before next turn..."
            )
            await asyncio.sleep(EVAL_SLEEP_BETWEEN_TURNS_SECONDS)

    return case_output


def format_score(value: Any) -> str:
    if value is None:
        return "n/a"
    return str(value)


def write_markdown_report(results: dict[str, Any], path: Path):
    lines = []

    lines.append("# Deep Research Agent Evaluation Report\n")
    lines.append(f"Generated at: `{results['generated_at']}`")
    lines.append(f"Judge provider: `Groq`\nJudge model(s): `{results['judge_model']}`\n")

    lines.append("## Summary\n")
    lines.append(f"- Cases evaluated: **{results['case_count']}**")
    lines.append(f"- Turns evaluated: **{results['turn_count']}**")
    lines.append("- Aggregate score: **not used**")
    lines.append("- This evaluator reports separate deterministic checks, Groq-judge scores, and hallucination probes.\n")

    lines.append("## Metrics Used\n")
    lines.append("### Deterministic checks")
    lines.append("- **Citation URL validity:** every cited URL must exist in the fetched/used sources.")
    lines.append("- **Search/fetch health:** whether the pipeline retrieved search results and fetched page content.")
    lines.append("- **Follow-up rewrite check:** for follow-ups, whether the standalone question changed objectively.\n")

    lines.append("### Groq judge dimensions, scored 1-5")
    lines.append("- **Factual grounding:** claim support and citation support.")
    lines.append("- **Conflict handling:** real source tension is identified accurately; fake conflict is not invented.")
    lines.append("- **Uncertainty calibration:** confidence matches the evidence quality.")
    lines.append("- **Answer completeness:** all parts of the question and key facts are addressed.")
    lines.append("- **Follow-up resolution quality:** references/pronouns are resolved correctly when applicable.\n")

    lines.append("### Hallucination probe")
    lines.append("- Extracts factual claims and checks whether each one is traceable to a fetched source chunk.\n")

    lines.append("## Case Results\n")

    for case in results["cases"]:
        lines.append(f"### {case['case_id']} - {case.get('type', '')}")
        lines.append(f"{case.get('description', '')}\n")

        for turn in case["turns"]:
            lines.append(f"#### Turn {turn['turn_index']}")
            lines.append(f"Query: `{turn['query']}`\n")

            if turn.get("error"):
                lines.append(f"Error: `{turn['error']}`\n")
                continue

            scorecard = turn.get("scorecard", {}) or {}
            deterministic = scorecard.get("deterministic", {}) or {}
            llm_judge = scorecard.get("llm_judge", {}) or {}
            hallucination_probe = scorecard.get("hallucination_probe", {}) or {}
            radar_scores = scorecard.get("radar_scores", {}) or {}
            result = turn.get("result", {}) or {}

            citation = deterministic.get("citation_url_validity", {}) or {}
            health = deterministic.get("health_checks", {}) or {}
            follow_check = deterministic.get("follow_up_rewrite_check", {}) or {}
            judge_scores = llm_judge.get("scores", {}) or {}
            hallucination_summary = hallucination_probe.get("summary", {}) or {}

            lines.append("**Scorecard**\n")
            lines.append("| Metric | Score / Status |")
            lines.append("|---|---:|")
            lines.append(f"| Citation URL validity | {format_score(citation.get('score'))} |")
            lines.append(f"| Search success | {format_score(health.get('search_success'))} |")
            lines.append(f"| Fetch success | {format_score(health.get('fetch_success'))} |")

            for dimension in JUDGE_DIMENSIONS:
                item = judge_scores.get(dimension, {}) or {}
                lines.append(f"| {dimension.replace('_', ' ').title()} | {format_score(item.get('score'))}/5 |")

            lines.append(f"| Hallucination risk | {format_score(hallucination_summary.get('hallucination_risk'))} |")
            lines.append("")

            lines.append("**Radar-ready normalized scores**")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(radar_scores, indent=2, ensure_ascii=False))
            lines.append("```\n")

            lines.append("**Deterministic details**")
            lines.append(f"- Citations: {citation.get('valid_citation_count', 0)}/{citation.get('citation_count', 0)} valid")
            lines.append(f"- Search results: {health.get('search_results_count', 0)}")
            lines.append(f"- Fetched pages: {health.get('fetched_pages_count', 0)}")
            lines.append(f"- Used sources: {health.get('used_sources_count', 0)}")
            lines.append(f"- Unique domains: {', '.join(health.get('unique_domains', []) or [])}")
            lines.append(f"- Expected follow-up: {follow_check.get('expected_follow_up')}")
            lines.append(f"- Planner marked follow-up: {follow_check.get('planner_marked_follow_up')}")
            lines.append(f"- Standalone question changed: {follow_check.get('standalone_question_changed')}\n")

            invalid_urls = citation.get("invalid_urls", []) or []
            if invalid_urls:
                lines.append("**Invalid citation URLs**")
                for url in invalid_urls:
                    lines.append(f"- {url}")
                lines.append("")

            lines.append("**Groq judge justifications**")
            for dimension in JUDGE_DIMENSIONS:
                item = judge_scores.get(dimension, {}) or {}
                lines.append(
                    f"- **{dimension.replace('_', ' ').title()} ({format_score(item.get('score'))}/5):** "
                    f"{item.get('justification', '')}"
                )
            lines.append("")

            key_fact_coverage = llm_judge.get("key_fact_coverage", []) or []
            if key_fact_coverage:
                lines.append("**Key fact coverage**")
                for item in key_fact_coverage:
                    covered = item.get("covered")
                    fact = item.get("fact", "")
                    justification = item.get("justification", "")
                    lines.append(f"- `{covered}` - {fact}: {justification}")
                lines.append("")

            lines.append("**Hallucination probe summary**")
            lines.append(f"- Supported claims: {hallucination_summary.get('supported_claim_count', 0)}")
            lines.append(f"- Unsupported claims: {hallucination_summary.get('unsupported_claim_count', 0)}")
            lines.append(f"- Unclear claims: {hallucination_summary.get('unclear_claim_count', 0)}")
            lines.append(f"- Justification: {hallucination_summary.get('justification', '')}\n")

            unsupported_claims = [
                claim for claim in hallucination_probe.get("claims", []) or []
                if claim.get("traceability") in {"unsupported", "unclear"}
            ]

            if unsupported_claims:
                lines.append("**Claims needing review**")
                for claim in unsupported_claims[:12]:
                    lines.append(f"- **{claim.get('traceability')}**: {claim.get('claim', '')}")
                    lines.append(f"  - {claim.get('explanation', '')}")
                lines.append("")

            lines.append("**Standalone question**")
            lines.append(result.get("standalone_question", "") or "")
            lines.append("")

            lines.append("**Answer preview**")
            answer = result.get("final_answer", "") or ""
            lines.append(answer[:1600].replace("\n", "\n\n"))
            lines.append("\n")

    path.write_text("\n".join(lines), encoding="utf-8")


async def main():
    session_store.init_db()

    dataset = load_dataset(DATASET_PATH)

    OUTPUT_DIR.mkdir(exist_ok=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    safe_timestamp = generated_at.replace(":", "-")

    cases = []

    for case_index, case in enumerate(dataset, start=1):
        case_result = await evaluate_case(case)
        cases.append(case_result)

        if case_index < len(dataset) and EVAL_SLEEP_BETWEEN_CASES_SECONDS > 0:
            print(
                f"\nWaiting {EVAL_SLEEP_BETWEEN_CASES_SECONDS:.1f}s before next case "
                "to reduce judge-provider quota pressure..."
            )
            await asyncio.sleep(EVAL_SLEEP_BETWEEN_CASES_SECONDS)

    turn_count = sum(len(case.get("turns", [])) for case in cases)

    output = {
        "generated_at": generated_at,
        "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_MODEL,
        "judge_models": JUDGE_MODELS,
        "retry_policy": {
            "max_judge_retries": MAX_JUDGE_RETRIES,
            "eval_retry_base_sleep_seconds": EVAL_RETRY_BASE_SLEEP_SECONDS,
            "eval_retry_max_sleep_seconds": EVAL_RETRY_MAX_SLEEP_SECONDS,
            "eval_judge_max_tokens": EVAL_JUDGE_MAX_TOKENS,
            "eval_sleep_after_pipeline_seconds": EVAL_SLEEP_AFTER_PIPELINE_SECONDS,
            "eval_sleep_before_judge_call_seconds": EVAL_SLEEP_BEFORE_JUDGE_CALL_SECONDS,
            "eval_sleep_between_turns_seconds": EVAL_SLEEP_BETWEEN_TURNS_SECONDS,
            "eval_sleep_between_cases_seconds": EVAL_SLEEP_BETWEEN_CASES_SECONDS,
        },
        "case_count": len(cases),
        "turn_count": turn_count,
        "scoring_policy": {
            "aggregate_score_used": False,
            "deterministic_metrics": [
                "citation_url_validity",
                "search_success",
                "fetch_success",
                "standalone_question_changed_on_follow_up",
            ],
            "llm_judge_dimensions": JUDGE_DIMENSIONS,
            "hallucination_probe": True,
            "dataset_fields_supported": [
                "reference_answer",
                "key_facts",
                "expected_behavior",
                "adversarial_type",
            ],
        },
        "cases": cases,
    }

    json_path = OUTPUT_DIR / f"eval_results_{safe_timestamp}.json"
    md_path = OUTPUT_DIR / f"eval_report_{safe_timestamp}.md"

    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_markdown_report(output, md_path)

    print("\nEvaluation complete.")
    print(f"JSON results: {json_path}")
    print(f"Markdown report: {md_path}")
    print("Aggregate score: disabled")


if __name__ == "__main__":
    asyncio.run(main())
