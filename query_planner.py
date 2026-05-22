import json
from google.genai import types


QUERY_PLANNER_PROMPT = """
You are a research query planner.

Your job is to rewrite the user's current question into a standalone research question
and create web search queries for Tavily.

You will receive:
1. The user's current question
2. Recent previous research turns from the same session



Return this JSON structure:
{
  "is_follow_up": true,
  "standalone_question": "...",
  "search_query": "...",
  "history_used": "short explanation of what previous context was used"
}
"""


def clean_json_response(text: str) -> str:
    text = text.strip()

    if text.startswith("```json"):
        text = text.removeprefix("```json").strip()

    if text.startswith("```"):
        text = text.removeprefix("```").strip()

    if text.endswith("```"):
        text = text.removesuffix("```").strip()

    return text


def format_turns_for_planner(turns: list[dict]) -> str:
    """
    Converts previous turns into compact text for the query planner.
    """

    if not turns:
        return "No previous turns."

    blocks = []

    for idx, turn in enumerate(turns, start=1):
        query = turn.get("query", "")
        answer = turn.get("final_answer", "")

        selected_urls = turn.get("selected_urls", [])
        selected_titles = [
            item.get("title", "")
            for item in selected_urls
            if item.get("title")
        ]

        block = f"""
Previous Turn {idx}
User query:
{query}

Assistant answer preview:
{answer[:1200]}

Sources selected:
{json.dumps(selected_titles[:5], indent=2)}
""".strip()

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


async def plan_research_query(
    aclient,
    current_query: str,
    previous_turns: list[dict],
    rolling_summary: str = "",
    model: str = "gemini-2.5-flash"
) -> dict:
    """
    Uses Gemini to create a standalone query and Tavily search queries.
    """

    history_text = format_turns_for_planner(previous_turns)

    prompt = f"""
            Current user question:
            {current_query}

            Rolling session summary:
            {rolling_summary if rolling_summary else "No rolling summary yet."}

            Recent previous turns:
            {history_text}

            Create a standalone research question and search queries.
            """

    response = await aclient.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=QUERY_PLANNER_PROMPT,
            response_mime_type="application/json",
        ),
    )

    cleaned = clean_json_response(response.text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"Query planner returned invalid JSON:\n{response.text}")

    search_query = data.get("search_query")

    if not search_query:
        search_query = current_query

    return {
        "is_follow_up": bool(data.get("is_follow_up", False)),
        "standalone_question": data.get("standalone_question", current_query),
        "search_query": search_query,
        "history_used": data.get("history_used", "")
    }