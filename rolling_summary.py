import session_store
from google.genai import types


ROLLING_SUMMARY_PROMPT = """
You update a rolling session summary for a web research agent.

You will receive:
1. Existing rolling summary
2. New older conversation messages that should be merged into the summary

Your job:
- Preserve the user's research topics
- Preserve important entities, dates, constraints, and preferences
- Preserve unresolved follow-up questions or open threads
- Preserve what prior answers were generally about
- Keep the summary compact and useful for future follow-up questions

Rules:
- Do not invent facts.
- Do not add new information beyond the provided conversation.
- Do not treat prior assistant answers as verified evidence.
- The summary is only for understanding conversation context.
- Keep it under 1200 words.
- Return only the updated summary as plain text.
"""


def format_messages_for_summary(messages: list[dict]) -> str:
    if not messages:
        return "No new messages."

    parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        created_at = msg.get("created_at", "")

        parts.append(f"""
[{created_at}] {role.upper()}:
{content}
""".strip())

    return "\n\n---\n\n".join(parts)


async def maybe_update_rolling_summary(
    aclient,
    session_id: str,
    model: str = "gemini-2.5-flash",
    min_messages_before_summary: int = 10,
    keep_recent_messages: int = 8,
    update_every_messages: int = 4
) -> str:
    """
    Updates rolling summary only when the conversation becomes long enough.

    It keeps recent messages untouched and summarizes older messages.
    """

    all_messages = session_store.get_all_messages(session_id)
    total_messages = len(all_messages)

    existing_summary = session_store.get_rolling_summary(session_id)
    already_summarized_count = session_store.get_summary_message_count(session_id)

    if total_messages < min_messages_before_summary:
        return existing_summary

    summarize_until = max(0, total_messages - keep_recent_messages)

    if summarize_until <= already_summarized_count:
        return existing_summary

    new_message_count = summarize_until - already_summarized_count

    if new_message_count < update_every_messages:
        return existing_summary

    messages_to_summarize = all_messages[already_summarized_count:summarize_until]

    formatted_messages = format_messages_for_summary(messages_to_summarize)

    prompt = f"""
Existing rolling summary:
{existing_summary if existing_summary else "No existing summary yet."}

New older messages to merge:
{formatted_messages}

Return the updated rolling summary.
"""

    response = await aclient.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            system_instruction=ROLLING_SUMMARY_PROMPT,
        ),
    )

    updated_summary = response.text.strip()

    session_store.update_rolling_summary(
        session_id=session_id,
        summary=updated_summary,
        summary_message_count=summarize_until
    )

    return updated_summary