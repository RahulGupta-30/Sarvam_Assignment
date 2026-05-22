import os
import asyncio
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from .env")

groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)


def generate_text_sync(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int = 1800,
) -> str:
    """
    Synchronous Groq text generation.

    Uses Groq's OpenAI-compatible chat completions API.
    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    kwargs = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = groq_client.chat.completions.create(**kwargs)

    return response.choices[0].message.content or ""


async def generate_text(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_tokens: int = 1800,
) -> str:
    """
    Async wrapper around the synchronous Groq API call.

    This lets the rest of your async pipeline keep using:
        await generate_text(...)
    """

    return await asyncio.to_thread(
        generate_text_sync,
        system_prompt,
        user_prompt,
        temperature,
        json_mode,
        max_tokens,
    )