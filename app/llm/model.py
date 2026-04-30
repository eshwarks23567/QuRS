import json
import os
import re
from functools import lru_cache

from openai import OpenAI


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1",
    )


def generate_completion(
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> str:
    """Send a prompt to xAI Grok and return the assistant message text."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def generate_structured_completion(
    prompt: str,
    model: str = "grok-3",
    temperature: float = 0.0,
) -> dict:
    """
    Generate a completion expected to return JSON.
    Strips markdown fences if present, then parses and returns a dict.
    Falls back to {"raw": <text>} on parse failure.
    """
    raw = generate_completion(prompt, model=model, temperature=temperature)
    cleaned = _strip_code_fence(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw": raw}


def get_embedding(text: str, model: str | None = None) -> list[float]:
    """Return a dense embedding vector for the given text."""
    embedding_model = model or os.getenv("EMBEDDING_MODEL", "v1")
    client = _get_client()
    response = client.embeddings.create(input=text, model=embedding_model)
    return response.data[0].embedding


def _strip_code_fence(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers from LLM output."""
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
