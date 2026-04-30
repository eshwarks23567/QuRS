import json
import re

from app.graph.schema import VALID_NODE_TYPES, VALID_RELATIONSHIP_TYPES
from app.llm.model import generate_completion
from app.llm.prompts import ENTITY_EXTRACTION_PROMPT


def extract_entities_and_relationships(text_chunk: str) -> dict:
    """
    Use an LLM to extract structured entities and relationships from a text chunk.
    Returns a validated dict: {"entities": [...], "relationships": [...]}.
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text_chunk)
    raw_response = generate_completion(prompt)
    return _parse_and_validate(raw_response)


def _parse_and_validate(raw: str) -> dict:
    """Parse LLM output as JSON and filter to schema-valid types."""
    cleaned = _strip_fence(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {"entities": [], "relationships": []}

    entities = [
        e for e in data.get("entities", [])
        if isinstance(e, dict)
        and e.get("type") in VALID_NODE_TYPES
        and isinstance(e.get("name"), str)
        and e["name"].strip()
    ]

    known_names = {e["name"] for e in entities}

    relationships = [
        r for r in data.get("relationships", [])
        if isinstance(r, dict)
        and r.get("type") in VALID_RELATIONSHIP_TYPES
        and r.get("source") in known_names
        and r.get("target") in known_names
    ]

    return {"entities": entities, "relationships": relationships}


def _strip_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
