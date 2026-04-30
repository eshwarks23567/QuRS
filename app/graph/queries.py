import re

from app.graph.neo4j_client import Neo4jClient
from app.graph.schema import DEFAULT_SCHEMA_CONTEXT
from app.llm.model import generate_completion
from app.llm.prompts import CYPHER_GENERATION_PROMPT
from app.utils.helpers import sanitize_cypher, validate_cypher


def generate_cypher_query(user_query: str, schema_context: str = "") -> str:
    """Use an LLM to convert a natural language query into a read-only Cypher statement."""
    prompt = CYPHER_GENERATION_PROMPT.format(
        schema_context=schema_context or DEFAULT_SCHEMA_CONTEXT,
        user_query=user_query,
    )
    raw = generate_completion(prompt)
    cypher = _extract_cypher_from_response(raw)

    if not validate_cypher(cypher):
        raise ValueError(f"LLM produced an invalid or unsafe Cypher query: {cypher!r}")

    return sanitize_cypher(cypher)


def execute_graph_query(cypher: str, params: dict | None = None) -> list[dict]:
    """Execute a validated Cypher query and return results."""
    client = Neo4jClient()
    try:
        return client.execute_cypher(cypher, params)
    finally:
        client.close()


def _extract_cypher_from_response(text: str) -> str:
    """Pull Cypher out of a fenced code block, or return the text as-is."""
    match = re.search(r"```(?:cypher)?\s*(.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


# ── Canned queries for the test flow ─────────────────────────────────────────

REGULATIONS_AFFECTING_CAPITAL_RATIOS = """
MATCH (r:Regulation)-[:AFFECTS]->(c:CapitalRatio)
RETURN r.name AS regulation, r.issuing_body AS issuing_body,
       c.name AS capital_ratio, c.minimum_threshold AS threshold
ORDER BY r.name
LIMIT 50
"""

REGULATIONS_BY_JURISDICTION = """
MATCH (r:Regulation)-[:APPLIES_IN]->(j:Jurisdiction)
WHERE toLower(j.name) CONTAINS toLower($jurisdiction)
RETURN r.name AS regulation, j.name AS jurisdiction
ORDER BY r.name
LIMIT 50
"""

ALL_REGULATIONS = """
MATCH (r:Regulation)
RETURN r.name AS name, r.issuing_body AS issuing_body,
       r.effective_date AS effective_date
ORDER BY r.name
LIMIT 100
"""
