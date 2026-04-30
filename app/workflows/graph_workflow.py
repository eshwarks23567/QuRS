import asyncio
import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.graph.queries import (
    _default_schema_context,
    _extract_cypher_from_response,
    execute_graph_query,
)
from app.graph.schema import DEFAULT_SCHEMA_CONTEXT
from app.llm.model import generate_completion, generate_structured_completion
from app.llm.prompts import (
    CYPHER_GENERATION_PROMPT,
    QUERY_UNDERSTANDING_PROMPT,
    REASONING_PROMPT,
)
from app.utils.helpers import (
    extract_confidence_score,
    flatten_graph_results,
    retrieve_faiss_context,
    sanitize_cypher,
    validate_cypher,
)


# ── State ─────────────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    user_query: str
    query_understanding: dict
    cypher_query: str
    graph_results: list[dict]
    additional_context: str
    answer: str
    graph_data: dict
    confidence: float
    retry_count: int
    error: str


# ── Node functions ─────────────────────────────────────────────────────────────

def parse_query(state: WorkflowState) -> WorkflowState:
    """Extract intent, entity types, relationships, and jurisdictions from the query."""
    prompt = QUERY_UNDERSTANDING_PROMPT.format(user_query=state["user_query"])
    understanding = generate_structured_completion(prompt)
    return {**state, "query_understanding": understanding}


def generate_cypher(state: WorkflowState) -> WorkflowState:
    """Translate the parsed query into a Cypher statement via LLM."""
    prompt = CYPHER_GENERATION_PROMPT.format(
        schema_context=DEFAULT_SCHEMA_CONTEXT,
        user_query=state["user_query"],
    )
    raw = generate_completion(prompt)
    cypher = _extract_cypher_from_response(raw)
    return {**state, "cypher_query": cypher}


def execute_query(state: WorkflowState) -> WorkflowState:
    """Run the Cypher query against Neo4j; increment retry counter on failure."""
    cypher = state.get("cypher_query", "")
    retry = state.get("retry_count", 0)

    if not validate_cypher(cypher):
        return {**state, "graph_results": [], "retry_count": retry + 1, "error": "invalid_cypher"}

    try:
        results = execute_graph_query(sanitize_cypher(cypher))
        return {**state, "graph_results": results, "error": ""}
    except Exception as exc:
        return {**state, "graph_results": [], "retry_count": retry + 1, "error": str(exc)}


def retrieve_context(state: WorkflowState) -> WorkflowState:
    """Augment graph results with semantically similar document chunks (FAISS)."""
    ctx = retrieve_faiss_context(state["user_query"])
    return {**state, "additional_context": ctx}


def reason_with_llm(state: WorkflowState) -> WorkflowState:
    """Synthesise graph data and document context into a natural language answer."""
    prompt = REASONING_PROMPT.format(
        user_query=state["user_query"],
        graph_results=json.dumps(state.get("graph_results", []), indent=2),
        additional_context=state.get("additional_context", "") or "None available.",
    )
    raw_answer = generate_completion(prompt)
    confidence = extract_confidence_score(raw_answer)
    clean_answer = re.sub(r"\n?Confidence:\s*[\d.]+\s*$", "", raw_answer, flags=re.IGNORECASE).strip()
    return {**state, "answer": clean_answer, "confidence": confidence}


def validate_output(state: WorkflowState) -> WorkflowState:
    """Ensure the answer field is populated; set a floor confidence if not."""
    if not state.get("answer", "").strip():
        return {
            **state,
            "answer": "No answer could be generated from the available graph data.",
            "confidence": 0.0,
        }
    return state


def return_response(state: WorkflowState) -> WorkflowState:
    """Package graph results into a frontend-friendly structure."""
    flat = flatten_graph_results(state.get("graph_results", []))
    graph_data = {
        **flat,
        "cypher_used": state.get("cypher_query", ""),
        "query_understanding": state.get("query_understanding", {}),
    }
    return {**state, "graph_data": graph_data}


# ── Routing ────────────────────────────────────────────────────────────────────

def _should_retry(state: WorkflowState) -> str:
    """Retry Cypher generation up to 2 times when the query fails validation."""
    if state.get("error") and state.get("retry_count", 0) < 2:
        return "retry"
    return "continue"


# ── Graph assembly ─────────────────────────────────────────────────────────────

def _build_workflow():
    wf = StateGraph(WorkflowState)

    wf.add_node("parse_query",      parse_query)
    wf.add_node("generate_cypher",  generate_cypher)
    wf.add_node("execute_query",    execute_query)
    wf.add_node("retrieve_context", retrieve_context)
    wf.add_node("reason_with_llm",  reason_with_llm)
    wf.add_node("validate_output",  validate_output)
    wf.add_node("return_response",  return_response)

    wf.set_entry_point("parse_query")
    wf.add_edge("parse_query",     "generate_cypher")
    wf.add_edge("generate_cypher", "execute_query")
    wf.add_conditional_edges(
        "execute_query",
        _should_retry,
        {"retry": "generate_cypher", "continue": "retrieve_context"},
    )
    wf.add_edge("retrieve_context", "reason_with_llm")
    wf.add_edge("reason_with_llm",  "validate_output")
    wf.add_edge("validate_output",  "return_response")
    wf.add_edge("return_response",  END)

    return wf.compile()


_COMPILED_WORKFLOW = None


async def run_query_workflow(user_query: str) -> dict:
    """
    Entry point for the full LangGraph pipeline.
    Runs synchronous nodes in a thread pool to keep FastAPI non-blocking.
    """
    global _COMPILED_WORKFLOW
    if _COMPILED_WORKFLOW is None:
        _COMPILED_WORKFLOW = _build_workflow()

    initial_state: WorkflowState = {
        "user_query":          user_query,
        "query_understanding": {},
        "cypher_query":        "",
        "graph_results":       [],
        "additional_context":  "",
        "answer":              "",
        "graph_data":          {},
        "confidence":          0.0,
        "retry_count":         0,
        "error":               "",
    }

    final_state = await asyncio.to_thread(_COMPILED_WORKFLOW.invoke, initial_state)

    return {
        "answer":     final_state["answer"],
        "graph_data": final_state["graph_data"],
        "confidence": final_state["confidence"],
    }


def _default_schema_context():
    return DEFAULT_SCHEMA_CONTEXT
