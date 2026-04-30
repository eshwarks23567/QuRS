import os
import pickle
import re
from typing import Any

import numpy as np

# Cypher keywords that must not appear in read-only queries
_WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|CALL\s+apoc\.periodic)\b",
    re.IGNORECASE,
)


# ── Cypher safety ─────────────────────────────────────────────────────────────

def validate_cypher(query: str) -> bool:
    """
    Return False if the query is empty, contains write operations,
    or is missing both MATCH and RETURN clauses.
    """
    if not query or not query.strip():
        return False
    if _WRITE_PATTERN.search(query):
        return False
    has_match = bool(re.search(r"\bMATCH\b", query, re.IGNORECASE))
    has_return = bool(re.search(r"\bRETURN\b", query, re.IGNORECASE))
    return has_match and has_return


def sanitize_cypher(query: str) -> str:
    """Strip inline comments and collapse whitespace."""
    query = re.sub(r"//[^\n]*", "", query)
    return " ".join(query.split())


# ── LLM output helpers ────────────────────────────────────────────────────────

def extract_confidence_score(text: str) -> float:
    """
    Parse a 'Confidence: <score>' line from the end of an LLM response.
    Falls back to 0.5 if the line is missing or malformed.
    """
    match = re.search(r"Confidence:\s*([\d.]+)", text, re.IGNORECASE)
    if match:
        try:
            return max(0.0, min(1.0, float(match.group(1))))
        except ValueError:
            pass
    return 0.5


# ── FAISS vector retrieval ────────────────────────────────────────────────────

def retrieve_faiss_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve semantically similar document chunks from a local FAISS index.
    Returns an empty string if the index files do not exist yet.
    """
    index_path = "data/faiss.index"
    meta_path = "data/faiss_meta.pkl"

    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return ""

    try:
        import faiss

        from app.llm.model import get_embedding

        embedding = get_embedding(query)
        vec = np.array([embedding], dtype="float32")

        index = faiss.read_index(index_path)
        _, indices = index.search(vec, top_k)

        with open(meta_path, "rb") as f:
            metadata: list[dict] = pickle.load(f)

        results = [
            metadata[i]["text"]
            for i in indices[0]
            if 0 <= i < len(metadata)
        ]
        return "\n\n---\n\n".join(results)
    except Exception:
        return ""


def build_faiss_index(chunks: list[str], metadata: list[dict]) -> None:
    """
    Build and persist a FAISS flat-L2 index from text chunks.
    Saves index to data/faiss.index and metadata to data/faiss_meta.pkl.
    """
    import faiss

    from app.llm.model import get_embedding

    os.makedirs("data", exist_ok=True)
    embeddings = [get_embedding(c) for c in chunks]
    matrix = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatL2(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, "data/faiss.index")
    with open("data/faiss_meta.pkl", "wb") as f:
        pickle.dump(metadata, f)


# ── Misc ──────────────────────────────────────────────────────────────────────

def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    """Split a list into fixed-size batches."""
    return [items[i : i + size] for i in range(0, len(items), size)]


def flatten_graph_results(results: list[dict]) -> dict:
    """
    Separate raw graph query records into distinct node and edge collections
    for frontend graph visualization.
    """
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    for record in results:
        for key, val in record.items():
            if isinstance(val, dict) and "name" in val:
                nodes[val["name"]] = {"id": val["name"], **val}
        # Surface explicit edge records
        if all(k in record for k in ("source", "target", "type")):
            edges.append(record)

    return {"nodes": list(nodes.values()), "edges": edges}
