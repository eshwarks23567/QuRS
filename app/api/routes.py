import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.graph.neo4j_client import Neo4jClient
from app.ingestion.chunker import split_into_chunks
from app.ingestion.extractor import extract_entities_and_relationships
from app.ingestion.parser import load_pdf, load_pdf_bytes
from app.utils.helpers import build_faiss_index
from app.workflows.graph_workflow import run_query_workflow

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    graph_data: dict
    confidence: float


class IngestResponse(BaseModel):
    message: str
    chunks_processed: int
    entities_extracted: int
    relationships_extracted: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Accept a natural language query and return a hybrid reasoning response.

    The pipeline: parse → generate Cypher → execute graph query →
    augment with vector context → LLM reasoning → validated answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query must not be empty.")

    try:
        result = await run_query_workflow(request.query)
        return QueryResponse(
            answer=result["answer"],
            graph_data=result.get("graph_data", {}),
            confidence=result.get("confidence", 0.0),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a regulatory PDF, extract entities/relationships,
    and persist them to Neo4j. Also indexes chunks in FAISS.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()

    try:
        text = load_pdf_bytes(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    chunks = split_into_chunks(text)
    client = Neo4jClient()
    if not client.verify_connection():
        client.close()
        raise HTTPException(status_code=503, detail="Neo4j is not reachable.")

    client.create_constraints()

    total_entities = 0
    total_relationships = 0
    faiss_chunks: list[str] = []
    faiss_meta: list[dict] = []

    try:
        for i, chunk in enumerate(chunks):
            extracted = extract_entities_and_relationships(chunk)
            client.ingest_graph_data(extracted)
            total_entities += len(extracted.get("entities", []))
            total_relationships += len(extracted.get("relationships", []))
            faiss_chunks.append(chunk)
            faiss_meta.append({"text": chunk, "source": file.filename, "chunk_index": i})
    finally:
        client.close()

    if faiss_chunks:
        build_faiss_index(faiss_chunks, faiss_meta)

    return IngestResponse(
        message="Ingestion complete.",
        chunks_processed=len(chunks),
        entities_extracted=total_entities,
        relationships_extracted=total_relationships,
    )


@router.get("/health/graph")
async def graph_health() -> dict:
    """Check Neo4j connectivity."""
    client = Neo4jClient()
    reachable = client.verify_connection()
    client.close()
    return {"neo4j": "ok" if reachable else "unreachable"}
