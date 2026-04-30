# Knowledge Graph + LLM Hybrid Reasoning System

Answers complex regulatory queries by combining a Neo4j knowledge graph with OpenAI LLMs, orchestrated via LangGraph.

**Example query:** *"Show all regulations impacting capital ratios across jurisdictions"*

---

## Architecture

```
PDF → Parser → Chunker → Extractor (LLM) → Neo4j
                                          ↓
User Query → FastAPI → LangGraph Workflow → LLM Answer
                            ↓
                     Cypher → Neo4j → Graph Results
                     FAISS → Vector Context (optional)
```

## Project Structure

```
app/
├── main.py                  # FastAPI app entry point
├── api/routes.py            # POST /query, POST /ingest, GET /health/graph
├── graph/
│   ├── neo4j_client.py      # Neo4j driver wrapper
│   ├── schema.py            # Node/relationship dataclasses + valid types
│   └── queries.py           # Cypher generation + execution helpers
├── ingestion/
│   ├── parser.py            # PDF text extraction (pdfplumber)
│   ├── chunker.py           # Token-aware chunking with overlap
│   └── extractor.py        # LLM entity/relationship extraction
├── llm/
│   ├── prompts.py           # All prompt templates
│   └── model.py             # OpenAI wrapper (completion + embeddings)
├── workflows/
│   └── graph_workflow.py    # LangGraph pipeline (7 nodes)
└── utils/helpers.py         # Cypher validation, FAISS, confidence parsing
```

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   OPENAI_API_KEY
#   NEO4J_URI  (e.g. bolt://localhost:7687 or Neo4j Aura URI)
#   NEO4J_USER
#   NEO4J_PASSWORD
```

### 3. Start Neo4j

**Local (Docker):**
```bash
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/yourpassword \
  neo4j:5
```

**Cloud:** Use [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/) and set `NEO4J_URI` to the Aura connection string.

### 4. Run the server

```bash
python -m app.main
# or
uvicorn app.main:app --reload
```

API docs: http://localhost:8000/docs

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/query` | Natural language query → hybrid answer |
| `POST` | `/api/v1/ingest` | Upload PDF → populate Neo4j + FAISS |
| `GET`  | `/api/v1/health/graph` | Neo4j connectivity check |
| `GET`  | `/health` | App health check |

### Query example

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which regulations affect capital ratios?"}'
```

Response:
```json
{
  "answer": "Basel III requires banks to maintain a minimum Common Equity Tier 1 (CET1) ratio of 4.5%...",
  "graph_data": {
    "nodes": [...],
    "edges": [...],
    "cypher_used": "MATCH (r:Regulation)-[:AFFECTS]->(c:CapitalRatio) RETURN r, c LIMIT 50"
  },
  "confidence": 0.87
}
```

### Ingest example

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@path/to/regulation.pdf"
```

---

## LangGraph Workflow

```
parse_query
    ↓
generate_cypher
    ↓
execute_query ──(invalid/error, ≤2 retries)──→ generate_cypher
    ↓ (success)
retrieve_context   ← FAISS vector lookup (optional)
    ↓
reason_with_llm
    ↓
validate_output
    ↓
return_response
    ↓
END
```

## Graph Schema

**Nodes:** `Regulation` · `CapitalRatio` · `Jurisdiction` · `Institution` · `Metric`

**Relationships:** `AFFECTS` · `APPLIES_IN` · `REFERENCES` · `DEFINES` · `REQUIRES`

---

## Guardrails

- **Cypher validation:** rejects queries without `MATCH`/`RETURN`, blocks all write operations
- **Schema filtering:** extracted entities and relationships must match known types
- **Retry loop:** re-generates Cypher up to 2 times on validation failure
- **Confidence scoring:** LLM reports a 0–1 score included in every response

## Optional Enhancements

- **LangSmith tracing:** set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in `.env`
- **FAISS vector search:** automatically enabled once a PDF is ingested
- **Graph visualization:** `graph_data.nodes` and `graph_data.edges` are ready for D3.js / Cytoscape.js
