from dotenv import load_dotenv

load_dotenv()  # must run before any module imports that read env vars

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Knowledge Graph + LLM Hybrid Reasoning System",
    description=(
        "Hybrid reasoning over regulatory documents using Neo4j knowledge graphs "
        "and OpenAI LLMs, orchestrated by LangGraph."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["reasoning"])


@app.get("/health", tags=["health"])
def health_check() -> dict:
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
