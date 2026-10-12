
from fastapi import FastAPI

app = FastAPI(
    title="RAG API",
    description=(
        "A document-grounded question-answering API "
        "with semantic retrieval and source evidence."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    return {"status": "meow"}
