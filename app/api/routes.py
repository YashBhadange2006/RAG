
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.services.ingestion import process_document
from app.services.rag import answer_question
from app.services.vector_store import add_chunks, collection_stats, get_embedding_model


from app.services.vector_store import (
    add_chunks,
    collection_stats,
    delete_document,
    get_embedding_model,
    get_document_chunks,
    list_documents,
)

router = APIRouter()


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class SourceResponse(BaseModel):
    filename: str
    document_id: str
    chunk_index: int
    excerpt: str
    distance: float


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    sources: list[SourceResponse]


@router.post("/documents", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    settings = get_settings()

    filename = (file.filename or "").replace("\\", "/").split("/")[-1]

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required.",
        )

    content_length = 0
    content_parts = []

    # Read incrementally to enforce the upload limit.
    while True:
        part = await file.read(1024 * 1024)

        if not part:
            break

        content_length += len(part)

        if content_length > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Uploaded file exceeds the size limit.",
            )

        content_parts.append(part)

    content = b"".join(content_parts)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        # Force model loading here so ingestion errors are reported.
        model = get_embedding_model()

        chunks = process_document(
            filename=filename,
            content=content,
            tokenizer=model.tokenizer,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        count = add_chunks(chunks)

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Document processing failed.",
        ) from exc
    finally:
        await file.close()

    return {
        "message": "Document indexed successfully.",
        "filename": filename,
        "document_id": chunks[0].document_id,
        "chunks_created": count,
    }


@router.post("/query", response_model=QueryResponse, tags=["RAG"])
def query_documents(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be blank.",
        )

    try:
        result = answer_question(question)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Question answering failed.",
        ) from exc


@router.get("/stats", tags=["Documents"])
def get_stats():
    return collection_stats()


@router.get("/documents", tags=["Documents"])
def list_documents_endpoint():
    return {
        "documents": list_documents()
    }


@router.delete("/documents/{document_id}", tags=["Documents"])
def delete_document_endpoint(document_id: str):
    deleted_chunks = delete_document(document_id)

    if deleted_chunks == 0:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "deleted_chunks": deleted_chunks,
    }




@router.get("/documents/{document_id}/chunks", tags=["Documents"])
def get_document_chunks_endpoint(document_id: str):
    chunks = get_document_chunks(document_id)

    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No chunks found for this document.",
        )

    return {
        "document_id": document_id,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

