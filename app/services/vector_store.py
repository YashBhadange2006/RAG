
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.services.ingestion import DocumentChunk


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


@lru_cache
def get_collection():
    settings = get_settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(settings.chroma_path)
    )

    return client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(chunks: list[DocumentChunk]) -> int:
    if not chunks:
        return 0

    model = get_embedding_model()
    collection = get_collection()

    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    ids = [
        f"{chunk.document_id}:{chunk.chunk_index}"
        for chunk in chunks
    ]

    metadatas = [
        {
            "document_id": chunk.document_id,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def search_chunks(
    query: str,
    top_k: int = 5,
) -> list[dict]:
    model = get_embedding_model()
    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    for chunk_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        matches.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": metadata["source"],
                "document_id": metadata["document_id"],
                "chunk_index": metadata["chunk_index"],
                "distance": float(distance),
            }
        )

    return matches


def collection_stats() -> dict:
    collection = get_collection()

    return {
        "collection": collection.name,
        "total_chunks": collection.count(),
    }



def get_document_chunks(document_id: str) -> list[dict]:
    """Retrieve all chunks belonging to a document."""

    collection = get_collection()

    results = collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    chunks = []

    for chunk_id, text, metadata in zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
    ):
        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "source": metadata["source"],
            "document_id": metadata["document_id"],
            "chunk_index": metadata["chunk_index"],
        })

    return sorted(
        chunks,
        key=lambda chunk: chunk["chunk_index"],
    )


def list_documents() -> list[dict]:
    """List indexed documents and count their chunks."""

    collection = get_collection()

    results = collection.get(include=["metadatas"])

    documents = {}

    for metadata in results["metadatas"] or []:
        document_id = metadata["document_id"]

        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata["source"],
                "chunk_count": 0,
            }

        documents[document_id]["chunk_count"] += 1

    return sorted(
        documents.values(),
        key=lambda document: document["filename"].lower(),
    )


def delete_document(document_id: str) -> int:
    """Delete all chunks belonging to a document."""

    collection = get_collection()

    existing = collection.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    chunk_count = len(existing["ids"])

    if chunk_count:
        collection.delete(
            where={"document_id": document_id}
        )

    return chunk_count
