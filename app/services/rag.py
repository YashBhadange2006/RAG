
from functools import lru_cache

from groq import Groq

from app.core.config import get_settings
from app.services.vector_store import search_chunks


SYSTEM_PROMPT = """
You are a document-grounded question-answering assistant.

Rules:
1. Answer only using the supplied evidence.
2. Do not use outside knowledge to fill missing information.
3. If the evidence does not support an answer, say that the
   available documents do not contain enough information.
4. Never invent quotations, filenames, page numbers or citations.
5. Distinguish facts from reasonable inferences.
6. Keep answers concise and directly address the question.
"""


@lru_cache
def get_llm_client() -> Groq:
    settings = get_settings()

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your .env file."
        )

    return Groq(api_key=settings.groq_api_key)


def answer_question(question: str) -> dict:
    settings = get_settings()

    matches = search_chunks(
        question,
        top_k=settings.retrieval_top_k,
    )

    if not matches:
        return {
            "answer": (
                "I couldn't find relevant evidence in the "
                "uploaded documents to answer this question."
            ),
            "abstained": True,
            "sources": [],
        }

    # Lower cosine distance means greater similarity.
    relevant_matches = [
        match
        for match in matches
        if match["distance"] <= settings.max_cosine_distance
    ]

    if not relevant_matches:
        return {
            "answer": (
                "The uploaded documents do not appear to contain "
                "enough relevant information to answer this question."
            ),
            "abstained": True,
            "sources": [],
        }

    evidence = "\n\n".join(
        (
            f"[Evidence {index}]\n"
            f"Source: {match['source']}\n"
            f"Chunk: {match['chunk_index']}\n"
            f"Content:\n{match['text']}"
        )
        for index, match in enumerate(relevant_matches, start=1)
    )

    client = get_llm_client()

    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved evidence:\n{evidence}\n\n"
                    "Answer using only the retrieved evidence."
                ),
            },
        ],
    )

    answer = response.choices[0].message.content or ""

    sources = [
        {
            "filename": match["source"],
            "document_id": match["document_id"],
            "chunk_index": match["chunk_index"],
            "excerpt": match["text"],
            "distance": round(match["distance"], 4),
        }
        for match in relevant_matches
    ]

    return {
        "answer": answer.strip(),
        "abstained": False,
        "sources": sources,
    }
