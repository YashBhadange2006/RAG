
from pathlib import Path

import pytest

from app.services.ingestion import (
    chunk_text,
    extract_text,
    process_document,
)


def test_extract_text_file(tmp_path: Path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "RAG combines retrieval with generation.",
        encoding="utf-8",
    )

    assert extract_text(file_path) == (
        "RAG combines retrieval with generation."
    )


def test_unsupported_file_type(tmp_path: Path):
    file_path = tmp_path / "sample.csv"
    file_path.write_text("a,b,c", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        extract_text(file_path)


def test_empty_file_is_rejected(tmp_path: Path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No extractable text"):
        extract_text(file_path)


def test_chunk_overlap_and_metadata():
    text = " ".join(f"word{i}" for i in range(100))

    chunks = chunk_text(
        text=text,
        document_id="doc-1",
        source="sample.txt",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0].source == "sample.txt"
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1


def test_process_document(tmp_path: Path):
    file_path = tmp_path / "sample.md"
    file_path.write_text(
        "# Retrieval\n\nRAG retrieves evidence.",
        encoding="utf-8",
    )

    chunks = process_document(file_path)

    assert chunks
    assert chunks[0].source == "sample.md"
    assert chunks[0].document_id
