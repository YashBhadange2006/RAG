
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class DocumentChunk:
    text: str
    document_id: str
    source: str
    chunk_index: int


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from PDF, TXT or Markdown bytes."""

    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        text = "\n\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must use UTF-8 encoding.") from exc

    text = text.strip()

    if not text:
        raise ValueError(
            "No extractable text found. Scanned PDFs may require OCR."
        )

    return text


def chunk_text(
    text: str,
    tokenizer,
    document_id: str,
    source: str,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[DocumentChunk]:
    """Split text into overlapping tokenizer-sized chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative "
            "and smaller than chunk_size."
        )

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    chunks = []
    step = chunk_size - chunk_overlap

    for index, start in enumerate(range(0, len(token_ids), step)):
        token_slice = token_ids[start:start + chunk_size]

        if not token_slice:
            continue

        chunk = tokenizer.decode(
            token_slice,
            skip_special_tokens=True,
        ).strip()

        if chunk:
            chunks.append(
                DocumentChunk(
                    text=chunk,
                    document_id=document_id,
                    source=source,
                    chunk_index=len(chunks),
                )
            )

        if start + chunk_size >= len(token_ids):
            break

    return chunks


def process_document(
    filename: str,
    content: bytes,
    tokenizer,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[DocumentChunk]:
    text = extract_text(filename, content)
    document_id = str(uuid4())

    return chunk_text(
        text=text,
        tokenizer=tokenizer,
        document_id=document_id,
        source=Path(filename).name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
