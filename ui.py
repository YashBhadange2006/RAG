
import os

import requests
import streamlit as st


API_BASE_URL = os.getenv(
    "RAG_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Reliable RAG",
    page_icon="📄",
    layout="wide",
)

st.title("RAG")
st.caption(
    "Upload documents, inspect indexed chunks, "
    "and verify the evidence behind generated answers."
)


def api_request(method, path, **kwargs):
    """Make a request to the FastAPI backend."""
    try:
        response = requests.request(
            method,
            f"{API_BASE_URL}{path}",
            timeout=120,
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get(
                "detail", exc.response.text
            )
        except ValueError:
            detail = exc.response.text

        st.error(f"API error: {detail}")

    except requests.ConnectionError:
        st.error(
            "Cannot connect to FastAPI. "
            "Start the backend on port 8000 first."
        )

    except requests.Timeout:
        st.error(
            "The request timed out. "
            "The embedding model may still be loading."
        )

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")

    return None


# --------------------------------------------------
# Sidebar: backend status and collection statistics
# --------------------------------------------------

with st.sidebar:
    st.header("System")

    if st.button("Check backend", use_container_width=True):
        health = api_request("GET", "/health")
        if health:
            st.success(f"Backend: {health['status']}")

    if st.button("Refresh statistics", use_container_width=True):
        stats = api_request("GET", "/stats")
        if stats:
            st.session_state["stats"] = stats

    stats = st.session_state.get("stats")

    if stats:
        st.metric("Indexed chunks", stats["total_chunks"])

    st.divider()
    st.caption(f"API: {API_BASE_URL}")


# --------------------------------------------------
# Document library
# --------------------------------------------------

st.header("Your documents")

documents_result = api_request("GET", "/documents")
documents = documents_result.get("documents", []) if documents_result else []

if not documents:
    st.info("No indexed documents yet. Upload one below.")
else:
    st.caption(f"{len(documents)} indexed document(s)")

    for document in documents:
        col_document, col_delete = st.columns([6, 1])

        with col_document:
            st.markdown(f"**{document['filename']}**")
            st.caption(
                f"{document['chunk_count']} chunks · "
                f"ID: {document['document_id'][:8]}..."
            )

        with col_delete:
            if st.button(
                "Delete",
                key=f"delete_{document['document_id']}",
                type="secondary",
                use_container_width=True,
            ):
                with st.spinner("Deleting document..."):
                    result = api_request(
                        "DELETE",
                        f"/documents/{document['document_id']}",
                    )

                if result:
                    deleted_id = document["document_id"]

                    # Clear the selected document if it was deleted.
                    if st.session_state.get("document_id") == deleted_id:
                        for key in (
                            "document_id",
                            "document_name",
                            "last_upload",
                        ):
                            st.session_state.pop(key, None)

                    # Clear potentially stale answer/evidence.
                    st.session_state.pop("last_query", None)
                    st.session_state.pop("stats", None)

                    st.success(
                        f"Deleted {document['filename']} "
                        f"and {result['deleted_chunks']} chunks."
                    )
                    st.rerun()

        st.divider()



# --------------------------------------------------
# Document upload
# --------------------------------------------------

st.header("1. Upload a document")

uploaded_file = st.file_uploader(
    "Choose a PDF, TXT or Markdown file",
    type=["pdf", "txt", "md"],
    help="Maximum upload size: 10 MB by default.",
)

if uploaded_file is not None:
    st.caption(
        f"{uploaded_file.name} · "
        f"{uploaded_file.size / 1024:.1f} KB"
    )

    if st.button("Upload and index", type="primary"):
        with st.spinner(
            "Extracting text, generating embeddings and indexing..."
        ):
            result = api_request(
                "POST",
                "/documents",
                files={
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                        or "application/octet-stream",
                    )
                },
            )

        if result:
            st.session_state["document_id"] = result["document_id"]
            st.session_state["document_name"] = result["filename"]
            st.session_state["last_upload"] = result

            st.success("Document indexed successfully.")
            st.rerun()


last_upload = st.session_state.get("last_upload")

if last_upload:
    col1, col2 = st.columns(2)

    col1.metric(
        "Chunks created",
        last_upload["chunks_created"],
    )
    col2.write(
        f"**Document:** {last_upload['filename']}\n\n"
        f"**Document ID:** `{last_upload['document_id']}`"
    )


# --------------------------------------------------
# Inspect all chunks for the selected document
# --------------------------------------------------

st.divider()
st.header("2. Inspect document chunks")

document_id = st.session_state.get("document_id")

if document_id:
    chunks_data = api_request(
        "GET",
        f"/documents/{document_id}/chunks",
    )

    if chunks_data:
        st.caption(
            f"{chunks_data['total_chunks']} chunks indexed "
            f"for {st.session_state.get('document_name', 'document')}"
        )

        for chunk in chunks_data["chunks"]:
            with st.expander(
                f"Chunk {chunk['chunk_index']} · "
                f"{len(chunk['text'])} characters",
                expanded=False,
            ):
                st.text_area(
                    "Chunk text",
                    value=chunk["text"],
                    height=160,
                    key=f"chunk_{chunk['chunk_id']}",
                    disabled=True,
                )

                st.caption(f"Source: {chunk['source']}")
                st.caption(
                    f"Document ID: {chunk['document_id']}"
                )
                st.caption(f"Chunk ID: {chunk['chunk_id']}")

else:
    st.info("Upload a document to inspect its chunks.")


# --------------------------------------------------
# Question answering
# --------------------------------------------------

st.divider()
st.header("3. Ask a question")

with st.form("query_form"):
    question = st.text_area(
        "Question",
        placeholder="Ask something about your uploaded documents...",
        max_chars=2000,
    )

    submitted = st.form_submit_button(
        "Retrieve evidence and answer",
        type="primary",
    )

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner(
            "Retrieving relevant chunks and generating an answer..."
        ):
            result = api_request(
                "POST",
                "/query",
                json={"question": question.strip()},
            )

        if result:
            st.session_state["last_query"] = result


# --------------------------------------------------
# Answer and retrieved evidence
# --------------------------------------------------

result = st.session_state.get("last_query")

if result:
    st.divider()
    st.header("4. Answer and evidence")

    if result["abstained"]:
        st.warning("Insufficient evidence to answer reliably.")
    else:
        st.subheader("Generated answer")
        st.markdown(result["answer"])

    sources = result.get("sources", [])

    st.subheader(f"Retrieved evidence ({len(sources)} chunks)")

    if not sources:
        st.info("No supporting chunks were returned.")
    else:
        for index, source in enumerate(sources, start=1):
            with st.expander(
                f"Evidence {index} · {source['filename']} · "
                f"Chunk {source['chunk_index']}",
                expanded=(index == 1),
            ):
                st.write(source["excerpt"])

                col1, col2 = st.columns(2)
                col1.metric(
                    "Cosine distance",
                    f"{source['distance']:.4f}",
                )
                col2.write(
                    f"**Document ID:** "
                    f"`{source['document_id']}`"
                )

                st.caption(
                    "Lower cosine distance indicates greater "
                    "embedding similarity; it does not prove "
                    "that the excerpt supports every answer claim."
                )
