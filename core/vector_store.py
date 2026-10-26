import os
import re
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_PERSIST_DIR = os.path.join("data", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embeddings_instance = None
_chroma_client_instance = None


def get_embeddings():
    """Singleton for HuggingFace embeddings to avoid reloading weights repeatedly."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings_instance


def get_chroma_client() -> chromadb.PersistentClient:
    """
    Singleton PersistentClient pointing to data/chroma_db.
    Using PersistentClient ensures SQLite schemas (tenants, collections)
    are always initialized and fully thread-safe across Streamlit worker threads.
    """
    global _chroma_client_instance
    if _chroma_client_instance is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _chroma_client_instance = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _chroma_client_instance


def _sanitize_collection_name(raw_name: str) -> str:
    """Chroma collection names must be 3-63 chars, alphanumeric, dots, hyphens or underscores."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name or "current_meeting").strip("_")
    if len(clean) < 3:
        clean = f"meeting_{clean}"
    return clean[:60]


def build_vector_store(
    transcript: str,
    summary: str = "",
    chapters: str = "",
    key_decisions: list | str = None,
    action_items: list | str = None,
    session_id: str = "current_meeting",
) -> Chroma:
    """
    Build a session-isolated Chroma vector store using PersistentClient.
    Indexes:
      1. Executive summary & key concepts
      2. Chapters & discussion timeline
      3. Key decisions & action items
      4. Transcript text chunks
    Using a persistent client ensures thread safety across Streamlit reruns,
    while unique collection names guarantee zero cross-contamination.
    """
    col_name = _sanitize_collection_name(session_id)
    print(f"[RAG] Building vector store for collection: '{col_name}'...")
    docs: list[Document] = []

    # 1. Summary Document(s)
    if summary and summary.strip():
        summary_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        summary_chunks = summary_splitter.split_text(summary.strip())
        for idx, sc in enumerate(summary_chunks):
            docs.append(Document(
                page_content=f"MEETING EXECUTIVE SUMMARY & KEY CONCEPTS:\n{sc}",
                metadata={"type": "summary", "chunk_index": idx}
            ))

    # 2. Timeline Chapters Document
    if chapters and chapters.strip():
        docs.append(Document(
            page_content=f"MEETING DISCUSSION TIMELINE & CHAPTERS:\n{chapters.strip()}",
            metadata={"type": "chapters", "chunk_index": 0}
        ))

    # 3. Key Decisions & Action Items
    dec_items = []
    if key_decisions:
        if isinstance(key_decisions, list):
            dec_items.extend([f"• Decision: {d}" for d in key_decisions if d])
        else:
            dec_items.append(f"• Decisions: {key_decisions}")
    if action_items:
        if isinstance(action_items, list):
            for a in action_items:
                if isinstance(a, dict):
                    dec_items.append(f"• Action: {a.get('task','')} (Owner: {a.get('owner','')}, Deadline: {a.get('deadline','')})")
                else:
                    dec_items.append(f"• Action: {a}")
        else:
            dec_items.append(f"• Actions: {action_items}")

    if dec_items:
        docs.append(Document(
            page_content="MEETING DECISIONS & ACTION ITEMS:\n" + "\n".join(dec_items),
            metadata={"type": "decisions_actions", "chunk_index": 0}
        ))

    # 4. Transcript Chunks
    if transcript and transcript.strip():
        transcript_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
        )
        t_chunks = transcript_splitter.split_text(transcript.strip())
        for i, chunk in enumerate(t_chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"type": "transcript", "chunk_index": i}
            ))

    if not docs:
        docs.append(Document(page_content="Empty meeting session.", metadata={"type": "empty"}))

    client = get_chroma_client()
    embeddings = get_embeddings()

    # Reset any existing collection with this name to prevent residual stale data
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        collection_name=col_name,
    )

    print(f"[RAG] Indexed {len(docs)} document chunks successfully into collection '{col_name}'.")
    return vector_store


def load_vector_store(session_id: str = "current_meeting") -> Chroma:
    col_name = _sanitize_collection_name(session_id)
    client = get_chroma_client()
    embeddings = get_embeddings()
    return Chroma(
        client=client,
        collection_name=col_name,
        embedding_function=embeddings,
    )


def get_retriever(vector_store: Chroma, k: int = 6):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
