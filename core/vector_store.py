import os
import shutil
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_embeddings_instance = None


def get_embeddings():
    """Singleton for HuggingFace embeddings to avoid reloading weights repeatedly."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings_instance


def build_vector_store(
    transcript: str,
    summary: str = "",
    chapters: str = "",
    key_decisions: list | str = None,
    action_items: list | str = None,
) -> Chroma:
    """
    Build an isolated in-memory Chroma vector store for the current meeting.
    Indexes:
      1. Executive summary & key concepts
      2. Chapters & discussion timeline
      3. Key decisions & action items
      4. Transcript text chunks
    Running in-memory ensures zero cross-contamination between different meetings.
    """
    print("[RAG] Building isolated in-memory vector store...")
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

    embeddings = get_embeddings()

    # Build fresh in-memory Chroma instance (no persist_directory prevents old data leakage)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )

    print(f"[RAG] Indexed {len(docs)} document chunks successfully into vector store.")
    return vector_store


def get_retriever(vector_store: Chroma, k: int = 6):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def clear_legacy_vector_db(dir_path: str = "vector_db"):
    """Clean up any old legacy Chroma disk folder to save space and avoid stale state."""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"[RAG] Cleared legacy {dir_path} disk directory.")
    except Exception as e:
        print(f"[RAG] Warning clearing {dir_path}: {e}")
