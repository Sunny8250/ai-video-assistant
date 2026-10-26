import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, get_retriever


def get_llm():
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "open-mistral-7b"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def build_rag_chain(
    transcript: str,
    summary: str = "",
    chapters: str = "",
    key_decisions: list | str = None,
    action_items: list | str = None,
):
    """
    Builds a grounded LCEL RAG pipeline with an in-memory vector store.
    Indexes both the high-level meeting summary/chapters and granular transcript chunks.
    """
    vector_store = build_vector_store(
        transcript=transcript,
        summary=summary,
        chapters=chapters,
        key_decisions=key_decisions,
        action_items=action_items,
    )

    retriever = get_retriever(vector_store, k=6)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert AI Meeting Assistant answering questions about this recorded meeting or video session.

CONTEXT FROM THIS MEETING SESSION (Summary, Timeline Chapters, Decisions & Transcript):
{context}

INSTRUCTIONS:
1. Answer the question accurately and helpfully based on the provided meeting context above.
2. Synthesize information from the executive summary, discussion chapters, and transcript excerpts.
3. The original speech or transcript may include Hindi, Hinglish, or English, and technical concepts may have phonetic or spelling variations (e.g., 'ड्रग' or 'ग्राफ' for Spark DAG / Directed Acyclic Graph). Always interpret terms in the context of the meeting's subject matter.
4. If a concept was explained or demonstrated in the meeting, provide a thorough, easy-to-understand explanation citing timestamps if available.
5. Only if the question is genuinely and completely unrelated to the topics covered in this meeting, state:
"I could not find this information in this meeting session."
""",
        ),
        ("human", "{question}"),
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:
    print(f"[RAG] Question: {question}")
    answer = rag_chain.invoke(question)
    print(f"[RAG] Answer: {answer[:100]}...")
    return answer
