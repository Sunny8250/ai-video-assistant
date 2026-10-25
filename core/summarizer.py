from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os
import time
import httpx


# ── Retry helper ─────────────────────────────────────────────────────────────
def _invoke_with_retry(chain, payload, max_retries: int = 3):
    """Invoke a LangChain chain and retry on Mistral 429 rate-limit errors."""
    wait_times = [60, 120, 180]  # seconds to wait between retries
    for attempt in range(max_retries + 1):
        try:
            return chain.invoke(payload)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < max_retries:
                wait = wait_times[min(attempt, len(wait_times) - 1)]
                print(f"\n⚠️  Mistral rate limit hit. Waiting {wait}s before retry "
                      f"({attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def get_llm():
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "open-mistral-7b"),
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str, status_callback=None) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = []
    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        pct = int(70 + (i / max(total_chunks, 1)) * 10)
        msg = f"Summarising transcript chunk {i + 1}/{total_chunks}..."
        print(f"  → {msg}")
        if status_callback:
            status_callback(pct, msg)
        chunk_summaries.append(_invoke_with_retry(map_chain, {"text": chunk}))

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into one final professional meeting summary in bullet points.",
        ),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    combine_msg = "Synthesizing executive meeting summary..."
    print(f"  → {combine_msg}")
    if status_callback:
        status_callback(80, combine_msg)
    return _invoke_with_retry(combined_chain, combined)


def generate_title(transcipt: str) -> str:
    llm = get_llm()

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    print("  → Generating title...")
    return _invoke_with_retry(title_chain, transcipt[:2000])
