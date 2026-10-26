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


def get_combine_prompt_for_mode(mode: str) -> str:
    """Return tailored system prompt based on user-selected summary preset."""
    mode_clean = mode.lower()
    if "executive" in mode_clean:
        return (
            "You are an executive chief of staff. Create an Executive Brief for leadership:\n"
            "### 🎯 Strategic Overview\n"
            "- High-level summary in 3-4 impactful bullet points\n\n"
            "### 💡 Key Outcomes & ROI\n"
            "- Strategic decisions made and business impact\n\n"
            "### ⚠️ Critical Blockers & Risks\n"
            "- Key dependencies, risks, or resources required"
        )
    elif "technical" in mode_clean:
        return (
            "You are a principal software engineer. Create a Technical Deep-Dive summary:\n"
            "### 🛠️ Architecture & System Design\n"
            "- Core concepts, frameworks, and architecture patterns discussed\n\n"
            "### ⚙️ Implementation & Technical Decisions\n"
            "- Technical mechanisms, APIs, data flows, and trade-offs\n\n"
            "### 🔍 Gotchas & Engineering Considerations\n"
            "- Performance implications, potential bugs, or technical debt"
        )
    elif "study" in mode_clean or "lecture" in mode_clean:
        return (
            "You are a top academic instructor. Create structured Study & Lecture Notes:\n"
            "### 📚 Core Concepts & Definitions\n"
            "- Essential terms, theories, and concepts explained simply\n\n"
            "### 📝 Key Takeaways & Explanations\n"
            "- Detailed breakdown of each topic with bulleted notes\n\n"
            "### 🃏 Flashcard Revision Q&A\n"
            "- 3-5 high-yield Question & Answer pairs for quick review"
        )
    elif "email" in mode_clean:
        return (
            "You are an executive communications specialist. Draft a polished, ready-to-send follow-up email:\n"
            "Subject: [Meeting Recap & Next Steps] <Topic>\n\n"
            "Hi Team,\n\n"
            "Thank you for attending today's session. Below is a concise recap:\n\n"
            "**Key Discussion Points:**\n"
            "- Point 1\n"
            "- Point 2\n\n"
            "**Immediate Next Steps:**\n"
            "- Action 1\n\n"
            "Best regards,"
        )
    else:
        return (
            "You are an expert meeting summarizer. Combine these partial summaries "
            "into a comprehensive, well-structured professional meeting summary in bullet points, "
            "covering all main topics, decisions, and nuances discussed."
        )


def summarize(transcript: str, mode: str = "comprehensive", status_callback=None) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely in clear bullet points."),
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

    system_instruction = get_combine_prompt_for_mode(mode)
    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{text}"),
    ])

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    combine_msg = f"Synthesizing {mode} summary..."
    print(f"  → {combine_msg}")
    if status_callback:
        status_callback(80, combine_msg)
    return _invoke_with_retry(combined_chain, combined)


def generate_chapters(transcript: str, status_callback=None) -> str:
    """Generate chronological chapters/milestones with timestamps for video/meeting navigation."""
    llm = get_llm()

    chapter_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting video editor. Break down this meeting/video transcript into chronological "
            "chapters with timestamps.\n"
            "Format each chapter on a new line as:\n"
            "- ⏱️ [MM:SS] - Chapter Title: Brief 1-sentence topic summary\n\n"
            "If the transcript has timestamps, use them accurately. Otherwise estimate sequential progression starting from [00:00]. "
            "Provide between 4 to 8 clear chapters. Return only the chapter list, no intro/outro.",
        ),
        ("human", "{text}"),
    ])

    chapter_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | chapter_prompt
        | llm
        | StrOutputParser()
    )

    print("  → Generating video chapters...")
    if status_callback:
        status_callback(83, "Generating meeting chapters & timeline...")
    sample_text = transcript[:8000] if len(transcript) > 8000 else transcript
    return _invoke_with_retry(chapter_chain, sample_text)


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

