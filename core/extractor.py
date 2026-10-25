#Actionableitems , decision , questions 

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os 
import time
import httpx


def _invoke_with_retry(chain, payload, max_retries: int = 3):
    """Invoke a LangChain chain and retry on Mistral 429 rate-limit errors."""
    wait_times = [60, 120, 180]
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
    return ChatMistralAI(model=os.getenv("MISTRAL_MODEL", "open-mistral-7b"), mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2)



def build_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )

def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return _invoke_with_retry(chain, transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return _invoke_with_retry(chain, transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return _invoke_with_retry(chain, transcript)