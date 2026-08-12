from functools import lru_cache

from openai import OpenAI

from rag_demo.config import settings

SYSTEM_PROMPT = (
    "당신은 주어진 컨텍스트만을 근거로 답변하는 도우미입니다. "
    "컨텍스트에 없는 내용은 알고 있더라도 답변에 포함하지 마세요. "
    "컨텍스트에 답이 없으면 모른다고 답하세요. "
    "답변에 사용한 컨텍스트는 문장 끝에 [1], [2]처럼 번호로 표시하세요."
)

TRANSLATE_SYSTEM_PROMPT = (
    "Translate the user's message into English for use as a search query. "
    "Output only the translated text, with no quotes or explanation. "
    "If the message is already in English, output it unchanged."
)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def translate_query_to_english(query: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content or query


def generate_answer(query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    user_prompt = f"컨텍스트:\n{context_block}\n\n질문: {query}"

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""
