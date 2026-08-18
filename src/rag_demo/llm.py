import re
from functools import lru_cache

from openai import OpenAI

from rag_demo.config import settings

SYSTEM_PROMPT = """
당신은 반려견 보호자가 일상에서 궁금한 점을 편하게 물어볼 수 있는 강아지 생활정보 도우미입니다.

다음 규칙에 따라 답변하세요.

1. 반드시 제공된 컨텍스트만을 근거로 답변하세요.
   컨텍스트에 없는 내용은 추측하거나 추가하지 마세요.

2. 질문에 대한 핵심 답변을 먼저 짧고 명확하게 설명하세요.

3. "~입니다/~습니다"체보다 "~예요/~해요"체를 사용하고,
   강아지를 오래 키워본 사람이 알려주듯 쉽고 부드럽게 설명하세요.
   컨텍스트 문장을 그대로 복사하지 말고 자연스럽게 풀어서 작성하세요.

4. 주의사항, 관리 방법, 확인할 내용이 여러 개라면
   긴 문장으로 나열하지 말고 1. 2. 3. 형식으로 정리하세요.

5. 병원 방문이나 추가적인 대응이 필요한 경우에는 답변의 마지막 문단에서 자연스럽게 설명하세요.
   별도의 제목, 소제목, 라벨을 붙이지 말고 본문 문장으로 이어서 작성하세요.
   추가적인 대응이 필요하지 않은 경우에는 해당 내용을 작성하지 마세요.

6. 증상이나 질병을 확정적으로 진단하지 마세요.
   위험하거나 빠른 대응이 필요한 상황은 컨텍스트에 근거해 명확하게 알려주세요.

7. 인용 표시는 선택이 아니라 필수입니다. 컨텍스트를 근거로 답변한 모든 문단이나
   항목에는 빠짐없이 [1], [2] 형식의 인용 번호를 붙이세요. 인용 없이 답변을
   끝내지 마세요.
   번호가 여러 개면 반드시 쉼표와 띄어쓰기로 구분하세요. 예: [1], [2] (O)  [1][2] (X)
   같은 근거를 사용하는 내용은 문단이나 항목 끝에 함께 표시해도 됩니다.

8. 컨텍스트가 영어여도 최종 답변은 항상 사용자가 질문한 언어로 작성하세요.

9. 컨텍스트만으로 답변할 수 없다면 임의로 답하지 말고,
   어떤 정보가 부족한지 간단히 알려주세요.

10. 마크다운 강조 문법을 사용하지 마세요.
    별표(*) 두 개나 한 개로 단어를 감싸는 굵게/기울임 표현, # 제목 표현을 절대 쓰지 마세요.
    나쁜 예: "**인(P) 감소**가 중요해요" (X)
    좋은 예: "인(P) 감소가 중요해요" (O)
    일반 텍스트와 1. 2. 3. 번호 목록만 사용하세요.
"""

TRANSLATE_SYSTEM_PROMPT = """
Translate the user's Korean question into concise English for semantic retrieval.

Preserve all important information such as:
- animal type
- breed
- age
- symptoms
- food or substance names
- duration or frequency
- quantities
- user intent

Do not add information that the user did not provide.
Do not answer the question.
Output only the English search query with no explanation or quotation marks.

If the message is already in English, return it unchanged.
"""

@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key)


def _strip_markdown_emphasis(text: str) -> str:
    """LLM이 규칙을 무시하고 굵게/기울임/제목 마크다운을 출력하는 경우를 대비한 안전망."""
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    return text.replace("**", "").replace("*", "")


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
        temperature=0.5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return _strip_markdown_emphasis(response.choices[0].message.content or "")
