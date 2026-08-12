# rag-demo

FastAPI 기반 프로젝트. 패키지/가상환경 관리는 [uv](https://docs.astral.sh/uv/)를 사용합니다.

## 요구사항

- Python 3.12
- uv (`winget install --id astral-sh.uv -e`)

## 시작하기

```powershell
uv sync                 # .venv 생성 + 의존성 설치
Copy-Item .env.example .env
uv run uvicorn rag_demo.main:app --reload
```

- API: http://127.0.0.1:8000
- 문서: http://127.0.0.1:8000/docs

## 의존성 관리

```powershell
uv add <package>            # 의존성 추가
uv add --dev <package>      # 개발 의존성 추가
uv remove <package>         # 제거
uv sync                     # uv.lock 기준으로 환경 동기화
```

## 구조

```
src/rag_demo/
├─ __init__.py
├─ config.py     # pydantic-settings 기반 설정 (.env 로딩)
└─ main.py       # FastAPI 앱 진입점
```
