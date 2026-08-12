# rag-demo

FastAPI 기반 프로젝트. 패키지/가상환경 관리는 [uv](https://docs.astral.sh/uv/)를 사용합니다.

## 요구사항

- Python 3.12
- uv (`winget install --id astral-sh.uv -e`)
- Docker Desktop (PostgreSQL + pgvector 컨테이너 실행용)
- Node.js 20+ (프론트엔드 `frontend/` 실행용)

## 시작하기

```powershell
docker compose up -d     # PostgreSQL + pgvector 컨테이너 기동
uv sync                  # .venv 생성 + 의존성 설치
Copy-Item .env.example .env
# .env에서 DATABASE_URL, HF_DATASET_PATH, HF_DATASET_TEXT_COLUMN 등을 채운다

uv run python -m rag_demo.ingest               # e5(기본) 임베딩으로 청킹 + 적재 (스키마도 자동 생성)
uv run python -m rag_demo.ingest --model bge_m3 # bge-m3 임베딩으로도 적재 (비교용, 별도 테이블)
uv run uvicorn rag_demo.main:app --reload
```

- API: http://127.0.0.1:8000
- 문서: http://127.0.0.1:8000/docs
- `POST /search` — `{"query": "...", "top_k": 5, "model": "e5"}` 로 유사도 검색만 수행 (`model`은 `e5` | `bge_m3`)
- `POST /ask` — 위와 동일한 요청으로 검색 + OpenAI 답변 생성까지 수행
- `POST /compare` — `{"query": "...", "top_k": 5}` 로 e5와 bge-m3 결과를 한 번에 비교 반환 (`results: [{model, answer, sources}, ...]`)

### 프론트엔드 (검색 데모 화면)

백엔드가 떠 있는 상태에서:

```powershell
cd frontend
npm install
npm run dev
```

- 데모 화면: http://localhost:3000
- API 주소는 `frontend/.env.local`의 `NEXT_PUBLIC_API_BASE_URL`로 설정 (기본값 `http://127.0.0.1:8000`)
- `/compare`를 호출해서 e5 / bge-m3 결과를 한 화면에 나란히 보여줌

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
├─ config.py         # pydantic-settings 기반 설정 (.env 로딩)
├─ model_registry.py # 임베딩 모델(e5, bge-m3) 메타데이터: 모델명/차원/프리픽스/테이블명
├─ db.py             # pgvector 커넥션 + 모델별 스키마 초기화
├─ chunking.py       # e5 토크나이저 기반 토큰 단위 청킹 (512 토큰 한도 대응)
├─ embeddings.py     # 모델별 임베딩 (passage/query 프리픽스 처리)
├─ ingest.py         # HuggingFace 데이터셋 적재 스크립트 (--model로 임베딩 모델 선택)
└─ main.py           # FastAPI 앱 진입점 (/search, /ask, /compare, CORS 포함)

frontend/        # Next.js 검색 데모 화면 (App Router + Tailwind)
```
