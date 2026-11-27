# LangChain AI 기술 보고서 에이전트 (Advanced) 구현 및 검증 결과 보고서

| 항목 | 내용 | 
| ----- | ----- | 
| **프로젝트명** | LangChain Report Assistant (`03_advanced`) | 
| **작성일** | 2026-11-27 | 
| **구현 난이도** | Advanced (빈 폴더 기반 풀스택 파이프라인 구축) | 
| **작성자** | Master Kim | 
| **핵심 스택** | Python, FastAPI, LangChain (`create_agent`, `@tool`, `ProviderStrategy`, LCEL), PostgreSQL 18 + `pgvector`, CUDA (`E5-small` + `BGE-reranker-v2-m3`) | 

## 1. 프로젝트 개요 및 아키텍처

본 프로젝트는 기존 `report_agent_lab` PostgreSQL 데이터베이스를 읽기 전용(`SET TRANSACTION READ ONLY`)으로 연동하여, 기술 보고서 본문 검색(RAG)과 보고서 상태 조회(DB)를 결합해 근거 기반 답변을 생성하는 **Advanced 등급 AI 에이전트 서버**입니다.

### 1.1 핵심 설계 원칙

* **무결성 및 읽기 전용 잠금**: 모든 DB 세션에 `SET TRANSACTION READ ONLY`와 `statement_timeout(DB_TIMEOUT_MS)`을 강제하고, `reports`·`report_embeddings` 조인 시 6중 `FRESH` 벡터 유효성 조건(`index_status='ready'`, `model_name`, `indexed_revision`, `vector_dims=384`, `vector_norm>0`, `overview_hash`)을 적용했습니다.

* **CUDA GPU 전용 추론 (CPU 폴백 금지)**: `intfloat/multilingual-e5-small` 임베딩과 `BAAI/bge-reranker-v2-m3` 리랭커(`batch_size=2`, `max_length=512`)를 모두 `cuda:0`에서 구동하며, GPU 미감지 시 `503 GPU_UNAVAILABLE`를 반환합니다.

* **단일 에이전트 + 네이티브 Structured Output**: `create_agent`에 `ProviderStrategy(StructuredModelAnswer)`를 적용하고 `ToolCallLimitMiddleware` · `ModelCallLimitMiddleware`로 무한 루프를 차단했습니다.

* **근거 검증 계층 (`validator.py`)**: 도구 실행 트레이스(`tool_trace`)에서 실제 관찰된 `report_id`와 모델의 `source_ids`를 대조해 환각 인용을 차단하고, `insufficient_evidence` 상태에서는 `sources=[]`를 강제했습니다.

### 1.2 시스템 계층 및 엔드포인트 구성

| 구분 | 경로 / 모듈 | 역할 | 
| ----- | ----- | ----- | 
| **API-00** | `GET /health` (`app/api/agent.py`) | DB 준비 상태(`ready`), GPU(`cuda:0`), FRESH 벡터 수, LLM 설정 여부 점검 | 
| **API-01** | `POST /api/v1/agent/ask` (`app/api/agent.py`) | `rag_search` · `lookup_status` 도구 선택 실행 및 검증된 `AgentAskResponse` 반환 | 
| **API-02** | `POST /api/v1/rag/debug` (`app/api/rag.py`) | 질의 계보(`query_original` → `query_search`) 및 리랭킹 전후(`before`/`after`) 비교 | 
| **API-03** | `GET /api/v1/agent/runs/{request_id}` (`app/api/agent.py`, `RunStore`) | 최근 100건 인메모리 실행 이력 및 `metrics` 조회 (미존재 시 `404 RUN_NOT_FOUND`) | 
| **Offline Eval** | `app/evaluation/retrieval_eval.py`, `debug_report.py` | `data/evaluation/gold.json` 기반 Hit@K, MRR 산출 및 `outputs/quality_report.md` 생성 | 

## 2. 라이브 API 통합 검증 결과 (Postman Collection Runner)

전체 컬렉션(`Day30 Report Agent - SRS v1.0`, `project_level=advanced`, `allow_paid_llm=true`) 실행 결과 **35개 테스트 항목 전체 통과**를 달성했습니다.

| 지표 | 실측 결과 | 
| ----- | ----- | 
| **전체 테스트 수 (All tests)** | **35** | 
| **통과 (Passed)** | **35 (100%)** | 
| **실패 / 건너뜀 / 오류** | **0 / 0 / 0** | 
| **총 소요 시간 (Duration)** | **35s 178ms** | 
| **평균 응답 시간 (Avg. Resp. Time)** | **2,825 ms** | 

### 2.1 폴더별 테스트 통과 요약

1. **01 · 무료 / 입력 검증**: `API-00 T-01 Health` (`200 OK`, `db="ready"`, `gpu="cuda:0"`), `API-01 T-08` 빈 질문 · 잘못된 Category · 서버 설정 변경 금지 (`422 VALIDATION_ERROR`, 표준 `error` 래퍼 검증 통과)

2. **02 · Basic 실제 Agent (유료)**: `RAG only`, `DB only`, `T-07/T-09 복합 질문`, `T-10 미존재 ID`, `T-10 No-answer` 모두 `200 OK` 및 `tool_trace`, `validation.citations_match_context`, `validation.required_tools_observed`, 거짓 인용 차단(`sources=[]`) 통과

3. **03 · Intermediate 검색 진단**: `API-02 T-11/T-12 Search A/B` (`200 OK`, `before`/`after` 동일 후보 집합 및 `query_search` 계보 검증 통과)

4. **04 · Advanced 실행 요약**: `API-03 T-19 현재 run` (`200 OK`, `request_id` 및 `metrics` 객체 검증 통과), `API-03 T-19 없는 run` (`404 Not Found`, `error.code == "RUN_NOT_FOUND"` 통과)

### 2.2 대표 실측 응답 분석

#### (1) `API-02` 검색·리랭킹 A/B 비교 (`query="RAG 근거 인용 정확도 향상"`, `category_code="RESEARCH_PLAN"`, `top_k=8`)

* **질의 계보**: `query_original`과 `query_search` 모두 `"RAG 근거 인용 정확도 향상"`으로 보존됨.

* **리랭킹 순위 재정렬 효과**:

  * `report_id=63` (`RAG 근거 인용 정확도 향상 계획`): 1차 벡터 유사도 `0.9213` (1위) → GPU 리랭커 점수 **`5.353`** (1위 확정, 타 후보 대비 압도적 양수 점수 분리)

  * `report_id=69` (`Embedding 언어별 검색 성능 연구`): 1차 벡터 `0.8285` (7위) → 리랭킹 후 **`-8.4776`** (**2위**로 상승)

  * `report_id=82` (`보고서 Overview 자동 생성 연구`): 1차 벡터 `0.8307` (5위) → 리랭킹 후 **`-8.6908`** (**3위**로 상승)

  * 최종 선별 출처: `source_ids = [63, 69, 82]`

#### (2) `API-03` 복합 질의 실행 이력 (`request_id="1451388a-3c68-4f36-a475-86a0b1f43d1f"`)

* **호출 도구**: `rag_search(query="보고서 63 핵심 내용 기술 요약", k=8)` + `lookup_status(report_id=63)` (총 2회 호출)

* **실측 메트릭**: `latency_ms = 9617.24`, `tool_call_count = 2`, `source_count = 1`, `status_report_count = 1`

## 3. 오프라인 검색·리랭킹 품질 평가 (`outputs/quality_report.md`)

`data/evaluation/gold.json` 전체 7건(정답 포함 6건 `G01~G06`, 무응답 검증 1건 `G07`)을 대상으로 오프라인 평가를 수행했습니다.

| 지표 | Before (1차 pgvector) | After (CUDA BGE 리랭커) | 
| ----- | ----- | ----- | 
| **Hit@K** (정답 포함 6건 기준) | **1.0** | **1.0** | 
| **MRR** (정답 포함 6건 기준) | **1.0** | **1.0** | 

| ID | 질의 | 기대 ID | Before Top-K | After Top-K | Before RR | After RR | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **G01** | 클래스 가중치 적용에 따른 소수 클래스 재현율과 Macro-F1을 어떻게 비교하나요? | `[1, 137]` | `[1, 61, 137]` | `[1, 7, 61]` | 1.0 | 1.0 | 
| **G02** | 결측치 대체 시 중앙값 및 통계량 기록 방법은? | `[16, 91]` | `[91, 16, 52]` | `[91, 16, 52]` | 1.0 | 1.0 | 
| **G03** | RAG 답변 근거 인용과 출처 ID 정확도를 개선하려면? | `[63, 131]` | `[63, 131, 122]` | `[63, 131, 122]` | 1.0 | 1.0 | 
| **G04** | 보고서 개요 Embedding 생성 시 정규화, E5 Prefix와 차원은? | `[25, 161]` | `[161, 35, 25]` | `[161, 35, 25]` | 1.0 | 1.0 | 
| **G05** | 벡터 검색 임계값과 검색 품질을 어떻게 검증하나요? | `[64, 152]` | `[64, 152, 149]` | `[152, 64, 29]` | 1.0 | 1.0 | 
| **G06** | CNN 학습률 0.01과 0.001의 수렴 속도를 어떻게 비교하나요? | `[2]` | `[2, 7, 29]` | `[2, 7, 9]` | 1.0 | 1.0 | 
| **G07** | 화성의 낙타 전용 미용실 운영시간을 알려 주세요. | `[]` | `[24, 52, 75]` | `[24, 45, 151]` | 0.0 | 0.0 | 

* **결과 해석**: 모든 정답 포함 케이스(`G01~G06`)에서 최상위 1순위에 정답 보고서가 적중하여 `Hit@K=1.0`, `MRR=1.0`을 기록했습니다. 무응답 테스트(`G07`)는 1차 검색에서 유사도 후보가 반환되더라도 에이전트 프롬프트 및 `validator` 단계에서 최종 `insufficient_evidence`와 `sources=[]`로 안전하게 차단됨을 Postman `T-10 No-answer`로 교차 확인했습니다.

## 4. 주요 트러블슈팅 및 해결 내역

| 발생 이슈 | 원인 | 해결 조치 | 
| ----- | ----- | ----- | 
| **LLM 호출 `400 Unsupported parameter: temperature`** | 사용 모델이 `temperature=0` 파라미터를 미지원 | `app/agent.py`, `app/rag/pipeline.py`의 `ChatOpenAI`에서 `temperature=0` 제거 | 
| **PostgreSQL 컬럼 불일치 (`r.id`, `r.title`, `r.category_code`)** | `reports` 테이블 기본키가 `report_id`이며, 상세 제목·카테고리 코드는 `report_read_view`에 존재 | `repository.py` SQL을 `r.report_id`, `SELECT *`, `report_read_view` 기반 `EXISTS` 필터로 교정 | 
| **빈 질문 입력 시 422 대신 `500 Internal Server Error`** | `RequestValidationError`의 `exc.errors()` 내부에 Python `ValueError` 객체가 남아 JSON 직렬화 실패 | `app/main.py` 예외 핸들러에서 오류 필드를 문자열로 안전 변환 후 `422` 반환 | 
| **Postman 계약 및 에러 래퍼 불일치** | 에러 응답의 `{"error": {"code": ...}}` 래퍼 및 `tool_trace`, `validation`, `metrics`, `before`/`after` 전체 후보 구조 요구 | `ErrorResponse` 래퍼 적용, `runtime.py` · `agent.py` · `api/rag.py` 응답 스키마 정렬 | 
| **근거 존재 시 보수적 `insufficient_evidence` 판정 (`34/35` → `35/35`)** | `RAG_PROMPT`와 `AGENT_SYSTEM_PROMPT`가 일부 세부 수치 누락 시 지나치게 `insufficient_evidence`로 회피 | 관련 보고서 존재 시 조회 범위 내 성실 요약 및 `completed` 반환 지침 명시 | 
| **오프라인 평가 `Hit@K / MRR = 0.0` 출력** | `gold.json`이 `case_id`, `question`, `grades`, `evidence` 키 구조를 사용 | `retrieval_eval.py`에 `case_id` · `grades` · `evidence` 파서 반영 후 `1.0` 실측 확인 | 

## 5. 결론

빈 폴더에서 시작해 FastAPI · LangChain `create_agent` · 읽기 전용 PostgreSQL(`pgvector`) · CUDA GPU(`E5-small` + `BGE-reranker-v2-m3`) · Advanced 실행 이력/디버그 API까지 전 계층을 구현하고, **Postman 라이브 통합 테스트 35/35 전체 통과** 및 **오프라인 평가 Hit@K 1.0 · MRR 1.0**으로 최종 검증을 완료했습니다.