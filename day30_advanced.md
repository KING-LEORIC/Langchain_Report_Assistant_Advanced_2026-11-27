# LangChain AI 기술 보고서 에이전트 (Advanced) 구현 및 검증 결과 보고서

| 항목 | 내용 | 
| ----- | ----- | 
| **프로젝트명** | LangChain Report Assistant (`03_advanced`) | 
| **작성일** | 2026-11-28 | 
| **구현 난이도** | Advanced (빈 폴더 기반 풀스택 파이프라인 구축) | 
| **작성자** | Master Kim | 
| **핵심 스택** | Python, FastAPI, LangChain (`create_agent`, `@tool`, `ProviderStrategy`, LCEL), PostgreSQL 18/19 + `pgvector`, `E5-small` + `BGE-reranker-v2-m3` | 

## 1. 프로젝트 개요 및 아키텍처

본 프로젝트는 `report_agent_lab` PostgreSQL 데이터베이스를 읽기 전용(`SET TRANSACTION READ ONLY`)으로 연동하여, 기술 보고서 본문 검색(RAG)과 보고서 상태 조회(DB)를 결합해 근거 기반 답변을 생성하는 **Advanced 등급 AI 에이전트 서버**입니다.

### 1.1 핵심 설계 원칙

* **무결성 및 읽기 전용 잠금**: 모든 DB 세션에 `SET TRANSACTION READ ONLY`와 `statement_timeout(DB_TIMEOUT_MS)`을 강제하고, `reports`·`report_embeddings` 조인 시 6중 `FRESH` 벡터 유효성 조건(`index_status='ready'`, `model_name`, `indexed_revision`, `vector_dims=384`, `vector_norm>0`, `overview_hash`)을 적용했습니다.

* **CUDA 우선 및 조건부 CPU 이식성 (`ALLOW_CPU_FALLBACK`)**: 기본값(`false`)에서는 `cuda:0`를 강제하여 CPU 폴백을 금지(`503 GPU_UNAVAILABLE`)하고, 비-CUDA 환경(`AMD Radeon(TM) 780M (4 GB)`)에서는 `.env`의 `ALLOW_CPU_FALLBACK=true` 설정만으로 코드 수정 없이 동작하도록 설계했습니다.

* **하이브리드 ID 보강 검색 (`VectorRetriever`)**: 순수 의미 임베딩이 `"보고서 63"` 같은 숫자 ID를 놓치는 한계를 보완하기 위해 `index_status == 'ready'`인 명시 ID를 1차 후보에 보강하고 리랭킹 후 `CONTEXT_K` 내에 보존했습니다.

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

전체 컬렉션(`Day30 Report Agent - SRS v1.0`, `project_level=advanced`, `allow_paid_llm=true`) 실행 결과 **두 하드웨어 환경 모두 35개 테스트 항목 100% 통과**를 달성했습니다.

| 지표 | RTX 5080 Laptop GPU (CUDA) | AMD Ryzen 7 PRO 7840U (CPU) | 
| ----- | ----- | ----- | 
| **전체 테스트 수 (All tests)** | **35** | **35** | 
| **통과 (Passed) / 실패 / 오류** | **35 (100%) / 0 / 0** | **35 (100%) / 0 / 0** | 
| **총 소요 시간 (Duration)** | **35s 178ms** | **37s 327ms** | 
| **평균 응답 시간 (Avg. Resp. Time)** | **2,825 ms** | **2,993 ms** | 

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

`data/evaluation/gold.json` 전체 7건(정답 포함 6건 `G01~G06`, 무응답 검증 1건 `G07`)을 대상으로 오프라인 평가를 수행했으며, **RTX 5080 Laptop GPU (CUDA) 환경과 AMD Ryzen 7 PRO 7840U (CPU) 환경에서 100% 동일한 지표와 Top-K 순위가 재현**되었습니다.

| 지표 | Before (1차 pgvector) | After (BGE 리랭커) | 
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

## 4. 이종 하드웨어 환경 비교 분석 (RTX 5080 Laptop GPU vs Ryzen 7 PRO 7840U CPU)

동일한 코드베이스와 180건 벡터화 DB를 두 환경에서 교차 실행해 성능 및 이식성을 검증했습니다.

* **환경 A (CUDA GPU)**: 인텔 코어 울트라 9 275HX (24코어, 최대 5.4GHz), **NVIDIA GeForce RTX 5080 Laptop GPU (16GB GDDR7, 최대 TGP 175W)**, 32GB DDR5-5600MHz (`ALLOW_CPU_FALLBACK=false`)

* **환경 B (비-CUDA CPU 폴백)**: **AMD Ryzen 7 PRO 7840U w/ Radeon 780M Graphics (3.30 GHz)**, **AMD Radeon(TM) 780M (4 GB)**, 32GB, Windows 11 26H2 (`ALLOW_CPU_FALLBACK=true`)

| 비교 항목 | 환경 A (RTX 5080 Laptop GPU) | 환경 B (Ryzen 7 PRO 7840U CPU) | 차이 및 분석 | 
| ----- | ----- | ----- | ----- | 
| **연산 장치** | `cuda:0` | `cpu` (`ALLOW_CPU_FALLBACK=true`) | Windows 780M(4 GB)의 ROCm 미지원 및 DirectML 연산 호환 이슈 회피 | 
| **`API-02` 순수 검색·리랭킹 시간** | **439 \~ 591 ms** | **1,822 ms** | LLM 제외 순수 E5 + BGE-reranker 구간은 RTX 5080 GPU가 **약 3.1\~4.2배** 빠름 | 
| **전체 35개 Runner 소요 시간** | **35s 178ms** | **37s 327ms** | 외부 LLM API 대기 시간이 전체를 지배하여 총 체감 차이는 **+2.1초(+6.1%)** | 
| **요청당 평균 응답 시간** | **2,825 ms** | **2,993 ms** | 평균 **+168 ms (+5.9%)** 차이로 로컬 개발·검증에 충분한 응답성 확보 | 
| **검색·리랭킹 정확도 (`Hit@K` / `MRR`)** | **1.0 / 1.0** | **1.0 / 1.0** | 하드웨어가 달라져도 Top-K 순위와 검색 품질 **100% 동일 재현** | 

## 5. 주요 트러블슈팅 및 해결 내역

| 발생 이슈 | 원인 | 해결 조치 | 
| ----- | ----- | ----- | 
| **LLM 호출 `400 Unsupported parameter: temperature`** | 사용 모델이 `temperature=0` 파라미터를 미지원 | `app/agent.py`, `app/rag/pipeline.py`의 `ChatOpenAI`에서 `temperature=0` 제거 | 
| **PostgreSQL 컬럼 불일치 (`r.id`, `r.title`, `r.category_code`)** | `reports` 테이블 기본키가 `report_id`이며, 상세 제목·카테고리 코드는 `report_read_view`에 존재 | `repository.py` SQL을 `r.report_id`, `SELECT *`, `report_read_view` 기반 `EXISTS` 필터로 교정 | 
| **빈 질문 입력 시 422 대신 `500 Internal Server Error`** | `RequestValidationError`의 `exc.errors()` 내부에 Python `ValueError` 객체가 남아 JSON 직렬화 실패 | `app/main.py` 예외 핸들러에서 오류 필드를 문자열로 안전 변환 후 `422` 반환 | 
| **Postman 계약 및 에러 래퍼 불일치** | 에러 응답의 `{"error": {"code": ...}}` 래퍼 및 `tool_trace`, `validation`, `metrics`, `before`/`after` 전체 후보 구조 요구 | `ErrorResponse` 래퍼 적용, `runtime.py` · `agent.py` · `api/rag.py` 응답 스키마 정렬 | 
| **비-CUDA 환경 기동 실패 (`503 GPU_UNAVAILABLE`)** | `AMD Ryzen 7 PRO 7840U / Radeon(TM) 780M (4 GB)` 환경에 NVIDIA CUDA 부재 | `.env`의 `ALLOW_CPU_FALLBACK` 플래그 도입으로 CUDA 강제와 로컬 CPU 호환 양립 | 
| **내용 질의 시 `lookup_status` 단독 호출 및 숫자 ID 누락** | 모델이 보고서 번호만 보고 상태 조회만 호출하거나, E5 임베딩이 `"63"`을 ID로 매칭하지 못함 | `AGENT_SYSTEM_PROMPT` 규칙 강화 및 `ready` 상태 보고서 대상 하이브리드 ID 보강 적용 | 
| **오프라인 평가 `Hit@K / MRR = 0.0` 출력** | `gold.json`이 `case_id`, `question`, `grades`, `evidence` 키 구조를 사용 | `retrieval_eval.py`에 `case_id` · `grades` · `evidence` 파서 반영 후 `1.0` 실측 확인 | 

## 6. 결론

빈 폴더에서 시작해 FastAPI · LangChain `create_agent` · 읽기 전용 PostgreSQL(`pgvector`) · `E5-small` + `BGE-reranker-v2-m3` · Advanced 실행 이력/디버그 API까지 전 계층을 구현했습니다. **RTX 5080 Laptop GPU (CUDA) 환경과 AMD Ryzen 7 PRO 7840U (CPU) 환경 모두에서 Postman 35/35 전체 통과** 및 **오프라인 평가 Hit@K 1.0 · MRR 1.0**을 달성해 기능 정확도와 이종 환경 이식성을 모두 입증했습니다.