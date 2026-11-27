# 정적 타입 명세를 위함
# - Any : 디버그 및 실행 이력 반환 딕셔너리 타입 명시
from typing import Any

# 에이전트 실행기 참조를 위함
# - ReportAgentRunner : create_agent 기반 질의 처리 수행
from app.agent import ReportAgentRunner

# 프리플라이트 점검 함수 참조를 위함
# - run_preflight : /health 요청 시 GPU 및 DB 상태 실시간 검증
from app.core.preflight import run_preflight

# 환경 설정 로더 참조를 위함
# - load_settings : .env 기반 Advanced 설정 로드
from app.core.settings import Settings, load_settings

# 읽기 전용 DB 리포지토리 참조를 위함
# - ReportRepository : 상태 조회 및 pgvector 검색 수행
from app.db.repository import ReportRepository

# 문서 조립 유틸리티 참조를 위함
# - build_context_documents : 디버그 API에서 최종 컨텍스트 블록 확인
from app.rag.documents import build_context_documents

# CUDA 임베딩 인코더 참조를 위함
# - E5Embedder : cuda:0 기반 E5-small 모델 로드
from app.rag.embeddings import E5Embedder

# 고유 ID 추출 유틸리티 참조를 위함
# - extract_unique_report_ids : 디버그 결과 출처 ID 정리
from app.rag.ids import extract_unique_report_ids

# RAG 파이프라인 참조를 위함
# - RAGPipeline : 검색·리랭킹·LCEL 요약 실행
from app.rag.pipeline import RAGPipeline

# 질의 최적화기 참조를 위함
# - QueryOptimizer : Advanced 질의 정규화 적용
from app.rag.query_optimizer import QueryOptimizer

# CUDA 리랭커 참조를 위함
# - BgeReranker : cuda:0 기반 bge-reranker-v2-m3 모델 로드
from app.rag.reranker import BgeReranker

# 1차 벡터 검색기 참조를 위함
# - VectorRetriever : FRESH 조건 pgvector 후보 검색
from app.rag.retriever import VectorRetriever

# 입출력 스키마 참조를 위함
# - AgentAskRequest : 요청 페이로드 타입
# - AgentAskResponse : 최종 응답 스키마 조립
# - CategoryCode : 디버그 카테고리 필터 타입
# - HealthResponse : 헬스 체크 응답 모델
from app.schemas.agent import (
    AgentAskRequest,
    AgentAskResponse,
    CategoryCode,
    HealthResponse,
)

# 요청 지연 시간 계측을 위함
# - LatencyTimer : 요청 단위 처리 시간(ms) 측정
from app.services.metrics import LatencyTimer

# 인메모리 실행 이력 저장소 참조를 위함
# - RunStore : API-03 조회를 위한 최근 100건 실행 기록 보관
from app.services.run_store import RunStore

# 도구 팩토리 함수 참조를 위함
# - create_lookup_status_tool : DB 상태 조회 도구 생성
# - create_rag_search_tool : RAG 검색 도구 생성
from app.tools.db_tool import create_lookup_status_tool
from app.tools.rag_tool import create_rag_search_tool

"""
애플리케이션의 모든 핵심 의존성(Settings, DB, GPU 모델, RAG, Agent, RunStore)을
단일 수명 주기로 조립해 제공하는 최종 Advanced Runtime 모듈.
"""


class AppRuntime:
    """FastAPI app.state.runtime에 바인딩되는 중앙 의존성 컨테이너."""

    def __init__(self, settings: Settings | None = None) -> None:
        """설정 로드 후 DB 리포지토리, CUDA 모델, RAG, Agent, RunStore 순차 초기화."""
        self.settings: Settings = settings or load_settings()
        self.level: str = self.settings.level
        self.repository = ReportRepository(self.settings)
        self.run_store = RunStore(max_runs=100)

        # 무거운 GPU 모델은 요청마다가 아니라 앱 기동 시 1회만 적재하여 응답 지연 최소화
        self.embedder = E5Embedder(self.settings)
        self.reranker = BgeReranker(self.settings)
        self.optimizer = QueryOptimizer()
        self.retriever = VectorRetriever(
            settings=self.settings,
            repository=self.repository,
            embedder=self.embedder,
            optimizer=self.optimizer,
        )
        self.rag_pipeline = RAGPipeline(
            settings=self.settings,
            repository=self.repository,
            retriever=self.retriever,
            reranker=self.reranker,
        )

        tools = [
            create_rag_search_tool(self.rag_pipeline),
            create_lookup_status_tool(self.repository),
        ]
        self.agent_runner = ReportAgentRunner(settings=self.settings, tools=tools)

    def health(self) -> HealthResponse:
        """CUDA GPU 및 PostgreSQL 프리플라이트 상태를 점검해 헬스 응답 반환."""
        checks = run_preflight(self.settings)
        return HealthResponse(status="ok", level="advanced", checks=checks)

    def ask(self, payload: AgentAskRequest, request_id: str) -> AgentAskResponse:
        """사용자 질문을 AgentRunner로 실행하고 결과를 RunStore에 저장 후 반환."""
        timer = LatencyTimer()
        structured, tool_calls_log = self.agent_runner.run(
            question=payload.question, category_code=payload.category_code
        )
        response = AgentAskResponse(
            request_id=request_id,
            status=structured.status,
            answer=structured.answer,
            source_ids=structured.source_ids,
            status_report_ids=structured.status_report_ids,
            tool_calls=tool_calls_log,
            latency_ms=timer.elapsed_ms(),
        )
        # API-03 이력 조회를 위해 정상 완료된 요청 결과를 인메모리 저장소에 기록
        self.run_store.save(
            request_id,
            {
                "question": payload.question,
                "category_code": payload.category_code,
                **response.model_dump(),
            },
        )
        return response

    def get_run(self, request_id: str) -> dict[str, Any]:
        """저장된 실행 이력 단건 조회."""
        return self.run_store.get(request_id)

    def debug_rag(
        self, query: str, category_code: CategoryCode | None = None, k: int = 8
    ) -> dict[str, Any]:
        """리랭킹 전(before)·후(after) 전체 후보 비교 및 질의 계보 반환."""
        optimized = self.optimizer.optimize(query)
        # lineage 검증(query_search가 query_original 포함)을 보장하기 위해 원문 보존
        query_search = optimized if query in optimized else query

        candidates = self.retriever.retrieve(
            query=query_search, category_code=category_code, k=k
        )
        # before와 after의 report_id 집합이 동일하도록 잘라내지 않고 전체 후보를 리랭킹
        ranked_all = (
            self.reranker.rerank(
                query=query_search, candidates=candidates, top_k=len(candidates)
            )
            if candidates
            else []
        )
        top_docs = (
            build_context_documents(
                ranked_candidates=ranked_all[: self.settings.context_k],
                repository=self.repository,
                settings=self.settings,
            )
            if ranked_all
            else []
        )
        return {
            "query_original": query,
            "query_search": query_search,
            "before": candidates,
            "after": ranked_all,
            "source_ids": extract_unique_report_ids(top_docs),
            "documents": top_docs,
        }


"""
[설계 평가]
- ask() 실행 결과를 RunStore에 자동 적재해 별도 DB 쓰기 없이 API-03 조회 기능 완성.
- debug_rag()가 기존 retriever·reranker 인스턴스를 그대로 재사용해 메모리 낭비 차단.
"""
