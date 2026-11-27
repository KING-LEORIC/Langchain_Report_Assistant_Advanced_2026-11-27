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

# CUDA 임베딩 인코더 참조를 위함
# - E5Embedder : cuda:0 기반 E5-small 모델 로드
from app.rag.embeddings import E5Embedder

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
# - HealthResponse : 헬스 체크 응답 모델
from app.schemas.agent import AgentAskRequest, AgentAskResponse, HealthResponse

# 요청 지연 시간 계측을 위함
# - LatencyTimer : 요청 단위 처리 시간(ms) 측정
from app.services.metrics import LatencyTimer

# 도구 팩토리 함수 참조를 위함
# - create_lookup_status_tool : DB 상태 조회 도구 생성
# - create_rag_search_tool : RAG 검색 도구 생성
from app.tools.db_tool import create_lookup_status_tool
from app.tools.rag_tool import create_rag_search_tool

"""
애플리케이션의 모든 핵심 의존성(Settings, DB, GPU 모델, RAG, Agent)을
단일 수명 주기로 조립해 제공하는 최종 Runtime(3/3) 모듈.
"""


class AppRuntime:
    """FastAPI app.state.runtime에 바인딩되는 중앙 의존성 컨테이너."""

    def __init__(self, settings: Settings | None = None) -> None:
        """설정 로드 후 DB 리포지토리, CUDA 모델, RAG, Agent를 순차 초기화."""
        self.settings: Settings = settings or load_settings()
        self.level: str = self.settings.level
        self.repository = ReportRepository(self.settings)

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
        """사용자 질문을 AgentRunner로 실행하고 지연 시간과 함께 최종 응답 반환."""
        timer = LatencyTimer()
        structured, tool_calls_log = self.agent_runner.run(
            question=payload.question, category_code=payload.category_code
        )
        return AgentAskResponse(
            request_id=request_id,
            status=structured.status,
            answer=structured.answer,
            source_ids=structured.source_ids,
            status_report_ids=structured.status_report_ids,
            tool_calls=tool_calls_log,
            latency_ms=timer.elapsed_ms(),
        )


"""
[설계 평가]
- 기동 시 모델 1회 적재(Singleton DI)로 매 요청마다 발생하는 CUDA 초기화 병목 제거.
- 라우터 코드를 전혀 수정하지 않고 Phase 01 스텁과 100% 동일한 인터페이스로 교체 완료.
"""
