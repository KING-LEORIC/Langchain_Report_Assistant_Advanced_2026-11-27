# 정적 타입 명세를 위함
# - Any : 검색 후보 딕셔너리 리스트 반환 타입 명시
from typing import Any

# 환경 설정 참조를 위함
# - Settings : 기본 RETRIEVE_K 값 참조
from app.core.settings import Settings

# 읽기 전용 DB 리포지토리 참조를 위함
# - ReportRepository : FRESH 벡터 조건이 적용된 pgvector 유사도 검색 수행
from app.db.repository import ReportRepository

# 질의 임베딩 생성을 위함
# - E5Embedder : 질의어를 384차원 벡터로 변환
from app.rag.embeddings import E5Embedder

# Advanced 질의 전처리를 위함
# - QueryOptimizer : 검색 정확도 향상을 위한 질의 정제
from app.rag.query_optimizer import QueryOptimizer

# 카테고리 타입 참조를 위함
# - CategoryCode : 선택적 카테고리 필터링 지원
from app.schemas.agent import CategoryCode

"""
QueryOptimizer · E5Embedder · ReportRepository를 결합해 1차 벡터 후보를 조회하는 리트리버 모듈.
"""


class VectorRetriever:
    """pgvector FRESH 인덱스 기반 1차 보고서 후보 검색기."""

    def __init__(
        self,
        settings: Settings,
        repository: ReportRepository,
        embedder: E5Embedder,
        optimizer: QueryOptimizer | None = None,
    ) -> None:
        """리트리버 의존성 주입 초기화."""
        self._settings = settings
        self._repository = repository
        self._embedder = embedder
        self._optimizer = optimizer or QueryOptimizer()

    def retrieve(
        self,
        query: str,
        category_code: CategoryCode | None = None,
        k: int | None = None,
    ) -> list[dict[str, Any]]:
        """질의를 최적화·임베딩한 뒤 FRESH pgvector 상위 k건 반환."""
        optimized_query = self._optimizer.optimize(query)
        query_vec = self._embedder.embed_query(optimized_query)
        top_k = k if k is not None else self._settings.retrieve_k
        return self._repository.search_vector_candidates(
            query_embedding=query_vec, category_code=category_code, k=top_k
        )


"""
[설계 평가]
- 임베딩, 질의 최적화, DB 검색 책임을 분리하고 오케스트레이션만 담당해 단위 테스트 용이.
- Repository의 FRESH 필터를 그대로 통과하므로 오염된 임베딩이 후보에 섞이지 않음.
"""
