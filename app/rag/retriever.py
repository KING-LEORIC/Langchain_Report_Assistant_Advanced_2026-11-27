# 질의어 내 명시적 보고서 ID 추출을 위함
# - IGNORECASE : 대소문자 구분 없이 report/ID 패턴 매칭
# - findall : '보고서 63' 등 지정된 ID를 놓치지 않고 후보에 포함
from re import IGNORECASE, findall

# 정적 타입 명세를 위함
# - Any : 검색 후보 딕셔너리 리스트 반환 타입 명시
from typing import Any

# 환경 설정 참조를 위함
# - Settings : 기본 RETRIEVE_K 값 참조
from app.core.settings import Settings

# 읽기 전용 DB 리포지토리 참조를 위함
# - ReportRepository : FRESH 벡터 검색 및 명시 ID 단건 보강 조회 수행
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
의미 기반 벡터 검색과 함께 질의에 명시된 보고서 ID를 보조 결합하여 재현율 보장.
"""


class VectorRetriever:
    """pgvector FRESH 인덱스 기반 1차 보고서 후보 검색기."""

    _EXPLICIT_ID_PATTERN = r"(?:보고서|report|id)\s*#?\s*(\d+)"

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
        """질의를 최적화·임베딩한 뒤 FRESH pgvector 상위 k건과 명시 ID 후보 반환."""
        optimized_query = self._optimizer.optimize(query)
        query_vec = self._embedder.embed_query(optimized_query)
        top_k = k if k is not None else self._settings.retrieve_k
        candidates = self._repository.search_vector_candidates(
            query_embedding=query_vec, category_code=category_code, k=top_k
        )

        # 벡터 모델이 '63' 같은 숫자 ID를 놓치거나 리랭커에서 밀리는 한계를 보완
        candidates_by_id = {int(c["report_id"]): c for c in candidates}
        explicit_ids = [
            int(m) for m in findall(self._EXPLICIT_ID_PATTERN, query, flags=IGNORECASE)
        ]
        for rid in explicit_ids:
            if rid in candidates_by_id:
                # 1차 후보에 이미 있더라도 리랭킹 후 탈락하지 않도록 explicit_match 표시
                candidates_by_id[rid]["explicit_match"] = True
                continue
            row = self._repository.get_report_status(rid)
            # ready 상태인 실제 보고서만 후보에 추가해 미존재 ID(999999)나 미색인 문서 유입 차단
            if row is not None and row.get("index_status") == "ready":
                new_candidate = {
                    "report_id": int(row["report_id"]),
                    "overview": row.get("overview", ""),
                    "similarity": 1.0,
                    "explicit_match": True,
                }
                candidates.insert(0, new_candidate)
                candidates_by_id[rid] = new_candidate

        return candidates


"""
[설계 평가]
- 순수 벡터 검색의 약점인 숫자 ID 매칭 누락과 리랭커 탈락을 explicit_match 플래그로 해결.
- ready 상태 검증을 거쳐 미존재 ID나 미색인 보고서가 후보에 섞이지 않도록 안전하게 통제.
"""
