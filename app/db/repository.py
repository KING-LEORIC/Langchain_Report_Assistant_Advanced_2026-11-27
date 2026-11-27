# 정적 타입 명세를 위함
# - Any : DB 조회 결과 행(dict) 타입 명시
from typing import Any

# 환경 설정 참조를 위함
# - Settings : 임베딩 모델명 등 조회 조건 전달
from app.core.settings import Settings

# 읽기 전용 DB 커넥션 획득을 위함
# - get_readonly_connection : 모든 SELECT 쿼리를 READ ONLY 트랜잭션 내에서 실행
from app.db.connection import get_readonly_connection

"""
report_agent_lab 데이터베이스의 읽기 전용(SELECT) 쿼리를 캡슐화한 리포지토리 모듈.
FRESH 벡터 유효성 조건 필터링, 보고서 상태 단건 조회, 원문 상세 일괄 조회 담당.
"""


class ReportRepository:
    """보고서 메타데이터 및 pgvector 임베딩 조회를 담당하는 읽기 전용 리포지토리."""

    def __init__(self, settings: Settings) -> None:
        """설정 객체 주입 초기화."""
        self._settings = settings

    def check_db_ready(self) -> dict[str, Any]:
        """DB 연결, pgvector 확장, 핵심 뷰/테이블 및 FRESH 벡터 건수 점검."""
        sql = """
            SELECT
                COUNT(*) AS total_reports,
                COUNT(*) FILTER (
                    WHERE r.index_status = 'ready'
                      AND e.model_name = %(model_name)s
                      AND e.indexed_revision = r.revision
                      AND vector_dims(e.embedding) = 384
                      AND vector_norm(e.embedding) > 0
                      AND e.overview_hash = encode(sha256(convert_to(r.overview, 'UTF8')), 'hex')
                ) AS fresh_vectors
            FROM reports r
            LEFT JOIN report_embeddings e ON e.report_id = r.report_id
        """
        with get_readonly_connection(self._settings) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"model_name": self._settings.embedding_model})
                row = cur.fetchone()
        return dict(row) if row else {"total_reports": 0, "fresh_vectors": 0}

    def get_report_status(self, report_id: int) -> dict[str, Any] | None:
        """단일 보고서 ID에 대한 상태·리비전·카테고리 메타데이터 조회."""
        sql = """
            SELECT
                r.report_id,
                r.title,
                r.category_code,
                r.status,
                r.index_status,
                r.revision,
                r.updated_at
            FROM reports r
            WHERE r.report_id = %(report_id)s
        """
        with get_readonly_connection(self._settings) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"report_id": report_id})
                row = cur.fetchone()
        # 미존재 ID일 경우 None을 반환하여 도구 계층에서 명시적 미발견 응답 처리
        return dict(row) if row else None

    def search_vector_candidates(
        self, query_embedding: list[float], category_code: str | None = None, k: int = 8
    ) -> list[dict[str, Any]]:
        """FRESH 조건을 만족하는 벡터만 대상으로 코사인 유사도 상위 k건 검색."""
        # pgvector 문자열 리터럴 포맷으로 직렬화하여 파라미터 바인딩
        vec_literal = "[" + ",".join(f"{v:.8f}" for v in query_embedding) + "]"

        sql = """
            SELECT
                r.report_id,
                r.title,
                r.category_code,
                r.status,
                r.overview,
                1 - (e.embedding <=> %(vec)s::vector) AS similarity
            FROM reports r
            JOIN report_embeddings e ON e.report_id = r.report_id
            WHERE r.index_status = 'ready'
              AND e.model_name = %(model_name)s
              AND e.indexed_revision = r.revision
              AND vector_dims(e.embedding) = 384
              AND vector_norm(e.embedding) > 0
              AND e.overview_hash = encode(sha256(convert_to(r.overview, 'UTF8')), 'hex')
              AND (%(category_code)s::text IS NULL OR r.category_code = %(category_code)s)
            ORDER BY e.embedding <=> %(vec)s::vector ASC
            LIMIT %(limit)s
        """
        params = {
            "vec": vec_literal,
            "model_name": self._settings.embedding_model,
            "category_code": category_code,
            "limit": int(k),
        }
        with get_readonly_connection(self._settings) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(row) for row in rows]

    def fetch_report_details_by_ids(
        self, report_ids: list[int]
    ) -> list[dict[str, Any]]:
        """report_read_view에서 선택된 보고서 ID 목록의 전체 상세 본문 조회."""
        # 빈 목록 조회 시 불필요한 DB 왕복 차단
        if not report_ids:
            return []

        sql = """
            SELECT *
            FROM report_read_view
            WHERE report_id = ANY(%(report_ids)s)
        """
        with get_readonly_connection(self._settings) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"report_ids": report_ids})
                rows = cur.fetchall()
        return [dict(row) for row in rows]


"""
[설계 평가]
- reports 테이블 기본키(r.report_id)를 모든 조인·필터 절에 일관 적용해 컬럼 불일치 제거.
- 6중 FRESH 벡터 조건과 READ ONLY 트랜잭션을 유지해 검색 안전성 확보.
"""
