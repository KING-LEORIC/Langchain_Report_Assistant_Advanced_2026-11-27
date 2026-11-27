# 정적 타입 명세를 위함
# - Any : report_read_view 행 딕셔너리 타입 명시
from typing import Any

# 환경 설정 참조를 위함
# - Settings : CONTEXT_CHARS 최대 문자 수 제한 참조
from app.core.settings import Settings

# 읽기 전용 DB 리포지토리 참조를 위함
# - ReportRepository : report_read_view 통합 본문 조회
from app.db.repository import ReportRepository

# ID 추출 유틸리티 참조를 위함
# - extract_unique_report_ids : 후보 목록에서 중복 없는 보고서 ID 추출
from app.rag.ids import extract_unique_report_ids

"""
리랭킹된 보고서 ID 목록을 바탕으로 report_read_view의 전체 본문을 조회하고
LLM 컨텍스트 예산(CONTEXT_CHARS) 이내로 안전하게 조립하는 모듈.
"""


def build_context_documents(
    ranked_candidates: list[dict[str, Any]],
    repository: ReportRepository,
    settings: Settings,
) -> list[dict[str, Any]]:
    """리랭킹 순서를 보존하면서 상세 본문을 조회하고 CONTEXT_CHARS 한도 내로 절단."""
    ordered_ids = extract_unique_report_ids(ranked_candidates)
    if not ordered_ids:
        return []

    detail_rows = repository.fetch_report_details_by_ids(ordered_ids)
    # DB의 ANY() 조회는 입력 ID 순서를 보장하지 않으므로 ID 기준 맵으로 재정렬
    detail_by_id = {int(row["report_id"]): row for row in detail_rows}

    documents: list[dict[str, Any]] = []
    remaining_chars = settings.context_chars

    for candidate in ranked_candidates:
        rid = int(candidate["report_id"])
        row = detail_by_id.get(rid)
        if row is None:
            continue

        # 본문과 요약을 결합해 리랭커 점수 및 LLM 근거로 활용 가능한 단일 텍스트 블록 구성
        body_text = (
            f"[보고서 ID: {rid}] 제목: {row.get('title', '')} | "
            f"카테고리: {row.get('category_code', '')} | 상태: {row.get('status', '')}\n"
            f"요약: {row.get('overview', '')}\n"
            f"상세: {row.get('body', row.get('content', ''))}"
        ).strip()

        # 전체 컨텍스트 예산 소진 시 더 이상 문서를 추가하지 않아 토큰 초과 방지
        if remaining_chars <= 0:
            break

        if len(body_text) > remaining_chars:
            body_text = body_text[:remaining_chars]

        remaining_chars -= len(body_text)
        documents.append(
            {
                "report_id": rid,
                "title": row.get("title", ""),
                "category_code": row.get("category_code", ""),
                "status": row.get("status", ""),
                "rerank_score": candidate.get("rerank_score"),
                "similarity": candidate.get("similarity"),
                "text": body_text,
            }
        )

    return documents


"""
[설계 평가]
- PostgreSQL ANY() 쿼리의 비결정적 정렬 문제를 Python 맵핑으로 복원해 리랭킹 순위 보존.
- CONTEXT_CHARS 하드 캡 적용으로 대용량 보고서 유입 시 LLM 컨텍스트 창 초과 방지.
"""
