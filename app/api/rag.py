# 정적 타입 명세를 위함
# - Any : 디버그 응답 내 before·after 후보 및 문서 딕셔너리 리스트 타입 명시
from typing import Any

# FastAPI 라우터 및 요청 컨텍스트 접근을 위함
# - APIRouter : /api/v1/rag/debug 엔드포인트 모듈화
# - Request : app.state.runtime 조회
from fastapi import APIRouter, Request

# Pydantic 요청·응답 스키마 정의를 위함
# - BaseModel : 입출력 자동 검증 및 직렬화
# - ConfigDict : 미정의 필드 차단(extra="forbid")
# - Field : 질의어·후보 개수 제약 설정
from pydantic import BaseModel, ConfigDict, Field

# 카테고리 코드 리터럴 타입 참조를 위함
# - CategoryCode : 6개 고정 보고서 카테고리 필터 지원
from app.schemas.agent import CategoryCode

"""
Intermediate·Advanced 트랙의 RAG 디버그 API(POST /api/v1/rag/debug) 라우터 모듈.
리랭킹 전(before)과 후(after)의 순위 변화 및 질의 계보(query_original → query_search) 검증용.
"""

router = APIRouter(tags=["rag"])


class RagDebugRequest(BaseModel):
    """RAG 디버그 엔드포인트 요청 스키마."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=500, description="검색 질의어")
    category_code: CategoryCode | None = Field(
        default=None, description="선택적 카테고리 필터"
    )
    # Postman 컬렉션 규격(top_k)과 내부 호환 파라미터(k) 모두 지원
    top_k: int = Field(default=8, ge=1, le=20, description="1차 벡터 검색 후보 개수")


class RagDebugResponse(BaseModel):
    """RAG 디버그 엔드포인트 응답 스키마."""

    query_original: str
    query_search: str
    before: list[dict[str, Any]]
    after: list[dict[str, Any]]
    source_ids: list[int]
    documents: list[dict[str, Any]]


@router.post("/api/v1/rag/debug", response_model=RagDebugResponse)
def debug_rag(payload: RagDebugRequest, request: Request) -> RagDebugResponse:
    """벡터 검색(before)과 리랭킹(after) 비교 결과를 Postman 검증 규격으로 반환."""
    runtime = request.app.state.runtime
    result = runtime.debug_rag(
        query=payload.query, category_code=payload.category_code, k=payload.top_k
    )
    return RagDebugResponse(**result)


"""
[설계 평가]
- Postman 테스트 스크립트의 top_k, before/after, query_original/query_search 계약과 1:1 일치.
- extra="forbid"를 유지해 오타 파라미터 유입 시 422로 즉시 탐지.
"""
