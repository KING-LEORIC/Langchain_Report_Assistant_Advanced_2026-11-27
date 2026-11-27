# 카테고리 코드 리터럴 타입 재사용을 위함
# - CategoryCode : 6개 고정 보고서 카테고리 외의 임의 문자열 입력 차단
# Pydantic 기반 도구 인수 스키마 정의를 위함
# - BaseModel : LangChain @tool의 args_schema로 변환되는 표준 모델
# - Field : LLM이 올바른 인수를 생성하도록 제약 조건과 설명 제공
# - field_validator : 공백 질의어 등 무의미한 도구 호출 사전 차단
from pydantic import BaseModel, Field, field_validator

from app.schemas.agent import CategoryCode

"""
LangChain Agent가 호출하는 2개 도구(rag_search, lookup_status)의 입력 스키마 정의 모듈.
모델이 도구를 선택할 때 정확한 JSON Schema를 제공하여 인수 오류율 최소화.
"""


class RagSearchInput(BaseModel):
    """rag_search 도구의 입력 파라미터 스키마."""

    query: str = Field(
        ..., min_length=1, max_length=500, description="검색할 핵심 질의어"
    )
    category_code: CategoryCode | None = Field(
        default=None, description="특정 보고서 카테고리로 검색 범위를 좁힐 때 지정"
    )
    k: int = Field(
        default=8, ge=1, le=20, description="1차 벡터 검색에서 가져올 후보 보고서 개수"
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_blank(cls, value: str) -> str:
        """공백 문자열로 검색 도구가 낭비 호출되는 것을 방지."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("검색 질의어는 공백일 수 없음")
        return stripped


class LookupStatusInput(BaseModel):
    """lookup_status 도구의 입력 파라미터 스키마."""

    report_id: int = Field(..., ge=1, description="상태를 조회할 보고서의 양의 정수 ID")


"""
[설계 평가]
- ge/le 및 CategoryCode 리터럴 제약을 통해 LLM의 환각 인수 생성 억제.
- 도구 스키마를 별도 모듈로 분리해 Phase 04·05 및 단위 검증에서 독립 재사용 가능.
"""
