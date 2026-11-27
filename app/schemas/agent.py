# 엄격한 문자열 리터럴 제약 및 타입 명세를 위함
# - Any : 도구 호출 로그 및 에러 상세 필드의 유연한 직렬화 지원
# - Literal : 허용된 상태 값(completed, insufficient_evidence 등) 외의 출력 원천 차단
from typing import Any, Literal

# 요청·응답 데이터 검증 및 ProviderStrategy용 스키마 생성을 위함
# - BaseModel : Pydantic v2 기반 자동 직렬화 및 JSON Schema 생성
# - ConfigDict : 정의되지 않은 여분 필드 유입 차단(extra="forbid") 설정
# - Field : 필드별 최소 길이, 기본값, 설명(LLM 스키마 힌트) 부여
# - field_validator : 공백만 입력된 질문 등 비정상 입력 사전 필터링
from pydantic import BaseModel, ConfigDict, Field, field_validator

"""
FastAPI 엔드포인트 입출력 계약과 LangChain Agent의 Structured Output 스키마를 정의하는 모듈.
Advanced 난이도 요구사항에 따라 ModelAnswer를 확장한 StructuredModelAnswer를 포함.
"""


# DB의 고정된 6개 보고서 카테고리 코드 제약
CategoryCode = Literal[
    "EXPERIMENT", "DAILY_WORK", "RESEARCH_PLAN", "DATA_PROCESSING", "PROMPT", "MEETING"
]


class AgentAskRequest(BaseModel):
    """에이전트 질의 API(POST /api/v1/agent/ask)의 요청 본문 스키마."""

    # 스키마에 없는 임의 필드를 거부하여 클라이언트 오타를 422로 즉시 탐지
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ..., min_length=1, max_length=2000, description="사용자 질문 원문"
    )
    category_code: CategoryCode | None = Field(
        default=None,
        description="특정 보고서 카테고리로 검색 범위를 제한할 때 사용하는 선택 코드",
    )

    @field_validator("question")
    @classmethod
    def validate_non_blank_question(cls, value: str) -> str:
        """공백 문자로만 구성된 빈 질문을 거부하기 위한 전처리 검증."""
        stripped = value.strip()
        # 길이 1 이상이더라도 공백만 있으면 무의미한 LLM 호출을 막기 위해 차단
        if not stripped:
            raise ValueError("질문은 공백만으로 구성될 수 없음")
        return stripped


class ModelAnswer(BaseModel):
    """에이전트 최종 답변의 기본 계약 스키마."""

    status: Literal["completed", "insufficient_evidence"] = Field(
        ..., description="근거 충분 여부에 따른 답변 완료 상태"
    )
    answer: str = Field(..., description="근거에 기반한 한국어 최종 답변")
    source_ids: list[int] = Field(
        default_factory=list, description="답변 작성에 인용된 보고서 ID 목록"
    )


class StructuredModelAnswer(ModelAnswer):
    """Advanced 트랙의 ProviderStrategy(Structured Output)용 확장 스키마."""

    status_report_ids: list[int] = Field(
        default_factory=list,
        description="lookup_status 도구로 상태를 직접 조회한 보고서 ID 목록",
    )


class AgentAskResponse(StructuredModelAnswer):
    """에이전트 질의 API의 최종 HTTP 응답 스키마."""

    request_id: str = Field(..., description="요청 추적용 고유 UUID")
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list, description="실행 중 호출된 도구 이름·인수·결과 요약 기록"
    )
    latency_ms: float = Field(..., ge=0.0, description="전체 요청 처리 시간(ms)")


class HealthResponse(BaseModel):
    """서버 상태 확인 API(GET /health)의 응답 스키마."""

    status: Literal["ok", "degraded"] = Field(..., description="서비스 가용 상태")
    level: Literal["basic", "intermediate", "advanced"] = Field(
        default="advanced", description="현재 구현 대상 난이도 식별자"
    )
    checks: dict[str, Any] = Field(
        default_factory=dict, description="세부 구성 요소(DB, GPU, 모델 등) 점검 결과"
    )


class ErrorResponse(BaseModel):
    """전역 예외 핸들러가 반환하는 표준 에러 응답 스키마."""

    code: str = Field(..., description="SRS v1.1 표준 에러 코드")
    message: str = Field(..., description="사용자 및 개발자용 오류 설명")
    request_id: str = Field(..., description="오류 추적용 요청 UUID")
    detail: dict[str, Any] = Field(
        default_factory=dict, description="디버깅에 필요한 부가 컨텍스트"
    )


"""
[설계 평가]
- extra="forbid"와 공백 검증기를 통해 잘못된 입력이 LLM·DB 계층으로 넘어가기 전 422로 차단.
- ModelAnswer → StructuredModelAnswer → AgentAskResponse 상속 구조로 계약 중복 제거.
"""
