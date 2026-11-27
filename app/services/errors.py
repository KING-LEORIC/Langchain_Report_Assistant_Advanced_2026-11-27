# HTTP 상태 코드 상수 및 표준 열거형 정의를 참조하기 위함
# - HTTPStatus : 매직 넘버 대신 표준 HTTP 상태 코드를 명시적으로 매핑하기 위함
from http import HTTPStatus

# 정적 타입 힌트 및 딕셔너리 구조 명세를 위함
# - Any : 에러 상세 정보(detail)에 다양한 직렬화 가능 객체를 수용하기 위함
from typing import Any

"""
서비스 전역에서 발생하는 도메인 예외와 HTTP 상태 코드 매핑을 관리하는 모듈.
SRS v1.1에 정의된 13개 표준 에러 코드를 단일 진입점으로 통제하여 API 응답 일관성 유지.
"""


# SRS v1.1 표준 에러 코드와 HTTP 상태 코드의 1:1 매핑 테이블 정의
ERROR_STATUS_MAP: dict[str, int] = {
    "VALIDATION_ERROR": HTTPStatus.UNPROCESSABLE_ENTITY,
    "RUN_NOT_FOUND": HTTPStatus.NOT_FOUND,
    "DB_UNAVAILABLE": HTTPStatus.SERVICE_UNAVAILABLE,
    "GPU_UNAVAILABLE": HTTPStatus.SERVICE_UNAVAILABLE,
    "EMBEDDING_UNAVAILABLE": HTTPStatus.SERVICE_UNAVAILABLE,
    "RERANKER_UNAVAILABLE": HTTPStatus.SERVICE_UNAVAILABLE,
    "LLM_UNAVAILABLE": HTTPStatus.SERVICE_UNAVAILABLE,
    "LLM_TIMEOUT": HTTPStatus.GATEWAY_TIMEOUT,
    "TOOL_CONTRACT_ERROR": HTTPStatus.INTERNAL_SERVER_ERROR,
    "OUTPUT_CONTRACT_ERROR": HTTPStatus.INTERNAL_SERVER_ERROR,
    "CONFIG_INVALID": HTTPStatus.SERVICE_UNAVAILABLE,
    "EXECUTION_LIMIT": HTTPStatus.SERVICE_UNAVAILABLE,
    "NOT_IMPLEMENTED": HTTPStatus.INTERNAL_SERVER_ERROR,
}


class ServiceError(Exception):
    """비즈니스 및 인프라 계층의 오류를 표준 HTTP 에러 응답으로 변환하기 위한 공통 예외 클래스."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """에러 코드 검증 및 기본 상태 코드 자동 결정 초기화."""
        # 미등록 에러 코드 유입 시 계약 위반을 조기에 드러내기 위한 방어 분기
        if code not in ERROR_STATUS_MAP:
            raise ValueError(f"등록되지 않은 ServiceError 코드: {code}")

        self.code = code
        self.message = message
        # 호출부가 상태 코드를 생략하면 표준 매핑 테이블 값을 기본 적용
        self.status_code = (
            status_code if status_code is not None else int(ERROR_STATUS_MAP[code])
        )
        self.detail = detail or {}
        super().__init__(message)


"""
[설계 평가]
- 13개 표준 에러 코드와 HTTP 상태 코드를 중앙 테이블로 묶어 계층 간 상태 코드 불일치 방지.
- 미등록 코드 전달 시 즉시 ValueError를 발생시켜 오타나 임의 코드 남용 차단.
"""
