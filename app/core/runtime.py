# 표준화된 미구현 오류 및 응답 스키마 참조를 위함
# - AgentAskRequest : 요청 타입 시그니처 일관성 유지
# - AgentAskResponse : 최종 Runtime 전환 시 인터페이스 호환성 보장
# - HealthResponse : 헬스 체크 응답 모델 반환
from app.schemas.agent import AgentAskRequest, AgentAskResponse, HealthResponse

# Phase 01 단계에서 미구현 엔드포인트 호출을 표준 예외로 알리기 위함
# - ServiceError : 500 NOT_IMPLEMENTED 에러 응답 생성
from app.services.errors import ServiceError

"""
Phase 01 골격 검증용 임시 Runtime(1/3) 모듈.
DB·RAG·Agent 연결 전 단계에서 FastAPI 라우터와 예외 핸들러의 정상 동작 확인 목적.
"""


class AppRuntime:
    """애플리케이션 수명 주기 동안 공유되는 의존성 컨테이너 스텁."""

    def __init__(self) -> None:
        """Phase 01 임시 상태 초기화."""
        self.level: str = "advanced"

    def health(self) -> HealthResponse:
        """서버 기동 여부만 확인하는 기본 헬스 체크 응답 반환."""
        # Phase 02에서 DB·GPU 프리플라이트 결과 연동 예정
        return HealthResponse(
            status="ok", level="advanced", checks={"phase": "01_skeleton"}
        )

    def ask(self, payload: AgentAskRequest, request_id: str) -> AgentAskResponse:
        """에이전트 미연결 상태에서 명시적 NOT_IMPLEMENTED 예외 발생."""
        # 빈 200 응답이나 무음 실패를 방지하고 계약된 500 에러 반환
        raise ServiceError(
            code="NOT_IMPLEMENTED",
            message="Agent 파이프라인이 아직 연결되지 않음 (Phase 01 스텁)",
            detail={"request_id": request_id, "question_length": len(payload.question)},
        )


"""
[설계 평가]
- 라우터가 직접 비즈니스 로직을 알지 않고 Runtime 인터페이스에만 의존하도록 분리(DI).
- Phase 02·05에서 동일 메서드 시그니처를 유지한 채 내부 구현만 점진적으로 교체 가능.
"""
