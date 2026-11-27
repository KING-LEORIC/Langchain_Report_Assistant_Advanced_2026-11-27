# 정적 타입 명세를 위함
# - Any : 실행 이력 상세 딕셔너리 반환 타입 명시
from typing import Any

# FastAPI 라우터 및 요청 컨텍스트 접근을 위함
# - APIRouter : 도메인별 엔드포인트를 모듈화하여 main.py에 결합
# - Request : app.state에 주입된 Runtime 및 요청별 request_id 추출
from fastapi import APIRouter, Request

# 최종 Runtime 타입 힌트 명세를 위함
# - AppRuntime : app.state.runtime의 정적 타입 명시
from app.core.runtime import AppRuntime

# 입출력 스키마 검증 및 OpenAPI 문서 자동 생성을 위함
# - AgentAskRequest : POST 본문 자동 파싱 및 422 검증
# - AgentAskResponse : 최종 응답 직렬화 계약
# - HealthResponse : 헬스 체크 응답 모델
from app.schemas.agent import AgentAskRequest, AgentAskResponse, HealthResponse

# 요청 ID 생성 계측을 위함
# - generate_request_id : 미들웨어 누락 시에도 안전한 UUID 확보
from app.services.metrics import generate_request_id

"""
헬스 체크(API-00), 에이전트 질의(API-01), 실행 이력 조회(API-03) 엔드포인트를 제공하는 라우터 모듈.
비즈니스 로직을 직접 처리하지 않고 app.state.runtime에 위임하여 테스트 용이성 확보.
"""

router = APIRouter(tags=["agent"])


def _get_runtime(request: Request) -> AppRuntime:
    """요청 컨텍스트에서 애플리케이션 Runtime 인스턴스 추출."""
    return request.app.state.runtime


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """서버 및 핵심 의존성 가용 상태 조회 엔드포인트."""
    runtime = _get_runtime(request)
    return runtime.health()


@router.post("/api/v1/agent/ask", response_model=AgentAskResponse)
def ask_agent(payload: AgentAskRequest, request: Request) -> AgentAskResponse:
    """사용자 질문을 에이전트에 전달하고 근거 기반 답변 반환."""
    runtime = _get_runtime(request)
    # 예외 핸들러와 동일한 request_id를 공유하기 위해 request.state 우선 참조
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    return runtime.ask(payload=payload, request_id=request_id)


@router.get("/api/v1/agent/runs/{request_id}")
def get_agent_run(request_id: str, request: Request) -> dict[str, Any]:
    """저장된 에이전트 실행 이력을 request_id로 조회하는 Advanced 전용 엔드포인트."""
    runtime = _get_runtime(request)
    return runtime.get_run(request_id)


"""
[설계 평가]
- 엔드포인트 함수를 얇게(Thin Controller) 유지하여 단위 테스트 및 Runtime 교체 비용 최소화.
- 미존재 request_id는 Runtime과 RunStore가 404 RUN_NOT_FOUND로 일관 처리.
"""
