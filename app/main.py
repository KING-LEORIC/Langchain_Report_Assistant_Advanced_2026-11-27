# 비동기 컨텍스트 매니저 기반 앱 수명 주기 관리를 위함
# - asynccontextmanager : FastAPI lifespan에서 Runtime 초기화 및 정리 보장
from contextlib import asynccontextmanager

# FastAPI 애플리케이션 인스턴스 및 요청/응답 객체 처리를 위함
# - FastAPI : ASGI 웹 애플리케이션 코어
# - Request : 예외 발생 시 요청 컨텍스트(request_id 등) 참조
from fastapi import FastAPI, Request

# Pydantic 입력 검증 실패 예외 포착을 위함
# - RequestValidationError : 기본 422 포맷을 SRS 표준 ErrorResponse로 변환
from fastapi.exceptions import RequestValidationError

# 일관된 JSON 에러 응답 반환을 위함
# - JSONResponse : 예외 핸들러에서 상태 코드와 직렬화된 JSON 본문 직접 반환
from fastapi.responses import JSONResponse

# 에이전트 및 헬스 체크 라우터 등록을 위함
# - router : /health, /api/v1/agent/ask 엔드포인트 바인딩
from app.api.agent import router as agent_router

# 애플리케이션 Runtime 컨테이너 초기화를 위함
# - AppRuntime : lifespan 시작 시 app.state.runtime에 바인딩
from app.core.runtime import AppRuntime

# 표준 에러 응답 직렬화를 위함
# - ErrorResponse : 422/500/503 등 모든 오류 응답의 단일 포맷 보장
from app.schemas.agent import ErrorResponse

# 서비스 도메인 예외 처리를 위함
# - ServiceError : 비즈니스 및 인프라 오류를 표준 코드로 매핑
from app.services.errors import ServiceError

# 요청별 고유 ID 발급을 위함
# - generate_request_id : 에러 응답에도 추적 가능한 request_id 포함
from app.services.metrics import generate_request_id

"""
FastAPI 애플리케이션 엔트리포인트 모듈.
lifespan 수명 주기 관리, 전역 예외 핸들러(ServiceError·422), 라우터 조립 담당.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작 시 Runtime을 초기화하고 종료 시 리소스 정리."""
    # 전역 변수 대신 app.state에 보관하여 테스트 시 독립된 앱 인스턴스 격리
    app.state.runtime = AppRuntime()
    yield


app = FastAPI(
    title="LangChain Report Assistant (Advanced)", version="0.1.0", lifespan=lifespan
)


@app.exception_handler(ServiceError)
async def handle_service_error(request: Request, exc: ServiceError) -> JSONResponse:
    """도메인 및 인프라 예외(ServiceError)를 SRS v1.1 표준 JSON 응답으로 변환."""
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    body = ErrorResponse(
        code=exc.code, message=exc.message, request_id=request_id, detail=exc.detail
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """FastAPI 기본 422 검증 오류를 표준 VALIDATION_ERROR 포맷으로 통일."""
    request_id = getattr(request.state, "request_id", None) or generate_request_id()
    # 내부 스택 트레이스 노출을 막고 검증 실패 필드 목록만 detail에 포함
    body = ErrorResponse(
        code="VALIDATION_ERROR",
        message="요청 입력값 검증 실패",
        request_id=request_id,
        detail={"errors": exc.errors()},
    )
    return JSONResponse(status_code=422, content=body.model_dump())


app.include_router(agent_router)


"""
[설계 평가]
- 모든 예외가 단일 ErrorResponse 스키마로 수렴하여 클라이언트 파싱 일관성 확보.
- lifespan과 app.state 기반 의존성 주입으로 전역 상태 오염 없이 테스트 실행 가능.
"""
