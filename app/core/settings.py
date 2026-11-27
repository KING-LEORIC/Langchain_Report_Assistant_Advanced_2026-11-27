# 운영체제 환경 변수 접근을 위함
# - environ : .env 로드 후 프로세스 환경 변수에서 설정값 조회
# 불변 설정 객체 정의를 위함
# - dataclass : 런타임 중 설정값이 임의로 변조되는 것을 방지(frozen=True)
from dataclasses import dataclass
from os import environ

# .env 파일 자동 로드를 위함
# - load_dotenv : 로컬 개발 시 별도 export 없이 .env 설정을 환경 변수로 주입
from dotenv import load_dotenv

# 설정 누락 또는 형식 오류를 표준 서비스 예외로 변환하기 위함
# - ServiceError : 503 CONFIG_INVALID 응답 반환
from app.services.errors import ServiceError

"""
애플리케이션 전역 환경 설정을 로드하고 유효성을 검증하는 모듈.
SRS v1.1의 Advanced 난이도 기준 및 모델·타임아웃·예산·디바이스 폴백 파라미터를 불변 객체로 제공.
"""


def _parse_int(name: str, default: int, minimum: int = 1) -> int:
    """정수형 환경 변수를 파싱하고 최소 허용치 검증."""
    raw = environ.get(name, str(default)).strip()
    # 숫자가 아닌 문자열 유입 시 런타임 중간 장애 대신 기동 단계에서 즉시 차단
    try:
        value = int(raw)
    except ValueError as exc:
        raise ServiceError(
            code="CONFIG_INVALID", message=f"환경 변수 {name}는 정수여야 함: {raw}"
        ) from exc

    if value < minimum:
        raise ServiceError(
            code="CONFIG_INVALID",
            message=f"환경 변수 {name}는 {minimum} 이상이어야 함: {value}",
        )
    return value


def _parse_bool(name: str, default: bool) -> bool:
    """불리언 문자열 환경 변수를 안전하게 변환."""
    raw = environ.get(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """애플리케이션 전역에서 공유하는 불변 환경 설정 컨테이너."""

    level: str
    database_url: str
    db_timeout_ms: int
    embedding_model: str
    reranker_model: str
    rerank_batch_size: int
    rerank_max_length: int
    retrieve_k: int
    context_k: int
    context_chars: int
    llm_model: str
    llm_timeout_seconds: int
    max_tool_calls: int
    max_model_calls: int
    langsmith_tracing: bool
    langchain_tracing_v2: bool
    allow_cpu_fallback: bool


def load_settings() -> Settings:
    """환경 변수에서 Settings 인스턴스를 생성하고 필수 값 누락 여부 검증."""
    # 기존 시스템 환경 변수가 있으면 덮어쓰지 않고 유지
    load_dotenv(override=False)

    database_url = environ.get("DATABASE_URL", "").strip()
    # DB 접속 문자열 미설정 시 이후 단계의 모호한 연결 오류를 막기 위해 즉시 차단
    if not database_url:
        raise ServiceError(
            code="CONFIG_INVALID", message="DATABASE_URL 환경 변수가 설정되지 않음"
        )

    llm_model = environ.get("LLM_MODEL", "gpt-4o-mini").strip()
    if not llm_model:
        raise ServiceError(
            code="CONFIG_INVALID", message="LLM_MODEL 환경 변수가 비어 있음"
        )

    retrieve_k = _parse_int("RETRIEVE_K", 8)
    context_k = _parse_int("CONTEXT_K", 3)
    # 최종 컨텍스트 개수가 1차 검색 후보 수보다 크면 리랭킹 의미가 사라지므로 검증
    if context_k > retrieve_k:
        raise ServiceError(
            code="CONFIG_INVALID", message="CONTEXT_K는 RETRIEVE_K보다 클 수 없음"
        )

    return Settings(
        level="advanced",
        database_url=database_url,
        db_timeout_ms=_parse_int("DB_TIMEOUT_MS", 10000, minimum=100),
        embedding_model=environ.get(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
        ).strip(),
        reranker_model=environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip(),
        rerank_batch_size=_parse_int("RERANK_BATCH_SIZE", 2),
        rerank_max_length=_parse_int("RERANK_MAX_LENGTH", 512, minimum=64),
        retrieve_k=retrieve_k,
        context_k=context_k,
        context_chars=_parse_int("CONTEXT_CHARS", 9000, minimum=500),
        llm_model=llm_model,
        llm_timeout_seconds=_parse_int("LLM_TIMEOUT_SECONDS", 30, minimum=5),
        max_tool_calls=_parse_int("MAX_TOOL_CALLS", 4),
        max_model_calls=_parse_int("MAX_MODEL_CALLS", 14),
        langsmith_tracing=_parse_bool("LANGSMITH_TRACING", False),
        langchain_tracing_v2=_parse_bool("LANGCHAIN_TRACING_V2", False),
        allow_cpu_fallback=_parse_bool("ALLOW_CPU_FALLBACK", False),
    )


"""
[설계 평가]
- allow_cpu_fallback 기본값을 False로 두어 별도 설정 없이는 기존 CUDA 필수 정책 유지.
- frozen=True 데이터클래스로 런타임 설정 변조 원천 차단.
"""
