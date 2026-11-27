# 제너레이터 기반 컨텍스트 매니저 작성을 위함
# - contextmanager : 커넥션 및 트랜잭션의 안전한 획득·반환 보장
# 반환 타입 힌트 명세를 위함
# - Iterator : with 블록에 전달되는 커넥션 객체 타입 명시
from collections.abc import Iterator
from contextlib import contextmanager

# PostgreSQL 18 및 pgvector 쿼리 실행을 위함
# - connect : psycopg3 네이티브 DB 커넥션 생성
# - Connection : 커넥션 객체 정적 타입 힌트
# - Error : DB 접속·타임아웃·SQL 실행 예외 포착
from psycopg import Connection, Error, connect

# 딕셔너리 형태 행 반환을 위함
# - dict_row : 컬럼명 기반 접근으로 인덱스 오타 방지
from psycopg.rows import dict_row

# 설정 객체 참조를 위함
# - Settings : DATABASE_URL 및 DB_TIMEOUT_MS 조회
from app.core.settings import Settings

# DB 장애를 표준 서비스 예외로 변환하기 위함
# - ServiceError : 503 DB_UNAVAILABLE 응답 반환
from app.services.errors import ServiceError

"""
PostgreSQL(report_agent_lab) 읽기 전용 커넥션을 제공하는 모듈.
모든 세션에 SET TRANSACTION READ ONLY와 statement_timeout을 강제하여 쓰기 및 장기 락 방지.
"""


def _normalize_dsn(database_url: str) -> str:
    """SQLAlchemy 형식 접두사(+psycopg)를 psycopg3 표준 DSN으로 정규화."""
    # .env의 postgresql+psycopg:// 형식을 psycopg.connect()가 인식하도록 변환
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return database_url


@contextmanager
def get_readonly_connection(settings: Settings) -> Iterator[Connection]:
    """읽기 전용 트랜잭션과 타임아웃이 적용된 PostgreSQL 커넥션 컨텍스트 제공."""
    dsn = _normalize_dsn(settings.database_url)
    conn: Connection | None = None
    try:
        conn = connect(dsn, row_factory=dict_row, connect_timeout=5)
        with conn.cursor() as cur:
            # 어떠한 DDL/DML도 실행되지 않도록 트랜잭션 수준에서 읽기 전용 잠금
            cur.execute("SET TRANSACTION READ ONLY")
            # 슬로우 쿼리로 인한 서버 스레드 고갈을 막기 위해 문장 타임아웃 강제
            cur.execute(f"SET LOCAL statement_timeout = {int(settings.db_timeout_ms)}")
        yield conn
        # 읽기 전용 트랜잭션 정상 종료
        conn.rollback()
    except Error as exc:
        if conn is not None:
            conn.rollback()
        # 내부 DB 드라이버 예외 메시지를 감추고 표준 503 에러로 변환
        raise ServiceError(
            code="DB_UNAVAILABLE",
            message="PostgreSQL 읽기 전용 연결 또는 쿼리 실행 실패",
            detail={"reason": str(exc)},
        ) from exc
    finally:
        if conn is not None:
            conn.close()


"""
[설계 평가]
- SET TRANSACTION READ ONLY 강제로 실수로라도 INSERT/UPDATE/DELETE 실행 원천 차단.
- 컨텍스트 매니저 종료 시 항상 rollback 및 close를 수행해 커넥션 누수 방지.
"""
