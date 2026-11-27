# JSON 형식의 깔끔한 콘솔 출력을 위함
# - dumps : 점검 결과를 사람이 읽기 쉬운 들여쓰기 JSON으로 출력
# 종료 코드 제어 및 에러 스트림 출력을 위함
# - exit : 프리플라이트 실패 시 0이 아닌 종료 코드로 CI/스크립트 실패 전파
import sys
from json import dumps

# 프리플라이트 점검 함수 호출을 위함
# - run_preflight : CUDA GPU 및 DB 연결 상태 일괄 확인
from app.core.preflight import run_preflight

# 환경 설정 로드를 위함
# - load_settings : .env 기반 Settings 인스턴스 생성
from app.core.settings import load_settings

# 표준 서비스 예외 포착을 위함
# - ServiceError : 실패 코드와 사유를 명확히 출력
from app.services.errors import ServiceError

"""
터미널에서 서버 기동 전 환경(CUDA GPU · PostgreSQL)을 단독 점검하는 CLI 스크립트.
"""


def main() -> None:
    """설정 로드 후 프리플라이트 점검을 실행하고 결과를 표준 출력으로 표시."""
    try:
        settings = load_settings()
        report = run_preflight(settings)
        print(dumps(report, ensure_ascii=False, indent=2))
    except ServiceError as exc:
        # 오류 코드와 메시지를 stderr로 출력하고 비정상 종료 코드 반환
        print(f"[PREFLIGHT FAILED] {exc.code}: {exc.message}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


"""
[설계 평가]
- FastAPI 서버와 독립적으로 실행 가능해 배포·개발 초기 환경 진단 비용 절감.
- ServiceError 포착 후 exit(1) 처리로 자동화 파이프라인 연동에 적합.
"""
