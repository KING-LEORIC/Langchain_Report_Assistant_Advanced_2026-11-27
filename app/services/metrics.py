# 고정밀 경과 시간 측정을 위함
# - perf_counter : 시스템 시계 변경에 영향받지 않는 단조 증가 타이머로 정확한 지연 시간 산출
from time import perf_counter

# 요청별 고유 추적 식별자 생성을 위함
# - uuid4 : 분산 로그 및 실행 이력 조회(API-03)에서 충돌 없는 요청 ID 발급
from uuid import uuid4

"""
요청 단위 추적 식별자 생성 및 실행 지연 시간(ms) 측정을 담당하는 경량 메트릭 유틸리티.
API 응답 스키마와 Advanced 실행 이력 저장소(run_store)가 공통으로 사용하는 관측 데이터 생성.
"""


def generate_request_id() -> str:
    """요청별 고유 식별자 문자열 생성."""
    # 하이픈 없는 32자리 16진수 대신 표준 UUID 문자열을 반환하여 가독성과 호환성 유지
    return str(uuid4())


class LatencyTimer:
    """컨텍스트 진입 시점부터 경과 시간을 밀리초 단위로 산출하는 타이머 유틸리티."""

    def __init__(self) -> None:
        """인스턴스 생성 시점의 기준 시각 기록."""
        self._start: float = perf_counter()

    def elapsed_ms(self) -> float:
        """시작 시점 대비 현재까지 경과한 밀리초 반환."""
        # 소수점 둘째 자리까지 반올림하여 응답 페이로드의 가독성 확보
        return round((perf_counter() - self._start) * 1000.0, 2)


"""
[설계 평가]
- time.time() 대신 perf_counter()를 사용해 NTP 동기화 등으로 인한 음수 지연 시간 발생 원천 차단.
- 최소 인터페이스로 분리되어 Phase 06의 run_store 및 평가 스크립트와 무리 없이 결합 가능.
"""
