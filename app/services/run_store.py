# FIFO 순서 유지 및 O(1) 오래된 항목 제거를 위함
# - OrderedDict : 최대 100건 초과 시 가장 오래된 실행 기록을 즉시 퇴거
from collections import OrderedDict

# 실행 기록 생성 시각 기록을 위함
# - datetime : ISO-8601 UTC 타임스탬프 생성
# - timezone : 시간대 정보가 포함된 표준 UTC 시각 보장
from datetime import UTC, datetime, timezone

# 멀티스레드 요청 동시성 보호를 위함
# - Lock : 동시 요청 시 OrderedDict 경합 상태 방지
from threading import Lock

# 정적 타입 명세를 위함
# - Any : 저장되는 실행 이력 페이로드 타입 명시
from typing import Any

# 미존재 실행 ID 조회 시 표준 404 예외 발생을 위함
# - ServiceError : 404 RUN_NOT_FOUND 응답 반환
from app.services.errors import ServiceError

"""
Advanced 트랙의 API-03(GET /api/v1/agent/runs/{request_id})을 지원하는
인메모리 실행 이력 저장소 모듈(최대 100건 보관).
"""


class RunStore:
    """스레드 안전한 FIFO 기반 인메모리 실행 기록 저장소."""

    def __init__(self, max_runs: int = 100) -> None:
        """최대 보관 건수 및 동기화 락 초기화."""
        self._max_runs = max_runs
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def save(self, request_id: str, record: dict[str, Any]) -> None:
        """실행 기록 저장 및 최대 보관 건수 초과 시 가장 오래된 항목 제거."""
        stored = {**record, "stored_at": datetime.now(UTC).isoformat()}
        with self._lock:
            # 동일 request_id 재저장 시 최신 순서를 유지하기 위해 기존 키 갱신
            if request_id in self._runs:
                self._runs.move_to_end(request_id)
            self._runs[request_id] = stored
            # 메모리 무한 증가를 막기 위해 max_runs(100건) 초과분 즉시 퇴거
            while len(self._runs) > self._max_runs:
                self._runs.popitem(last=False)

    def get(self, request_id: str) -> dict[str, Any]:
        """request_id로 저장된 실행 기록 조회, 미존재 시 404 RUN_NOT_FOUND 발생."""
        with self._lock:
            record = self._runs.get(request_id)
        if record is None:
            raise ServiceError(
                code="RUN_NOT_FOUND",
                message=f"실행 이력을 찾을 수 없음: {request_id}",
                detail={"request_id": request_id},
            )
        return dict(record)


"""
[설계 평가]
- OrderedDict와 Lock을 결합해 스레드 안전성과 O(1) 퇴거 성능을 동시에 확보.
- 미존재 키 조회 시 404 RUN_NOT_FOUND를 즉시 발생시켜 API-03 계약 준수.
"""
