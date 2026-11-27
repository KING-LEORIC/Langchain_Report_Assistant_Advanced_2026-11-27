# 정적 타입 명세를 위함
# - Any : 검색 결과 딕셔너리 리스트의 유연한 수용
from typing import Any

"""
검색·리랭킹된 보고서 컨텍스트 목록에서 출처 ID(report_id)를 순서 보존 방식으로 추출하는 유틸리티.
"""


def extract_unique_report_ids(items: list[dict[str, Any]]) -> list[int]:
    """입력 순서를 유지하면서 중복 없는 정수형 report_id 목록 반환."""
    seen: set[int] = set()
    unique_ids: list[int] = []

    for item in items:
        raw_id = item.get("report_id")
        # 정수 변환 불가능하거나 누락된 비정상 항목은 조용히 건너뛰어 파이프라인 중단 방지
        if raw_id is None:
            continue
        report_id = int(raw_id)
        if report_id not in seen:
            seen.add(report_id)
            unique_ids.append(report_id)

    return unique_ids


"""
[설계 평가]
- 리랭킹 점수 순서를 훼손하지 않으면서 중복 보고서 ID 인용 차단.
- 단순 유틸리티로 분리해 리트리버·리랭커·검증기 어디서든 재사용 가능.
"""
