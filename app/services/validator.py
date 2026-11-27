# 정적 타입 명세를 위함
# - Iterable : 관찰된 ID 컬렉션을 유연하게 전달받기 위함
from collections.abc import Iterable

# Advanced 최종 출력 스키마 참조를 위함
# - StructuredModelAnswer : status, answer, source_ids, status_report_ids 계약 검증 대상
from app.schemas.agent import StructuredModelAnswer

# 계약 위반 시 표준 500 예외 발생을 위함
# - ServiceError : OUTPUT_CONTRACT_ERROR 및 TOOL_CONTRACT_ERROR 반환
from app.services.errors import ServiceError

"""
도구 실행 결과와 에이전트 최종 Structured Output의 계약 정합성을 검증하는 모듈.
근거 없는 보고서 ID 인용(환각) 및 빈 답변 반환을 차단하여 출력 신뢰성 보장.
"""


def validate_tool_result_dict(payload: object, tool_name: str) -> dict:
    """도구 반환 페이로드가 딕셔너리 계약 형태를 만족하는지 검증."""
    # 비정상 반환 타입으로 후속 파싱이 깨지는 것을 막기 위한 방어 검사
    if not isinstance(payload, dict):
        raise ServiceError(
            code="TOOL_CONTRACT_ERROR",
            message=f"{tool_name} 도구 반환 형식이 dict가 아님",
            detail={"actual_type": type(payload).__name__},
        )
    return payload


def validate_structured_answer(
    answer: StructuredModelAnswer,
    observed_rag_ids: Iterable[int],
    observed_status_ids: Iterable[int],
) -> StructuredModelAnswer:
    """최종 StructuredModelAnswer가 실제 도구 실행 결과에 근거하는지 검증."""
    rag_id_set = set(observed_rag_ids)
    status_id_set = set(observed_status_ids)

    # 답변 텍스트가 비어 있으면 정상 응답으로 간주할 수 없으므로 계약 오류 처리
    if not answer.answer.strip():
        raise ServiceError(
            code="OUTPUT_CONTRACT_ERROR",
            message="에이전트 최종 답변 텍스트가 비어 있음",
        )

    # 모델이 도구로 조회하지 않은 임의의 보고서 ID를 출처로 날조했는지 검사
    ungrounded_sources = [
        rid
        for rid in answer.source_ids
        if rid not in rag_id_set and rid not in status_id_set
    ]
    if ungrounded_sources:
        raise ServiceError(
            code="OUTPUT_CONTRACT_ERROR",
            message="도구 실행 결과에 없는 보고서 ID가 source_ids에 포함됨",
            detail={"ungrounded_source_ids": ungrounded_sources},
        )

    # completed 상태인데 인용 출처와 상태 조회 ID가 모두 비어 있으면 근거 부족으로 보정
    if (
        answer.status == "completed"
        and not answer.source_ids
        and not answer.status_report_ids
    ):
        return StructuredModelAnswer(
            status="insufficient_evidence",
            answer=answer.answer,
            source_ids=[],
            status_report_ids=[],
        )
    # 근거 부족 상태에서는 어떠한 출처 ID도 남기지 않아 거짓 인용 차단
    if answer.status != "completed":
        return StructuredModelAnswer(
            status="insufficient_evidence",
            answer=answer.answer,
            source_ids=[],
            status_report_ids=answer.status_report_ids,
        )
    return answer


"""
[설계 평가]
- 도구 출력과 최종 응답을 분리 검증해 모델의 출처 ID 환각을 코드 수준에서 차단.
- 근거 없는 completed 응답을 insufficient_evidence로 안전하게 강등해 거짓 확신 방지.
"""
