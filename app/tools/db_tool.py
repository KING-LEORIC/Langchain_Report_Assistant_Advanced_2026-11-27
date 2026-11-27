# JSON 직렬화를 위함
# - dumps : 조회 결과를 LLM과 로그 파서가 모두 읽기 쉬운 UTF-8 JSON 문자열로 변환
from json import dumps

# Annotated 타입 기반 인자 주입 선언을 위함
# - Annotated : LangChain InjectedToolCallId 메타데이터를 파라미터에 결합
from typing import Annotated, Any

# LangChain 도구 데코레이터 및 호출 ID 주입을 위함
# - InjectedToolCallId : LLM 스키마에서 숨긴 채 런타임 tool_call_id를 자동 주입받기 위함
# - tool : 함수를 LangChain Tool 객체로 등록
from langchain_core.tools import InjectedToolCallId, tool

# 읽기 전용 DB 리포지토리 참조를 위함
# - ReportRepository : reports 테이블 단건 상태 조회 위임
from app.db.repository import ReportRepository

# 도구 입력 스키마 참조를 위함
# - LookupStatusInput : report_id 양의 정수 제약 적용
from app.schemas.tools import LookupStatusInput

"""
PostgreSQL reports 테이블에서 특정 보고서의 상태·리비전·인덱싱 여부를 조회하는 도구 모듈.
팩토리 패턴으로 ReportRepository를 주입받아 전역 상태 없이 테스트 및 DI 지원.
"""


def create_lookup_status_tool(repository: ReportRepository) -> Any:
    """ReportRepository가 바인딩된 lookup_status @tool 인스턴스 생성."""

    @tool("lookup_status", args_schema=LookupStatusInput)
    def lookup_status(
        report_id: int, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
        """보고서 ID로 현재 상태(status), 색인 상태(index_status), 카테고리, 리비전을 조회."""
        row = repository.get_report_status(report_id)

        # 존재하지 않는 보고서 ID 조회 시 예외 대신 found=False JSON을 반환해 모델이 정상 안내하도록 유도
        if row is None:
            payload = {
                "tool_call_id": tool_call_id,
                "found": False,
                "report_id": report_id,
                "message": f"보고서 ID {report_id}를 찾을 수 없음",
            }
            return dumps(payload, ensure_ascii=False, default=str)

        payload = {
            "tool_call_id": tool_call_id,
            "found": True,
            "report_id": row["report_id"],
            "title": row["title"],
            "category_code": row["category_code"],
            "status": row["status"],
            "index_status": row["index_status"],
            "revision": row["revision"],
            "updated_at": row["updated_at"],
        }
        return dumps(payload, ensure_ascii=False, default=str)

    return lookup_status


"""
[설계 평가]
- InjectedToolCallId를 사용해 LLM이 tool_call_id를 잘못 생성할 여지를 원천 제거.
- 미존재 ID를 오류가 아닌 found=False 구조화 응답으로 내려 에이전트의 우아한 폴백 유도.
"""
