# JSON 직렬화를 위함
# - dumps : RAG 검색 결과와 출처 ID를 구조화된 JSON 문자열로 반환
from json import dumps

# Annotated 및 인터페이스 타입 명세를 위함
# - Annotated : InjectedToolCallId 메타데이터 선언
# - Any : Phase 04 RAG 파이프라인 객체와의 느슨한 결합 유지
from typing import Annotated, Any

# LangChain 도구 데코레이터 및 호출 ID 주입을 위함
# - InjectedToolCallId : 런타임 도구 호출 식별자 자동 주입
# - tool : rag_search 함수를 LangChain Tool로 래핑
from langchain_core.tools import InjectedToolCallId, tool

# 카테고리 리터럴 타입 참조를 위함
# - CategoryCode : 선택적 카테고리 필터 타입 일치
from app.schemas.agent import CategoryCode

# 도구 입력 스키마 참조를 위함
# - RagSearchInput : query, category_code, k 검증 스키마 적용
from app.schemas.tools import RagSearchInput

"""
E5 임베딩 · pgvector · GPU 리랭커 기반 RAG 파이프라인을 호출하는 rag_search 도구 모듈.
에이전트가 기술 보고서 본문 근거를 탐색할 때 사용하는 핵심 검색 진입점.
"""


def create_rag_search_tool(rag_pipeline: Any) -> Any:
    """RAG 파이프라인이 바인딩된 rag_search @tool 인스턴스 생성."""

    @tool("rag_search", args_schema=RagSearchInput)
    def rag_search(
        query: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        category_code: CategoryCode | None = None,
        k: int = 8,
    ) -> str:
        """질의어와 선택적 카테고리로 기술 보고서를 검색하고 리랭킹된 근거 요약과 출처 ID 반환."""
        result = rag_pipeline.run(query=query, category_code=category_code, k=k)

        # 에이전트가 tool_call_id와 source_ids를 함께 추적할 수 있도록 결합 반환
        payload = {
            "tool_call_id": tool_call_id,
            "status": result.get("status", "insufficient_evidence"),
            "answer": result.get("answer", ""),
            "source_ids": result.get("source_ids", []),
            "contexts": result.get("contexts", []),
        }
        return dumps(payload, ensure_ascii=False, default=str)

    return rag_search


"""
[설계 평가]
- 팩토리 패턴으로 RAG 파이프라인을 주입해 단위 테스트 시 목(Mock) 파이프라인으로 손쉽게 대체 가능.
- 도구 출력에 source_ids와 contexts를 명시해 최종 Agent의 근거 인용 정확도 향상.
"""
