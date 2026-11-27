# ToolMessage 본문 JSON 파싱을 위함
# - loads : 도구 실행 결과에서 관찰된 report_id·source_ids 추출
from json import loads

# 정적 타입 명세를 위함
# - Any : 메시지 기록 및 도구 호출 로그 딕셔너리 타입 명시
from typing import Any

# LangChain 에이전트 생성 팩토리 호출을 위함
# - create_agent : 도구 호출 루프와 Structured Output을 통합 실행하는 표준 에이전트 구성
from langchain.agents import create_agent

# 무한 반복 및 과다 호출 방지를 위함
# - ModelCallLimitMiddleware : 모델 호출 횟수를 MAX_MODEL_CALLS 이하로 강제 제한
# - ToolCallLimitMiddleware : 도구 호출 횟수를 MAX_TOOL_CALLS 이하로 강제 제한
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)

# Provider 네이티브 Structured Output 강제를 위함
# - ProviderStrategy : ToolStrategy 대신 모델 네이티브 스키마 출력으로 파싱 오류 최소화
from langchain.agents.structured_output import ProviderStrategy

# 실행 기록에서 도구 호출 및 도구 응답 메시지 식별을 위함
# - AIMessage : 모델이 선택한 도구 이름과 인수 추출
# - ToolMessage : 실제 도구가 반환한 결과에서 근거 ID 수집
from langchain_core.messages import AIMessage, ToolMessage

# OpenAI 호환 Chat 모델 연결을 위함
# - ChatOpenAI : max_retries=0 및 타임아웃 설정으로 실패 즉시 노출
from langchain_openai import ChatOpenAI

# 환경 설정 참조를 위함
# - Settings : LLM 모델명, 타임아웃, 호출 한도 조회
from app.core.settings import Settings

# Advanced 최종 출력 스키마 참조를 위함
# - StructuredModelAnswer : status, answer, source_ids, status_report_ids 계약 적용
from app.schemas.agent import CategoryCode, StructuredModelAnswer

# 표준 서비스 예외 변환을 위함
# - ServiceError : 503 EXECUTION_LIMIT, 504 LLM_TIMEOUT, 500 OUTPUT_CONTRACT_ERROR 반환
from app.services.errors import ServiceError

# 출력 근거 정합성 검증을 위함
# - validate_structured_answer : 모델의 출처 ID 환각 차단
from app.services.validator import validate_structured_answer

"""
LangChain create_agent와 ProviderStrategy(StructuredModelAnswer)를 결합해
기술 보고서 검색(rag_search) 및 상태 조회(lookup_status)를 오케스트레이션하는 에이전트 모듈.
"""


AGENT_SYSTEM_PROMPT = (
    "너는 기술 보고서 분석 어시스턴트다.\n"
    "도구 사용 규칙:\n"
    "1. 보고서 내용·기술 요약·실험 결과·회의록 등 본문 근거가 필요하면 rag_search를 호출하라.\n"
    "2. 특정 보고서 ID의 진행 상태(status), 색인 상태(index_status), 리비전이 필요하면 lookup_status를 호출하라.\n"
    "3. 두 정보가 모두 필요한 복합 질문이면 두 도구를 함께 활용하라.\n"
    "4. 도구 결과에 없는 보고서 ID를 절대 source_ids나 status_report_ids에 넣지 마라.\n"
    "5. 근거가 불충분하거나 질문과 일치하는 자료가 없으면 status='insufficient_evidence'로 답하라."
)


class ReportAgentRunner:
    """create_agent 기반 보고서 질의 응답 실행기."""

    def __init__(self, settings: Settings, tools: list[Any]) -> None:
        """모델, 도구, 예산 미들웨어, ProviderStrategy를 결합해 에이전트 그래프 생성."""
        self._settings = settings
        self._model = ChatOpenAI(
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        # 무한 루프 및 과도한 API 과금을 막기 위해 도구·모델 호출 상한 미들웨어 장착
        self._agent = create_agent(
            model=self._model,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            response_format=ProviderStrategy(StructuredModelAnswer),
            middleware=[
                ToolCallLimitMiddleware(run_limit=settings.max_tool_calls),
                ModelCallLimitMiddleware(run_limit=settings.max_model_calls),
            ],
        )

    def _extract_execution_trace(
        self, messages: list[Any]
    ) -> tuple[list[dict[str, Any]], set[int], set[int]]:
        """메시지 히스토리에서 도구 호출 로그와 실제 관찰된 보고서 ID 집합 추출."""
        tool_calls_log: list[dict[str, Any]] = []
        observed_rag_ids: set[int] = set()
        observed_status_ids: set[int] = set()

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_log.append(
                        {
                            "id": tc.get("id"),
                            "name": tc.get("name"),
                            "args": tc.get("args", {}),
                        }
                    )
            elif isinstance(msg, ToolMessage):
                try:
                    data = loads(str(msg.content))
                except Exception:
                    continue

                if msg.name == "rag_search":
                    for rid in data.get("source_ids", []):
                        if isinstance(rid, int):
                            observed_rag_ids.add(rid)
                    for ctx in data.get("contexts", []):
                        rid = ctx.get("report_id")
                        if isinstance(rid, int):
                            observed_rag_ids.add(rid)
                elif msg.name == "lookup_status" and data.get("found"):
                    rid = data.get("report_id")
                    if isinstance(rid, int):
                        observed_status_ids.add(rid)

        return tool_calls_log, observed_rag_ids, observed_status_ids

    def run(
        self, question: str, category_code: CategoryCode | None = None
    ) -> tuple[StructuredModelAnswer, list[dict[str, Any]]]:
        """사용자 질문을 에이전트에 전달하고 검증된 StructuredModelAnswer와 도구 로그 반환."""
        user_content = question
        # 요청에 category_code가 명시되면 에이전트가 rag_search 호출 시 해당 필터를 쓰도록 힌트 부여
        if category_code is not None:
            user_content = f"[카테고리 필터: {category_code}] {question}"

        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": user_content}]}
            )
        except ServiceError:
            # 하위 도구(DB, GPU, 임베딩, 리랭커)에서 발생한 표준 ServiceError는 그대로 전파
            raise
        except Exception as exc:
            err_msg = str(exc).lower()
            # 미들웨어 호출 한도 초과 시 SRS 명세에 따라 503 EXECUTION_LIMIT 반환
            if "limit" in err_msg or "call limit" in err_msg:
                raise ServiceError(
                    code="EXECUTION_LIMIT",
                    message="에이전트 도구 또는 모델 호출 상한 초과",
                    detail={"reason": str(exc)},
                ) from exc
            if "timeout" in err_msg:
                raise ServiceError(
                    code="LLM_TIMEOUT",
                    message="에이전트 LLM 응답 시간 초과",
                    detail={"timeout_seconds": self._settings.llm_timeout_seconds},
                ) from exc
            raise ServiceError(
                code="LLM_UNAVAILABLE",
                message="에이전트 실행 중 LLM 호출 실패",
                detail={"reason": str(exc)},
            ) from exc

        structured = result.get("structured_response")
        # ProviderStrategy 결과 객체가 비어 있으면 AIMessage 재파싱 없이 계약 오류로 처리
        if not isinstance(structured, StructuredModelAnswer):
            raise ServiceError(
                code="OUTPUT_CONTRACT_ERROR",
                message="에이전트가 StructuredModelAnswer 형식을 반환하지 않음",
            )

        messages = result.get("messages", [])
        tool_calls_log, observed_rag_ids, observed_status_ids = (
            self._extract_execution_trace(messages)
        )

        validated = validate_structured_answer(
            answer=structured,
            observed_rag_ids=observed_rag_ids,
            observed_status_ids=observed_status_ids,
        )
        return validated, tool_calls_log


"""
[설계 평가]
- ProviderStrategy로 스키마 준수율을 높이고 validate_structured_answer로 출처 ID 환각을 2중 차단.
- ToolCallLimitMiddleware와 ModelCallLimitMiddleware로 무한 루프 방지 및 503 EXECUTION_LIMIT 매핑 일관성 확보.
"""
