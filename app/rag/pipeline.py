# JSON 파싱 및 정규표현식 추출을 위함
# - loads : LLM JSON 응답 파싱
from json import loads

# 코드 블록 제거용 정규표현식 처리를 위함
# - sub : 마크다운 펜스(```json ... ```) 제거
from re import sub

# 정적 타입 명세를 위함
# - Any : 파이프라인 입출력 딕셔너리 타입 명시
from typing import Any

# LangChain 프롬프트 및 LCEL 체인 구성을 위함
# - ChatPromptTemplate : 시스템·사용자 프롬프트 템플릿 정의
from langchain_core.prompts import ChatPromptTemplate

# LCEL 입력 전처리 람다 노드 구성을 위함
# - RunnableLambda : 컨텍스트 포맷팅과 체인 연결
from langchain_core.runnables import RunnableLambda

# OpenAI 호환 Chat 모델 호출을 위함
# - ChatOpenAI : max_retries=0 및 타임아웃 설정으로 실패 즉시 노출
from langchain_openai import ChatOpenAI

# 환경 설정 참조를 위함
# - Settings : LLM_MODEL, LLM_TIMEOUT_SECONDS 조회
from app.core.settings import Settings

# DB 리포지토리 참조를 위함
# - ReportRepository : 리랭킹 후 원문 상세 조회
from app.db.repository import ReportRepository

# 문서 조립 유틸리티 참조를 위함
# - build_context_documents : CONTEXT_CHARS 한도 내 본문 블록 생성
from app.rag.documents import build_context_documents

# 고유 ID 추출 유틸리티 참조를 위함
# - extract_unique_report_ids : 근거 문서 ID 목록 정리
from app.rag.ids import extract_unique_report_ids

# GPU 리랭커 참조를 위함
# - BgeReranker : 1차 후보 정밀 재정렬
from app.rag.reranker import BgeReranker

# 1차 벡터 검색기 참조를 위함
# - VectorRetriever : pgvector FRESH 후보 검색
from app.rag.retriever import VectorRetriever

# 카테고리 타입 참조를 위함
# - CategoryCode : 선택적 카테고리 필터 전달
from app.schemas.agent import CategoryCode

# LLM 장애 표준 예외 변환을 위함
# - ServiceError : 504 LLM_TIMEOUT 및 503 LLM_UNAVAILABLE 반환
from app.services.errors import ServiceError

"""
VectorRetriever → BgeReranker → build_context_documents → LCEL(RunnableLambda | RAG_PROMPT | model)
흐름으로 동작하는 RAG 파이프라인 모듈.
"""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "너는 기술 보고서 검색 근거만 바탕으로 답하는 분석기다.\n"
            "규칙:\n"
            "1. 제공된 [근거 문서]에 있는 사실만 사용하여 질문에 답하라.\n"
            "2. [근거 문서]에 질문과 관련된 보고서가 1건이라도 있으면 조회된 내용을 바탕으로 요약하고 반드시 status='completed'와 해당 report_id를 source_ids에 포함하라.\n"
            "3. [근거 문서]가 질문의 주제와 완전히 무관할 때만 status='insufficient_evidence', source_ids=[]로 답하라('incomplete' 금지).\n"
            '4. 반드시 JSON 객체만 출력하라: {{"status": "...", "answer": "...", "source_ids": [1, 2]}}',
        ),
        ("human", "질문: {question}\n\n[근거 문서]\n{context_block}"),
    ]
)


def parse_answer(raw_text: str, allowed_ids: set[int]) -> dict[str, Any]:
    """LLM 원시 출력 문자열을 파싱하고 허용된 상태 및 출처 ID 범위 검증."""
    cleaned = sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip())
    try:
        parsed = loads(cleaned)
    except Exception:
        # JSON 파싱 실패 시에도 파이프라인을 죽이지 않고 근거 부족 폴백 반환
        return {
            "status": "insufficient_evidence",
            "answer": raw_text.strip() or "근거 요약 파싱 실패",
            "source_ids": [],
        }

    status = parsed.get("status", "insufficient_evidence")
    # SRS 명세상 'incomplete' 등 비표준 상태는 허용하지 않으므로 교정
    if status not in {"completed", "insufficient_evidence"}:
        status = "insufficient_evidence"

    raw_ids = parsed.get("source_ids", [])
    # 실제 검색 컨텍스트에 포함된 ID만 통과시켜 환각 ID 제거
    valid_ids = [
        int(rid) for rid in raw_ids if isinstance(rid, int) and rid in allowed_ids
    ]

    if status == "completed" and not valid_ids:
        status = "insufficient_evidence"

    return {
        "status": status,
        "answer": str(parsed.get("answer", "")).strip(),
        "source_ids": valid_ids,
    }


class RAGPipeline:
    """검색·리랭킹·근거 기반 답변 생성을 통합 실행하는 RAG 파이프라인."""

    def __init__(
        self,
        settings: Settings,
        repository: ReportRepository,
        retriever: VectorRetriever,
        reranker: BgeReranker,
    ) -> None:
        """RAG 구성 요소와 max_retries=0 LLM 클라이언트 초기화."""
        self._settings = settings
        self._repository = repository
        self._retriever = retriever
        self._reranker = reranker
        # 타임아웃·장애가 즉시 드러나도록 자동 재시도를 0으로 고정
        self._model = ChatOpenAI(
            model=settings.llm_model,
            timeout=settings.llm_timeout_seconds,
            max_retries=0,
        )
        self._chain = (
            RunnableLambda(
                lambda x: {
                    "question": x["question"],
                    "context_block": x["context_block"],
                }
            )
            | RAG_PROMPT
            | self._model
        )

    def run(
        self,
        query: str,
        category_code: CategoryCode | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """질의에 대한 벡터 검색·리랭킹·LCEL 요약 실행."""
        candidates = self._retriever.retrieve(
            query=query, category_code=category_code, k=k
        )
        # 1차 검색 결과가 없으면 리랭커와 LLM 비용을 쓰지 않고 즉시 근거 부족 반환
        if not candidates:
            return {
                "status": "insufficient_evidence",
                "answer": "질의와 일치하는 유효한 보고서 근거를 찾지 못함",
                "source_ids": [],
                "contexts": [],
            }

        ranked = self._reranker.rerank(query=query, candidates=candidates)
        documents = build_context_documents(
            ranked_candidates=ranked,
            repository=self._repository,
            settings=self._settings,
        )
        if not documents:
            return {
                "status": "insufficient_evidence",
                "answer": "조회 가능한 보고서 상세 본문이 없음",
                "source_ids": [],
                "contexts": [],
            }

        context_block = "\n\n---\n\n".join(doc["text"] for doc in documents)
        allowed_ids = set(extract_unique_report_ids(documents))

        try:
            ai_message = self._chain.invoke(
                {"question": query, "context_block": context_block}
            )
        except TimeoutError as exc:
            raise ServiceError(
                code="LLM_TIMEOUT",
                message="RAG 요약 LLM 호출 시간 초과",
                detail={"timeout_seconds": self._settings.llm_timeout_seconds},
            ) from exc
        except Exception as exc:
            # httpx/openai 타임아웃 예외 문자열도 함께 검사해 504로 정확히 매핑
            if "timeout" in str(exc).lower():
                raise ServiceError(
                    code="LLM_TIMEOUT",
                    message="RAG 요약 LLM 호출 시간 초과",
                    detail={"reason": str(exc)},
                ) from exc
            raise ServiceError(
                code="LLM_UNAVAILABLE",
                message="RAG 요약 LLM 호출 실패",
                detail={"reason": str(exc)},
            ) from exc

        parsed = parse_answer(str(ai_message.content), allowed_ids=allowed_ids)
        parsed["contexts"] = documents
        return parsed


"""
[설계 평가]
- 검색 결과 0건 시 조기 반환(Short-circuit)하여 무의미한 리랭킹 및 LLM 비용 차단.
- parse_answer()에서 허용된 상태('completed'·'insufficient_evidence')와 실제 컨텍스트 ID만 통과시켜 안전성 확보.
"""
