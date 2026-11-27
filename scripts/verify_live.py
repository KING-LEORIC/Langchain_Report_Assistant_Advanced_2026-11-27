# JSON 결과 출력을 위함
# - dumps : 각 엔드포인트 검증 결과를 콘솔에 가독성 있게 출력
# CLI 종료 코드 반환을 위함
# - exit : 하나라도 검증 실패 시 비정상 종료 코드 반환
import sys
from json import dumps

# 실행 중인 FastAPI 서버 HTTP 호출을 위함
# - Client : 단일 HTTP 클라이언트 세션으로 연결 재사용 및 타임아웃 제어
from httpx import Client

"""
기동 중인 FastAPI 서버(http://127.0.0.1:8000)의 핵심 엔드포인트
(API-00 /health, API-01 /ask, API-02 /rag/debug, API-03 /runs)를 순차 점검하는 스크립트.
"""

BASE_URL = "http://127.0.0.1:8000"


def main() -> None:
    """라이브 서버 대상 4종 API 스모크 테스트 실행."""
    with Client(base_url=BASE_URL, timeout=60.0) as client:
        # 1. API-00 헬스 체크 확인
        health_res = client.get("/health")
        assert health_res.status_code == 200, f"/health 실패: {health_res.text}"

        # 2. API-02 RAG 디버그 확인 (유료 LLM 없이 검색·리랭킹 동작 검증)
        debug_res = client.post(
            "/api/v1/rag/debug",
            json={
                "query": "RAG 근거 인용 정확도 향상",
                "category_code": "RESEARCH_PLAN",
                "top_k": 8,
            },
        )
        assert debug_res.status_code == 200, f"/rag/debug 실패: {debug_res.text}"

        # 3. API-01 에이전트 질의 확인
        ask_res = client.post(
            "/api/v1/agent/ask",
            json={"question": "최근 실험 보고서의 주요 결과를 요약해 줘"},
        )
        assert ask_res.status_code == 200, f"/agent/ask 실패: {ask_res.text}"
        request_id = ask_res.json()["request_id"]

        # 4. API-03 실행 이력 단건 조회 확인
        run_res = client.get(f"/api/v1/agent/runs/{request_id}")
        assert run_res.status_code == 200, f"/agent/runs 실패: {run_res.text}"

        print(
            dumps(
                {
                    "health": health_res.status_code,
                    "rag_debug": debug_res.status_code,
                    "agent_ask": ask_res.status_code,
                    "agent_run": run_res.status_code,
                    "verified_request_id": request_id,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[VERIFY FAILED] {exc}", file=sys.stderr)
        sys.exit(1)


"""
[설계 평가]
- 서버 외부에서 실제 HTTP 계약을 검증해 TestClient와 다른 환경 변수·네트워크 문제 조기 발견.
- API-01에서 받은 request_id를 API-03 조회에 연쇄 사용해 상태 전파까지 함께 검증.
"""
