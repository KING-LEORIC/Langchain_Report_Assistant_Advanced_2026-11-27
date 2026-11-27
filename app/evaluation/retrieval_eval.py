# 평가 데이터셋 JSON 로드를 위함
# - loads : gold.json의 질의·정답 ID·무응답 기대 케이스 파싱
from json import loads

# 파일 경로 안전 처리를 위함
# - Path : OS 독립적인 평가 데이터셋 경로 확인
from pathlib import Path

# 정적 타입 명세를 위함
# - Any : 평가 케이스 및 요약 메트릭 딕셔너리 타입 명시
from typing import Any

# 환경 설정 로더 참조를 위함
# - Settings : 평가에 사용할 RETRIEVE_K, CONTEXT_K 참조
# - load_settings : 단독 실행 시 .env 설정 로드
from app.core.settings import Settings, load_settings

# 읽기 전용 DB 리포지토리 참조를 위함
# - ReportRepository : FRESH 벡터 검색 실행
from app.db.repository import ReportRepository

# CUDA 임베딩 인코더 참조를 위함
# - E5Embedder : 평가 질의어 벡터 변환
from app.rag.embeddings import E5Embedder

# ID 추출 유틸리티 참조를 위함
# - extract_unique_report_ids : 검색 결과에서 보고서 ID 순위 목록 추출
from app.rag.ids import extract_unique_report_ids

# CUDA 리랭커 참조를 위함
# - BgeReranker : 리랭킹 후 순위 상승(MRR 개선) 측정
from app.rag.reranker import BgeReranker

# 1차 벡터 검색기 참조를 위함
# - VectorRetriever : 리랭킹 전 기준 순위 확보
from app.rag.retriever import VectorRetriever

"""
data/evaluation/gold.json을 이용해 1차 벡터 검색(Before)과 GPU 리랭킹(After)의
Hit@K 및 MRR을 정량 평가하는 오프라인 평가 모듈.
"""


def _compute_reciprocal_rank(ranked_ids: list[int], expected_ids: set[int]) -> float:
    """정답 ID가 처음 등장한 순위의 역수(MRR) 계산."""
    for idx, rid in enumerate(ranked_ids, start=1):
        if rid in expected_ids:
            return 1.0 / idx
    return 0.0


def evaluate_retrieval(
    gold_path: Path, settings: Settings | None = None
) -> dict[str, Any]:
    """gold.json 전체 케이스를 실행해 Before/After 검색 성능 요약 반환."""
    cfg = settings or load_settings()
    if not gold_path.exists():
        raise FileNotFoundError(f"평가 데이터셋을 찾을 수 없음: {gold_path}")

    raw = loads(gold_path.read_text(encoding="utf-8"))
    cases = raw if isinstance(raw, list) else raw.get("cases", [])

    repo = ReportRepository(cfg)
    embedder = E5Embedder(cfg)
    retriever = VectorRetriever(settings=cfg, repository=repo, embedder=embedder)
    reranker = BgeReranker(cfg)

    case_results: list[dict[str, Any]] = []
    positive_count = 0
    before_hits = 0
    after_hits = 0
    before_mrr_sum = 0.0
    after_mrr_sum = 0.0

    for case in cases:
        case_id = case.get("id", "")
        query = case.get("query", "")
        category_code = case.get("category_code")
        expected_ids = {int(x) for x in case.get("expected_report_ids", [])}

        before_candidates = retriever.retrieve(query=query, category_code=category_code)
        after_candidates = (
            reranker.rerank(
                query=query, candidates=before_candidates, top_k=cfg.context_k
            )
            if before_candidates
            else []
        )

        before_ids = extract_unique_report_ids(before_candidates)[: cfg.context_k]
        after_ids = extract_unique_report_ids(after_candidates)

        # 정답 ID가 비어 있는 G07(No-answer) 케이스는 Hit/MRR 분모에서 제외하고 별도 기록
        if expected_ids:
            positive_count += 1
            b_hit = any(rid in expected_ids for rid in before_ids)
            a_hit = any(rid in expected_ids for rid in after_ids)
            b_rr = _compute_reciprocal_rank(before_ids, expected_ids)
            a_rr = _compute_reciprocal_rank(after_ids, expected_ids)

            before_hits += int(b_hit)
            after_hits += int(a_hit)
            before_mrr_sum += b_rr
            after_mrr_sum += a_rr
        else:
            b_hit = False
            a_hit = False
            b_rr = 0.0
            a_rr = 0.0

        case_results.append(
            {
                "id": case_id,
                "query": query,
                "expected_report_ids": sorted(expected_ids),
                "before_ids": before_ids,
                "after_ids": after_ids,
                "before_hit": b_hit,
                "after_hit": a_hit,
                "before_rr": round(b_rr, 4),
                "after_rr": round(a_rr, 4),
            }
        )

    denom = max(positive_count, 1)
    return {
        "total_cases": len(cases),
        "positive_cases": positive_count,
        "before_hit_at_k": round(before_hits / denom, 4),
        "after_hit_at_k": round(after_hits / denom, 4),
        "before_mrr": round(before_mrr_sum / denom, 4),
        "after_mrr": round(after_mrr_sum / denom, 4),
        "cases": case_results,
    }


"""
[설계 평가]
- 정답 ID가 없는 무응답(G07) 케이스를 Hit/MRR 분모에서 분리해 왜곡 없는 지표 산출.
- Before(1차 벡터)와 After(리랭커)를 동일 context_k 기준으로 공정 비교.
"""
