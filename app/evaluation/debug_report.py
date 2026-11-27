# CLI 인자 및 종료 코드 제어를 위함
# - Path : 평가 입력(gold.json)과 출력(quality_report.md) 경로 관리
from pathlib import Path

# 정적 타입 명세를 위함
# - Any : 평가 요약 딕셔너리 타입 명시
from typing import Any

# 검색·리랭킹 오프라인 평가 실행을 위함
# - evaluate_retrieval : 리포트 본문에 삽입할 정량 지표 및 케이스별 결과 산출
from app.evaluation.retrieval_eval import evaluate_retrieval

"""
오프라인 검색·리랭킹 평가 결과를 사람이 읽기 쉬운 Markdown 품질 보고서
(outputs/quality_report.md)로 생성하는 모듈.
"""


def render_quality_markdown(summary: dict[str, Any]) -> str:
    """평가 요약 딕셔너리를 Markdown 형식 리포트 문자열로 변환."""
    lines: list[str] = [
        "# RAG 검색·리랭킹 품질 평가 리포트 (Advanced)",
        "",
        "## 1. 전체 요약 지표",
        f"- 전체 케이스 수: {summary['total_cases']}건 (정답 포함: {summary['positive_cases']}건)",
        f"- Before Hit@K (1차 벡터): {summary['before_hit_at_k']}",
        f"- After Hit@K (GPU 리랭킹): {summary['after_hit_at_k']}",
        f"- Before MRR: {summary['before_mrr']}",
        f"- After MRR: {summary['after_mrr']}",
        "",
        "## 2. 케이스별 상세 결과",
        "| ID | 질의 | 기대 ID | Before Top-K | After Top-K | Before RR | After RR |",
        "|---|---|---|---|---|---|---|",
    ]

    for case in summary["cases"]:
        lines.append(
            f"| {case['id']} | {case['query']} | {case['expected_report_ids']} | "
            f"{case['before_ids']} | {case['after_ids']} | {case['before_rr']} | {case['after_rr']} |"
        )

    return "\n".join(lines) + "\n"


def generate_quality_report(
    gold_path: Path = Path("data/evaluation/gold.json"),
    output_path: Path = Path("outputs/quality_report.md"),
) -> Path:
    """평가 실행 후 outputs/quality_report.md 파일 저장."""
    summary = evaluate_retrieval(gold_path=gold_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_quality_markdown(summary), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    saved_path = generate_quality_report()
    print(f"품질 리포트 생성 완료: {saved_path}")


"""
[설계 평가]
- 평가 로직(retrieval_eval)과 마크다운 직렬화(debug_report)를 분리해 단일 책임 원칙 준수.
- outputs 디렉터리가 없어도 자동 생성하여 초기 실행 실패 방지.
"""
