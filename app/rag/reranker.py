# 정적 타입 명세를 위함
# - Any : 후보 딕셔너리 리스트 입출력 타입 명시
from typing import Any

# CUDA GPU 연산 및 그래디언트 비활성화를 위함
# - torch : cuda:0 또는 cpu 디바이스 선택 및 inference_mode 메모리 절약
import torch

# HuggingFace Cross-Encoder 리랭커 모델 로드를 위함
# - AutoModelForSequenceClassification : 질의-문서 쌍의 관련도 로짓 산출
# - AutoTokenizer : RERANK_MAX_LENGTH 기준 토큰화 및 패딩
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 환경 설정 참조를 위함
# - Settings : 모델명, 배치 크기, 최대 토큰 길이, CONTEXT_K, allow_cpu_fallback 조회
from app.core.settings import Settings

# GPU·리랭커 장애 표준 예외 변환을 위함
# - ServiceError : 503 GPU_UNAVAILABLE 및 503 RERANKER_UNAVAILABLE 반환
from app.services.errors import ServiceError

"""
BAAI/bge-reranker-v2-m3 리랭커 모듈.
1차 벡터 검색 후보를 정밀 교차 인코딩(Cross-Encoding)하여 상위 CONTEXT_K건 선별.
"""


class BgeReranker:
    """bge-reranker-v2-m3 기반 교차 인코딩 리랭커."""

    def __init__(self, settings: Settings) -> None:
        """디바이스 가용성 확인 후 토크나이저와 시퀀스 분류 모델을 대상 장치에 로드."""
        if torch.cuda.is_available():
            self._device = "cuda:0"
        elif settings.allow_cpu_fallback:
            # 비-CUDA PC에서 개발 실행을 허용하기 위한 조건부 CPU 선택
            self._device = "cpu"
        else:
            raise ServiceError(
                code="GPU_UNAVAILABLE", message="리랭커 로드에 필요한 CUDA GPU 미감지"
            )

        self._settings = settings
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.reranker_model)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                settings.reranker_model
            ).to(self._device)
            self._model.eval()
        except Exception as exc:
            raise ServiceError(
                code="RERANKER_UNAVAILABLE",
                message=f"리랭커 모델 로드 실패: {settings.reranker_model}",
                detail={"reason": str(exc)},
            ) from exc

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """후보 목록을 배치 단위로 점수화하고 상위 top_k건 반환."""
        if not candidates:
            return []

        limit = top_k if top_k is not None else self._settings.context_k
        batch_size = self._settings.rerank_batch_size
        max_length = self._settings.rerank_max_length

        pairs = [
            [query, f"{item.get('title', '')}\n{item.get('overview', '')}".strip()]
            for item in candidates
        ]
        scores: list[float] = []

        try:
            # 메모리 폭증을 막기 위해 설정된 RERANK_BATCH_SIZE 단위로 분할 추론
            with torch.inference_mode():
                for start in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[start : start + batch_size]
                    inputs = self._tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(self._device)
                    logits = self._model(**inputs).logits.view(-1).float()
                    scores.extend(logits.detach().cpu().tolist())
        except Exception as exc:
            raise ServiceError(
                code="RERANKER_UNAVAILABLE",
                message="리랭커 추론 중 오류 발생",
                detail={"reason": str(exc)},
            ) from exc

        scored_items: list[dict[str, Any]] = []
        for item, score in zip(candidates, scores):
            copied = dict(item)
            copied["rerank_score"] = round(float(score), 4)
            scored_items.append(copied)

        scored_items.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_items[:limit]


"""
[설계 평가]
- 모델과 입력 텐서가 모두 동일한 self._device를 참조해 장치 불일치 런타임 오류 차단.
- allow_cpu_fallback=False 기본값을 유지해 CUDA 전용 환경의 명세 검증과 충돌 없음.
"""
