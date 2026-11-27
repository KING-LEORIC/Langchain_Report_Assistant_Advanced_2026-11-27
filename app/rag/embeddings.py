# CUDA GPU 가용성 및 추론 컨텍스트 확인을 위함
# - torch : cuda:0 또는 허용 시 cpu 디바이스 동적 선택
import torch

# HuggingFace SentenceTransformer 임베딩 모델 로드를 위함
# - SentenceTransformer : E5-small 384차원 벡터 인코딩 수행
from sentence_transformers import SentenceTransformer

# 환경 설정 참조를 위함
# - Settings : EMBEDDING_MODEL 이름 및 allow_cpu_fallback 조회
from app.core.settings import Settings

# GPU·임베딩 장애 표준 예외 변환을 위함
# - ServiceError : 503 GPU_UNAVAILABLE 및 503 EMBEDDING_UNAVAILABLE 반환
from app.services.errors import ServiceError

"""
E5 임베딩 생성 모듈.
기본적으로 CUDA GPU(cuda:0)를 강제하며, allow_cpu_fallback=True일 때만 CPU 추론 허용.
"""


class E5Embedder:
    """intfloat/multilingual-e5-small 기반 질의어 임베딩 인코더."""

    def __init__(self, settings: Settings) -> None:
        """디바이스 가용성 확인 후 선택된 장치(cuda:0 또는 cpu)에 임베딩 모델 로드."""
        if torch.cuda.is_available():
            self._device = "cuda:0"
        elif settings.allow_cpu_fallback:
            # 비-CUDA PC에서 개발 실행을 허용하기 위한 조건부 CPU 선택
            self._device = "cpu"
        else:
            raise ServiceError(
                code="GPU_UNAVAILABLE",
                message="임베딩 모델 로드에 필요한 CUDA GPU 미감지",
            )

        try:
            self._model = SentenceTransformer(
                settings.embedding_model, device=self._device
            )
        except Exception as exc:
            raise ServiceError(
                code="EMBEDDING_UNAVAILABLE",
                message=f"임베딩 모델 로드 실패: {settings.embedding_model}",
                detail={"reason": str(exc)},
            ) from exc

    def embed_query(self, query: str) -> list[float]:
        """사용자 질의를 E5 query 접두사 규격으로 384차원 정규화 벡터로 변환."""
        cleaned = query.strip()
        if not cleaned:
            raise ServiceError(
                code="VALIDATION_ERROR", message="임베딩할 질의어가 비어 있음"
            )

        # E5 모델은 비대칭 검색 시 질의어 앞에 'query: ' 접두사를 붙여야 코사인 유사도 정확도 유지
        prefixed_query = f"query: {cleaned}"
        try:
            vector = self._model.encode(
                prefixed_query, normalize_embeddings=True, show_progress_bar=False
            )
            return [float(x) for x in vector.tolist()]
        except Exception as exc:
            raise ServiceError(
                code="EMBEDDING_UNAVAILABLE",
                message="질의어 임베딩 생성 실패",
                detail={"reason": str(exc)},
            ) from exc


"""
[설계 평가]
- allow_cpu_fallback=False일 때는 기존처럼 CUDA 미감지 시 즉시 차단해 명세 준수.
- self._device 속성으로 모델 로드 장치를 일관 관리해 이식성 확보.
"""
