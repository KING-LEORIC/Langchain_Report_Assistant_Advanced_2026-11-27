# CUDA GPU 가용성 및 추론 컨텍스트 확인을 위함
# - torch : CPU 폴백 차단 및 cuda:0 강제 검증
import torch

# HuggingFace SentenceTransformer 임베딩 모델 로드를 위함
# - SentenceTransformer : E5-small 384차원 벡터 인코딩 수행
from sentence_transformers import SentenceTransformer

# 환경 설정 참조를 위함
# - Settings : EMBEDDING_MODEL 이름 조회
from app.core.settings import Settings

# GPU·임베딩 장애 표준 예외 변환을 위함
# - ServiceError : 503 GPU_UNAVAILABLE 및 503 EMBEDDING_UNAVAILABLE 반환
from app.services.errors import ServiceError

"""
CUDA GPU(cuda:0) 전용 E5 임베딩 생성 모듈.
CPU 폴백을 금지하며 E5 모델 규격인 'query: ' 접두사와 정규화를 강제.
"""


class E5Embedder:
    """intfloat/multilingual-e5-small 기반 질의어 임베딩 인코더."""

    def __init__(self, settings: Settings) -> None:
        """GPU 가용성 확인 후 cuda:0에 임베딩 모델 로드."""
        # CPU 폴백으로 인한 극심한 지연 및 명세 위반을 막기 위해 즉시 차단
        if not torch.cuda.is_available():
            raise ServiceError(
                code="GPU_UNAVAILABLE",
                message="임베딩 모델 로드에 필요한 CUDA GPU 미감지",
            )

        try:
            self._model = SentenceTransformer(settings.embedding_model, device="cuda:0")
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
- device='cuda:0' 명시와 사전 가용성 검증으로 CPU 폴백 가능성 원천 차단.
- 'query: ' 접두사와 L2 정규화를 인코더 내부에 캡슐화해 호출부의 실수 방지.
"""
