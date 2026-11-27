# 정적 타입 명세를 위함
# - Any : 점검 결과 리포트 딕셔너리 반환 타입 명시
from typing import Any

# CUDA GPU 가용성 검증을 위함
# - torch : cuda.is_available() 및 디바이스 이름 조회로 CPU 폴백 원천 차단
import torch

# 환경 설정 객체 참조를 위함
# - Settings : DB 및 모델 설정 검증 대상
from app.core.settings import Settings

# DB 가용성 및 FRESH 벡터 점검을 위함
# - ReportRepository : 실제 DB 쿼리 기반 준비 상태 확인
from app.db.repository import ReportRepository

# GPU 미가용 시 표준 예외 발생을 위함
# - ServiceError : 503 GPU_UNAVAILABLE 응답 반환
from app.services.errors import ServiceError

"""
애플리케이션 기동 전 CUDA GPU와 PostgreSQL 가용성을 검증하는 프리플라이트 모듈.
SRS v1.1의 CPU 폴백 금지 원칙에 따라 CUDA 미감지 시 즉시 503 GPU_UNAVAILABLE 발생.
"""


def check_cuda_gpu() -> dict[str, Any]:
    """CUDA GPU(cuda:0) 사용 가능 여부 확인 및 디바이스 정보 반환."""
    # GPU가 없으면 임베딩·리랭커를 CPU로 내리지 않고 즉시 중단
    if not torch.cuda.is_available():
        raise ServiceError(
            code="GPU_UNAVAILABLE", message="CUDA GPU를 사용할 수 없음 (CPU 폴백 금지)"
        )
    device_name = torch.cuda.get_device_name(0)
    return {"cuda_available": True, "device": "cuda:0", "device_name": device_name}


def run_preflight(settings: Settings) -> dict[str, Any]:
    """GPU 및 PostgreSQL 사전 점검을 순차 실행하고 요약 결과 반환."""
    gpu_info = check_cuda_gpu()
    repo = ReportRepository(settings)
    db_info = repo.check_db_ready()
    return {"level": settings.level, "gpu": gpu_info, "db": db_info}


"""
[설계 평가]
- 기동 시점에 GPU와 DB를 함께 검증해 요청 처리 도중 발생하는 지연 장애 예방.
- 점검 결과를 구조화된 딕셔너리로 반환해 /health 및 CLI 스크립트에서 공통 재사용.
"""
