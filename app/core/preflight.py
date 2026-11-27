# 정적 타입 명세를 위함
# - Any : 점검 결과 리포트 딕셔너리 반환 타입 명시
from typing import Any

# CUDA GPU 가용성 검증을 위함
# - torch : cuda.is_available() 및 디바이스 이름 조회로 실행 장치 결정
import torch

# 환경 설정 객체 참조를 위함
# - Settings : DB, 모델 및 allow_cpu_fallback 설정 조회
from app.core.settings import Settings

# DB 가용성 및 FRESH 벡터 점검을 위함
# - ReportRepository : 실제 DB 쿼리 기반 준비 상태 확인
from app.db.repository import ReportRepository

# GPU 미가용 시 표준 예외 발생을 위함
# - ServiceError : 503 GPU_UNAVAILABLE 응답 반환
from app.services.errors import ServiceError

"""
애플리케이션 기동 전 연산 디바이스(CUDA 또는 허용 시 CPU)와 PostgreSQL 가용성을 검증하는 프리플라이트 모듈.
기본값은 CPU 폴백 금지이며, allow_cpu_fallback=True일 때만 비-CUDA 로컬 실행 허용.
"""


def check_cuda_gpu(allow_cpu_fallback: bool = False) -> dict[str, Any]:
    """연산 디바이스(cuda:0 또는 허용 시 cpu) 사용 가능 여부 확인 및 정보 반환."""
    if not torch.cuda.is_available():
        # 비-CUDA 개발 PC에서 명시적으로 허용한 경우에만 CPU 디바이스 정보 반환
        if allow_cpu_fallback:
            return {
                "cuda_available": False,
                "device": "cpu",
                "device_name": "CPU (fallback)",
            }
        raise ServiceError(
            code="GPU_UNAVAILABLE", message="CUDA GPU를 사용할 수 없음 (CPU 폴백 금지)"
        )
    device_name = torch.cuda.get_device_name(0)
    return {"cuda_available": True, "device": "cuda:0", "device_name": device_name}


def run_preflight(settings: Settings) -> dict[str, Any]:
    """연산 디바이스 및 PostgreSQL 사전 점검을 순차 실행하고 요약 결과 반환."""
    gpu_info = check_cuda_gpu(allow_cpu_fallback=settings.allow_cpu_fallback)
    repo = ReportRepository(settings)
    db_info = repo.check_db_ready()
    return {"level": settings.level, "gpu": gpu_info, "db": db_info}


"""
[설계 평가]
- allow_cpu_fallback 플래그에 따라 엄격한 CUDA 검증과 로컬 CPU 호환을 모두 지원.
- 점검 결과를 구조화된 딕셔너리로 반환해 /health 및 CLI 스크립트에서 공통 재사용.
"""
