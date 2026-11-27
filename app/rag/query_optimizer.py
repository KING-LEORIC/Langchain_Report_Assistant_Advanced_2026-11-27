# 정규표현식 기반 질의어 정제를 위함
# - sub : 연속 공백 축소 및 불필요한 특수문자 정리
from re import sub

"""
Advanced 트랙용 질의 최적화(Query Optimizer) 모듈.
검색 품질을 떨어뜨리는 군더더기 표현을 정리하되 핵심 기술 키워드와 보고서 번호는 온전히 보존.
"""


class QueryOptimizer:
    """벡터 검색 및 리랭킹 효율을 높이기 위한 경량 질의어 정규화기."""

    # 검색에 불필요한 한국어 질문 종결어미·완곡 표현 패턴
    _TRAILING_NOISE_PATTERN = r"(에\s*대해\s*알려\s*줘|알려\s*주세요|설명해\s*줘|정리해\s*줘|찾아\s*줘|무엇인가요\??|뭐야\??)$"

    def optimize(self, query: str) -> str:
        """원본 질의에서 불필요한 종결 표현과 중복 공백을 제거해 검색용 질의 반환."""
        normalized = sub(r"\s+", " ", query).strip()
        stripped = sub(self._TRAILING_NOISE_PATTERN, "", normalized).strip()
        # 과도한 절삭으로 질의가 비어 버리면 원본 정제 문자열을 그대로 유지
        return stripped if stripped else normalized


"""
[설계 평가]
- 추가 LLM 호출 없이 결정론적 규칙으로 지연 시간(latency) 0ms 수준 유지.
- 절삭 후 빈 문자열 방지 폴백으로 짧은 키워드 질의 손실 예방.
"""
