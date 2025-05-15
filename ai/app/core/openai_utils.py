import os
import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, TypeVar, Awaitable, Type
from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# 제네릭 타입 정의
T = TypeVar('T')

class OpenAIRateLimiter:
    """
    OpenAI API 요청에 대한 속도 제한(Rate Limit)을 관리하는 클래스
    - 요청을 큐에 넣고 처리
    - 속도 제한 오류 발생 시 자동 재시도
    - TPM(Tokens Per Minute) 추적 및 관리
    """
    
    def __init__(self, 
                 max_tpm: int = 30000,  # 기본 Tier 1 한도
                 max_rpm: int = 500,    # 기본 Tier 1 한도
                 max_retries: int = 5,
                 initial_backoff: float = 1.0,
                 backoff_multiplier: float = 2.0):
        """
        Args:
            max_tpm: 분당 최대 토큰 수 (기본값: 30000, Tier 1 한도)
            max_rpm: 분당 최대 요청 수 (기본값: 500, Tier 1 한도)
            max_retries: 최대 재시도 횟수
            initial_backoff: 초기 재시도 대기 시간(초)
            backoff_multiplier: 재시도 시 대기 시간 증가 배수
        """
        self.max_tpm = max_tpm
        self.max_rpm = max_rpm
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.backoff_multiplier = backoff_multiplier
        
        # TPM 및 요청 추적
        self.request_timestamps: List[float] = []
        self.token_usage: List[Dict[str, float]] = []
        
        # 요청 처리를 위한 세마포어 (동시 요청 제한)
        self.semaphore = asyncio.Semaphore(5)  # 최대 5개 동시 요청
        
        # API 클라이언트
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    async def _prune_old_data(self, current_time: float) -> None:
        """1분 이상 지난 요청 및 토큰 사용 데이터 제거"""
        one_minute_ago = current_time - 60.0
        
        # 1분 이상 지난 요청 타임스탬프 제거
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > one_minute_ago]
        
        # 1분 이상 지난 토큰 사용 데이터 제거
        self.token_usage = [usage for usage in self.token_usage 
                           if usage.get('timestamp', 0) > one_minute_ago]
    
    def _count_current_tpm(self) -> int:
        """현재 분당 토큰 사용량 계산"""
        return sum(usage.get('tokens', 0) for usage in self.token_usage)
    
    def _count_current_rpm(self) -> int:
        """현재 분당 요청 수 계산"""
        return len(self.request_timestamps)
    
    def _calculate_tokens(self, messages: List[Dict[str, str]], max_tokens: int) -> int:
        """
        메시지와 max_tokens를 기반으로 예상 토큰 수 계산
        매우 대략적인 추정값 (실제 값과 차이가 있을 수 있음)
        """
        # 입력 메시지 토큰 수 추정 (4자당 약 1 토큰으로 대략 계산)
        input_tokens = sum(len(msg.get('content', '')) // 4 + 3 for msg in messages)
        
        # 최대 출력 토큰 수 (응답을 위해 max_tokens 사용)
        output_tokens = max_tokens
        
        return input_tokens + output_tokens
    
    async def _wait_for_capacity(self, estimated_tokens: int) -> None:
        """현재 사용량에 따라 적절한 시간 대기"""
        while True:
            current_time = time.time()
            await self._prune_old_data(current_time)
            
            current_tpm = self._count_current_tpm()
            current_rpm = self._count_current_rpm()
            
            # TPM과 RPM 모두 한도 이내인지 확인
            if (current_tpm + estimated_tokens <= self.max_tpm and 
                current_rpm + 1 <= self.max_rpm):
                break
                
            # 대기 시간 계산 (1초 단위로 대기)
            await asyncio.sleep(1.0)
    
    async def chat_completion(self, 
                             model: str, 
                             messages: List[Dict[str, str]], 
                             temperature: float = 0.7,
                             max_tokens: int = 1500,
                             **kwargs) -> ChatCompletion:
        """
        OpenAI 채팅 완성 API 호출 (속도 제한 처리)
        """
        # response_format이 BaseModel 클래스인 경우 beta.chat.completions.parse로 처리
        if 'response_format' in kwargs and isinstance(kwargs['response_format'], type) and issubclass(kwargs['response_format'], BaseModel):
            return await self.chat_completion_with_parse(model, messages, temperature, max_tokens, **kwargs)
            
        # 예상 토큰 수 계산
        estimated_tokens = self._calculate_tokens(messages, max_tokens)
        
        # 처리 용량이 생길 때까지 대기
        await self._wait_for_capacity(estimated_tokens)
        
        # 세마포어로 동시 요청 제한
        async with self.semaphore:
            current_time = time.time()
            retry_count = 0
            backoff_time = self.initial_backoff
            
            while retry_count <= self.max_retries:
                try:
                    # API 요청 시도
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    
                    # 실제 사용된 토큰 수 추적
                    actual_tokens = (
                        response.usage.prompt_tokens + 
                        response.usage.completion_tokens
                    )
                    
                    # 요청 및 토큰 사용 기록
                    self.request_timestamps.append(current_time)
                    self.token_usage.append({
                        'timestamp': current_time,
                        'tokens': actual_tokens
                    })
                    
                    return response
                    
                except Exception as e:
                    error_message = str(e)
                    
                    # 속도 제한 오류 처리
                    if 'rate_limit_exceeded' in error_message:
                        retry_count += 1
                        
                        # 마지막 재시도인 경우 예외 발생
                        if retry_count > self.max_retries:
                            logger.error(f"최대 재시도 횟수 초과: {error_message}")
                            raise
                        
                        # 대기 시간 추출 (오류 메시지에서 직접 파싱)
                        wait_time = backoff_time
                        try:
                            # "Please try again in 6.904s" 형식에서 시간 추출
                            import re
                            time_match = re.search(r'try again in (\d+\.\d+)s', error_message)
                            if time_match:
                                wait_time = float(time_match.group(1))
                        except:
                            pass
                        
                        logger.warning(f"속도 제한 도달. {wait_time}초 대기 후 재시도 ({retry_count}/{self.max_retries})")
                        await asyncio.sleep(wait_time)
                        
                        # 다음 재시도를 위한 백오프 시간 증가
                        backoff_time *= self.backoff_multiplier
                    else:
                        # 속도 제한 외 다른 오류는 바로 예외 발생
                        logger.error(f"OpenAI API 오류: {error_message}")
                        raise
                        
    async def chat_completion_with_parse(self, 
                             model: str, 
                             messages: List[Dict[str, str]], 
                             temperature: float = 0.7,
                             max_tokens: int = 1500,
                             **kwargs) -> Any:
        """
        OpenAI 채팅 완성 API 호출 - beta.chat.completions.parse 사용 (Pydantic 모델 직접 파싱)
        """
        # 파라미터에서 response_format 추출
        response_format = kwargs.pop('response_format', None)
        if response_format is None or not isinstance(response_format, type) or not issubclass(response_format, BaseModel):
            raise ValueError("response_format은 반드시 BaseModel 클래스를 상속한 클래스여야 합니다.")
        
        # 예상 토큰 수 계산
        estimated_tokens = self._calculate_tokens(messages, max_tokens)
        
        # 처리 용량이 생길 때까지 대기
        await self._wait_for_capacity(estimated_tokens)
        
        # 세마포어로 동시 요청 제한
        async with self.semaphore:
            current_time = time.time()
            retry_count = 0
            backoff_time = self.initial_backoff
            
            while retry_count <= self.max_retries:
                try:
                    # beta.chat.completions.parse API 사용
                    response = self.client.beta.chat.completions.parse(
                        response_format=response_format,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    )
                    
                    # 토큰 사용량 추정 (정확한 토큰 수를 알 수 없으므로 대략 추정)
                    # 이 함수는 사용량을 직접 반환하지 않으므로 추정해야 함
                    response_str = str(response)
                    response_tokens = len(response_str) // 4
                    
                    # 요청 및 토큰 사용 기록
                    self.request_timestamps.append(current_time)
                    self.token_usage.append({
                        'timestamp': current_time,
                        'tokens': estimated_tokens + response_tokens  # 대략적인 추정값
                    })
                    
                    return response
                    
                except Exception as e:
                    error_message = str(e)
                    
                    # 속도 제한 오류 처리
                    if 'rate_limit_exceeded' in error_message:
                        retry_count += 1
                        
                        # 마지막 재시도인 경우 예외 발생
                        if retry_count > self.max_retries:
                            logger.error(f"최대 재시도 횟수 초과: {error_message}")
                            raise
                        
                        # 대기 시간 추출 (오류 메시지에서 직접 파싱)
                        wait_time = backoff_time
                        try:
                            # "Please try again in 6.904s" 형식에서 시간 추출
                            import re
                            time_match = re.search(r'try again in (\d+\.\d+)s', error_message)
                            if time_match:
                                wait_time = float(time_match.group(1))
                        except:
                            pass
                        
                        logger.warning(f"속도 제한 도달. {wait_time}초 대기 후 재시도 ({retry_count}/{self.max_retries})")
                        await asyncio.sleep(wait_time)
                        
                        # 다음 재시도를 위한 백오프 시간 증가
                        backoff_time *= self.backoff_multiplier
                    else:
                        # 속도 제한 외 다른 오류는 바로 예외 발생
                        logger.error(f"OpenAI API 오류: {error_message}")
                        raise

# 싱글톤 인스턴스
_rate_limiter: Optional[OpenAIRateLimiter] = None

def get_rate_limiter() -> OpenAIRateLimiter:
    """OpenAIRateLimiter 인스턴스 반환 (싱글톤 패턴)"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = OpenAIRateLimiter()
    return _rate_limiter 