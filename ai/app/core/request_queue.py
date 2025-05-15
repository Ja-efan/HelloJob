import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 제네릭 타입 정의
T = TypeVar('T')

@dataclass
class QueuedRequest(Generic[T]):
    """큐에 저장될 요청 객체"""
    id: str
    func: Callable[..., Awaitable[T]]
    args: tuple
    kwargs: Dict[str, Any]
    priority: int  # 우선순위 (낮을수록 우선)
    enqueue_time: float  # 요청이 큐에 들어온 시간
    estimated_tokens: int  # 예상 토큰 수
    future: asyncio.Future  # 결과를 저장할 Future 객체

class APIRequestQueue:
    """
    OpenAI API 요청들을 단일 큐에서 관리하는 클래스
    - 모든 API 요청을 중앙 큐에 넣고 순차적으로 처리
    - 속도 제한 준수를 위한 처리 간격 조절
    - 우선순위 기반 처리 (낮은 숫자가 높은 우선순위)
    """
    
    def __init__(self, 
                 max_tpm: int = 30000,  # 기본 Tier 1 한도
                 max_rpm: int = 500,    # 기본 Tier 1 한도
                 min_interval: float = 0.1):  # 요청 간 최소 간격(초)
        """
        Args:
            max_tpm: 분당 최대 토큰 수
            max_rpm: 분당 최대 요청 수
            min_interval: 연속된 요청 간 최소 간격(초)
        """
        self.max_tpm = max_tpm
        self.max_rpm = max_rpm
        self.min_interval = min_interval
        
        # 요청 큐
        self._queue: List[QueuedRequest] = []
        
        # 처리 상태 및 토큰 사용량 관리
        self.request_timestamps: List[float] = []
        self.token_usage: List[Dict[str, float]] = []
        self.last_request_time: float = 0
        
        # 캐시
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 캐시 유효 시간(초), 5분
        
        # 큐 처리 작업
        self._worker_task = None
        self._running = False
        self._lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """큐 처리 워커 시작"""
        if self._running:
            return
            
        self._running = True
        self._shutdown_event.clear()
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("APIRequestQueue 워커가 시작되었습니다.")
    
    async def stop(self):
        """큐 처리 워커 중지 (진행 중인 작업이 모두 완료될 때까지 대기)"""
        if not self._running:
            return
            
        logger.info("APIRequestQueue 워커를 안전하게 종료합니다...")
        self._running = False
        self._shutdown_event.set()
        
        if self._worker_task:
            try:
                # 워커가 정상적으로 종료될 때까지 대기
                await self._worker_task
            except asyncio.CancelledError:
                logger.warning("APIRequestQueue 워커가 강제로 취소되었습니다.")
            finally:
                self._worker_task = None
                
        logger.info("APIRequestQueue 워커가 중지되었습니다.")
    
    async def enqueue(self, 
                     func: Callable[..., Awaitable[T]], 
                     args: tuple = (),
                     kwargs: Dict[str, Any] = None,
                     priority: int = 10,
                     estimated_tokens: int = 1000,
                     cache_key: Optional[str] = None) -> T:
        """
        API 호출 함수와 매개변수를 큐에 넣고, 결과가 준비되면 반환
        
        Args:
            func: 실행할 비동기 함수
            args: 함수의 위치 인자
            kwargs: 함수의 키워드 인자
            priority: 우선순위 (낮을수록 우선)
            estimated_tokens: 예상 토큰 사용량
            cache_key: 캐시 키 (지정 시 캐싱 사용)
            
        Returns:
            함수의 실행 결과
        """
        if kwargs is None:
            kwargs = {}
            
        # 캐시 확인 (cache_key가 있는 경우)
        if cache_key and cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            expiry_time = cache_entry.get('expiry_time', 0)
            
            # 캐시가 유효한지 확인
            if time.time() < expiry_time:
                logger.info(f"캐시에서 결과 반환: {cache_key}")
                return cache_entry['result']
        
        # 큐에 요청 추가
        request_id = f"{time.time()}_{id(func)}"
        future = asyncio.Future()
        
        request = QueuedRequest(
            id=request_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            enqueue_time=time.time(),
            estimated_tokens=estimated_tokens,
            future=future
        )
        
        async with self._lock:
            self._queue.append(request)
            # 우선순위에 따라 정렬
            self._queue.sort(key=lambda x: x.priority)
        
        # 워커가 실행 중이 아니면 시작
        if not self._running:
            await self.start()
        
        # 결과가 준비될 때까지 대기
        try:
            result = await future
            
            # 캐싱 (cache_key가 있는 경우)
            if cache_key:
                expiry_time = time.time() + self.cache_ttl
                self.cache[cache_key] = {
                    'result': result,
                    'expiry_time': expiry_time
                }
                
                # 캐시 크기 관리 (100개 초과 시 가장 오래된 항목 제거)
                if len(self.cache) > 100:
                    oldest_key = min(self.cache.keys(), 
                                    key=lambda k: self.cache[k]['expiry_time'])
                    del self.cache[oldest_key]
            
            return result
        except asyncio.CancelledError:
            logger.warning(f"요청 {request_id}가 취소되었습니다.")
            raise
    
    async def _prune_old_data(self, current_time: float) -> None:
        """1분 이상 지난 토큰 사용량 및 요청 데이터 제거"""
        one_minute_ago = current_time - 60.0
        
        # 1분 이상 지난 요청 타임스탬프 제거
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if ts > one_minute_ago]
        
        # 1분 이상 지난 토큰 사용 데이터 제거
        self.token_usage = [usage for usage in self.token_usage 
                           if usage.get('timestamp', 0) > one_minute_ago]
    
    def _count_current_tpm(self) -> int:
        """현재 분당 토큰 사용량 계산"""
        return sum(usage.get('tokens', 0) for usage in self.token_usage)
    
    def _count_current_rpm(self) -> int:
        """현재 분당 요청 수 계산"""
        return len(self.request_timestamps)
    
    async def _wait_for_capacity(self, estimated_tokens: int) -> None:
        """현재 사용량에 따라 적절한 시간 대기"""
        while True:
            # 종료 이벤트가 설정된 경우 종료
            if self._shutdown_event.is_set():
                raise asyncio.CancelledError("큐 워커가 종료 중입니다.")
            
            current_time = time.time()
            await self._prune_old_data(current_time)
            
            current_tpm = self._count_current_tpm()
            current_rpm = self._count_current_rpm()
            
            # 마지막 요청 이후 최소 간격 확인
            time_since_last_request = current_time - self.last_request_time
            
            # TPM, RPM 및 최소 간격 모두 확인
            if (current_tpm + estimated_tokens <= self.max_tpm and 
                current_rpm + 1 <= self.max_rpm and
                time_since_last_request >= self.min_interval):
                break
                
            # 대기 시간 결정 (0.1초 간격으로 체크)
            await asyncio.sleep(0.1)
    
    async def _worker(self):
        """큐에서 요청을 처리하는 워커 루프"""
        logger.info("API 요청 큐 워커 시작")
        
        try:
            while self._running or self._queue:  # 종료 신호를 받아도 큐의 모든 항목 처리
                # 종료 이벤트가 설정되었고 큐가 비어있으면 종료
                if self._shutdown_event.is_set() and not self._queue:
                    logger.info("큐가 모두 처리되어 워커를 종료합니다.")
                    break
                
                if not self._queue:
                    # 큐가 비어있으면 잠시 대기
                    await asyncio.sleep(0.1)
                    continue
                
                # 큐에서 다음 요청 가져오기
                async with self._lock:
                    if not self._queue:
                        continue
                    request = self._queue[0]
                    self._queue.pop(0)
                
                try:
                    # 용량이 충분해질 때까지 대기
                    try:
                        await self._wait_for_capacity(request.estimated_tokens)
                    except asyncio.CancelledError:
                        # 종료 중이지만 큐에 있는 작업은 계속 처리
                        if not self._shutdown_event.is_set():
                            raise
                        logger.info("워커가 종료 중이지만 현재 요청은 계속 처리합니다.")
                    
                    # 요청 실행 시간 업데이트
                    self.last_request_time = time.time()
                    self.request_timestamps.append(self.last_request_time)
                    
                    # 요청 실행
                    result = await request.func(*request.args, **request.kwargs)
                    
                    # 토큰 사용량 추정 및 기록
                    response_size = len(str(result)) if result else 0
                    tokens_used = request.estimated_tokens  # 실제로는 응답 크기에 따라 조정할 수 있음
                    
                    self.token_usage.append({
                        'timestamp': self.last_request_time,
                        'tokens': tokens_used
                    })
                    
                    # Future에 결과 설정
                    if not request.future.done():
                        request.future.set_result(result)
                    
                except asyncio.CancelledError as e:
                    if self._shutdown_event.is_set():
                        # 종료 중인데 이 작업이 실패하면 실패로 처리
                        logger.warning(f"워커 종료 중 요청 {request.id}가 취소되었습니다.")
                        if not request.future.done():
                            request.future.set_exception(e)
                    else:
                        # 종료 중이 아니면 다시 큐에 넣음
                        raise
                
                except Exception as e:
                    logger.error(f"요청 처리 중 오류 발생: {str(e)}")
                    if not request.future.done():
                        request.future.set_exception(e)
        
        except asyncio.CancelledError:
            logger.info("API 요청 큐 워커가 취소되었습니다.")
            # 큐에 남아있는 작업들은 처리하지 않고 모두 취소
            await self._cancel_remaining_requests("워커가 취소되었습니다.")
            raise
        
        except Exception as e:
            logger.error(f"API 요청 큐 워커에서 예상치 못한 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 큐에 남아있는 모든 작업에 오류 설정
            await self._cancel_remaining_requests(f"워커 오류: {str(e)}")
    
    async def _cancel_remaining_requests(self, reason: str):
        """큐에 남아있는 모든 요청 취소"""
        async with self._lock:
            for request in self._queue:
                if not request.future.done():
                    request.future.set_exception(
                        asyncio.CancelledError(reason)
                    )
            self._queue.clear()

# 싱글톤 인스턴스
_request_queue: Optional[APIRequestQueue] = None

def get_request_queue() -> APIRequestQueue:
    """APIRequestQueue 인스턴스 반환 (싱글톤 패턴)"""
    global _request_queue
    if _request_queue is None:
        _request_queue = APIRequestQueue()
    return _request_queue

# 애플리케이션 시작/종료 유틸리티 함수
async def init_request_queue():
    """애플리케이션 시작 시 요청 큐 초기화"""
    queue = get_request_queue()
    await queue.start()
    return queue

async def cleanup_request_queue():
    """애플리케이션 종료 시 요청 큐 정리"""
    if _request_queue is not None:
        await _request_queue.stop() 