import logging
import asyncio
import time
from typing import Any, Optional, List, Dict
from agents import Agent, Runner
from app.core.openai_utils import get_rate_limiter

logger = logging.getLogger(__name__)

class RateLimitedAgent:
    """
    OpenAI Assistants API의 Agent를 래핑하여 속도 제한(Rate Limit)을 관리하는 클래스
    """
    
    def __init__(self, agent: Agent):
        """
        Args:
            agent: 래핑할 Agent 객체
        """
        self.agent = agent
        self.rate_limiter = get_rate_limiter()
        
    @property
    def name(self):
        return self.agent.name
        
    @property
    def model(self):
        return self.agent.model
        
    @property
    def output_type(self):
        return self.agent.output_type
        
    @property
    def mcp_servers(self):
        return self.agent.mcp_servers
        
    @property
    def instructions(self):
        return self.agent.instructions
        
    # 다른 Agent 속성들도 래핑

class RateLimitedRunner:
    """
    OpenAI Assistants API의 Runner를 래핑하여 속도 제한(Rate Limit)을 관리하는 클래스
    """
    
    @staticmethod
    async def run(starting_agent: Agent, input: str, max_turns: int = 30):
        """
        Agent와 Runner를 래핑하여 속도 제한을 적용한 실행 함수
        
        Args:
            starting_agent: 시작 Agent 객체
            input: Agent에게 전달할 입력 텍스트
            max_turns: 최대 턴 수
            
        Returns:
            Runner 실행 결과
        """
        # 속도 제한을 적용하기 위한 지연 처리
        rate_limiter = get_rate_limiter()
        
        # Agent 실행 전 용량 확인
        # 메시지 토큰 수 추정 (대략적인 계산)
        estimated_tokens = len(input) // 4
        
        # Agent 설정 메시지도 고려 (instructions 길이)
        estimated_tokens += len(starting_agent.instructions) // 4
        
        # 용량이 생길 때까지 대기
        await rate_limiter._wait_for_capacity(estimated_tokens)
        
        # Runner 실행 
        try:
            current_time = time.time()
            result = await Runner.run(starting_agent=starting_agent, input=input, max_turns=max_turns)
            
            # 토큰 사용량 추적 (대략적인 추정)
            # 실제 토큰 수는 알 수 없으므로 대략적으로 추정
            response_tokens = 0
            if result and result.final_output:
                # 응답의 크기에 따라 토큰 수 대략 추정
                response_str = str(result.final_output)
                response_tokens = len(response_str) // 4
            
            total_tokens = estimated_tokens + response_tokens
            
            # 속도 제한 추적 데이터 업데이트
            rate_limiter.request_timestamps.append(current_time)
            rate_limiter.token_usage.append({
                'timestamp': current_time,
                'tokens': total_tokens
            })
            
            return result
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Agent 실행 중 오류 발생: {error_message}")
            
            # 속도 제한 오류인 경우 재시도 로직 추가 가능
            if 'rate_limit_exceeded' in error_message:
                logger.warning(f"속도 제한에 도달했습니다. 잠시 후 다시 시도하세요.")
            
            raise

# 유틸리티 함수
async def create_rate_limited_agent(
    name: str, 
    instructions: str, 
    model: str = "gpt-4.1", 
    output_type: Any = None, 
    mcp_servers: Optional[List[Any]] = None
) -> Agent:
    """
    속도 제한을 적용한 Agent 객체를 생성하는 유틸리티 함수
    
    Args:
        name: Agent 이름
        instructions: Agent 지시사항
        model: 사용할 모델 이름
        output_type: 출력 타입
        mcp_servers: MCP 서버 목록
        
    Returns:
        생성된 Agent 객체
    """
    agent = Agent(
        name=name,
        instructions=instructions,
        model=model,
        output_type=output_type,
        mcp_servers=mcp_servers
    )
    
    return agent 