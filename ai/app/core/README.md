# OpenAI API 속도 제한(Rate Limit) 관리 유틸리티

OpenAI API를 사용할 때 발생하는 속도 제한(Rate Limit) 오류를 효과적으로 관리하기 위한 유틸리티 모듈입니다.

## 기능

- OpenAI API 요청에 대한 속도 제한(TPM, RPM) 관리
- 속도 제한 초과 시 자동 대기 및 재시도
- 토큰 사용량 추적 및 관리
- 에러 메시지에서 대기 시간 자동 추출
- Assistants API(Agent, Runner)에 대한 래핑 클래스 제공

## 구성 요소

### 1. OpenAIRateLimiter

OpenAI API 호출에 대한 속도 제한을 관리하는 핵심 클래스입니다.

#### 주요 특징:
- TPM(Tokens Per Minute) 및 RPM(Requests Per Minute) 추적
- 속도 제한 초과 시 자동 대기
- 지수 백오프(Exponential Backoff) 알고리즘으로 재시도
- 토큰 및 요청 수 최적화

### 2. RateLimitedRunner

OpenAI Assistants API의 Runner를 래핑하여 속도 제한을 적용하는 클래스입니다.

#### 주요 특징:
- Agent 호출 전 토큰 사용량 추정
- 용량이 충분할 때까지 대기
- 오류 발생 시 재시도 로직

## 사용 방법

### Direct API 호출

```python
from app.core.openai_utils import get_rate_limiter

async def call_openai_api():
    rate_limiter = get_rate_limiter()
    
    response = await rate_limiter.chat_completion(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "시스템 메시지"},
            {"role": "user", "content": "사용자 메시지"}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    return response
```

### Assistants API 호출

```python
from app.core.agent_utils import RateLimitedRunner

async def call_assistants_api(agent, input_text):
    result = await RateLimitedRunner.run(
        starting_agent=agent,
        input=input_text,
        max_turns=30
    )
    
    return result
```

## 구성 파일

- `openai_utils.py`: OpenAI API 직접 호출에 대한 속도 제한 관리
- `agent_utils.py`: Assistants API(Agent, Runner)에 대한 속도 제한 관리

## 주요 매개변수

- `max_tpm`: 분당 최대 토큰 수 (기본값: 30,000, Tier 1 한도)
- `max_rpm`: 분당 최대 요청 수 (기본값: 500, Tier 1 한도)
- `max_retries`: 최대 재시도 횟수 (기본값: 5)
- `initial_backoff`: 초기 재시도 대기 시간(초) (기본값: 1.0)
- `backoff_multiplier`: 재시도 시 대기 시간 증가 배수 (기본값: 2.0)

## 에러 처리

속도 제한 오류(`rate_limit_exceeded`)가 발생하면 다음 단계로 처리됩니다:

1. 오류 메시지에서 대기 시간 추출 (예: "Please try again in 6.904s")
2. 추출된 시간만큼 대기 후 재시도
3. 최대 재시도 횟수까지 반복
4. 재시도 실패 시 예외 발생 