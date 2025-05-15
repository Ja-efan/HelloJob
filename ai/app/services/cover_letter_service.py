import json
import hashlib
from app.schemas.cover_letter import *
from app.core.openai_utils import get_rate_limiter
from app.core.request_queue import get_request_queue
import logging

logger = logging.getLogger(__name__)

async def create_cover_letter(
    content: ContentItem, 
    company_analysis: CompanyAnalysis, 
    job_role_analysis: Optional[JobRoleAnalysis] = None) -> CoverLetterItem:
    
    # 요청 큐 가져오기
    request_queue = get_request_queue()
    
    # 캐시 키 생성
    cache_key = f"cover_letter_{content.content_number}_{hashlib.md5(content.content_question.encode()).hexdigest()}"
    
    # 경험 정보 텍스트 구성
    experiences_text = ""
    for idx, exp in enumerate(content.experiences, 1):
        experiences_text += f"""
        경험 {idx}:
        - 경험명: {exp.experience_name}
        - 경험 상세: {exp.experience_detail}
        - 역할: {exp.experience_role or '정보 없음'}
        - 기간: {exp.experience_start_date} ~ {exp.experience_end_date}
        - 클라이언트: {exp.experience_client or '정보 없음'}
        """
    
    # 프로젝트 정보 텍스트 구성
    projects_text = ""
    for idx, proj in enumerate(content.projects, 1):
        projects_text += f"""
        프로젝트 {idx}:
        - 프로젝트명: {proj.project_name}
        - 프로젝트 소개: {proj.project_intro}
        - 역할: {proj.project_role or '정보 없음'}
        - 사용 기술: {proj.project_skills or '정보 없음'}
        - 상세 내용: {proj.project_detail or '정보 없음'}
        - 클라이언트: {proj.project_client or '정보 없음'}
        - 기간: {proj.project_start_date} ~ {proj.project_end_date}
        """
    
    # 프롬프트 구성
    prompt = ""
    if job_role_analysis:
        prompt = f"""다음 정보를 바탕으로 자기소개서 항목에 대한 초안을 작성해주세요:
    
    ## 자기소개서 항목
    - 항목 번호: {content.content_number}
    - 항목 질문: {content.content_question}
    - 글자수 제한: {content.content_length}
    - 요청 사항: {content.content_prompt}
    
    ## 기업 분석
    - 기업명: {company_analysis.company_name}
    - 기업 브랜드: {company_analysis.company_brand}
    - 기업 분석: {company_analysis.company_analysis}
    - 비전: {company_analysis.company_vision}
    - 재무 상태: {company_analysis.company_finance}
    - 뉴스 분석: {company_analysis.news_analysis_data}
    
    ## 직무 분석
    - 직무명: {job_role_analysis.job_role_name}
    - 직무 제목: {job_role_analysis.job_role_title or '정보 없음'}
    - 업무 내용: {job_role_analysis.job_role_work or '정보 없음'}
    - 필요 스킬: {job_role_analysis.job_role_skills or '정보 없음'}
    - 자격 요건: {job_role_analysis.job_role_requirements or '정보 없음'}
    - 우대 사항: {job_role_analysis.job_role_preferences or '정보 없음'}
    - 기타 정보: {job_role_analysis.job_role_etc or '정보 없음'}
    - 직무 카테고리: {job_role_analysis.job_role_category}
    
    ## 지원자 경험 정보
    {experiences_text}
    
    ## 지원자 프로젝트 정보
    {projects_text}
    
    ## 작성 가이드라인
    1. 기업의 비전과 분석 내용에 부합하는 내용으로 작성해주세요.
    2. 직무에 필요한 역량과 스킬을 지원자의 경험/프로젝트와 연결하여 작성해주세요.
    3. 지원자의 경험과 프로젝트 중 해당 직무와 가장 관련성이 높은 내용을 중심으로 작성해주세요.
    4. **지원자가 요청한 사항({content.content_prompt})을 반드시 반영해주세요.**
    5. {content.content_length}자 내외로 작성해주세요.
    6. **자기소개서 내용만을 작성해주세요. 자기소개서 내용 외의 내용은 작성하지 마세요.**
    7. 한국어로 작성해주세요.
    """
    else: 
        prompt = f"""다음 정보를 바탕으로 자기소개서 항목에 대한 초안을 작성해주세요:
    
    ## 자기소개서 항목
    - 항목 번호: {content.content_number}
    - 항목 질문: {content.content_question}
    - 글자수 제한: {content.content_length}
    - 요청 사항: {content.content_prompt}
    
    ## 기업 분석
    - 기업명: {company_analysis.company_name}
    - 기업 브랜드: {company_analysis.company_brand}
    - 기업 분석: {company_analysis.company_analysis}
    - 비전: {company_analysis.company_vision}
    - 재무 상태: {company_analysis.company_finance}
    - 뉴스 분석: {company_analysis.news_analysis_data}
    
    ## 지원자 경험 정보
    {experiences_text}
    
    ## 지원자 프로젝트 정보
    {projects_text}
    
    ## 작성 가이드라인
    1. 기업의 비전과 분석 내용에 부합하는 내용으로 작성해주세요.
    2. 직무에 필요한 역량과 스킬을 지원자의 경험/프로젝트와 연결하여 작성해주세요.
    3. 지원자의 경험과 프로젝트 중 해당 직무와 가장 관련성이 높은 내용을 중심으로 작성해주세요.
    4. **지원자가 요청한 사항({content.content_prompt})을 반드시 반영해주세요.**
    5. {content.content_length}자 내외로 작성해주세요.
    6. **자기소개서 내용만을 작성해주세요. 자기소개서 내용 외의 내용은 작성하지 마세요.**
    7. 한국어로 작성해주세요.
    """
    
    # 실제 OpenAI API 호출을 수행하는 함수
    async def perform_api_call():
        rate_limiter = get_rate_limiter()
        response = await rate_limiter.chat_completion(
            model="gpt-4.1", 
            messages=[
                {"role": "system", "content": "당신은 '제트'라는 이름의 AI로, 전문적인 자기소개서 작성 도우미입니다. 기업과 직무 분석을 바탕으로 지원자의 경험과 프로젝트를 잘 활용하여 맞춤형 자기소개서를 작성해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format=CoverLetterItem
        )
        
        # API 응답에서 자기소개서 내용 추출
        cover_letter = response.choices[0].message.parsed
        return cover_letter
    
    # 요청 큐에 넣고 실행 (캐싱 적용)
    cover_letter = await request_queue.enqueue(
        perform_api_call,
        priority=5,  # 우선순위 (낮을수록 우선)
        estimated_tokens=3000,
        cache_key=cache_key
    )
    
    return cover_letter


async def create_cover_letter_all(request: CreateCoverLetterRequest) -> list[CoverLetterItem]:  
    
    logger.info(f"자기소개서 초안 작성 요청")
    
    # 기업 분석 정보
    company_analysis = request.company_analysis
    # 직무 분석 정보
    job_role_analysis = request.job_role_analysis
    # 자기소개서 항목 정보
    contents = request.contents
    
    # 자기소개서 항목 생성
    cover_letters = []                              
    for idx, content in enumerate(contents, 1):
        
        logger.info(f"자기소개서 항목 {idx} 생성 시작")
        cover_letter = await create_cover_letter(content, company_analysis, job_role_analysis)
        logger.info(f"자기소개서 항목 {idx} 생성 완료")
        cover_letters.append(cover_letter)
        
    return cover_letters


async def parse_edit_suggestion(ai_message: EditSuggestionList) -> str:
    #TODO: 수정 제안 파싱
    
    edit_suggestions_str = ""
    for idx, suggestion in enumerate(ai_message.suggestions, 1):
        edit_suggestions_str += f"========== {idx}번째 수정 제안 ==========\n"
        edit_suggestions_str += f"  원본 내용: {suggestion.original_content}\n"
        edit_suggestions_str += f"  수정 이유: {suggestion.edit_reason}\n"
        edit_suggestions_str += f"  수정 제안: {suggestion.edit_suggestion}\n"
        
    return edit_suggestions_str


async def edit_cover_letter_service(request: EditCoverLetterRequest) -> str:
    
    # 요청 큐 가져오기
    request_queue = get_request_queue()
    
    # 캐시 키 생성
    user_msg_hash = hashlib.md5(request.edit_content.user_message.encode()).hexdigest()
    cache_key = f"edit_cover_letter_{request.edit_content.content_number}_{user_msg_hash}"
    
    # 기업 분석 정보
    company_analysis = request.company_analysis
    # 직무 분석 정보
    job_role_analysis = request.job_role_analysis
    # 경험 정보
    experiences = request.experiences
    # 프로젝트 정보
    projects = request.projects
    # 수정할 내용
    edit_content = request.edit_content
    
    # 경험 정보 텍스트 구성
    experiences_text = ""
    for idx, exp in enumerate(experiences, 1):
        experiences_text += f"""
        경험 {idx}:
        - 경험명: {exp.experience_name}
        - 경험 상세: {exp.experience_detail}
        - 역할: {exp.experience_role or '정보 없음'}
        - 기간: {exp.experience_start_date} ~ {exp.experience_end_date}
        - 클라이언트: {exp.experience_client or '정보 없음'}
        """
    
    # 프로젝트 정보 텍스트 구성
    projects_text = ""
    for idx, proj in enumerate(projects, 1):
        projects_text += f"""
        프로젝트 {idx}:
        - 프로젝트명: {proj.project_name}
        - 프로젝트 소개: {proj.project_intro}
        - 역할: {proj.project_role or '정보 없음'}
        - 사용 기술: {proj.project_skills or '정보 없음'}
        - 상세 내용: {proj.project_detail or '정보 없음'}
        - 클라이언트: {proj.project_client or '정보 없음'}
        - 기간: {proj.project_start_date} ~ {proj.project_end_date}
        """
    
    # 프롬프트 구성
    prompt = f"""
    다음 정보를 바탕으로 자기소개서 수정 방향을 제시해주세요:
    
    ## 현재 자기소개서 항목
    - 항목 번호: {edit_content.content_number}
    - 항목 질문: {edit_content.content_question}
    - 글자수 제한: {edit_content.content_length}
    - 현재 내용: {edit_content.cover_letter}
    - 수정 요청사항: {edit_content.user_message}
    
    ## 기업 분석
    - 기업명: {company_analysis.company_name}
    - 기업 브랜드: {company_analysis.company_brand}
    - 기업 분석: {company_analysis.company_analysis}
    - 비전: {company_analysis.company_vision}
    - 재무 상태: {company_analysis.company_finance}
    - 뉴스 분석: {company_analysis.news_analysis_data}
    
    ## 직무 분석
    - 직무명: {job_role_analysis.job_role_name}
    - 직무 제목: {job_role_analysis.job_role_title or '정보 없음'}
    - 업무 내용: {job_role_analysis.job_role_work or '정보 없음'}
    - 필요 스킬: {job_role_analysis.job_role_skills or '정보 없음'}
    - 자격 요건: {job_role_analysis.job_role_requirements or '정보 없음'}
    - 우대 사항: {job_role_analysis.job_role_preferences or '정보 없음'}
    - 기타 정보: {job_role_analysis.job_role_etc or '정보 없음'}
    - 직무 카테고리: {job_role_analysis.job_role_category}
    
    ## 지원자 경험 정보
    {experiences_text}
    
    ## 지원자 프로젝트 정보
    {projects_text}
    
    ## 수정 방향 제시 가이드라인
    1. 사용자의 수정 요청사항을 정확히 반영한 구체적인 수정 방향을 제시해주세요.
    2. 기업의 비전과 분석 내용에 부합하는 방향으로 수정 방향을 제시해주세요.
    3. 직무에 필요한 역량과 스킬을 지원자의 경험/프로젝트와 연결하여 수정 방향을 제시해주세요.
    4. 수정 제안 시 예시 문장을 제시해도 좋습니다. 예시 문장을 작성하는 경우 예시 문장은 정답이 아님을 명시해주세요.
    5. 글자수 제한({edit_content.content_length}자)을 고려한 수정 방향을 제시해주세요.
    6. 항목 질문({edit_content.content_question})에 직접적으로 답하는 방식으로 수정 방향을 제시해주세요.
    7. 한국어로 작성해주세요.
    8. 수정 제안은 다음 JSON 형식으로 작성해주세요:
    [
        {{
            "original_content": "수정이 필요한 원본 자기소개서의 특정 부분",
            "edit_reason": "수정 이유",
            "edit_suggestion": "수정 제안 내용 및 예시 문장"
        }},
        ...
    ]
    
    ## 중요 사항
    - 직접적인 수정은 절대 하지 마세요. 자기소개서의 어떤 부분을 어떤 방향으로 수정하면 좋을지에 대한 제안만 해주세요.
    - "이렇게 써보세요"와 같은 직접적인 텍스트 제시가 아닌, "이런 방향으로 수정하면 좋을 것 같습니다"와 같은 방향성 제시를 해주세요.
    - 각 수정 제안(edit_suggestion)에는 구체적인 문장이 아닌 수정 방향과 고려 사항만 작성해주세요.
    """
    
    # 실제 OpenAI API 호출을 수행하는 함수
    async def perform_api_call():
        rate_limiter = get_rate_limiter()
        response = await rate_limiter.chat_completion(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "당신은 '제트'라는 이름의 AI로, 전문적인 자기소개서 수정 도우미입니다. 기업과 직무 분석을 바탕으로 지원자의 경험과 프로젝트를 잘 활용하여 맞춤형 자기소개서 수정 방향을 제시해주세요. 직접적인 수정은 절대 하지 말고 수정 방향만 제안해주세요. 수정 제안은 반드시 JSON 형식으로 작성해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format=EditSuggestionList
        )
        
        # API 응답 JSON 추출 및 파싱
        suggestion_json = response.choices[0].message.content.strip()
        return suggestion_json
    
    # 요청 큐에 넣고 실행 (캐싱 적용)
    suggestion_json = await request_queue.enqueue(
        perform_api_call,
        priority=5,  # 우선순위 (낮을수록 우선)
        estimated_tokens=3000,
        cache_key=cache_key
    )
    
    # JSON 파싱 및 변환
    edit_suggestions = json.loads(suggestion_json)
    
    print(f"********** edit_suggestions: {edit_suggestions}")
    
    # 수정 제안 변환
    suggestions_list = EditSuggestionList(suggestions=[
        EditSuggestion(
            original_content=suggestion.get("original_content", ""),
            edit_reason=suggestion.get("edit_reason", ""),
            edit_suggestion=suggestion.get("edit_suggestion", "")
        )
        for suggestion in edit_suggestions["suggestions"]
    ])
    
    # 포맷팅된 결과 반환
    ai_message_str = await parse_edit_suggestion(suggestions_list)
    
    return ai_message_str


async def get_chat_system_prompt(chat_type: str, request: ChatCoverLetterRequest) -> str:
    
    # chat_history 파싱
    parsed_chat_history = ""
    if request.chat_history: # chat_history가 None이거나 비어있지 않은 경우에만 파싱
        for entry in request.chat_history:
            # Pydantic 모델의 필드에 직접 접근
            sender = entry.sender if hasattr(entry, 'sender') else "unknown"
            message = entry.message if hasattr(entry, 'message') else ""
            if sender == "user_message":
                parsed_chat_history += f"사용자: {message}\\n"
            elif sender == "ai_message":
                parsed_chat_history += f"AI: {message}\\n"
            else:
                parsed_chat_history += f"{sender}: {message}\\n" # 혹시 다른 sender 타입이 있을 경우 대비
    
    base_prompt = f"""
    당신은 '제트'라는 이름의 AI로, 취업 도움 사이트 Hello Job의 자기소개서 작성 도움 AI 어시스턴트입니다. 최근 대화 기록과 추가 정보를 바탕으로 사용자의 메시지에 대한 답변을 작성해주세요.
    
    ## 주의 사항 
    - 추가 정보가 존재하는 경우 추가 정보를 활용하여 적절한 답변을 제공하세요.
    - 추가 정보가 없는 경우 대화 기록을 활용하여 사용자의 메시지에 대한 답변을 작성해주세요.
    
    ## 최근 대화 기록
{parsed_chat_history.strip()}

    """
    
    if chat_type.lower() == "coverletter":
        
        # 기업 분석 정보
        company_analysis = request.company_analysis
        # 직무 분석 정보
        job_role_analysis = request.job_role_analysis
        # 경험 정보
        experiences = request.experiences
        # 프로젝트 정보
        projects = request.projects
        # 수정할 내용
        cover_letter = request.cover_letter
        
        # 경험 정보 텍스트 구성
        experiences_text = ""
        if experiences: # experiences가 None이거나 비어있지 않은 경우
            for idx, exp in enumerate(experiences, 1):
                experiences_text += f"""
            경험 {idx}:
            - 경험명: {exp.experience_name}
            - 경험 상세: {exp.experience_detail}
            - 역할: {exp.experience_role or '정보 없음'}
            - 기간: {exp.experience_start_date} ~ {exp.experience_end_date}
            - 클라이언트: {exp.experience_client or '정보 없음'}
            """
        
        # 프로젝트 정보 텍스트 구성
        projects_text = ""
        if projects: # projects가 None이거나 비어있지 않은 경우
            for idx, proj in enumerate(projects, 1):
                projects_text += f"""
            프로젝트 {idx}:
            - 프로젝트명: {proj.project_name}
            - 프로젝트 소개: {proj.project_intro}
            - 역할: {proj.project_role or '정보 없음'}
            - 사용 기술: {proj.project_skills or '정보 없음'}
            - 상세 내용: {proj.project_detail or '정보 없음'}
            - 클라이언트: {proj.project_client or '정보 없음'}
            - 기간: {proj.project_start_date} ~ {proj.project_end_date}
            """
        
        # 프롬프트 구성
        # edit_content가 None일 경우를 대비하여 None 체크 추가
        additional_info_prompt = "## 추가 정보\\n\\n"
        if cover_letter:
            additional_info_prompt += f"""
        ### 현재 자기소개서 항목
        - 항목 질문: {cover_letter.content_question if cover_letter.content_question else '정보 없음'}
        - 글자수 제한: {cover_letter.content_length if cover_letter.content_length else '정보 없음'}
        - 현재 내용: {cover_letter.cover_letter if cover_letter.cover_letter else '정보 없음'}
        """
        else:
            additional_info_prompt += "### 현재 자기소개서 항목 정보 없음\\n"

        if company_analysis:
            additional_info_prompt += f"""
        ### 기업 분석
        - 기업명: {company_analysis.company_name or '정보 없음'}
        - 기업 브랜드: {company_analysis.company_brand or '정보 없음'}
        - 기업 분석: {company_analysis.company_analysis or '정보 없음'}
        - 비전: {company_analysis.company_vision or '정보 없음'}
        - 재무 상태: {company_analysis.company_finance or '정보 없음'}
        - 뉴스 분석: {company_analysis.news_analysis_data or '정보 없음'}
        """
        else:
            additional_info_prompt += "### 기업 분석 정보 없음\\n"
            
        if job_role_analysis:
            additional_info_prompt += f"""
        ### 직무 분석
        - 직무명: {job_role_analysis.job_role_name or '정보 없음'}
        - 직무 제목: {job_role_analysis.job_role_title or '정보 없음'}
        - 업무 내용: {job_role_analysis.job_role_work or '정보 없음'}
        - 필요 스킬: {job_role_analysis.job_role_skills or '정보 없음'}
        - 자격 요건: {job_role_analysis.job_role_requirements or '정보 없음'}
        - 우대 사항: {job_role_analysis.job_role_preferences or '정보 없음'}
        - 기타 정보: {job_role_analysis.job_role_etc or '정보 없음'}
        - 직무 카테고리: {job_role_analysis.job_role_category or '정보 없음'}
        """
        else:
            additional_info_prompt += "### 직무 분석 정보 없음\\n"

        if experiences_text:
            additional_info_prompt += f"""
        ### 지원자 경험 정보
        {experiences_text}
        """
        else:
            additional_info_prompt += "### 지원자 경험 정보 없음\\n"
            
        if projects_text:
            additional_info_prompt += f"""
        ### 지원자 프로젝트 정보
        {projects_text}
        """
        else:
            additional_info_prompt += "### 지원자 프로젝트 정보 없음\\n"
            
        return base_prompt + additional_info_prompt

    else:
        return base_prompt
    
async def get_chat_type(user_message: str) -> str:
    """
    사용자 메시지를 분석하여 대화 타입을 결정하는 함수
    
    Args:
        user_message: str - 사용자의 메시지
    
    """
    
    request_queue = get_request_queue()
    
    system_prompt = """
당신은 '제트'라는 이름의 AI로, 사용자의 메시지를 분석하여 대화의 목적을 파악하는 전문가입니다. 다음 중 가장 적절한 대화 타입을 선택하고, 선택한 타입만을 반환하세요.

대화 타입:
- general: 일상적인 대화 
- coverletter: 자기소개서 관련 대화

주의 사항:
- 대화 타입은 반드시 general 또는 coverletter 중 하나여야 합니다.
- 자기소개서 혹은 취업 관련 내용이 조금이라도 포함되어 있으면 coverletter 타입으로 판단합니다.
"""
    
    async def perform_api_call(
        model="gpt-4.1",
        system_prompt=None,
        user_message=None,
        temperature=0.3,
        max_tokens=100,
        response_format=None):
        
        rate_limiter = get_rate_limiter()
        
        response = await rate_limiter.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format if response_format else None
        )
        
        # API 응답 반환
        return response
    
    chat_type_response = await request_queue.enqueue(
        perform_api_call,
        kwargs={
            "model": "gpt-4.1",
            "system_prompt": system_prompt,
            "user_message": user_message,
            "temperature": 0.3,
            "max_tokens": 100,
            "response_format": ChatTypeResponse
        },
        priority=3,  # 우선순위 (낮을수록 우선)
    )
    chat_type = chat_type_response.choices[0].message.parsed.chat_type
    logger.info(f"chat_type: {chat_type}")
    return chat_type


async def chat_with_cover_letter_service(request: ChatCoverLetterRequest) -> dict:
    """
    자기소개서 관련 채팅 기능을 제공하는 서비스
    
    Args:
        request: ChatCoverLetterRequest - 채팅 요청 정보
        
    Returns:
        str - AI 응답 메시지
    """
    try: 
        logger.info(f"chat_with_cover_letter_service 호출")
        logger.info(f"user_message: {request.user_message}")
        
        request_queue = get_request_queue()
    
        async def perform_api_call(
            model="gpt-4.1", 
            system_prompt="", 
            user_message="", 
            temperature=0.3,
            max_tokens=100, 
            response_format=None):
            
            try: 
                rate_limiter = get_rate_limiter()
            
                response = await rate_limiter.chat_completion(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format if response_format else None
                )
            
                return response 
            except Exception as e:
                logger.error(f"OpenAI API 호출 중 오류 발생: {e}")
                return {
                    "status": "error",
                    "content": f"OpenAI API 호출 중 오류 발생: {e}"
                }
    
        try:
            chat_type = await get_chat_type(request.user_message)
        except Exception as e:
            logger.error(f"chat_type 결정 중 오류 발생: {e}")
            return {
                "status": "error",
                "content": f"chat_type 결정 중 오류 발생: {e}"
            }
        
        try:
            system_prompt_chat = await get_chat_system_prompt(chat_type=chat_type, request=request)
        except Exception as e:
            logger.error(f"system_prompt_chat 생성 중 오류 발생: {e}")
            return {
                "status": "error",
                "content": f"system_prompt_chat 생성 중 오류 발생: {e}"
            }

        try:
            # step2: 프롬프트 분기에 따른 응답 반환
            chat_response = await request_queue.enqueue(
                perform_api_call,
                kwargs={
                    "model": "gpt-4.1",
                    "system_prompt": system_prompt_chat,
                    "user_message": request.user_message,
                    "temperature": 0.3,
                    "max_tokens": 3000,
                },
                priority=3,  # 우선순위 (낮을수록 우선)
            )
            
            # 응답 형식에 따른 파싱
            chat_response_str = chat_response.choices[0].message.content.strip()
            logger.info(f"chat_response_str: {chat_response_str}")
            # 응답 반환
            return {
                "status": "success",
                "content": chat_response_str
            }
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {e}")
            return {
                "status": "error",
                "content": f"응답 생성 중 오류 발생: {e}"
            }
    except Exception as e:
        logger.error(f"cover_letter_service 오류: {e}")
        return {
            "status": "error",
            "content": f"cover_letter_service 오류: {e}"
        }