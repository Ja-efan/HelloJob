import os
from openai import OpenAI
from agents import Agent, Runner
from app.schemas.cover_letter import *
from app.core.mcp_core import get_mcp_servers
from app.core.logger import app_logger


logger = app_logger


async def create_cover_letter(
    content: ContentItem, 
    company_analysis: CompanyAnalysis, 
    job_role_analysis: Optional[JobRoleAnalysis] = None) -> CoverLetterItem:
    
    # mcp 서버 가져오기
    mcp_servers = get_mcp_servers()
    
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
    
    # SWOT 분석이 비어있는지 확인
    is_swot_empty = False
    if hasattr(company_analysis, 'swot') and company_analysis.swot:
        # SWOT의 각 항목이 모두 빈 배열인지 확인
        is_empty_strengths = True if not company_analysis.swot.strengths else False
        
        is_empty_weaknesses = True if not company_analysis.swot.weaknesses else False
        
        is_empty_opportunities = True if not company_analysis.swot.opportunities else False
        
        is_empty_threats = True if not company_analysis.swot.threats else False
        
        # 모든 항목이 빈 배열이면 SWOT가 비어있다고 판단
        is_swot_empty = is_empty_strengths and is_empty_weaknesses and is_empty_opportunities and is_empty_threats
    else:
        is_swot_empty = True
        
    logger.info(f"is_swot_empty: {is_swot_empty}")
    
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
    - 뉴스 분석: {company_analysis.news_analysis_data}"""
    
        # SWOT가 비어있지 않은 경우에만 추가
        if not is_swot_empty:
            prompt += f"""
    - SWOT 분석:
        * 강점(Strengths): {', '.join(company_analysis.swot.strengths) if company_analysis.swot.strengths else '정보 없음'}
        * 약점(Weaknesses): {', '.join(company_analysis.swot.weaknesses) if company_analysis.swot.weaknesses else '정보 없음'}
        * 기회(Opportunities): {', '.join(company_analysis.swot.opportunities) if company_analysis.swot.opportunities else '정보 없음'}
        * 위협(Threats): {', '.join(company_analysis.swot.threats) if company_analysis.swot.threats else '정보 없음'}
        * 종합 분석: {company_analysis.swot.swot_summary if company_analysis.swot.swot_summary else '정보 없음'}"""
        
        prompt += f"""
    
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
    - 뉴스 분석: {company_analysis.news_analysis_data}"""
    
        # SWOT가 비어있지 않은 경우에만 추가
        if not is_swot_empty:
            prompt += f"""
    - SWOT 분석:
        * 강점(Strengths): {', '.join(company_analysis.swot.strengths) if company_analysis.swot.strengths else '정보 없음'}
        * 약점(Weaknesses): {', '.join(company_analysis.swot.weaknesses) if company_analysis.swot.weaknesses else '정보 없음'}
        * 기회(Opportunities): {', '.join(company_analysis.swot.opportunities) if company_analysis.swot.opportunities else '정보 없음'}
        * 위협(Threats): {', '.join(company_analysis.swot.threats) if company_analysis.swot.threats else '정보 없음'}
        * 종합 분석: {company_analysis.swot.swot_summary if company_analysis.swot.swot_summary else '정보 없음'}"""
            
        prompt += f"""
    
    ## 지원자 경험 정보
    {experiences_text}
    
    ## 지원자 프로젝트 정보
    {projects_text}
"""
    
    # 시스템 프롬프트
    system_prompt = """당신은 한국 채용 시장에 특화된 자기소개서 작성 및 개선 전문 AI 어시스턴트입니다. 사용자가 제공하는 정보(경험, 역량, 지원 회사/직무 등)를 바탕으로 자기소개서 초안을 작성하거나 기존 내용을 개선하는 것을 주된 기능으로 합니다.

**당신의 주요 목표:**

1.  사용자의 경험과 역량을 효과적으로 부각하고, 지원하는 회사 및 직무에 최적의 'fit'을 이루는 내용을 작성/개선합니다.
2.  사용자가 자신의 강점을 발견하고 매력적인 스토리로 발전시키도록 돕습니다.
3.  채용 담당자에게 긍정적인 첫인상을 주고, 사용자의 잠재력과 입사 의지를 효과적으로 전달합니다.
4.  면접 질문으로 자연스럽게 이어질 수 있는 흥미로운 내용을 포함하여 사용자가 면접을 유리하게 이끌도록 지원합니다.

**핵심 원칙 및 제약 조건:**

1.  **작성 스타일 및 어조:**
    * 문장은 간결하고 명확하게, 핵심 위주로 작성해야 합니다 (두괄식 권장).
    * 긍정적이고 적극적인 어조를 유지해야 합니다.
    * 진솔하고 진정성 있는 내용으로 구성하되, 지나친 과장이나 허위 사실은 배제해야 합니다.
    * 표준 한국어 문법 및 맞춤법을 엄격하게 준수해야 합니다.
    * 지원 기업/직무 특성에 맞는 적절한 전문 용어를 사용하되, 남용하지 않도록 주의해야 합니다.
2.  **적합성 및 관련성:** 작성하는 모든 내용이 사용자가 지원하는 회사와 직무의 요구사항 및 문화와 직접적으로 연결되고 부합하는지 지속적으로 확인해야 합니다.

**항목별 작성 지침 및 전략 (내부 지식):**

당신은 자기소개서의 주요 항목별 질문 의도와 각 항목에 대한 최적의 답변 전략에 대한 내부 지식을 갖추고 있습니다. 사용자 요청 시 해당 지침에 따라 내용을 구성합니다.

* **성장 과정:** 지원자의 가치관, 인성, 태도 형성 배경 및 직무/기업 문화 적합성을 파악하는 항목임을 이해하고, 현재 자신에게 영향을 준 핵심 경험/사건 중심으로 서술합니다. 경험을 통해 형성된 가치관/성격/태도가 직무와 어떻게 연결되는지 제시하도록 안내합니다.
* **성격의 장단점:** 자기 객관화 능력, 직무 적합성, 단점 극복 의지 및 성장 가능성을 평가하는 항목임을 인식하고, 장점은 직무 관련 강점과 구체적 사례(경험)로 제시합니다. 단점은 솔직히 인정하되 직무 수행에 치명적이지 않은 요소를 선택하고 구체적인 개선 노력과 결과(혹은 계획)를 제시하여 자기 객관화 능력과 성장 가능성을 동시에 어필하도록 합니다.
* **지원 동기:** 회사/직무에 대한 이해도, 입사 의지, 지원자와 회사 간의 적합성(Fit)을 확인하는 항목임을 이해하고, '왜 이 회사/직무인가?'에 대한 명확하고 설득력 있는 답변을 제시합니다. 기업/직무 심층 분석 기반으로 작성하며, 자신의 경험/역량/가치관과 회사/직무의 연결고리를 구체적으로 설명합니다. 단순 칭찬/막연한 관심이 아닌 '전략적 선택'임을 어필하도록 돕습니다.
* **직무 경험/역량:** 직무 수행 능력, 전문성, 문제 해결 능력, 잠재적 성과 예측을 평가하는 항목임을 인식하고, 직무 직접 관련 경험(인턴, 프로젝트 등) 중심으로 구체적으로 기술합니다. **STAR 기법(Situation 상황, Task 과제, Action 행동, Result 결과)을 핵심적으로 적용**하여 구체적인 사례, 정량적인 성과, 배운 점 등이 명확히 드러나도록 구성해야 합니다. 사용자의 역할, 기여도, 성과(수치화), 배운 점을 강조하여 '과거 성과'를 통해 '미래 기여'를 예측하게 하는 핵심 근거를 제시하도록 돕습니다.
* **입사 후 포부:** 미래 성장 계획, 회사 기여 의지, 직무/회사에 대한 장기적 비전을 확인하는 항목임을 이해하고, 막연한 다짐 대신 구체적이고 실현 가능한 단기(1~3년)/중장기(5년+) 목표와 달성 계획을 제시하도록 안내합니다. 회사 비전/사업 방향과 개인 목표를 연결하고 구체적인 기여 방안을 명시하여, 회사와 함께 성장하겠다는 '약속'을 전달하도록 돕습니다.
* **기타 빈출 항목 (협업, 갈등관리, 문제해결, 창의성, 가치관 등):** 각 항목의 질문 의도를 정확히 파악하고, 그에 맞는 핵심 경험을 선정하여 **STAR 기법을 활용**해 구체적인 상황, 본인의 행동, 결과, 그리고 배운 점을 기술하도록 안내합니다.

**차별화 및 심화 전략:**

1.  **스토리텔링:** 단순 사실 나열을 넘어, 사용자의 경험을 흥미롭고 설득력 있는 스토리로 구성해야 합니다. 'FACT + 생각'을 결합하여 경험의 의미와 교훈을 강조하고, 사용자만의 고유한 관점과 개성이 드러나도록 유도해야 합니다.
2.  **진정성 있는 열정/비전:** 막연한 다짐(예: '열심히 하겠습니다') 대신, 구체적인 행동, 경험, 미래 계획을 통해 열정과 비전을 간접적으로 보여주도록 가이드해야 합니다. 실질적인 기여 방안을 제시하도록 돕습니다.
3.  **실패 경험 작성:** 실패를 통해 배운 점, 성장한 점, 개선 노력을 중심으로 긍정적으로 재해석하여 작성하도록 합니다. 외부 요인 탓으로 돌리지 않고 자기 성찰적 모습을 보여주도록 하며, 직무와 무관하거나 인성적 결함으로 비춰질 수 있는 소재는 피하도록 안내해야 합니다.

**지원 분야 및 경력 수준별 맞춤 전략:**

* **신입 지원자:** 잠재력, 성장 가능성, 학습 의지, 적극성을 강조해야 합니다. 학업, 대외활동, 인턴 등 다양한 경험을 직무와 연결하여 의미를 부여하도록 돕습니다.
* **경력 지원자:** 구체적이고 정량화된 성과 중심으로 전문성을 어필해야 합니다 (즉시 전력감 강조). 긍정적이고 미래지향적인 이직 사유(해당하는 경우) 작성을 유도합니다. 지원하는 회사와 직무에 대한 깊이 있는 통찰력을 보여주도록 돕습니다.
* **산업별 특징 반영:** 사용자가 산업 정보(IT, 금융권, 공기업(NCS 기반), 제조업 등)를 제공하는 경우, 해당 산업의 특성에 맞는 언어를 사용하고 산업별로 중요하게 여기는 역량(예: IT-기술 스택/문제 해결, 금융-신뢰성/윤리, 공기업-NCS 역량/직업윤리, 제조-전문지식/공정 이해)을 부각하도록 내용을 맞춤화해야 합니다. NCS 기반 채용의 경우 직무기술서 기반 역량 및 직업기초능력을 강조하고 블라인드 채용 원칙을 준수해야 합니다.

**최종 점검 및 피해야 할 사항 (내부 점검 및 사용자 가이드):**

당신은 자기소개서 작성 시 자주 발생하는 실수를 인식하고 사용자가 이를 개선하도록 돕는 역할을 수행합니다. 다음의 실수 유형과 피해야 할 표현/어구를 인지하고 작성 과정 및 피드백에 반영해야 합니다.

* **자주 하는 실수 유형:**
    * 회사명/직무명 오류 (제출 전 지원하는 회사/직무에 맞춰 내용을 수정하도록 안내)
    * 오탈자 및 문법 오류 (꼼꼼한 검토와 맞춤법 검사기 활용 권장 안내)
    * 추상적/일반적 표현 남발 (구체적인 경험, 사례, 성과(수치화) 제시 유도)
    * 지원 회사/직무 이해 부족 (기업/직무 심층 분석 필요성 안내)
    * 단순 경험 나열 (의미/배운 점/결과를 연결하도록 안내 - STAR 기법 활용)
    * 수동적 태도 및 자신감 부족 (능동적인 표현 사용 및 강점 자신 있게 제시 유도)
    * 질문 의도 파악 실패 (동문서답 방지, 질문의 핵심에 집중하도록 안내)
    * 과장 및 허위 사실 기재 (사실 기반 진솔한 작성 강조, 면접에서 검증될 수 있음을 주지)
    * 가독성 저하 (짧고 간결한 문장, 두괄식, 적절한 문단 나누기 권장)
    * 진부하고 상투적인 표현 사용 (자신만의 경험/생각 기반 개성 있는 표현 유도)
* **피해야 할 표현 및 어구:** 다음은 피해야 할 표현 또는 어구 유형입니다. 이와 같은 내용 생성을 방지하고 사용자가 사용했을 경우 개선을 제안해야 합니다.
    * 1인칭 주어 남발 ('저는', '제가' 등)
    * 모호한 회사 지칭 ('귀사', '당사') - 공식적인 문맥 외 지양
    * 비공식적 표현 및 줄임말
    * 추측성 어조 ('~인 것 같습니다')
    * 지나치게 포괄적이거나 단정적인 표현 ('무조건', '항상')
    * 개선 노력 없이 부정적인 자기 평가
    * 선천적인 능력만 강조 (노력/경험 기반 역량 강조 필요)
    * 진부하고 상투적인 표현 ('물 맑고 공기 좋은...', '뼈를 묻겠습니다')
    * 연봉, 워라밸 등 조건을 지원 동기의 주된 이유로 제시 (특히 신입)
    
이 모든 지침을 따라 당신은 채용 시장에서 합격 가능성을 높이는 맞춤형이고 전략적인 자기소개서를 작성합니다.

답변은 반드시 자기소개서만 작성합니다. 자기소개서 이외 내용은 포함하면 안됩니다."""

    # 실제 OpenAI API 호출을 수행하는 함수
    async def perform_api_call():
        cover_letter_agent = Agent(
            name="CoverLetter Draft Assistant",
            instructions=system_prompt,
            model="gpt-4.1",
            output_type=CoverLetterItem
        )
        
        response = await Runner.run(
            starting_agent=cover_letter_agent,
            input=prompt,
            max_turns=30
        )
        
        return response.final_output
    
    # 직접 API 호출 실행
    cover_letter = await perform_api_call()
    
    logger.info(f"자기소개서 초안 작성 완료: {cover_letter}")
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
    def perform_api_call():
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": "당신은 '제트'라는 이름의 AI로, 전문적인 자기소개서 수정 도우미입니다. 기업과 직무 분석을 바탕으로 지원자의 경험과 프로젝트를 잘 활용하여 맞춤형 자기소개서 수정 방향을 제시해주세요. 직접적인 수정은 절대 하지 말고 수정 방향만 제안해주세요. 수정 제안은 반드시 JSON 형식으로 작성해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            response_format=EditSuggestionList
        )
        
        # API 응답에서 파싱된 객체 반환
        return response.choices[0].message.parsed
    
    # 직접 API 호출 실행
    suggestions_list = perform_api_call()
    
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
        - 뉴스 분석: {company_analysis.news_analysis_data or '정보 없음'}"""
            
            # SWOT 분석 추가
            if hasattr(company_analysis, 'swot') and company_analysis.swot:
                # SWOT의 각 항목이 모두 빈 배열인지 확인
                is_empty_strengths = True if not company_analysis.swot.strengths else False
                is_empty_weaknesses = True if not company_analysis.swot.weaknesses else False
                is_empty_opportunities = True if not company_analysis.swot.opportunities else False
                is_empty_threats = True if not company_analysis.swot.threats else False
                
                # 모든 항목이 빈 배열이면 SWOT가 비어있다고 판단
                is_swot_empty = is_empty_strengths and is_empty_weaknesses and is_empty_opportunities and is_empty_threats
                
                # SWOT가 비어있지 않은 경우에만 추가
                if not is_swot_empty:
                    additional_info_prompt += f"""
        - SWOT 분석:
            * 강점(Strengths): {', '.join(company_analysis.swot.strengths) if company_analysis.swot.strengths else '정보 없음'}
            * 약점(Weaknesses): {', '.join(company_analysis.swot.weaknesses) if company_analysis.swot.weaknesses else '정보 없음'}
            * 기회(Opportunities): {', '.join(company_analysis.swot.opportunities) if company_analysis.swot.opportunities else '정보 없음'}
            * 위협(Threats): {', '.join(company_analysis.swot.threats) if company_analysis.swot.threats else '정보 없음'}
            * 종합 분석: {company_analysis.swot.swot_summary if company_analysis.swot.swot_summary else '정보 없음'}"""
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
    
    system_prompt = """
당신은 '제트'라는 이름의 AI로, 사용자의 메시지를 분석하여 대화의 목적을 파악하는 전문가입니다. 다음 중 가장 적절한 대화 타입을 선택하고, 선택한 타입만을 반환하세요.

대화 타입:
- general: 일상적인 대화 
- coverletter: 자기소개서 관련 대화

주의 사항:
- 대화 타입은 반드시 general 또는 coverletter 중 하나여야 합니다.
- 자기소개서 혹은 취업 관련 내용이 조금이라도 포함되어 있으면 coverletter 타입으로 판단합니다.
"""
    
    def perform_api_call(
        model="gpt-4.1",
        system_prompt=None,
        user_message=None,
        temperature=0.3,
        max_tokens=100,
        response_format=None):
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        if response_format:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format
            )
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        
        # API 응답 반환
        return response
    
    # 직접 API 호출 실행
    chat_type_response = perform_api_call(
        model="gpt-4.1",
        system_prompt=system_prompt,
        user_message=user_message,
        temperature=0.3,
        max_tokens=100,
        response_format=ChatTypeResponse
    )
    
    # 응답에서 chat_type 추출
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
        
        def perform_api_call(
            model="gpt-4.1", 
            system_prompt="", 
            user_message="", 
            temperature=0.5,
            max_tokens=1500, 
            response_format=None):
            
            try: 
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                if response_format:
                    response = client.beta.chat.completions.parse(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format
                    )
                else:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
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
            chat_response = perform_api_call(
                model="gpt-4.1",
                system_prompt=system_prompt_chat,
                user_message=request.user_message,
                temperature=0.3,
                max_tokens=3000,
                response_format=ChatCoverLetterResponse
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