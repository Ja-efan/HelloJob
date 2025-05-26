import json
import os
import hashlib
from pydantic import BaseModel, create_model
from typing import List, Optional, Any, Dict, Tuple
from agents import Agent, WebSearchTool

from app.schemas import company
from app.core.agent_utils import RateLimitedRunner
from app.core.request_queue import get_request_queue
from app.core.mcp_core import get_mcp_servers
from app.core.logger import app_logger

logger = app_logger


# 기업 분석 결과 포맷팅 
async def format_company_analysis(result_obj: Any) -> str:
    """
    company_analysis_dart 함수의 Agent 결과 객체를 파싱하여
    지정된 형식의 문자열로 반환합니다. (default 섹션 제외)
    재무 정보(sales_revenue, operating_profit, net_income, fin 섹션 전체)는 제외합니다.

    Args:
        result_obj: company_analysis_dart 함수의 Agent 최종 결과 객체 (Pydantic 모델 인스턴스).

    Returns:
        파싱된 정보를 포함하는 형식화된 문자열.
    """
    if not result_obj:
        return "분석 결과를 찾을 수 없습니다."

    output_lines = []

    # 영문 키 -> 한글 항목명 매핑 (default 키는 여기서 제외해도 되지만, 일단 유지)
    key_to_korean = {
        # base 섹션 (재무 정보 제외)
        "business_overview": "사업 개요",
        "main_products_services": "주요 제품/서비스",
        "major_contracts_rd_activities": "주요 계약 및 R&D 활동",
        "other_references": "기타 참고사항",
        # plus 섹션
        "raw_materials_facilities": "원재료 및 설비",
        "sales_order_status": "수주 상황",
        "risk_management_derivatives": "리스크 관리 및 파생상품",
        # 재무 정보 관련 키 (제외 대상)
        "sales_revenue": None,
        "operating_profit": None,
        "net_income": None,
        "total_assets": None,
        "total_liabilities": None,
        "total_equity": None,
        "operating_cash_flow": None,
        "investing_cash_flow": None,
        "financing_cash_flow": None,
        # default 섹션 (파싱에는 사용되지 않음)
        "company_brand": "기업 브랜드",
        "company_vision": "기업 비전"
    }

    # 섹션 처리 함수 (Pydantic 모델 객체 직접 처리 및 재무 정보 제외)
    def process_section(section_obj: Optional[BaseModel], section_title: str):
        # 섹션 객체가 존재하고 None이 아닌지 확인
        if section_obj:
            # 섹션 객체를 딕셔너리로 변환하여 유효한 값이 있는지 확인
            section_data = section_obj.model_dump(exclude_none=True) # None 값 제외
            
            # 재무 정보 제외
            filtered_data = {}
            for key, value in section_data.items():
                if key_to_korean.get(key) is not None:  # None이 아닌 키만 포함
                    filtered_data[key] = value
                    
            if filtered_data:  # 실제 데이터가 있는 경우에만 섹션 추가
                # 기본 또는 심화 섹션 제목 추가
                output_lines.append(f"(({section_title}))")
                
                # 하위 카테고리 추가
                for key, value in filtered_data.items():
                    korean_key = key_to_korean.get(key, key)  # 매핑 없으면 원래 키 사용
                    output_lines.append(f"({korean_key}) : {{{value}}}")

    # 각 섹션 처리 (default 제외)
    # getattr을 사용하여 해당 속성이 없거나 None일 경우 안전하게 처리
    process_section(getattr(result_obj, 'base', None), "기본")
    process_section(getattr(result_obj, 'plus', None), "심화")
    # fin 섹션은 company_finance에 포함되므로 여기서 제외

    return "\n".join(output_lines)


# 기업 분석 및 뉴스 데이터 분석 결과 반환 -> 현재 사용
async def company_analysis_all(company_name, base, plus, fin, swot, user_prompt):
    """OpenAI Agent 와 MCP를 활용하여 기업 분석 및 뉴스 데이터를 한번에 분석하여 반환합니다.

    Args:
        company_name (str): 기업 이름
        base (bool): 기본 정보 포함 여부
        plus (bool): 추가 정보 포함 여부
        fin (bool): 재무 정보 포함 여부
        swot (bool): SWOT 분석 포함 여부
        user_prompt (str): 사용자 프롬프트
    Returns:
        dict: 기업 분석 및 뉴스 분석 결과를 포함한 딕셔너리
    """
    logger.info(f"company_analysis_all 함수 호출: company_name={company_name}, base={base}, plus={plus}, fin={fin}, swot={swot}, user_prompt={user_prompt}")
    
    # 요청 큐 가져오기
    request_queue = get_request_queue()
    
    # 전역 MCP 서버 목록 가져오기
    mcp_servers = get_mcp_servers()
    
    # 캐시 키 생성 (company_name + 옵션 조합)
    options_hash = hashlib.md5(f"{base}_{plus}_{fin}_{swot}".encode()).hexdigest()
    cache_key_base = f"company_analysis_{company_name}_{options_hash}"
    
    
    # 0. 기업 기본 정보 검색 (주요 제품 및 브랜드, 비전)
    default_info_cache_key = f"{cache_key_base}_default_info"
    
    # 기업 기본 정보 검색 (주요 제품 및 브랜드, 비전)
    async def perform_default_info_search():
        instructions = """당신은 '제트'라는 이름의 AI로, 최신의 기업 정보 검색도구를 찾아 반환합니다.
검색도구: search-webkr, google-search
반환할 기본 정보는 '주요 제품 및 브랜드(서비스)'와 '기업 비전(핵심가치)' 입니다. 
두 정보에 대한 내용을 찾을 때 까지 적절한 도구를 활용하여 검색하세요.
"""
        default_info_agent = Agent(
            name=f"Company Default Info Searcher: {company_name}",
            instructions=instructions,
            model="gpt-4.1",
            output_type=company.CompanyAnalysisDefault,
            mcp_servers=mcp_servers  # 기존에 설정된 MCP 서버 사용
        )
        
        default_info_result = await RateLimitedRunner.run(
            starting_agent=default_info_agent,
            input=f"최신 정보를 활용하여 {company_name} 기업의 기본 정보를 찾아 반환하세요.",
            max_turns=30
        )
        
        return default_info_result.final_output
    
    # 1. DART 기업 분석 수행 (우선순위 높음)
    dart_cache_key = f"{cache_key_base}_dart"
    
    # DART 분석을 위한 비동기 함수 정의
    async def perform_dart_analysis():
        # 분석 타입 설정
        analysis_types = []
        if base:
            analysis_types.append("base")
        if plus:
            analysis_types.append("plus")
        if fin:
            analysis_types.append("fin")
        
        # Pydantic 모델 필드 정의
        model_fields = {
            "used_docs": (List[str], ...),
        }
        
        if "base" in analysis_types:
            model_fields["base"] = (Optional[company.CompanyAnalysisBase], None)
        if "plus" in analysis_types:
            model_fields["plus"] = (Optional[company.CompanyAnalysisPlus], None)
        if "fin" in analysis_types:
            model_fields["fin"] = (Optional[company.CompanyAnalysisFin], None)
            
        # 동적으로 Pydantic 모델 생성
        CompanyAnalysisOutput = create_model('CompanyAnalysisOutput', **model_fields)
        
        instructions = ""
        if user_prompt:
            instructions = f"""당신은 '제트'라는 이름의 AI로, 기업 정보를 분석하고 상세한 리포트를 작성하는 전문가입니다. 
사용자의 요청사항에 적합한 기업분석을 제공하기 위한 전략(sequential-thinking)을 세우고, 다양한 MCP를 활용하여 기업 분석 결과 데이터를 반환합니다. 
사용자 요청사항: {user_prompt}
**반드시 분석 전략을 세운 뒤 분석을 진행하세요.**
            """
        else:
            instructions = """당신은 '제트'라는 이름의 AI로, 기업 정보를 분석하고 상세한 리포트를 작성하는 전문가입니다. 
기업 분석을 제공하기 위한 전략(sequential-thinking)을 세우고, 다양한 MCP를 활용하여 기업 분석 결과 데이터를 반환합니다.
사용자의 요청 사항에 특별한 연도가 있다면 해당 연도를 반영하여 분석하고, 그렇지 않은 경우에는 현재 날짜를 기준으로 최신 데이터를 분석하세요.
**반드시 분석 전략을 세운 뒤 분석을 진행하세요.**"""
            
        # Agent 생성 - 미리 설정된 MCP 서버 사용
        dart_agent = Agent(
            name=f"Company Dart Assistant: {company_name}",
            instructions=instructions,
            model="gpt-4.1",
            output_type=CompanyAnalysisOutput,
            mcp_servers=mcp_servers  # 기존에 설정된 MCP 서버 사용
        )
        
        # 분석 컨텍스트 구성
        dart_context = f"DART API를 활용하여 {company_name}의 기업 분석 내용을 제공하세요.\n"
        dart_context += f"기업명은 반드시 주어진 그대로 사용하세요. 기업명: {company_name}\n"
        dart_context += "분석에 사용한 문서는 문서명과 문서등록일을 포함하여 used_docs에 추가하세요. \n"
        dart_context += "포함할 내용은 다음과 같습니다: \n"
        
        if "base" in analysis_types:
            dart_context += "사업의 개요(business_overview), 주요 제품 및 서비스(main_products_services), 주요계약 및 연구개발활동(major_contracts_rd_activities), 기타 참고사항(other_references), \n"
            dart_context += "매출액(sales_revenue), 영업이익(operating_profit), 당기순이익(net_income), \n"
            
        if "plus" in analysis_types:
            dart_context += "원재료 및 생산설비(raw_materials_facilities), 매출 및 수주상황(sales_order_status), 위험관리 및 파생거래(risk_management_derivatives), \n"
            
        if "fin" in analysis_types:
            dart_context += "자산 총계(total_assets), 부채 총계(total_liabilities), 자본 총계(total_equity), \n"
            dart_context += "영업활동 현금흐름(operating_cash_flow), 투자활동 현금흐름(investing_cash_flow), 재무활동 현금흐름(financing_cash_flow) \n\n" 

        # 정보 부재 시 처리 방법 추가
        dart_context += "해당 정보가 없다면 '정보 없음'이라고 명시적으로 값에 포함하여 출력하세요.\n\n"
        
        # DART 분석 실행
        dart_result = await RateLimitedRunner.run(
            starting_agent=dart_agent, 
            input=dart_context,
            max_turns=30
        )
        
        # 결과 포맷팅
        if not dart_result or not dart_result.final_output:
            return {
                "company_brand": "기업 브랜드 정보를 가져오지 못했습니다.",
                "company_analysis": "기업 분석 정보를 가져오지 못했습니다.",
                "company_vision": "기업 비전 정보를 가져오지 못했습니다.",
                "company_finance": "기업 재정상황 정보를 가져오지 못했습니다.",
            }
        
        # 포맷팅
        formatted_result = await format_company_analysis(dart_result.final_output)
        
        # 재무 정보 포맷팅
        company_finance = ""
        if base or fin:
            finance_lines = []
            
            # base=True인 경우 기본 재무 정보 추가
            if base and hasattr(dart_result.final_output, 'base') and dart_result.final_output.base:
                # 기본 섹션 제목 추가
                finance_lines.append("((기본))")
                
                base_finance = dart_result.final_output.base.model_dump(exclude_none=True)
                if "sales_revenue" in base_finance:
                    finance_lines.append(f"(매출액) : {{{base_finance['sales_revenue']}}}")
                if "operating_profit" in base_finance:
                    finance_lines.append(f"(영업이익) : {{{base_finance['operating_profit']}}}")
                if "net_income" in base_finance:
                    finance_lines.append(f"(당기순이익) : {{{base_finance['net_income']}}}")
                    
            # fin=True인 경우 심화 재무 정보 추가
            if fin and hasattr(dart_result.final_output, 'fin') and dart_result.final_output.fin:
                # 재무 섹션 제목 추가
                finance_lines.append("((재무))")
                
                fin_finance = dart_result.final_output.fin.model_dump(exclude_none=True)
                
                # 재무상태 정보
                if "total_assets" in fin_finance:
                    finance_lines.append(f"(자산 총계) : {{{fin_finance['total_assets']}}}")
                if "total_liabilities" in fin_finance:
                    finance_lines.append(f"(부채 총계) : {{{fin_finance['total_liabilities']}}}")
                if "total_equity" in fin_finance:
                    finance_lines.append(f"(자본 총계) : {{{fin_finance['total_equity']}}}")
                    
                # 현금흐름 정보
                if "operating_cash_flow" in fin_finance:
                    finance_lines.append(f"(영업활동 현금흐름) : {{{fin_finance['operating_cash_flow']}}}")
                if "investing_cash_flow" in fin_finance:
                    finance_lines.append(f"(투자활동 현금흐름) : {{{fin_finance['investing_cash_flow']}}}")
                if "financing_cash_flow" in fin_finance:
                    finance_lines.append(f"(재무활동 현금흐름) : {{{fin_finance['financing_cash_flow']}}}")
                    
            company_finance = "\n".join(finance_lines) if finance_lines else "재무 정보가 포함되지 않았습니다."

        # 결과 객체 및 속성 접근 시 None 확인 추가
        company_brand = "정보 없음"
        company_vision = "정보 없음"
        if hasattr(dart_result.final_output, 'default') and dart_result.final_output.default:
            company_brand = dart_result.final_output.default.company_brand or "정보 없음"
            company_vision = dart_result.final_output.default.company_vision or "정보 없음"

        return {
            "company_brand": company_brand,
            "company_analysis": formatted_result,
            "company_vision": company_vision,
            "company_finance": company_finance,
        }
    
    # 2. 뉴스 데이터 분석 (우선순위 낮음)
    news_cache_key = f"{cache_key_base}_news"
    
    # 뉴스 분석을 위한 비동기 함수 정의
    async def perform_news_analysis():
        
        
        instructions = ""
        if user_prompt:
            instructions += f"""당신은 '제트'라는 이름의 AI로, 기업 뉴스를 분석하고 요약하는 전문가입니다. 다양한 MCP를 활용하여 사용자의 요청 사항을 반영하는 뉴스 기사 분석 결과를 반환합니다.
            사용자 요청사항: {user_prompt}"""
        else:
            instructions = f"당신은 '제트'라는 이름의 AI로, 기업 뉴스를 분석하고 요약하는 전문가입니다. 다양한 MCP를 활용하여 뉴스 기사 분석 결과를 반환합니다."
        
        # Agent 생성 - 미리 설정된 MCP 서버 사용
        news_agent = Agent(
            name=f"Company News Analyzer: {company_name}",
            instructions=instructions,
            model="gpt-4.1",
            output_type=company.CompanyNews,
            mcp_servers=mcp_servers  # 기존에 설정된 MCP 서버 사용
        )
        
        # 분석 컨텍스트 구성
        num_news = 30
        news_context = f"뉴스 데이터를 활용하여 {company_name} 기업의 기업 분석 내용을 제공하세요. 뉴스 기사는 최신 순으로 {num_news}개를 가져오며, 뉴스 제목이 중복되는 경우에는 제외합니다. "
        news_context += "포함할 내용은 다음과 같습니다: \n"
        news_context += "summary, urls"
        
        # 뉴스 분석 실행
        news_result = await RateLimitedRunner.run(
            starting_agent=news_agent,
            input=news_context,
            max_turns=30
        )
        
        # 결과 포맷팅
        if not news_result or not news_result.final_output:
            return {
                "summary": "뉴스 요약 정보를 가져오지 못했습니다.",
                "urls": []
            }
        
        return {
            "summary": news_result.final_output.summary or "뉴스 요약 정보가 없습니다.",
            "urls": news_result.final_output.urls or []
        }
        

    # 3. SWOT 분석
    swot_cache_key = f"{cache_key_base}_swot"
    
    # SWOT 분석을 위한 비동기 함수 정의
    async def perform_swot_analysis():
        from pydantic import BaseModel, Field
        from typing import List, Dict, Optional

        # 단순화된 SWOT 분석 모델 정의 (동적으로)
        class SimpleSwot(BaseModel):
            strengths: List[str] = Field(default_factory=list, description="기업의 강점 목록")
            weaknesses: List[str] = Field(default_factory=list, description="기업의 약점 목록")
            opportunities: List[str] = Field(default_factory=list, description="기업의 기회 목록")
            threats: List[str] = Field(default_factory=list, description="기업의 위협 목록")
            strength_tags: List[str] = Field(default_factory=list, description="강점 관련 태그 목록")
            weakness_tags: List[str] = Field(default_factory=list, description="약점 관련 태그 목록")
            opportunity_tags: List[str] = Field(default_factory=list, description="기회 관련 태그 목록")
            threat_tags: List[str] = Field(default_factory=list, description="위협 관련 태그 목록")
            swot_summary: Optional[str] = Field(None, description="SWOT 분석 종합 요약")
        
        instructions = ""
        if user_prompt:
            instructions += f"""당신은 '제트'라는 이름의 AI로, 기업 SWOT 분석을 하는 전문가입니다.
            
swot_analysis MCP를 활용하여 사용자의 요청 사항을 반영하는 SWOT 분석 결과를 반환합니다.

**반드시 swot_analysis MCP를 활용하여 분석하세요.**

사용자 요청사항: {user_prompt}"""
        else:
            instructions = f"""당신은 '제트'라는 이름의 AI로, 기업 SWOT 분석을 하는 전문가입니다. swot_analysis MCP를 활용하여 SWOT 분석 결과를 반환합니다.

**반드시 swot_analysis MCP를 활용하여 분석하세요.**
"""
        
        # Agent 생성 - 미리 설정된 MCP 서버 사용
        swot_agent = Agent(
            name=f"Company SWOT Analyzer: {company_name}",
            instructions=instructions,
            model="gpt-4.1",
            output_type=SimpleSwot,
            mcp_servers=mcp_servers  # 기존에 설정된 MCP 서버 사용
        )
        
        swot_context = f"최신 정보를 활용하여 {company_name} 기업을 SWOT 분석하고 결과를 반환하세요."
        
        try:
            swot_result = await RateLimitedRunner.run(
                starting_agent=swot_agent,
                input=swot_context,
                max_turns=30
            )
            
            if not swot_result or not swot_result.final_output:
                raise ValueError("SWOT 분석 결과가 없습니다.")
                
            # 출력 형식을 원래 형식으로 변환
            result = swot_result.final_output
            return {
                "strengths": {
                    "contents": result.strengths,
                    "tags": result.strength_tags
                },
                "weaknesses": {
                    "contents": result.weaknesses,
                    "tags": result.weakness_tags
                },
                "opportunities": {
                    "contents": result.opportunities,
                    "tags": result.opportunity_tags
                },
                "threats": {
                    "contents": result.threats,
                    "tags": result.threat_tags
                },
                "swot_summary": result.swot_summary or ""
            }
        except Exception as e:
            logger.error(f"SWOT 분석 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 값 반환
            return {
                "strengths": {
                    "contents": [],
                    "tags": []
                },
                "weaknesses": {
                    "contents": [],
                    "tags": []
                },
                "opportunities": {
                    "contents": [],
                    "tags": []
                },
                "threats": {
                    "contents": [],
                    "tags": []
                },
                "swot_summary": f"SWOT 분석 중 오류가 발생했습니다: {str(e)}"
            }
        
    
    # 3. 큐를 통해 분석 작업 실행 (DART 분석 우선순위 높게)
    try:
        # 세 작업을 큐에 넣고 실행
        default_info_result_task = request_queue.enqueue(
            perform_default_info_search,
            priority=1,  # 낮은 숫자 = 높은 우선순위
            estimated_tokens=10000,
            cache_key=default_info_cache_key
        )
        
        dart_result_task = request_queue.enqueue(
            perform_dart_analysis,
            priority=5,  # 낮은 숫자 = 높은 우선순위
            estimated_tokens=20000,
            cache_key=dart_cache_key
        )
        
        news_result_task = request_queue.enqueue(
            perform_news_analysis,
            priority=10,  # 낮은 숫자 = 높은 우선순위
            estimated_tokens=10000,
            cache_key=news_cache_key
        )
        if swot:
            swot_result_task = request_queue.enqueue(
                perform_swot_analysis,
                priority=15,  # 낮은 숫자 = 높은 우선순위
                estimated_tokens=20000,
                cache_key=swot_cache_key
            )
        
        
        # 모든 작업 완료 대기
        default_info_result = await default_info_result_task
        dart_result = await dart_result_task
        dart_result["company_brand"] = default_info_result.company_brand
        dart_result["company_vision"] = default_info_result.company_vision
        news_result = await news_result_task
        if swot:
            swot_result = await swot_result_task
        else:
            swot_result = {
                "strengths": {
                    "contents": [],
                    "tags": []
                },
                "weaknesses": {
                    "contents": [],
                    "tags": []
                },
                "opportunities": {
                    "contents": [],
                    "tags": []
                },
                "threats": {
                    "contents": [],
                    "tags": []
                },
                "swot_summary": ""
            }
        
            
        # 4. 결과 합치기
        response = {
            "default": default_info_result,
            **dart_result,
            "news_summary": news_result["summary"],
            "news_urls": news_result["urls"],
            "swot": swot_result
        }
        
        return response
        
    except Exception as e:
        # 오류 발생 시 기본 결과 반환
        import traceback
        traceback.print_exc()
        
        # 기본 결과
        return {
            "company_brand": "처리 중 오류 발생",
            "company_analysis": f"기업 분석 중 오류가 발생했습니다: {str(e)}",
            "company_vision": "처리 중 오류 발생",
            "company_finance": "처리 중 오류 발생",
            "news_summary": "처리 중 오류 발생",
            "news_urls": [],
            "swot": {
                "strengths": {
                    "contents": [],
                    "tags": []
                },
                "weaknesses": {
                    "contents": [],
                    "tags": []
                },
                "opportunities": {
                    "contents": [],
                    "tags": []
                },
                "threats": {
                    "contents": [],
                    "tags": []
                },
                "swot_summary": ""
            }
        }