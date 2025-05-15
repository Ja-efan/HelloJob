from fastapi import APIRouter
from datetime import datetime

from app.schemas import company
from app.services.company_analysis_service import company_analysis_all

router = APIRouter(prefix="/company-analysis", tags=["company-analysis"])


@router.post("")
async def company_analysis(request: company.CompanyAnalysisRequest):
    """기업 분석 및 뉴스 데이터를 반환합니다.

    Args:
        request (company.CompanyAnalysisRequest): 기업 분석 요청 정보
    """
    
    company_name = request.company_name
    base = request.base
    plus = request.plus
    fin = request.fin
    user_prompt = request.user_prompt
    
    # company_analysis_all 함수를 호출하여 MCP 서버 설정을 한 번만 수행
    result = await company_analysis_all(company_name, base, plus, fin, user_prompt)
    
    response = {
        "company_name": company_name,  # 기업 명
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),  # 기업 분석 일자 
         
        "company_brand": result["company_brand"],  # 기업 주요 브랜드 및 제품
        "company_vision": result["company_vision"],  # 기업 비전
        "company_analysis": result["company_analysis"],  # 기업 분석 결과
        "company_finance": result["company_finance"],  # 기업 재무 상태
        
        "news_summary": result["news_summary"],  # 기업 뉴스 요약
        "news_urls": result["news_urls"]  # 기업 뉴스 링크
    }
    return response
