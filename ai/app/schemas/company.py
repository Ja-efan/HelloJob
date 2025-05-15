from pydantic import BaseModel
from typing import Optional, List


class CompanyAnalysisRequest(BaseModel):
    company_name: str
    base: bool # 사업 보고서 기본 분석 포함 여부
    plus: bool  # 사업 보고서 심화 분석 포함 여부
    fin: bool  # 재무 정보 포함 여부
    user_prompt: Optional[str] = None  # 사용자 프롬프트


# Default: 기본 제공 정보 
class CompanyAnalysisDefault(BaseModel):
    company_brand: str = None  # 주요 제품 및 브랜드
    company_vision: str = None  # 기업 비전

# Base: 사업 보고서 + 재무 정보
class CompanyAnalysisBase(BaseModel):
    # 사업 보고서 (기본)
    business_overview: str = None  # 사업 개요
    main_products_services: str = None  # 주요 제품 및 서비스
    major_contracts_rd_activities: str = None  # 주요 계약 및 연구개발 활동
    other_references: str = None  # 기타 참고사항
    
    # 재무 정보 (기본)
    sales_revenue: str = None  # 매출액
    operating_profit: str = None  # 영업이익
    net_income: str = None  # 당기순이익


# Plus: 사업 보고서 (심화)
class CompanyAnalysisPlus(BaseModel):
    raw_materials_facilities: str = None  # 원재료 및 생산설비
    sales_order_status: str = None  # 매출 및 수주상황 
    risk_management_derivatives: str = None  # 위험관리 및 파생거래


# Fin: 재무 정보 (심화)
class CompanyAnalysisFin(BaseModel):
    # 재무상태
    total_assets: str = None  # 자산 총계
    total_liabilities: str = None  # 부채 총계
    total_equity: str = None  # 자본 총계
    
    # 현금흐름
    operating_cash_flow: str = None  # 영업활동 현금흐름
    investing_cash_flow: str = None  # 투자활동 현금흐름
    financing_cash_flow: str = None  # 재무활동 현금흐름
    


class CompanyNews(BaseModel):
    summary: str  # 기업 뉴스 요약
    urls: list[str]  # 기업 뉴스 링크 리스트 
    
    
class CompanyAnalysisResponse(BaseModel):
    company_name: str  # 기업 명 
    analysis_date: str  # 기업 분석 일자 
    company_brand: str  # 주요 제품 및 브랜드
    company_analysis: str  # 기업 분석 내용
    company_vision: str  # 기업 비전
    company_finance: str  # 재정상황
    news_summary: str  # 뉴스 기사 분석 요약
    news_urls: list[str]  # 뉴스 링크 리스트 