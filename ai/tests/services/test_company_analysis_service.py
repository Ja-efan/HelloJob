import pytest
from app.services.company_analysis_service import company_analysis_dart

async def test_company_analysis_with_dart():
    # 준비 (Arrange)
    company_name = "삼성전자"
    base = True
    plus = True
    fin = True

    # 실행 (Act)
    result = await company_analysis_dart(company_name, base, plus, fin)

    # 검증 (Assert)
    assert isinstance(result, dict)
    assert "used_docs" in result
    
    # base 정보 검증
    assert "base" in result
    assert isinstance(result["base"], dict)
    assert "business_overview" in result["base"]
    assert "main_products_services" in result["base"]
    assert "major_contracts_rd_activities" in result["base"]
    assert "other_references" in result["base"]
    assert "sales_revenue" in result["base"]
    assert "operating_profit" in result["base"]
    assert "net_income" in result["base"]
    
    # plus 정보 검증
    assert "plus" in result
    assert isinstance(result["plus"], dict)
    assert "raw_materials_facilities" in result["plus"]
    assert "sales_order_status" in result["plus"]
    assert "risk_management_derivatives" in result["plus"]
    
    # fin 정보 검증
    assert "fin" in result
    assert isinstance(result["fin"], dict)
    assert "total_assets" in result["fin"]
    assert "total_liabilities" in result["fin"]
    assert "total_equity" in result["fin"]
    assert "operating_cash_flow" in result["fin"]
    assert "investing_cash_flow" in result["fin"]
    assert "financing_cash_flow" in result["fin"]