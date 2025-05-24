from fastapi import APIRouter
from datetime import datetime
import os

from app.core.logger import app_logger
from app.core.mcp_core import setup_mcp_servers, get_mcp_servers

from app.mcp.dart_mcp import dart

from app.services import interview_service, multi_agent_service
from app.services.gms import cover_letter_service as gms_cover_letter_service
from app.schemas import interview, cover_letter, company


logger = app_logger

router = APIRouter(prefix="/tests", tags=["tests"])

@router.get("")
async def tests():
    return {"message": "Hello, World!"}

@router.get("/mcp-servers")
async def test_mcp_servers():
    # 현재 초기화된 MCP 서버 목록 반환
    current_servers = get_mcp_servers()
    if current_servers:
        return [server.name for server in current_servers]
    
    # 현재 서버가 없는 경우 새로 설정
    servers = await setup_mcp_servers()
    return [server.name for server in servers]

@router.get("/dart-mcp/get_corp_code_by_name")
async def get_corp_code_by_name(company_name: str):
    return await dart.get_corp_code_by_name(company_name)


@router.post("/interview/parse-user-info")
async def test_parse_user_info(request: interview.CreateQuestionRequest):
    result = await interview_service.parse_user_info(request)
    return result


@router.post("/gms")
async def test_gms(user_input: str):
    import os 
    from agents import Agent, Runner, OpenAIChatCompletionsModel
    from openai import AsyncOpenAI
    
    GMS_KEY  = os.getenv("GMS_KEY")
    GMS_API_BASE = os.getenv("GMS_API_BASE")
    
    if not GMS_KEY or not GMS_API_BASE:
        return {"message": "GMS_KEY or GMS_API_BASE is not set"}
    
    gms_client = AsyncOpenAI(api_key=GMS_KEY, base_url=GMS_API_BASE)
    
    logger.info(f"OpenAI Client: {gms_client}")
    logger.info(f"Client api key: {gms_client.api_key}")
    logger.info(f"Client base url: {gms_client.base_url}")
    
    # gms_client 를 사용한 모델 지정 
    gms_model = OpenAIChatCompletionsModel(
        model="gpt-4.1",
        openai_client=gms_client
    )
    
    agent = Agent(
        name="GMS Agent",
        instructions="You are a helpful assistant that can answer questions and help with tasks.",
        model=gms_model
    )
    
    result = await Runner.run(agent, user_input)
    
    # 직렬화 가능한 형태로 결과 변환
    serializable_result = {
        "message": result.response if hasattr(result, "response") else str(result),
        "status": "success"
    }
    
    return serializable_result

@router.post("/gms/mcp-server-config")
async def test_mcp_server_config(server_list: str|list[str] = 'all'):
    
    
    from app.services.gms.utils import gms_utils
    
    logger.info(f"server_list: {server_list}")
    
    try:
        config = await gms_utils._get_mcp_server_config(server_list)
        if server_list == 'all':
            # 딕셔너리 items()를 리스트로 변환하여 반환
            return {key: value for key, value in config}
        else:
            return {"server_list": config}
    except Exception as e:
        return {"error": str(e)}
    
    
@router.post("/gms/cover-letter")
async def test_gms_create_cover_letter(request: cover_letter.CreateCoverLetterRequest):
    cover_letters_result = await gms_cover_letter_service.create_cover_letter_all(request)
    logger.info(f"CreateCoverLetterResponse: {cover_letters_result}")
    return cover_letter.CreateCoverLetterResponse(cover_letters=cover_letters_result)


@router.post("/gms/cover-letter/chat")
async def test_gms_chat_cover_letter(request: cover_letter.ChatCoverLetterRequest):
    response = await gms_cover_letter_service.chat_with_cover_letter_service(request)
    # logger.info(f"ChatCoverLetterResponse: {response}")
    if response["status"] == "success":
        response = cover_letter.ChatCoverLetterResponse(
            status=response["status"],
            user_message=request.user_message,
            ai_message=response["content"]
        )
        logger.info(f"ChatCoverLetterResponse: {response}")
        return response
    else:
        response = cover_letter.ChatCoverLetterResponse(
            status=response["status"],
            user_message=request.user_message,
            ai_message=response["content"]
        )
        logger.info(f"ChatCoverLetterResponse: {response}")
        return response
    



########################################################
# 멀티 에이전트 테스트
########################################################

@router.post("/multi-agent/enhanced")
async def test_multi_agent_enhanced(request: company.CompanyAnalysisRequest):
    """
    개선된 멀티 에이전트 기업 분석 테스트 엔드포인트
    - 상세한 로깅
    - 오류 처리
    - 실행 시간 측정
    - LangSmith 추적 정보
    """
    import time
    from datetime import datetime
    
    start_time = time.time()
    logger.info(f"🚀 멀티 에이전트 기업 분석 시작: {request.company_name}")
    logger.info(f"분석 옵션 - 기본:{request.base}, 추가:{request.plus}, 재무:{request.fin}, SWOT:{request.swot}")
    logger.info(f"사용자 요청: {request.user_prompt}")
    
    try:
        # MCP 서버 상태 확인
        mcp_servers = get_mcp_servers()
        logger.info(f"사용 가능한 MCP 서버: {[server.name for server in mcp_servers] if mcp_servers else '없음'}")
        
        # 멀티 에이전트 실행
        response = await multi_agent_service.company_analysis_multi_agent(
            company_name=request.company_name,
            base=request.base,
            plus=request.plus,
            fin=request.fin,
            swot=request.swot,
            user_prompt=request.user_prompt
        )
        
        execution_time = time.time() - start_time
        logger.info(f"✅ 멀티 에이전트 분석 완료 - 실행시간: {execution_time:.2f}초")
        
        # 응답에 메타데이터 추가
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "company_name": request.company_name,
                         "langsmith_tracking": "활성화" if os.getenv("LANGSMITH_TRACING") else "비활성화",
             "mcp_servers_available": [server.name for server in mcp_servers] if mcp_servers else [],
            "analysis_result": response
        }
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ 멀티 에이전트 분석 실패: {str(e)}")
        logger.error(f"실행시간: {execution_time:.2f}초")
        
        import traceback
        logger.error(f"오류 상세: {traceback.format_exc()}")
        
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": round(execution_time, 2),
            "company_name": request.company_name,
            "error_message": str(e),
            "error_type": type(e).__name__,
                "langsmith_tracking": "활성화" if os.getenv("LANGSMITH_TRACING") else "비활성화",
                "mcp_servers_available": [server.name for server in get_mcp_servers()] if get_mcp_servers() else []
        }

@router.get("/multi-agent/simple/{company_name}")
async def test_multi_agent_simple(company_name: str):
    """
    간단한 멀티 에이전트 테스트 엔드포인트 (GET 요청)
    기본 설정으로 빠른 테스트 수행
    """
    import time
    
    start_time = time.time()
    logger.info(f"🔬 간단 테스트 시작: {company_name}")
    
    try:
        # 기본 설정으로 테스트
        response = await multi_agent_service.company_analysis_multi_agent(
            company_name=company_name,
            base=True,
            plus=True, 
            fin=True,
            swot=True,
            user_prompt=f"{company_name} 기업에 대한 종합 분석을 해주세요."
        )
        
        execution_time = time.time() - start_time
        
        return {
            "status": "success",
            "company_name": company_name,
            "execution_time_seconds": round(execution_time, 2),
            "summary": {
                "company_name": response.company_basic_information.company_name,
                "analysis_summary": response.company_basic_information.company_analysis_summary[:200] + "..." if len(response.company_basic_information.company_analysis_summary) > 200 else response.company_basic_information.company_analysis_summary,
                "has_swot": bool(response.company_basic_information.company_swot.swot_summary),
                "has_finance": bool(response.company_basic_information.company_finance_summary),
                "has_news": bool(response.company_basic_information.company_news_summary)
            },
            "full_result": response
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"❌ 간단 테스트 실패: {str(e)}")
        
        return {
            "status": "error",
            "company_name": company_name,
            "execution_time_seconds": round(execution_time, 2),
                         "error_message": str(e),
             "error_type": type(e).__name__
         }

@router.get("/multi-agent/status")
async def check_multi_agent_status():
    """
    멀티 에이전트 시스템 상태 확인 엔드포인트
    - LangSmith 설정 확인
    - MCP 서버 상태 확인
    - 환경 변수 확인
    """
    import os
    
    try:
        # LangSmith 설정 확인
        langsmith_config = {
            "api_key_set": bool(os.getenv("LANGSMITH_API_KEY")),
            "tracing_enabled": os.getenv("LANGSMITH_TRACING") == "true",
            "project_name": os.getenv("LANGSMITH_PROJECT", "company_analysis_multi_agent")
        }
        
        # MCP 서버 상태 확인
        mcp_servers = get_mcp_servers()
        mcp_status = {
             "servers_available": bool(mcp_servers),
             "server_count": len(mcp_servers) if mcp_servers else 0,
             "server_names": [server.name for server in mcp_servers] if mcp_servers else []
         }
        
        # OpenAI 설정 확인
        openai_config = {
            "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "model": "gpt-4o"
        }
        
        # LangGraph 도구 테스트
        from app.services import multi_agent_service
        try:
            tools, _ = await multi_agent_service.setup_mcp_tools()
            tools_status = {
                "tools_loaded": len(tools),
                "tool_names": [tool.name for tool in tools[:5]]  # 처음 5개만 표시
            }
        except Exception as e:
            tools_status = {
                "tools_loaded": 0,
                "error": str(e)
            }
        
        status = "healthy" if (
            langsmith_config["api_key_set"] and 
            mcp_status["servers_available"] and 
            openai_config["api_key_set"]
        ) else "warning"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "langsmith": langsmith_config,
            "mcp_servers": mcp_status,
            "openai": openai_config,
            "tools": tools_status,
            "recommendations": []
        }
        
    except Exception as e:
        logger.error(f"상태 확인 중 오류: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error_message": str(e),
            "error_type": type(e).__name__
        }

@router.get("/multi-agent/config")
async def get_multi_agent_config():
    """
    멀티 에이전트 설정 정보 반환
    """
    import os
    from app.services import multi_agent_service
    
    try:
        # LangSmith 설정 가이드 가져오기
        config_guide = multi_agent_service.get_langsmith_config_guide()
        
        # 대시보드 URL 생성
        dashboard_url = multi_agent_service.get_langsmith_dashboard_url()
        
        config_info = {
            "langsmith_dashboard": dashboard_url,
            "setup_guide": config_guide,
            "environment_variables": {
                "LANGSMITH_API_KEY": "설정됨" if os.getenv("LANGSMITH_API_KEY") else "설정 필요",
                "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", "false"),
                "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "company_analysis_multi_agent"),
                "OPENAI_API_KEY": "설정됨" if os.getenv("OPENAI_API_KEY") else "설정 필요"
            },
            "workflow_info": {
                "agents": ["기본정보", "공시정보", "뉴스분석", "SWOT분석", "통합분석"],
                "supervisor": "슈퍼바이저",
                "flow": "START → 슈퍼바이저 → [에이전트들] → 통합분석 → END"
            }
        }
        
        return config_info
        
    except Exception as e:
        logger.error(f"설정 정보 조회 중 오류: {str(e)}")
        return {
            "error": str(e),
            "error_type": type(e).__name__
        }