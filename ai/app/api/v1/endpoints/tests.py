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