from fastapi import APIRouter
from app.core.mcp_core import setup_mcp_servers, get_mcp_servers
from app.mcp.dart_mcp import dart
from app.services import interview_service
from app.schemas import interview

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