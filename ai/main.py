from fastapi import FastAPI
from app.api.v1.endpoints import company_analysis, cover_letter, interview, tests
from app.core.request_queue import init_request_queue, cleanup_request_queue
from app.core.mcp_core import init_mcp_servers, cleanup_mcp_servers, get_mcp_servers

app = FastAPI()

API_V1_STR = "/api/v1/ai"

# 라우터 추가
app.include_router(company_analysis.router, prefix=API_V1_STR)
app.include_router(cover_letter.router, prefix=API_V1_STR)
app.include_router(interview.router, prefix=API_V1_STR)
app.include_router(tests.router, prefix=API_V1_STR)

@app.on_event("startup")
async def startup_event():
    """애플리케이션 시작 시 실행되는 이벤트 핸들러"""
    # API 요청 큐 초기화
    await init_request_queue()
    print("API 요청 큐가 초기화되었습니다.")
    
    # MCP 서버 초기화
    try:
        servers = await init_mcp_servers()
        print(f"MCP 서버 {len(servers)}개가 초기화되었습니다.")
    except Exception as e:
        print(f"MCP 서버 초기화 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    """애플리케이션 종료 시 실행되는 이벤트 핸들러"""
    # API 요청 큐 정리
    await cleanup_request_queue()
    print("API 요청 큐가 정리되었습니다.")
    
    # MCP 서버 연결 닫기
    await cleanup_mcp_servers()
    print("MCP 서버 연결이 종료되었습니다.")

@app.get(f"{API_V1_STR}")
async def root():
    return {"message": "Hello Job"} 