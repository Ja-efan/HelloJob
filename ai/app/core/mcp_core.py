import json
import os
import logging
from typing import List, Any
from agents.mcp import MCPServerStdio

logger = logging.getLogger(__name__)

async def setup_mcp_servers():
    """MCP 서버 설정 및 연결

    Returns:
        List[MCPServerStdio]: 연결된 MCP 서버 목록
    """
    servers = []
    
    try:
        # mcp.json 파일에서 설정 읽기
        with open('app/services/mcp.json', 'r') as f:
            config = json.load(f)
        
        # 구성된 MCP 서버들을 순회
        for server_name, server_config in config.get('mcpServers', {}).items():
            try:
                # 환경 변수 설정
                env_vars = server_config.get("env", {}).copy()
                
                # 모든 환경 변수를 .env 파일에서 로드
                if "env" in server_config:
                    for key in list(env_vars.keys()):
                        # .env 파일에서 환경 변수 가져오기
                        env_value = os.getenv(key)
                        if env_value:
                            env_vars[key] = env_value
                        else:
                            logger.warning(f"경고: {key} 환경 변수가 .env 파일에 설정되지 않았습니다.")
                
                mcp_server = MCPServerStdio(
                    name=server_name,
                    params={
                        "command": server_config.get("command"),
                        "args": server_config.get("args", []),
                        "env": env_vars
                    },
                    client_session_timeout_seconds=60,
                    cache_tools_list=True
                )
                logger.info(f"MCP 서버 연결 시도: {server_name}")
                logger.info(f"명령어: {server_config.get('command')}")
                logger.info(f"인자: {server_config.get('args', [])}")
                
                await mcp_server.connect()
                servers.append(mcp_server)
            except FileNotFoundError as e:
                logger.error(f"MCP 서버 {server_name} 연결 중 파일을 찾을 수 없음: {e}")
                logger.error(f"찾을 수 없는 파일 경로: {server_config.get('command')} 또는 {server_config.get('args', [])}")
                raise
            except Exception as e:
                logger.error(f"MCP 서버 {server_name} 연결 중 오류 발생: {e}")
                import traceback
                traceback.print_exc()
                raise
    except Exception as e:
        logger.error(f"MCP 서버 설정 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

    return servers

# 싱글톤 인스턴스
_mcp_servers: List[Any] = []

def get_mcp_servers() -> List[Any]:
    """전역 MCP 서버 인스턴스 반환 (싱글톤 패턴)"""
    global _mcp_servers
    return _mcp_servers

def set_mcp_servers(servers: List[Any]) -> None:
    """전역 MCP 서버 인스턴스 설정"""
    global _mcp_servers
    _mcp_servers = servers

# 유틸리티 함수
async def init_mcp_servers() -> List[Any]:
    """애플리케이션 시작 시 MCP 서버 초기화"""
    servers = await setup_mcp_servers()
    set_mcp_servers(servers)
    return servers

async def cleanup_mcp_servers() -> None:
    """애플리케이션 종료 시 MCP 서버 정리"""
    servers = get_mcp_servers()
    for server in servers:
        try:
            await server.disconnect()
        except Exception as e:
            logger.error(f"MCP 서버 연결 종료 중 오류 발생: {e}")
    _mcp_servers.clear() 